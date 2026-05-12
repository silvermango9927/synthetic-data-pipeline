"""Push a bulk-generated language's outputs to a private Hugging Face Datasets repo.

Layout pushed to the HF repo (one repo per language):

    <repo>/
      data/short_clean/audio/*.wav
      data/short_clean/manifest.jsonl       # train rows
      data/short_clean/val.jsonl            # 5% held-out
      data/short_aug/audio/*.wav
      data/short_aug/manifest.jsonl
      data/short_aug/val.jsonl
      data/long_clean/...
      data/long_aug/...
      README.md                              # dataset card

The manifest schema (NeMo-compatible plus a few metadata cols):

    {"audio_filepath": "audio/<filename>.wav",
     "text": "...",
     "duration": 4.35,
     "language": "hi",
     "source": "synthetic",
     "voice_id": "sarvam_neha",
     "augmentation": null | "<transform>"}

Resumability: `upload_folder` diffs against the remote and only sends changed files.
A re-run after partial upload picks up exactly where it left off.

Auth: either run `hf auth login` once, or export `HF_TOKEN` in the env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install with: .venv/bin/pip install -e '.[tts-cloud]'"
    ) from e


VAL_FRACTION = 0.05
LANG_DIRNAME = {"zh": "chinese", "hi": "hindi"}
BUCKETS = ("short", "long")
AUG_STATES = ("clean", "augmented")


def stable_val_flag(audio_filepath: str, val_fraction: float) -> bool:
    """Deterministic per-row train/val assignment based on hash of audio_filepath."""
    h = int(hashlib.md5(audio_filepath.encode()).hexdigest(), 16)
    return (h % 1000) < int(val_fraction * 1000)


def load_manifest_for_bucket(lang_dir: Path, bucket: str, aug_state: str) -> list[dict]:
    """Read all manifests for a (bucket, aug_state) combo.

    aug_state='clean'     → manifest_clean.jsonl from clean/
    aug_state='augmented' → manifest_augmented.jsonl from augmented/
    """
    if aug_state == "clean":
        manifest = lang_dir / bucket / "clean" / "manifest_clean.jsonl"
    else:
        manifest = lang_dir / bucket / "augmented" / "manifest_augmented.jsonl"
    if not manifest.exists():
        print(f"  [WARN] {manifest} missing; skipping {bucket}/{aug_state}")
        return []
    rows = []
    with open(manifest) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def stage_split(
    rows: list[dict],
    audio_src_dir: Path,
    staging_dir: Path,
    val_fraction: float = VAL_FRACTION,
) -> dict:
    """Materialise per-bucket layout under staging_dir.

    Copies are AVOIDED — we symlink the audio files into staging to keep the upload
    deterministic without doubling disk usage. HF's upload_folder follows symlinks.
    """
    audio_out = staging_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for entry in rows:
        src_path_str = entry["audio_filepath"]
        src = Path(src_path_str)
        # Some manifests use absolute, some relative. Resolve both.
        if not src.is_absolute() and not src.exists():
            src = audio_src_dir / src.name
        if not src.exists():
            print(f"  [WARN] missing WAV referenced in manifest: {src_path_str}")
            continue
        wav_name = src.name
        link = audio_out / wav_name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src.resolve(), link)

        # Rewrite audio_filepath to repo-relative path
        new_entry = dict(entry)
        new_entry["audio_filepath"] = f"audio/{wav_name}"
        if stable_val_flag(new_entry["audio_filepath"], val_fraction):
            val_rows.append(new_entry)
        else:
            train_rows.append(new_entry)

    train_path = staging_dir / "manifest.jsonl"
    val_path = staging_dir / "val.jsonl"
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "total_seconds": sum(r.get("duration", 0.0) for r in (train_rows + val_rows)),
    }


def write_readme(staging_root: Path, lang: str, bucket_stats: dict) -> None:
    """Dataset card."""
    total_hours = sum(s["total_seconds"] for s in bucket_stats.values()) / 3600
    lines = [
        "---",
        "language:",
        f"  - {lang}",
        "task_categories:",
        "  - automatic-speech-recognition",
        "size_categories:",
        "  - 1K<n<10K",
        "tags:",
        "  - synthetic",
        "  - tts-generated",
        "---",
        "",
        f"# Synthetic ASR data — `{lang}`",
        "",
        "Generated by [`Valsea-ASR/synthetic-data-pipeline`](https://github.com/Valsea-ASR/synthetic-data-pipeline).",
        "Audio is **synthetic** (TTS), targeted as training data for downstream ASR finetuning.",
        "",
        f"**Total audio: {total_hours:.1f} hr** across short (~5s) and long (~30s) length buckets,",
        "each in clean and augmented variants.",
        "",
        "## Splits",
        "",
        "| Bucket | Aug state | Train | Val | Audio (sec) |",
        "|---|---|---:|---:|---:|",
    ]
    for (bucket, aug_state), s in bucket_stats.items():
        lines.append(
            f"| {bucket} | {aug_state} | {s['train']} | {s['val']} | {s['total_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "## Schema",
        "",
        "Each row in `manifest.jsonl` / `val.jsonl`:",
        "",
        "```json",
        '{"audio_filepath": "audio/<filename>.wav",',
        ' "text": "...",',
        ' "duration": 4.35,',
        f' "language": "{lang}",',
        ' "source": "synthetic",',
        ' "voice_id": "...",',
        ' "augmentation": null | "<transform>"}',
        "```",
        "",
        "Audio is 16 kHz mono WAV. Val split is a deterministic ~5% hash-based hold-out.",
        "",
    ])
    (staging_root / "README.md").write_text("\n".join(lines))


def push(
    repo_id: str,
    staging_root: Path,
    private: bool = True,
    token: str | None = None,
) -> None:
    api = HfApi(token=token)
    try:
        create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
        print(f"  repo ready: https://huggingface.co/datasets/{repo_id}")
    except HfHubHTTPError as e:
        print(f"  [WARN] create_repo: {e}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        commit_message="bulk synthetic ASR data — auto-generated",
        ignore_patterns=[".DS_Store", "*.tmp"],
    )
    print(f"  uploaded: https://huggingface.co/datasets/{repo_id}/tree/main")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=("zh", "hi"), help="Language code")
    ap.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. valsea/synthetic-asr-zh")
    ap.add_argument(
        "--outputs-root",
        default="outputs",
        help="Local outputs root (default: outputs)",
    )
    ap.add_argument(
        "--staging-dir",
        default=None,
        help="Where to materialise the upload tree. Default: outputs/<lang>/_hf_staging",
    )
    ap.add_argument("--public", action="store_true", help="Make the dataset public (default: private)")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage files locally but skip the HF upload step",
    )
    args = ap.parse_args()

    lang_dir = Path(args.outputs_root) / LANG_DIRNAME[args.lang]
    if not lang_dir.exists():
        raise SystemExit(f"No outputs at {lang_dir}; run the prototype scripts first.")

    staging_root = Path(args.staging_dir) if args.staging_dir else lang_dir / "_hf_staging"
    if staging_root.exists():
        # Clean stale staging to avoid stale symlinks; safe — we rebuild from manifests.
        import shutil
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    bucket_stats = {}
    for bucket in BUCKETS:
        for aug_state in AUG_STATES:
            rows = load_manifest_for_bucket(lang_dir, bucket, aug_state)
            if not rows:
                continue
            bucket_staging = staging_root / "data" / f"{bucket}_{aug_state}"
            audio_src = lang_dir / bucket / ("augmented" if aug_state == "augmented" else "clean")
            stats = stage_split(rows, audio_src, bucket_staging, args.val_fraction)
            bucket_stats[(bucket, aug_state)] = stats
            print(
                f"  staged {bucket}/{aug_state}: train={stats['train']}, "
                f"val={stats['val']}, audio={stats['total_seconds']:.0f}s"
            )

    if not bucket_stats:
        raise SystemExit("No data staged — manifests empty or missing.")

    write_readme(staging_root, args.lang, bucket_stats)
    print(f"  staging tree: {staging_root}")

    if args.dry_run:
        print("  --dry-run set, skipping HF upload")
        return

    token = os.environ.get("HF_TOKEN")
    push(repo_id=args.repo_id, staging_root=staging_root, private=not args.public, token=token)


if __name__ == "__main__":
    main()
