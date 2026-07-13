"""Apply audio augmentations to make synthetic speech more realistic.

Standard offline ASR augmentation chain (background noise + room reverb +
speaking-rate / pitch / gain / codec perturbation). See docs/AUGMENTATION_DESIGN.md
for the design rationale and how it maps onto the augmentation literature.

The two "real-world acoustics" transforms — additive background noise and room
impulse response (reverb) — only run if the noise bank is populated. Populate it
with `bash scripts/download_noise_bank.sh` (MUSAN + RIRS_NOISES). Pass
`--require-noise` to make a run fail loudly instead of silently skipping them.
"""
import json
import random
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from audiomentations import (
        AddBackgroundNoise,
        ApplyImpulseResponse,
        BandPassFilter,
        Compose,
        Gain,
        Mp3Compression,
        PitchShift,
        TimeStretch,
    )
    _AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    _AUDIOMENTATIONS_AVAILABLE = False

# Mp3Compression shells out through pydub+ffmpeg. Without pydub the whole Compose
# raises mid-loop, leaving a partial output dir — so detect it up front and drop
# just that transform instead of crashing.
try:
    import pydub  # noqa: F401
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False


def _identity(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """No-op augmenter used when audiomentations is not installed."""
    return samples


def _audio_files(directory: Path) -> list:
    """WAVs anywhere under `directory` (recursive — matches audiomentations'
    own recursive search, so structured corpora like MUSAN/RIRS work)."""
    return list(directory.rglob("*.wav")) if directory.exists() else []


def build_augmentation_pipeline(
    noise_bank: str,
    *,
    min_snr_db: float = 10.0,
    max_snr_db: float = 25.0,
    require_noise: bool = False,
):
    """Build the augmentation pipeline.

    Falls back to a no-op if audiomentations is not installed, and skips
    noise/RIR transforms if the noise bank directories are empty (unless
    `require_noise`, in which case it is a hard error).
    """
    if not _AUDIOMENTATIONS_AVAILABLE:
        msg = (
            "audiomentations not installed — augmentation would be a file copy only.\n"
            "  To enable on Apple Silicon: pip install 'audiomentations<0.36' pydub"
        )
        if require_noise:
            raise SystemExit(f"[ERROR] {msg}")
        print(f"[WARN] {msg}")
        return _identity

    transforms = []
    noise_path = Path(noise_bank)

    # Background noise — the single most important "real-world" augmentation.
    noise_dir = noise_path / "ambient"
    if _audio_files(noise_dir):
        transforms.append(
            AddBackgroundNoise(
                sounds_path=str(noise_dir), min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=0.7
            )
        )
    else:
        hint = (
            f"No ambient noise files in {noise_dir}. Background-noise augmentation will NOT run.\n"
            "  Populate it: bash scripts/download_noise_bank.sh"
        )
        if require_noise:
            raise SystemExit(f"[ERROR] {hint}")
        print(f"[WARN] {hint}")

    # Room impulse response (reverb).
    rir_dir = noise_path / "rir"
    if _audio_files(rir_dir):
        transforms.append(ApplyImpulseResponse(ir_path=str(rir_dir), p=0.5))
    else:
        hint = (
            f"No RIR files in {rir_dir}. Reverb augmentation will NOT run.\n"
            "  Populate it: bash scripts/download_noise_bank.sh"
        )
        if require_noise:
            raise SystemExit(f"[ERROR] {hint}")
        print(f"[WARN] {hint}")

    # These need no external files.
    transforms.extend([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    ])
    if _PYDUB_AVAILABLE:
        transforms.append(Mp3Compression(min_bitrate=32, max_bitrate=64, p=0.3))
    else:
        print("[WARN] pydub not installed — skipping Mp3Compression. Install: pip install pydub")
    transforms.append(BandPassFilter(min_center_freq=200, max_center_freq=4000, p=0.2))

    return Compose(transforms)


def _format_param(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def describe_applied(augmenter) -> str:
    """Per-clip provenance: which transforms actually fired this call, and with
    what parameters. Reads each transform's `.parameters` (populated by
    audiomentations after __call__). Returns e.g.
    'AddBackgroundNoise(snr_db=12.4);PitchShift(semitones=-1.3)'. Empty -> 'none'.
    """
    if not _AUDIOMENTATIONS_AVAILABLE or not hasattr(augmenter, "transforms"):
        return "passthrough"
    parts = []
    for t in augmenter.transforms:
        params = getattr(t, "parameters", None) or {}
        if not params.get("should_apply"):
            continue
        kv = ",".join(
            f"{k}={_format_param(v)}"
            for k, v in params.items()
            if k != "should_apply"
        )
        parts.append(f"{type(t).__name__}({kv})" if kv else type(t).__name__)
    return ";".join(parts) if parts else "none"


@click.command()
@click.option("--input-dir", required=True, help="Directory with clean WAVs")
@click.option("--output-dir", required=True, help="Output directory for augmented WAVs")
@click.option("--noise-bank", default="data_generation/03_augmentation/noise_bank", help="Noise bank directory")
@click.option("--variants", default=2, help="Number of augmented variants per clean file")
@click.option("--sample-rate", default=16000, help="Target sample rate")
@click.option("--seed", default=42, help="RNG seed for reproducible augmentation")
@click.option("--min-snr-db", default=10.0, help="Min SNR for background noise (lower = harder/noisier)")
@click.option("--max-snr-db", default=25.0, help="Max SNR for background noise")
@click.option("--require-noise", is_flag=True, help="Fail if the noise bank is empty instead of silently skipping noise/reverb")
def main(input_dir, output_dir, noise_bank, variants, sample_rate, seed, min_snr_db, max_snr_db, require_noise):
    # Seed both RNGs audiomentations draws from, so a run is reproducible.
    random.seed(seed)
    np.random.seed(seed)

    in_path = Path(input_dir)
    out_path = Path(output_dir)

    wav_files = sorted(in_path.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {in_path}")
        return

    # Build (and validate) the pipeline before creating the output dir, so a
    # --require-noise failure leaves no empty output dir behind.
    augmenter = build_augmentation_pipeline(
        noise_bank, min_snr_db=min_snr_db, max_snr_db=max_snr_db, require_noise=require_noise
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # Load clean manifest
    clean_manifest_path = in_path / "manifest_clean.jsonl"
    clean_manifest: dict = {}
    if clean_manifest_path.exists():
        with open(clean_manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                clean_manifest[Path(entry["audio_filepath"]).name] = entry

    augmented_manifest = []

    print(f"Augmenting {len(wav_files)} files × {variants} variants (seed={seed})...")

    for wav_path in tqdm(wav_files):
        audio, sr = sf.read(str(wav_path), dtype="float32")

        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate

        for v in range(variants):
            aug_audio = augmenter(samples=audio, sample_rate=sr)
            applied = describe_applied(augmenter)

            aug_filename = f"{wav_path.stem}_aug{v:02d}.wav"
            aug_path = out_path / aug_filename
            sf.write(str(aug_path), aug_audio, sr)

            base_entry = clean_manifest.get(wav_path.name, {})
            augmented_manifest.append({
                "audio_filepath": str(aug_path),
                "text": base_entry.get("text", ""),
                "duration": len(aug_audio) / sr,
                "language": base_entry.get("language", ""),
                "source": "synthetic",
                "voice_id": base_entry.get("voice_id", ""),
                "augmentation": applied,
                "variant": v,
                "seed": seed,
            })

    manifest_path = out_path / "manifest_augmented.jsonl"
    with open(manifest_path, "w") as f:
        for entry in augmented_manifest:
            f.write(json.dumps(entry) + "\n")

    n_passthrough = sum(1 for e in augmented_manifest if e["augmentation"] in ("passthrough", "none"))
    print(f"Created {len(augmented_manifest)} augmented files. Manifest: {manifest_path}")
    if n_passthrough:
        print(f"[WARN] {n_passthrough} clip(s) had no transform applied (passthrough/none).")


if __name__ == "__main__":
    main()
