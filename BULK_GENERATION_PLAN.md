# Bulk Synthetic Data Generation — Chinese & Hindi (3hr / bucket / aug-state)

## Target outputs

Per (language × length-bucket × aug-state): **~3 hours of 16 kHz mono WAV** with an accompanying NeMo-format JSONL manifest. Aug is **derived 1:1 from clean** (apply 1 augmentation variant to each clean clip).

| Language | Bucket | Clean target | Aug target | Total |
|---|---|---|---|---|
| Chinese (`zh`) | short (~5s) | 3 hr | 3 hr | 6 hr |
| Chinese (`zh`) | long (~30s) | 3 hr | 3 hr | 6 hr |
| Hindi (`hi`) | short (~5s) | 3 hr | 3 hr | 6 hr |
| Hindi (`hi`) | long (~30s) | 3 hr | 3 hr | 6 hr |
| **Total** | | **12 hr** | **12 hr** | **24 hr** |

Audio size budget: 24 hr × ~115 MB/hr = **~2.8 GB total WAVs**.

## Feasibility: is the current pipeline ready?

Yes for the core pipeline, with **five concrete fixes** before kicking off. The setup that produced the 192 min in the last PR scales linearly to this target — the gaps are around **time, resumability, augmentation, and remote storage**.

### Gaps the bulk run exposes

1. **Augmentation today is pass-through.** `audiomentations` isn't installed, so the aug bucket is currently file-copy duplicates of clean. **Must install before bulk run.**
2. **No resumability.** Sarvam credits ran out mid-run last time and lost ~17 long Hindi sentences. The current `synthesize.py` re-synthesises everything on re-run — wastes API spend and time. Need a "skip if output exists" check.
3. **Single-threaded TTS calls.** Sequential edge-tts + Sarvam at this volume will take many hours. Need a small worker pool.
4. **No remote sync.** Outputs only land locally. With ~3 GB of audio, that's too big for git-on-the-PR-branch. Need a push step to HuggingFace Datasets.
5. **Default voice rotation.** Current `random.sample(voices, k=n_voices)` picks 2 of N at random per sentence. At scale, prefer **deterministic round-robin** so every voice gets equal coverage.

## Sentence-count math (overshoot for safety)

Real durations vary; sample medians from the small-scale run were:
- Short clean: ~4.1 s/clip → need ~2,640 clips to hit 3 hr
- Long clean: ~29.5 s/clip → need ~370 clips to hit 3 hr

With 2 voices/sentence and a ~10% overshoot to absorb dedup + duration-sanity rejection:

| Bucket | Sentences per language | Clips/language (×2 voices) | Both languages |
|---|---|---|---|
| short | **1,500** | 3,000 | 6,000 |
| long  | **250**   | 500   | 1,000 |

**Total unique sentences to generate: 3,500. Total clean clips: 7,000. Plus 1:1 aug → 14,000 clips total.**

## Time budget (with parallelism)

| Stage | Sequential | With workers | Notes |
|---|---|---|---|
| OpenAI text gen (350 batch calls) | ~50 min | ~12 min | 4 concurrent batches |
| Chinese TTS (edge-tts, ~3,500 clips) | ~2 hr | ~20–30 min | 6–8 workers, network-bound |
| Hindi TTS (Sarvam, ~3,500 clips) | ~3 hr | ~40–60 min | 4 workers, rate-limit-cautious |
| Augmentation (7,000 clips) | ~1 hr | ~15 min | CPU-local, easily parallel |
| Duration QC + manifest export | ~5 min | ~5 min | fast either way |
| HF Datasets push (~3 GB) | n/a | ~10–20 min | network-bound |
| **End-to-end** | ~7 hr | **~2.5–3.5 hr** | wall-clock on this machine |

## Cost budget

| Item | Estimate | Notes |
|---|---|---|
| OpenAI gpt-4.1 (text, 3,500 sentences) | **~$3.50** | ~350 batch calls × $0.01 avg |
| Chinese edge-tts | **$0** | free |
| Hindi Sarvam bulbul:v3 | **$5–30** | ~540 k chars across both buckets × 2 voices; rate varies by plan |
| Hindi edge-tts (fallback if Sarvam not topped up) | $0 | 2 hi-IN voices: `hi-IN-SwaraNeural` (F), `hi-IN-MadhurNeural` (M). Less natural Hinglish, but free. |
| MiniMax / Qwen (unused) | $0 | skipped per earlier decision |
| HuggingFace Datasets | $0 | free for private datasets at this size |
| **Total** | **~$10–35** | dominated by Sarvam uncertainty — verify plan tier before launch |

## Storage strategy

**Local-first, then push to a private Hugging Face Datasets repo.** You'll provide the repo name (e.g. `valsea/synthetic-asr-zh` and `valsea/synthetic-asr-hi`, or a single combined repo). I'll wire the push script once you confirm.

Two files per (lang, bucket, aug-state) ship to HF:
- `data/<bucket>_<aug-state>/audio/*.wav` — the WAVs
- `data/<bucket>_<aug-state>/manifest.jsonl` — NeMo schema `{audio_filepath, text, duration, voice_id, language, augmentation}` (paths rewritten to `audio/<filename>` once on HF)

Optional: a 95/5 train/val split is added as `train.jsonl` + `val.jsonl` per bucket (we have enough samples per bucket to make val meaningful).

The local repo's `.gitignore` will continue to exclude `outputs/`. The bulk audio **does not go through GitHub** — that's the role of HF Datasets here.

## Code changes needed (in this order)

1. **Install audiomentations** (one-time, on this machine):
   ```bash
   sudo xcodebuild -license accept   # you run this once
   .venv/bin/pip install 'audiomentations<0.36'   # 0.36+ has numpy-minmax AVX bug on Apple Silicon
   ```
   Verify by running `03_augmentation/augment.py` on a small sample and confirming the manifest has `"augmentation": "<transform_name>"` instead of `"passthrough"`.

2. **Add resumability** to `02_tts_synthesis/synthesize.py`:
   - Skip sentence/voice pairs whose output WAV already exists on disk.
   - Append-only writes to `manifest_clean.jsonl` (don't rewrite).
   - New `--resume` flag (default on).

3. **Add a worker pool** to `02_tts_synthesis/synthesize.py`:
   - New `--workers N` flag (default 4).
   - Use `concurrent.futures.ThreadPoolExecutor` for I/O-bound TTS calls.
   - Edge-tts is `asyncio`-native already; wrap in a queue with N tasks in flight.
   - Sarvam is sync `httpx` — straight ThreadPoolExecutor works.

4. **Deterministic voice rotation** in `synthesize.py`:
   - Replace `random.sample(voices, k=n)` with a `(idx, voice_idx) → voice` mapping so each voice gets `(total_clips / n_voices)` ± 1 samples across the run.

5. **HF push step** as a new file `07_hf_push/push_to_hf.py`:
   - Reads `outputs/{lang}/{bucket}/{clean|augmented}/manifest_*.jsonl`.
   - Uses `huggingface_hub.HfApi` and `upload_folder` to push WAVs + a rewritten manifest.
   - Computes train/val split deterministically (hash of sentence id).
   - Idempotent — re-runs only upload changed files.
   - Auth via `HF_TOKEN` env var or `huggingface-cli login`.
   - Add `huggingface_hub>=0.20.0` to `pyproject.toml` `tts-cloud` extra.

6. **Bulk orchestrator** as a new file `scripts/run_bulk.sh`:
   - Generates text → synthesises TTS (resume-safe) → augments → filters (duration-only) → exports → HF push.
   - Per (lang, bucket): respects `COUNT_SHORT` / `COUNT_LONG` env vars (default 1500 / 250).
   - Per stage: logs to `logs/bulk_<timestamp>/<stage>.log` so a failed Sarvam call can be diagnosed.
   - Exits non-zero if any stage fails so it can be wrapped in `screen` / `nohup` cleanly.

## Pipeline run sequence

```bash
# One-time setup (already done except step 1):
sudo xcodebuild -license accept
.venv/bin/pip install -e '.[prototype,tts-cloud,augment]'
huggingface-cli login   # or export HF_TOKEN=...

# Confirm balances before launch (script will print them as a preflight)
# - OpenAI: $5+ available
# - Sarvam: enough credits for ~540k chars (~$5–30)

# Launch
nohup bash scripts/run_bulk.sh > bulk_run.log 2>&1 &
tail -f bulk_run.log
```

The script will:
1. Generate ~1,500 short + ~250 long sentences for zh and hi (~$3.50 OpenAI).
2. Synthesise via edge-tts (zh) + Sarvam (hi) with 4–6 workers and `--resume` on.
3. Augment with audiomentations: noise (if `noise_bank/` populated) + RIR + time-stretch + pitch + gain + MP3 codec.
4. Run duration-sanity filter, export NeMo manifests.
5. Push to `valsea/synthetic-asr-zh` and `valsea/synthetic-asr-hi` (or your specified repo paths).

## Failure recovery

Every step is safe to re-run from any failure point:
- **Text gen fails midway** → corpus.jsonl is partial; re-run with the same `--count`; dedup keeps unique sentences.
- **TTS fails midway** (the Sarvam-credits-ran-out case) → `--resume` skips existing WAVs; re-run after topping up. Manifest is append-only.
- **Augmentation fails** → re-run; existing aug WAVs are reused.
- **HF push fails** → re-run `07_hf_push/push_to_hf.py`; `upload_folder` diffs and only sends changes.

## Verification checklist (after the run)

- [ ] `find outputs/{chinese,hindi}/{short,long}/clean -name '*.wav' | wc -l` ≥ 7,000.
- [ ] `python -c "import json, glob; print(sum(json.loads(l)['duration'] for p in glob.glob('outputs/*/{short,long}/train_manifest.json') for l in open(p))/3600)"` reports ~24.0 hours (give or take 5%).
- [ ] At least 5 spot-checked aug WAVs sound audibly augmented (noise/pitch/timing differ from the corresponding clean clip).
- [ ] Aug manifest entries have `"augmentation": "<transform>"`, never `"passthrough"`.
- [ ] HF dataset(s) browsable: `huggingface-cli` reports the right file count and size.
- [ ] `datasets.load_dataset('valsea/synthetic-asr-zh', 'short_clean')` loads with audio decoding working.

## Sarvam-fallback path (if no top-up)

If you don't top up Sarvam, **switch Hindi TTS to edge-tts** without changing the rest of the pipeline:

```bash
TTS_BACKEND_HI=edge bash scripts/run_bulk.sh
```

- Uses 2 voices: `hi-IN-SwaraNeural` (F), `hi-IN-MadhurNeural` (M) — already in `EDGE_VOICES`.
- $0 TTS cost, ~30 min Hindi synthesis time (network-bound).
- Tradeoffs: less natural Hindi prosody, weaker Hinglish handling (Microsoft tends to mis-stress code-switched English). Acceptable as a baseline; you can re-run Hindi only via `TTS_BACKEND_HI=sarvam` once topped up — the resume logic skips existing WAVs so only the missing voice files get re-synthesised.
- Bulk runner reads `TTS_BACKEND_ZH` (default `edge`) and `TTS_BACKEND_HI` (default `sarvam`) so either language can be flipped independently.

## Decisions locked in

| Item | Decision |
|---|---|
| HF dataset layout | Two separate repos, one per language. Created under your HF namespace once `huggingface-cli whoami` confirms login. |
| Sarvam | Default; falls back to edge-tts hi-IN if no top-up (see above). |
| `audiomentations` | Installed via `pip install 'audiomentations<0.36' pydub` (Xcode license already accepted). |
| MUSAN / RIR | Optional — pipeline gracefully skips noise/RIR if `noise_bank/` is empty. Drop in later for a re-aug pass without re-synthesising. |

## Resume strategy in detail

Every long-running stage writes incrementally and skips already-done work:

1. **Text gen (`generate_*.py`)**: if `corpus.jsonl` already has ≥ `--count` unique sentences, exits with success. Re-runs are cheap.
2. **TTS (`synthesize.py --resume`)**:
   - Walks the corpus + voice rotation deterministically, building expected output filenames first.
   - For each expected `(idx, voice) → wav_path`: if file exists *and* is a non-empty WAV, skip; otherwise synthesise and append to `manifest_clean.jsonl`.
   - `manifest_clean.jsonl` is opened in append mode; dedup happens on next manifest read.
3. **Augmentation**: same pattern — skip if augmented WAV exists.
4. **Filter / Export**: idempotent — rewrites the manifest from current disk state every time.
5. **HF push**: `upload_folder` diffs against remote and only sends changed files.

So if Sarvam dies at clip 1,500 of 3,500, you just top up and re-run the same script — it picks up at clip 1,501.
