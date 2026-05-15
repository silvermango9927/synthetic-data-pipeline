# VALSEA Synthetic ASR Data Pipeline

Cross-session context for Claude. This file is read at the start of every conversation in this repo.

## Purpose

Generate synthetic ASR training data (audio + transcript pairs) for VALSEA's downstream finetuning of language-specific ASR models. Outputs are NeMo-format JSONL manifests consumed by the VALSEA monorepo's training scripts.

## Languages & downstream targets

| Language | Code | Downstream ASR target | TTS provider | Reference dataset (HF) |
|---|---|---|---|---|
| Singlish | `en` | MERaLiON | edge-tts (Microsoft Neural) | `cesinsingapore/singlish` |
| Vietnamese | `vi` | Blaze | edge-tts / Fish Speech / XTTS | `doof-ferb/infore1_25hours` |
| Chinese (Mandarin) | `zh` | **Qwen2-Audio** | **edge-tts zh-CN** (default, free); MiniMax `speech-02-hd` and Qwen TTS are paid opt-ins | `AISHELL/AISHELL-1` |
| Hindi | `hi` | **Whisper** | **edge-tts hi-IN** (default, free); Sarvam `bulbul:v3` is a paid opt-in | `google/fleurs` (`hi_in`) |

## Length conventions

Two corpora per language for zh/hi:
- **short**: 8–15 words / ~50–90 chars (Devanagari) or 12–28 Hanzi → ~5 seconds spoken.
- **long**: 75–110 words → ~30 seconds spoken.

Selected via `--length-target {short,long}` on `01_text_corpus/generate_{chinese,hindi}.py`. The two buckets land in `outputs/{chinese,hindi}/{short,long}/`.

## Audio format invariant

**Every TTS backend must emit 16 kHz mono WAV.** Downstream augmentation, UTMOS, and Whisper QC all assume this.
- edge-tts → MP3, converted to 16 kHz mono in code.
- MiniMax `speech-02-hd` → request `audio_setting.sample_rate=16000`, `format=wav` (native, no resample). Hex-encoded payload.
- Qwen TTS → returns 24 kHz; we resample via librosa.
- Sarvam TTS → request `speech_sample_rate=16000` (native, no resample).
- Fish Speech / XTTS → check per-call.

## Environment

- **Python**: use `.venv` (Python 3.10.x at `/usr/local/bin/python3.10`). System Python is Anaconda 3.8 — DO NOT use it.
- **Setup**:
  ```bash
  /usr/local/bin/python3.10 -m venv .venv
  .venv/bin/pip install -e ".[prototype,tts-cloud]"
  ```
- **`make` is broken** on this machine (Xcode license not accepted). Use `bash scripts/run_prototype*.sh` or call `.venv/bin/python` directly.
- **`audiomentations` on Apple Silicon**: pin to `<0.36`. Versions ≥0.36 pull `numpy-minmax`, which has a buggy `IS_X86_64` macro that tries to compile AVX intrinsics on arm64 and fails. The Xcode license is unrelated — accepting it does not fix this. Install with:
  ```bash
  .venv/bin/pip install 'audiomentations<0.36' pydub
  ```
  `pydub` is needed for `Mp3Compression` and shells out to `ffmpeg` (already at `/opt/homebrew/bin/ffmpeg`). Without these, `03_augmentation/augment.py` falls back to pass-through (literal file copies). On x86_64 the upstream extras (`pip install -e '.[augment]'`) may work, but on this arm64 box do not use it.
- **GPU**: required for fast UTMOS, Whisper-large-v3-turbo, and Qwen2-Audio-7B QC. Qwen2-Audio needs ~14 GB VRAM.

## API keys (in `.env`, gitignored)

```
OPENAI_API_KEY=...      # Stage 01 text generation (gpt-4.1) — required
SARVAM_API_KEY=...      # Sarvam TTS for Hindi — only if TTS_BACKEND_HI=sarvam (opt-in)
MINIMAX_API_KEY=...     # MiniMax T2A v2 for Chinese — only if TTS_BACKEND_ZH=minimax
DASHSCOPE_API_KEY=...   # Qwen TTS for Chinese — only if TTS_BACKEND_ZH=qwen (needs Model Studio activation)
HF_TOKEN=...            # only if pushing the bulk dataset to HuggingFace (07_hf_push)
```

Default Chinese and Hindi paths both use free edge-tts and need no key beyond OpenAI.

Loaded automatically via `python-dotenv` at script entry.

## Pipeline stages

| Stage | Path | Notes |
|---|---|---|
| 01 Text corpus | `01_text_corpus/generate_{singlish,vietnamese,chinese,hindi}.py` | OpenAI gpt-4.1 + few-shot reference + lexicon. Outputs JSONL. |
| 02 TTS synthesis | `02_tts_synthesis/synthesize.py` | Backends: `edge`, `fish`, `xtts`, `qwen`, `minimax`, `sarvam`. Selected via `--backend`. |
| 03 Augmentation | `03_augmentation/augment.py` | Language-agnostic. Time-stretch, pitch, gain, bandpass, MP3 codec; noise/RIR if `noise_bank/{ambient,rir}` populated. Requires `audiomentations<0.36` + `pydub`; otherwise pass-through. Fallback: `scripts/augment_simple.py` (no audiomentations dep). |
| 04 Quality filter | `04_quality_filter/filter.py` | 3 layers: duration sanity + UTMOS + roundtrip WER. `--qc-asr` selects whisper or qwen2-audio. |
| 05 Real-data curation | `05_real_data_curation/` | Placeholder (unimplemented). |
| 06 Export | `06_dataset_export/export_nemo_manifest.py` | Strips metadata to NeMo JSONL `{audio_filepath, text, duration}`. |

## QC defaults per language

WER thresholds (auto-applied in `04_quality_filter/filter.py`):

| Language | Threshold | Notes |
|---|---|---|
| `vi` | 0.15 | Strictest — Blaze is mature. |
| `en` | 0.25 | Singlish hard to recognize. |
| `zh` | 0.20 | Mandarin Hanzi WER tends to be higher than syllable-level CER; tune after measurement. |
| `hi` | 0.25 | Hinglish code-switching inflates Whisper WER. |

The Chinese roundtrip can use Qwen2-Audio instead of Whisper:
```bash
04_quality_filter/filter.py --lang zh --qc-asr qwen2-audio ...
```

## How to run

End-to-end prototype (50 short + 50 long per language, ~$1–5 in API cost):
```bash
bash scripts/run_prototype_zh.sh   # Chinese (edge-tts zh-CN by default)
bash scripts/run_prototype_hi.sh   # Hindi (Sarvam by default; set TTS_BACKEND=edge for free hi-IN)
```

### Bulk production run

```bash
# Default: edge-tts for both zh and hi (free, no provider keys beyond OpenAI).
nohup bash scripts/run_bulk.sh > bulk_run.log 2>&1 &
tail -f bulk_run.log

# Before launching, run the smoke test to validate end-to-end wiring (~5–10 min):
bash scripts/smoke_test_bulk.sh             # against existing outputs/
ISOLATE=1 bash scripts/smoke_test_bulk.sh   # hermetic, exercises text gen + TTS from scratch
```

Sarvam is still wired up for opt-in (`TTS_BACKEND_HI=sarvam`), but credits must be funded
first — synthesize.py now aborts the run after 10 consecutive 429s rather than silently
producing nothing.

Override sample counts:
```bash
COUNT_SHORT=200 COUNT_LONG=200 bash scripts/run_prototype_zh.sh
```

Output layout per language:
```
outputs/<lang>/
  short/{corpus.jsonl, clean/, augmented/, manifest_filtered.jsonl, train_manifest.json}
  long/{...}
```

### Augmentation only (re-run stage 03)

Once `audiomentations<0.36` + `pydub` are installed (see Environment), augment a bucket like:
```bash
.venv/bin/python 03_augmentation/augment.py \
  --input-dir outputs/chinese/short/clean \
  --output-dir outputs/chinese/short/augmented \
  --variants 2
```
- Default 2 variants per clean clip. WAVs land alongside the clean ones; `manifest_augmented.jsonl` records `variant_0` / `variant_1`.
- Verify it actually augmented (vs silently passing through): `grep -c '"augmentation": "passthrough"' outputs/<lang>/<bucket>/augmented/manifest_augmented.jsonl` should return `0`.
- Empty `noise_bank/{ambient,rir}` means noise + reverb stages are skipped (warning, not error). Drop MUSAN into `03_augmentation/noise_bank/ambient/` and RIRS_NOISES into `03_augmentation/noise_bank/rir/` to enable them.

No-deps fallback (when audiomentations won't install): `scripts/augment_simple.py` mirrors the non-noise/non-RIR transforms using only librosa + scipy + numpy. Same CLI shape:
```bash
.venv/bin/python scripts/augment_simple.py \
  --input-dir outputs/<lang>/<bucket>/clean \
  --output-dir outputs/<lang>/<bucket>/augmented_real \
  --variants 2 --limit 10
```

## Adding a new language (6-step checklist)

1. Add to `Language` enum in `scripts/config.py:8-12`.
2. Add WER threshold in `scripts/config.py` and `WER_THRESHOLDS` dict in `04_quality_filter/filter.py`.
3. Create `01_text_corpus/prompts/<lang>_{short,long}_system.txt`.
4. Create `01_text_corpus/lexicons/<lang>_*.csv`.
5. Create `01_text_corpus/generate_<lang>.py` (copy nearest existing, swap `REFERENCE_DATASET`, `MODEL`, language tag).
6. If new TTS provider: add `synthesize_<provider>_tts(...)` to `02_tts_synthesis/synthesize.py`, add voice list constant, register in `--backend` `click.Choice`, wire into `main()` dispatch.

## Cost notes

- gpt-4.1: ~$0.005 per 10-sentence batch. Negligible for prototype.
- MiniMax `speech-02-hd`: ~$30 / 1M chars (HD tier). Speech-02-turbo is cheaper if you don't need HD quality.
- Qwen `qwen3-tts-flash`: $0.115 / 10k chars. Requires Model Studio activation before first call.
- Sarvam `bulbul:v3`: published rate varies by plan — verify before scaling. Range $0.05–$0.50 / 10k chars used in plan estimates.
- Local QC (UTMOS, Whisper, Qwen2-Audio) has zero API cost but burns GPU time.

## Known issues / tripwires

- `audiomentations` ≥0.36 fails to install on Apple Silicon (numpy-minmax AVX/arm64 bug — *not* an Xcode license issue, despite the install spam). Pin `audiomentations<0.36`. Pipeline degrades to pass-through (literal file copies labeled `"augmentation": "passthrough"`) if the import fails — silent, easy to miss. Always grep the augmented manifest after stage 03 to confirm.
- `Mp3Compression` in audiomentations requires `pydub` + `ffmpeg`. Without `pydub` the whole `Compose` raises mid-loop, leaving the augmented dir in a partial state (manifest never written). Install pydub before running.
- AISHELL streaming download is large; first run pulls metadata. Use `--skip-reference` for offline iteration.
- FLEURS `hi_in` config name uses underscore (`hi_in`), not hyphen.
- Sarvam max 2,500 chars per call — long-form Hindi (~110 words ≈ 600 chars) fits fine.
- Sarvam credits exhaust silently: every call 429s with `insufficient_quota_error`. As of 2026-05-15 the default Hindi backend is edge-tts (free); synthesize.py aborts the run after 10 consecutive failures so a credit-out doesn't burn hours producing zero audio. Edge-tts hi-IN voices: `hi-IN-SwaraNeural` (F), `hi-IN-MadhurNeural` (M).
- `04_quality_filter/filter.py` reads `clean/manifest_clean.jsonl` and `augmented/manifest_augmented.jsonl` explicitly — earlier versions used `rglob("manifest_*.jsonl")` and ended up double-counting their own output. If you re-introduce more manifests inside `outputs/<lang>/<bucket>/`, mirror the explicit listing.
- MiniMax max 10,000 chars per call. Hex-encoded WAV in `data.audio` of the response — decode with `bytes.fromhex(...)`.
- MiniMax errors can come back inside a 200 response; check `body["base_resp"]["status_code"] == 0` before treating as success.
- Qwen3-TTS-Flash requires a one-click activation per Alibaba Cloud account at https://modelstudio.console.alibabacloud.com (Singapore region). Default zh path is MiniMax until that's done.
- Qwen2-Audio QC takes ~5–10 s per sample; use `--skip-whisper` during iteration.
- The Sarvam SDK / endpoint shape may change; current code uses the documented REST endpoint with `api-subscription-key` header.

## Related repos

- `/Users/abhayganti/valsea/<monorepo>` — consumes `outputs/<lang>/{short,long}/train_manifest.json` for ASR finetuning.
