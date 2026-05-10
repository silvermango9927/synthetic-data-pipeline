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
| Hindi | `hi` | **Whisper** | **Sarvam `bulbul:v3`** | `google/fleurs` (`hi_in`) |

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
- **`audiomentations`** install fails until `sudo xcodebuild -license` is run (numpy-minmax compile). Pipeline falls back to pass-through if unavailable.
- **GPU**: required for fast UTMOS, Whisper-large-v3-turbo, and Qwen2-Audio-7B QC. Qwen2-Audio needs ~14 GB VRAM.

## API keys (in `.env`, gitignored)

```
OPENAI_API_KEY=...      # Stage 01 text generation (gpt-4.1) — required
SARVAM_API_KEY=...      # Sarvam TTS for Hindi — required for hi runs
MINIMAX_API_KEY=...     # MiniMax T2A v2 for Chinese — only if TTS_BACKEND=minimax
DASHSCOPE_API_KEY=...   # Qwen TTS for Chinese — only if TTS_BACKEND=qwen (needs Model Studio activation)
```

Default Chinese path uses free edge-tts and needs no key beyond OpenAI.

Loaded automatically via `python-dotenv` at script entry.

## Pipeline stages

| Stage | Path | Notes |
|---|---|---|
| 01 Text corpus | `01_text_corpus/generate_{singlish,vietnamese,chinese,hindi}.py` | OpenAI gpt-4.1 + few-shot reference + lexicon. Outputs JSONL. |
| 02 TTS synthesis | `02_tts_synthesis/synthesize.py` | Backends: `edge`, `fish`, `xtts`, `qwen`, `minimax`, `sarvam`. Selected via `--backend`. |
| 03 Augmentation | `03_augmentation/augment.py` | Language-agnostic. Adds noise, RIR, time-stretch, pitch, MP3 codec. |
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
bash scripts/run_prototype_zh.sh   # Chinese (Qwen TTS)
bash scripts/run_prototype_hi.sh   # Hindi (Sarvam TTS)
```

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

- `audiomentations` fails to install pre-Xcode-license. The pipeline degrades gracefully (pass-through) if it's missing.
- AISHELL streaming download is large; first run pulls metadata. Use `--skip-reference` for offline iteration.
- FLEURS `hi_in` config name uses underscore (`hi_in`), not hyphen.
- Sarvam max 2,500 chars per call — long-form Hindi (~110 words ≈ 600 chars) fits fine.
- MiniMax max 10,000 chars per call. Hex-encoded WAV in `data.audio` of the response — decode with `bytes.fromhex(...)`.
- MiniMax errors can come back inside a 200 response; check `body["base_resp"]["status_code"] == 0` before treating as success.
- Qwen3-TTS-Flash requires a one-click activation per Alibaba Cloud account at https://modelstudio.console.alibabacloud.com (Singapore region). Default zh path is MiniMax until that's done.
- Qwen2-Audio QC takes ~5–10 s per sample; use `--skip-whisper` during iteration.
- The Sarvam SDK / endpoint shape may change; current code uses the documented REST endpoint with `api-subscription-key` header.

## Related repos

- `/Users/abhayganti/valsea/<monorepo>` — consumes `outputs/<lang>/{short,long}/train_manifest.json` for ASR finetuning.
