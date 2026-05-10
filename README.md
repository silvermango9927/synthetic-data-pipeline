# VALSEA ASR Data Generation Pipeline

Generates synthetic training data for finetuning:
- **Blaze** (Vietnamese ASR) — edge-tts / Fish Speech
- **MERaLiON** (Singlish ASR) — edge-tts
- **Qwen2-Audio** (Mandarin Chinese ASR) — edge-tts zh-CN (default, free); MiniMax `speech-02-hd` or Qwen `qwen3-tts-flash` available as paid opt-ins
- **Whisper** (Hindi ASR) — Sarvam `bulbul:v3`

Per-language corpora come in two length buckets: ~5s (short) and ~30s (long).

## Pipeline

```
Text Corpus (OpenAI gpt-4.1) → TTS (per-language provider) → Augmentation → Quality Filter → NeMo Manifest
```

## Quick Start

```bash
# 1. Setup
.venv/bin/pip install -e ".[prototype,tts-cloud,quality,dev]"

# 2. Set API keys in .env
echo 'OPENAI_API_KEY=...'    >> .env
echo 'MINIMAX_API_KEY=...'   >> .env   # Chinese (MiniMax T2A v2 — default)
echo 'DASHSCOPE_API_KEY=...' >> .env   # Chinese (Qwen TTS — alternative)
echo 'SARVAM_API_KEY=...'    >> .env   # Hindi (Sarvam TTS)

# 3. Run prototype per language (50 short + 50 long, end-to-end)
bash scripts/run_prototype.sh       # Singlish (legacy)
bash scripts/run_prototype_zh.sh    # Chinese
bash scripts/run_prototype_hi.sh    # Hindi

# Override sample counts:
COUNT_SHORT=200 COUNT_LONG=200 bash scripts/run_prototype_zh.sh
```

For the legacy Vietnamese / Singlish full pipeline (Fish Speech voice-bank route), see `Makefile` targets — `make` requires Xcode license accepted.

## Directory Structure

```
01_text_corpus/          # OpenAI-generated sentence corpora
  lexicons/              # Per-language vocab CSVs
  prompts/               # System prompts (one per lang, plus short/long for zh+hi)
  generate_singlish.py
  generate_vietnamese.py
  generate_chinese.py    # --length-target {short,long}
  generate_hindi.py      # --length-target {short,long}

02_tts_synthesis/        # Text-to-speech synthesis
  synthesize.py          # Backends: edge | fish | xtts | qwen | minimax | sarvam
  voice_bank/            # Reference speaker WAVs (only for fish / xtts)
    singlish/
    vietnamese/

03_augmentation/         # Audio augmentation
  augment.py             # Noise, reverb, time-stretch, pitch-shift, codec
  noise_bank/
    ambient/             # Background noise (MUSAN)
    rir/                 # Room impulse responses

04_quality_filter/
  filter.py              # Duration sanity + UTMOS + roundtrip WER (--qc-asr whisper|qwen2-audio)

06_dataset_export/
  export_nemo_manifest.py  # Convert to NeMo JSONL format

outputs/
  singlish/{clean,augmented}/
  vietnamese/{clean,augmented}/
  chinese/{short,long}/{clean,augmented}/
  hindi/{short,long}/{clean,augmented}/

scripts/
  config.py              # Shared Pydantic models + PipelineConfig
  download_noise.sh      # Download MUSAN + RIR noise corpora
```

## Output

NeMo-format JSONL manifests at `outputs/{language}/train_manifest.json`, ready for VALSEA monorepo finetuning scripts. Each line:

```json
{"audio_filepath": "...", "text": "...", "duration": 3.2}
```

## Configuration

Edit [scripts/config.py](scripts/config.py) to adjust quality thresholds, TTS URL, and augmentation settings.

## Noise Bank (Optional but Recommended)

```bash
bash scripts/download_noise.sh   # downloads MUSAN (~11GB) + RIR corpus
```

Without noise files, the augmentation pipeline falls back to codec/pitch/time transforms only.

## Optional Dependencies

| Extra | Purpose |
|---|---|
| `pip install -e ".[prototype]"` | edge-tts + datasets + python-dotenv (zero-infra prototype) |
| `pip install -e ".[tts-cloud]"` | DashScope + Sarvam SDKs (Chinese + Hindi TTS) |
| `pip install -e ".[quality]"` | UTMOS scoring + Whisper WER filter |
| `pip install -e ".[tts-local]"` | XTTS-v2 local TTS fallback |
| `pip install -e ".[real-data]"` | WhisperX for real data curation |

For the full Chinese + Hindi prototype path: `pip install -e ".[prototype,tts-cloud]"`.
