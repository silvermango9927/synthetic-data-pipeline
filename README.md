# VALSEA ASR Data Generation Pipeline

Generates synthetic training data for finetuning:
- **Blaze** (Vietnamese ASR)
- **MERaLiON** (Singlish ASR)

## Pipeline

```
Text Corpus (Claude API) → TTS (Fish Speech S2) → Augmentation → Quality Filter → NeMo Manifest
```

## Quick Start

```bash
# 1. Setup
pip install -e ".[quality,dev]"
export ANTHROPIC_API_KEY="your-key-here"

# 2. Start Fish Speech TTS server (or use --backend xtts)
docker run -p 8080:8080 --gpus all fishaudio/fish-speech:latest

# 3. Add voice references
# Put 10-30s WAV clips in:
#   02_tts_synthesis/voice_bank/singlish/
#   02_tts_synthesis/voice_bank/vietnamese/

# 4. Run prototype (50 sentences, end-to-end)
make prototype

# 5. Or run full pipeline
make generate-text
make synthesize
make augment
make filter
make export
```

## Directory Structure

```
01_text_corpus/          # Claude-generated sentence corpora
  lexicons/              # Singlish particles + Vietnamese tonal vocab CSVs
  prompts/               # System prompts for corpus generation
  generate_singlish.py
  generate_vietnamese.py

02_tts_synthesis/        # Text-to-speech synthesis
  synthesize.py          # Fish Speech S2 (primary) + XTTS-v2 (fallback)
  voice_bank/            # Reference speaker WAVs for voice cloning
    singlish/
    vietnamese/

03_augmentation/         # Audio augmentation
  augment.py             # Noise, reverb, time-stretch, pitch-shift, codec
  noise_bank/
    ambient/             # Background noise (MUSAN)
    rir/                 # Room impulse responses

04_quality_filter/
  filter.py              # Duration sanity + UTMOS + Whisper WER roundtrip

06_dataset_export/
  export_nemo_manifest.py  # Convert to NeMo JSONL format

outputs/
  singlish/{clean,augmented}/
  vietnamese/{clean,augmented}/

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
| `pip install -e ".[quality]"` | UTMOS scoring + Whisper WER filter |
| `pip install -e ".[tts-local]"` | XTTS-v2 local TTS fallback |
| `pip install -e ".[real-data]"` | WhisperX for real data curation |
