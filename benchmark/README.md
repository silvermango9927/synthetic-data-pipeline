# Benchmark — ASR Scaling Laws Framework

End-to-end pipeline for **fine-tuning** and **scaling law evaluation** of ASR models on synthetic speech data.

Supports: `Whisper`, `Qwen2-Audio`, `Wav2Vec2/MMS (CTC)`  
Languages: `zh` (Chinese), `hi` (Hindi)  
Data modes: `synthetic` (clean TTS), `augmented`, `both`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e ".[benchmark]"
# or
pip install torch transformers datasets jiwer librosa soundfile peft click pandas matplotlib python-dotenv
```

### 2. Set HuggingFace token (if using private gated datasets)

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Or clone the HF dataset repos locally (supports offline operation):

```bash
# Install git-lfs first
git lfs install

git clone https://huggingface.co/datasets/silvermango9927/synthetic-asr-zh outputs/hf_datasets/synthetic-asr-zh
git clone https://huggingface.co/datasets/silvermango9927/synthetic-asr-hi outputs/hf_datasets/synthetic-asr-hi
```

> Once cloned locally, the pipeline runs **100% offline** — no HF token needed.

---

## Modules

| File | Description |
|------|-------------|
| `config.py` | `TrainConfig` — Pydantic config schema for all training parameters |
| `dataset.py` | Data loader — HF Hub, local cloned repo, or local WAV scan fallback |
| `train.py` | Fine-tuning loop for Whisper / Qwen2-Audio / Wav2Vec2 CTC |
| `evaluate.py` | WER & CER evaluation on validation set |
| `scaling.py` | Automated sweep across data fractions, outputs CSV + PNG plots |

---

## Scaling Law Sweep (`scaling.py`)

Trains the model across `--fractions` of the training data, evaluates WER/CER at each fraction, saves a CSV stats table and a PNG scaling curve.

### Usage

```bash
python -m benchmark.scaling \
  --model whisper \
  --model-name openai/whisper-tiny \
  --lang zh \
  --data-type synthetic \
  --epochs 3 \
  --batch-size 4 \
  --fractions 0.1,0.25,0.5,1.0 \
  --device cuda
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | `whisper\|qwen\|ctc` | `whisper` | Model architecture |
| `--model-name` | str | `openai/whisper-tiny` | HF model identifier |
| `--lang` | `zh\|hi` | `zh` | Target language |
| `--data-type` | `synthetic\|augmented\|both` | `synthetic` | Dataset split/mode to use |
| `--epochs` | int | `1` | Training epochs per fraction |
| `--batch-size` | int | `4` | Batch size per device |
| `--lr` | float | `5e-5` | Learning rate |
| `--fractions` | str | `0.1,0.25,0.5,1.0` | Comma-separated list of data fractions |
| `--dataset-name` | str | `None` | Override HF dataset name (auto-resolves to `silvermango9927/synthetic-asr-{lang}`) |
| `--device` | `cuda\|cpu` | `cuda` | Device to train on |
| `--use-lora / --no-lora` | bool | `True` | Enable LoRA (auto-skipped for Whisper) |

### Outputs

```
outputs/benchmark/
  stats/
    scaling_{lang}_{model}_{data_type}.csv   # metrics table
    scaling_{lang}_{model}_{data_type}.png   # scaling curve plot
  scaling_{lang}_{model}_{data_type}_f{frac}/
    final_model/                             # saved checkpoint
```

---

## Training Only (`train.py`)

```bash
python -m benchmark.train \
  --model whisper \
  --model-name openai/whisper-tiny \
  --lang zh \
  --data-type synthetic \
  --epochs 5 \
  --batch-size 4 \
  --output-dir outputs/my_run \
  --device cuda
```

---

## Evaluation Only (`evaluate.py`)

```bash
python -m benchmark.evaluate \
  --model-path outputs/benchmark/scaling_zh_whisper_synthetic_f1.0/final_model \
  --model whisper \
  --lang zh \
  --data-type synthetic \
  --device cpu
```

---

## Windows / GPU Notes

- **Blackwell GPU (sm_120) driver mismatch**: Force CPU with `$env:CUDA_VISIBLE_DEVICES="-1"` before running.
- **Offline mode**: Clone HF datasets to `outputs/hf_datasets/` (see Quick Start).
- **Hindi token limit**: Devanagari text can exceed Whisper's 448-token decoder limit — handled automatically via truncation in `dataset.py`.

---

## Results (Whisper-Tiny, Clean Synthetic, 3 Epochs)

### Chinese (ZH)

| Fraction | Hours | Train Loss | WER | CER |
|----------|-------|------------|-----|-----|
| 10% | 0.37h | 1.41 | 101.7% | 27.6% |
| 25% | 0.93h | 0.99 | 94.9% | 12.5% |
| 50% | 1.88h | 0.76 | 94.9% | 19.0% |
| 100% | 3.75h | 0.54 | **84.7%** | **5.5%** |

### Hindi (HI)

| Fraction | Hours | Train Loss | WER | CER |
|----------|-------|------------|-----|-----|
| 10% | 0.45h | 3.20 | 99.1% | 72.8% |
| 25% | 1.12h | 2.12 | 61.7% | 39.3% |
| 50% | 2.25h | 1.53 | 47.1% | 31.9% |
| 100% | 4.51h | 1.13 | **32.3%** | **21.2%** |

> **Observation**: WER and CER are still declining at 100% data — the model has not plateaued yet. More synthetic data is needed for a full scaling law curve.

---

## Dataset Format

The pipeline expects HF datasets with these columns in `manifest.jsonl`:

```json
{
  "audio_filepath": "audio/clip_0001.wav",
  "text": "这是一个示例句子",
  "duration": 28.4,
  "language": "zh",
  "source": "synthetic",
  "voice_id": "zh_speaker_01",
  "augmentation": "none"
}
```

Audio files must be `16kHz mono WAV`.
