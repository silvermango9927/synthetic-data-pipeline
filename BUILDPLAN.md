# VALSEA ASR Data Generation Prototype — Claude Code Build Plan

## Context

You are building a prototype repository called `valsea-asr-datagen` that generates synthetic training data for finetuning two ASR models:

1. **Blaze** — Vietnamese ASR
2. **MERaLiON** — Singlish (Singapore English) ASR

The pipeline: Text Corpus → TTS Synthesis → Audio Augmentation → Quality Filtering → Dataset Export (NeMo manifest format).

This repo is SEPARATE from the VALSEA monorepo. It produces output datasets (WAV files + JSON manifests) that the monorepo consumes for finetuning.

---

## Phase 1: Repository Scaffold

Create the full directory structure and config files.

### 1.1 Create directory structure

```bash
mkdir -p valsea-asr-datagen/{01_text_corpus/{prompts,lexicons},02_tts_synthesis/{voice_bank/{singlish,vietnamese},configs},03_augmentation/noise_bank,04_quality_filter,05_real_data_curation,06_dataset_export,outputs/{vietnamese/{clean,augmented},singlish/{clean,augmented}},scripts,tests}
cd valsea-asr-datagen
git init
```

### 1.2 Create `pyproject.toml`

```toml
[project]
name = "valsea-asr-datagen"
version = "0.1.0"
description = "Synthetic data generation pipeline for VALSEA ASR finetuning"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",
    "httpx>=0.27.0",
    "soundfile>=0.12.1",
    "librosa>=0.10.2",
    "audiomentations>=0.36.0",
    "jiwer>=3.0.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "tqdm>=4.66.0",
    "pydantic>=2.0.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
quality = [
    "openai-whisper>=20231117",
    "torch>=2.1.0",
    "torchaudio>=2.1.0",
]
tts-local = [
    "TTS>=0.22.0",  # Coqui XTTS-v2 (idiap fork)
]
real-data = [
    "whisperx>=3.1.0",
]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.4.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
```

### 1.3 Create `Makefile`

```makefile
.PHONY: setup generate-text synthesize augment filter export clean

setup:
	pip install -e ".[quality,dev]"
	@echo "For local TTS: pip install -e '.[tts-local]'"
	@echo "For real data curation: pip install -e '.[real-data]'"

generate-text:
	python 01_text_corpus/generate_singlish.py --output outputs/singlish/corpus.jsonl --count 1000
	python 01_text_corpus/generate_vietnamese.py --output outputs/vietnamese/corpus.jsonl --count 1000

synthesize:
	python 02_tts_synthesis/synthesize.py --corpus outputs/singlish/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/singlish --output-dir outputs/singlish/clean --lang en
	python 02_tts_synthesis/synthesize.py --corpus outputs/vietnamese/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/vietnamese --output-dir outputs/vietnamese/clean --lang vi

augment:
	python 03_augmentation/augment.py --input-dir outputs/singlish/clean --output-dir outputs/singlish/augmented --noise-bank 03_augmentation/noise_bank
	python 03_augmentation/augment.py --input-dir outputs/vietnamese/clean --output-dir outputs/vietnamese/augmented --noise-bank 03_augmentation/noise_bank

filter:
	python 04_quality_filter/filter.py --input-dir outputs/singlish --output outputs/singlish/manifest_filtered.jsonl --lang en
	python 04_quality_filter/filter.py --input-dir outputs/vietnamese --output outputs/vietnamese/manifest_filtered.jsonl --lang vi

export:
	python 06_dataset_export/export_nemo_manifest.py --input outputs/singlish/manifest_filtered.jsonl --output outputs/singlish/train_manifest.json
	python 06_dataset_export/export_nemo_manifest.py --input outputs/vietnamese/manifest_filtered.jsonl --output outputs/vietnamese/train_manifest.json

# Quick prototype: generate 50 sentences, synthesize, filter — end to end
prototype:
	python 01_text_corpus/generate_singlish.py --output outputs/singlish/corpus.jsonl --count 50
	python 02_tts_synthesis/synthesize.py --corpus outputs/singlish/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/singlish --output-dir outputs/singlish/clean --lang en --voices-per-sentence 1
	python 03_augmentation/augment.py --input-dir outputs/singlish/clean --output-dir outputs/singlish/augmented --noise-bank 03_augmentation/noise_bank --variants 1
	python 04_quality_filter/filter.py --input-dir outputs/singlish --output outputs/singlish/manifest_filtered.jsonl --lang en
	@echo "Done. Check outputs/singlish/manifest_filtered.jsonl"

clean:
	rm -rf outputs/singlish/clean/* outputs/singlish/augmented/* outputs/vietnamese/clean/* outputs/vietnamese/augmented/*
```

### 1.4 Create `.gitignore`

```
outputs/singlish/clean/
outputs/singlish/augmented/
outputs/vietnamese/clean/
outputs/vietnamese/augmented/
03_augmentation/noise_bank/*.wav
02_tts_synthesis/voice_bank/**/*.wav
__pycache__/
*.egg-info/
.venv/
```

### 1.5 Create shared config `scripts/config.py`

```python
"""Shared configuration and data models for the pipeline."""
from pydantic import BaseModel
from pathlib import Path
from enum import Enum


class Language(str, Enum):
    SINGLISH = "en"
    VIETNAMESE = "vi"


class SampleMetadata(BaseModel):
    """Metadata for a single audio-text pair."""
    audio_filepath: str
    text: str
    duration: float
    language: str
    source: str  # "synthetic" or "real"
    voice_id: str | None = None
    augmentation: str | None = None  # description of augmentation applied
    utmos_score: float | None = None
    roundtrip_wer: float | None = None


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    # TTS
    tts_api_url: str = "http://localhost:8080/v1/tts"  # Fish Speech local server
    voices_per_sentence: int = 3
    
    # Quality thresholds
    utmos_threshold: float = 3.5
    wer_threshold_vi: float = 0.15
    wer_threshold_en: float = 0.25
    
    # Augmentation
    augmentation_variants: int = 2  # augmented copies per clean sample
    snr_range: tuple[float, float] = (10.0, 25.0)
    
    # Paths
    noise_bank: Path = Path("03_augmentation/noise_bank")
    output_base: Path = Path("outputs")


# Singleton config, override with env vars or CLI args
CONFIG = PipelineConfig()
```

---

## Phase 2: Text Corpus Generation

### 2.1 Create Singlish lexicon file `01_text_corpus/lexicons/singlish_particles.csv`

```csv
term,category,example_position,notes
lah,particle,sentence-final,"Emphatic or softening: 'Can lah', 'Don't worry lah'"
lor,particle,sentence-final,"Resigned acceptance: 'Like that lor'"
leh,particle,sentence-final,"Mild assertion: 'Not bad leh'"
hor,particle,sentence-final,"Seeking agreement: 'This one nice hor'"
meh,particle,sentence-final,"Skeptical: 'Got meh?'"
sia,particle,sentence-final,"Surprise/emphasis: 'Damn good sia'"
arh,particle,sentence-final,"Confirmation seeking: 'You coming arh?'"
walao,exclamation,sentence-initial,"Exasperation: 'Walao eh, so expensive'"
alamak,exclamation,sentence-initial,"Surprise: 'Alamak, forgot already'"
shiok,adjective,mid-sentence,"Great/pleasurable: 'The food damn shiok'"
kiasu,adjective,mid-sentence,"Fear of losing out: 'So kiasu, queue since 5am'"
bojio,verb,any,"Didn't invite: 'Eh you all go makan bojio me'"
tapao,verb,mid-sentence,"Takeaway: 'Help me tapao chicken rice'"
chope,verb,mid-sentence,"Reserve: 'I chope the seat already'"
makan,verb,mid-sentence,"Eat: 'Let's go makan'"
jialat,adjective,any,"Terrible: 'This one very jialat'"
sian,adjective,any,"Bored/frustrated: 'So sian today'"
paiseh,adjective,any,"Embarrassed: 'Paiseh to ask'"
kaypoh,adjective,any,"Nosy: 'Don't so kaypoh lah'"
atas,adjective,any,"High-class/posh: 'This restaurant very atas'"
```

### 2.2 Create Vietnamese lexicon `01_text_corpus/lexicons/vietnamese_tonal.csv`

```csv
term,tone,confusable_with,domain,notes
ma,level,"má(rising),mà(falling),mả(dipping),mã(broken),mạ(heavy)","general","Ghost vs cheek vs but vs grave vs seedling vs rice"
bán,rising,ban(level),"commerce","Sell vs board/committee"
bàn,falling,ban(level),"furniture","Table vs board/committee"
phở,dipping,pho(level),"food","Must preserve dipping tone mark"
bánh mì,rising+level,,"food","Bread — common client term"
cà phê,falling+level,,"food","Coffee"
điện thoại,falling+falling,,"tech","Phone"
cuộc họp,heavy+heavy,,"business","Meeting"
báo cáo,rising+rising,,"business","Report"
thanh toán,level+rising,,"finance","Payment"
hóa đơn,rising+level,,"finance","Invoice"
giao hàng,level+falling,,"logistics","Delivery"
```

### 2.3 Create `01_text_corpus/prompts/singlish_system.txt`

```text
You are a Singlish sentence generator for training speech recognition models. Generate realistic conversational Singlish utterances that a Singaporean would naturally say in everyday situations.

Rules:
- Every sentence MUST include at least one Singlish-specific term from the provided list
- Use natural code-switching between English and Hokkien/Malay/Mandarin where appropriate
- Vary the scenarios: hawker centre ordering, office chat, MRT commute, clinic visit, call centre interaction, family conversation, kopitiam talk
- Include discourse particles (lah, lor, leh, hor, meh, sia) naturally at sentence boundaries
- Mix formal and informal registers
- Keep sentences between 5 and 30 words
- Vary speaker demographics: young/old, male/female, different ethnic backgrounds
- Include some sentences with numbers, dates, prices ("$5.50 only lah")
- Include some sentences with proper nouns (MRT station names, food names, Singapore locations)

Return ONLY a JSON array of strings. No markdown, no explanation, no preamble.
```

### 2.4 Create `01_text_corpus/prompts/vietnamese_system.txt`

```text
You are a Vietnamese sentence generator for training speech recognition models. Generate realistic conversational Vietnamese utterances covering everyday situations.

Rules:
- Every sentence MUST include at least one term from the provided vocabulary list
- Include both Northern (Hanoi) and Southern (HCMC) dialect patterns
- Vary scenarios: customer service calls, business meetings, medical consultations, food ordering, logistics/delivery, banking, tech support
- Include natural code-switching with English where common ("Em cần check lại cái report")
- Include sentences with numbers, dates, currency ("hai trăm năm mươi nghìn đồng")
- Keep sentences between 5 and 30 words
- Ensure proper Vietnamese diacritics on all words
- Include some sentences with proper nouns (Vietnamese company names, place names)
- Include tonal minimal pairs in context to challenge ASR

Return ONLY a JSON array of strings. No markdown, no explanation, no preamble.
```

### 2.5 Create `01_text_corpus/generate_singlish.py`

```python
"""Generate Singlish text corpus using Claude API."""
import json
import csv
import random
import click
from pathlib import Path
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise ImportError("pip install anthropic")


def load_lexicon(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def chunk_list(lst: list, n: int) -> list[list]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def generate_batch(
    client: anthropic.Anthropic,
    terms: list[str],
    system_prompt: str,
    batch_size: int = 20,
) -> list[str]:
    user_msg = (
        f"Generate {batch_size} Singlish sentences. "
        f"Each must include at least one of these terms: {', '.join(terms)}\n\n"
        f"Return ONLY a JSON array of strings."
    )
    
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    
    text = resp.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    
    try:
        sentences = json.loads(text)
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, str) and len(s.strip()) > 0]
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse batch, skipping. Raw: {text[:200]}")
    return []


@click.command()
@click.option("--output", "-o", required=True, help="Output JSONL path")
@click.option("--count", "-n", default=1000, help="Target sentence count")
@click.option("--batch-size", default=20, help="Sentences per API call")
@click.option(
    "--lexicon",
    default="01_text_corpus/lexicons/singlish_particles.csv",
    help="Lexicon CSV path",
)
def main(output: str, count: int, batch_size: int, lexicon: str):
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    
    lex = load_lexicon(lexicon)
    terms = [row["term"] for row in lex]
    
    system_prompt = Path("01_text_corpus/prompts/singlish_system.txt").read_text()
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_sentences = []
    n_calls = (count // batch_size) + 1
    
    print(f"Generating ~{count} Singlish sentences in {n_calls} API calls...")
    
    for i in tqdm(range(n_calls)):
        # Sample 5-8 terms per batch for variety
        batch_terms = random.sample(terms, k=min(random.randint(5, 8), len(terms)))
        sentences = generate_batch(client, batch_terms, system_prompt, batch_size)
        all_sentences.extend(sentences)
        
        if len(all_sentences) >= count:
            break
    
    # Deduplicate
    all_sentences = list(dict.fromkeys(all_sentences))[:count]
    
    # Write JSONL
    with open(output_path, "w") as f:
        for sent in all_sentences:
            f.write(json.dumps({"text": sent, "language": "en-SG"}) + "\n")
    
    print(f"Wrote {len(all_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
```

### 2.6 Create `01_text_corpus/generate_vietnamese.py`

Same structure as `generate_singlish.py` but:
- Uses `vietnamese_tonal.csv` lexicon
- Uses `vietnamese_system.txt` system prompt
- Language tag: `"vi"`
- Model: same `claude-sonnet-4-20250514`

```python
"""Generate Vietnamese text corpus using Claude API."""
import json
import csv
import random
import click
from pathlib import Path
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise ImportError("pip install anthropic")


def load_lexicon(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def generate_batch(
    client: anthropic.Anthropic,
    terms: list[str],
    system_prompt: str,
    batch_size: int = 20,
) -> list[str]:
    user_msg = (
        f"Generate {batch_size} Vietnamese sentences. "
        f"Each must include at least one of these terms: {', '.join(terms)}\n\n"
        f"Return ONLY a JSON array of strings."
    )
    
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    
    try:
        sentences = json.loads(text)
        if isinstance(sentences, list):
            return [s for s in sentences if isinstance(s, str) and len(s.strip()) > 0]
    except json.JSONDecodeError:
        print(f"[WARN] Failed to parse batch, skipping. Raw: {text[:200]}")
    return []


@click.command()
@click.option("--output", "-o", required=True, help="Output JSONL path")
@click.option("--count", "-n", default=1000, help="Target sentence count")
@click.option("--batch-size", default=20, help="Sentences per API call")
@click.option(
    "--lexicon",
    default="01_text_corpus/lexicons/vietnamese_tonal.csv",
    help="Lexicon CSV path",
)
def main(output: str, count: int, batch_size: int, lexicon: str):
    client = anthropic.Anthropic()
    
    lex = load_lexicon(lexicon)
    terms = [row["term"] for row in lex]
    
    system_prompt = Path("01_text_corpus/prompts/vietnamese_system.txt").read_text()
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_sentences = []
    n_calls = (count // batch_size) + 1
    
    print(f"Generating ~{count} Vietnamese sentences in {n_calls} API calls...")
    
    for i in tqdm(range(n_calls)):
        batch_terms = random.sample(terms, k=min(random.randint(4, 7), len(terms)))
        sentences = generate_batch(client, batch_terms, system_prompt, batch_size)
        all_sentences.extend(sentences)
        
        if len(all_sentences) >= count:
            break
    
    all_sentences = list(dict.fromkeys(all_sentences))[:count]
    
    with open(output_path, "w") as f:
        for sent in all_sentences:
            f.write(json.dumps({"text": sent, "language": "vi"}) + "\n")
    
    print(f"Wrote {len(all_sentences)} sentences to {output_path}")


if __name__ == "__main__":
    main()
```

---

## Phase 3: TTS Synthesis

### 3.1 Create `02_tts_synthesis/synthesize.py`

This script supports two backends:
- **Fish Speech S2** (primary) — via HTTP API (self-hosted or fish.audio cloud)
- **XTTS-v2** (fallback) — via local Python inference

```python
"""Synthesize audio from text corpus using TTS."""
import json
import hashlib
import random
from pathlib import Path

import click
import httpx
import soundfile as sf
from tqdm import tqdm


def get_voice_files(voice_bank_dir: str) -> list[Path]:
    """Get all WAV files from voice bank directory."""
    vb = Path(voice_bank_dir)
    voices = list(vb.glob("*.wav")) + list(vb.glob("*.mp3")) + list(vb.glob("*.flac"))
    if not voices:
        raise FileNotFoundError(
            f"No audio files in {vb}. Add reference speaker WAVs (10-30s each)."
        )
    return voices


def synthesize_fish_speech(
    text: str,
    voice_path: Path,
    output_path: Path,
    api_url: str = "http://localhost:8080/v1/tts",
    language: str = "en",
) -> bool:
    """Synthesize using Fish Speech S2 API (local or cloud)."""
    try:
        with open(voice_path, "rb") as f:
            resp = httpx.post(
                api_url,
                files={"reference_audio": (voice_path.name, f, "audio/wav")},
                data={"text": text, "language": language},
                timeout=120.0,
            )
        if resp.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            return True
        else:
            print(f"[WARN] Fish Speech returned {resp.status_code}: {resp.text[:200]}")
            return False
    except httpx.ConnectError:
        print("[ERROR] Cannot connect to Fish Speech. Is the server running?")
        print("  Start with: docker run -p 8080:8080 fishaudio/fish-speech:latest")
        return False


def synthesize_xtts(
    text: str,
    voice_path: Path,
    output_path: Path,
    language: str = "en",
) -> bool:
    """Synthesize using XTTS-v2 (local inference, fallback)."""
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("pip install -e '.[tts-local]'  # for XTTS-v2")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=str(voice_path),
        language=language,
        file_path=str(output_path),
    )
    return True


def make_output_filename(text: str, voice_id: str, idx: int) -> str:
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"synth_{idx:06d}_{voice_id}_{text_hash}.wav"


@click.command()
@click.option("--corpus", required=True, help="Input JSONL corpus path")
@click.option("--voice-bank", required=True, help="Directory of reference speaker WAVs")
@click.option("--output-dir", required=True, help="Output directory for WAVs")
@click.option("--lang", default="en", help="Language code (en for Singlish, vi for Vietnamese)")
@click.option("--voices-per-sentence", default=3, help="Number of voice variants per sentence")
@click.option("--backend", default="fish", type=click.Choice(["fish", "xtts"]))
@click.option("--api-url", default="http://localhost:8080/v1/tts", help="Fish Speech API URL")
@click.option("--max-sentences", default=None, type=int, help="Limit sentences to process")
def main(
    corpus: str,
    voice_bank: str,
    output_dir: str,
    lang: str,
    voices_per_sentence: int,
    backend: str,
    api_url: str,
    max_sentences: int | None,
):
    voices = get_voice_files(voice_bank)
    print(f"Found {len(voices)} voice profiles in {voice_bank}")
    
    # Load corpus
    sentences = []
    with open(corpus) as f:
        for line in f:
            sentences.append(json.loads(line))
    
    if max_sentences:
        sentences = sentences[:max_sentences]
    
    print(f"Synthesizing {len(sentences)} sentences × {voices_per_sentence} voices = "
          f"{len(sentences) * voices_per_sentence} audio files")
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    synth_fn = synthesize_fish_speech if backend == "fish" else synthesize_xtts
    
    for idx, item in enumerate(tqdm(sentences)):
        text = item["text"]
        selected_voices = random.sample(voices, k=min(voices_per_sentence, len(voices)))
        
        for voice_path in selected_voices:
            voice_id = voice_path.stem
            filename = make_output_filename(text, voice_id, idx)
            output_path = out / filename
            
            if backend == "fish":
                ok = synthesize_fish_speech(text, voice_path, output_path, api_url, lang)
            else:
                ok = synthesize_xtts(text, voice_path, output_path, lang)
            
            if ok and output_path.exists():
                info = sf.info(str(output_path))
                manifest.append({
                    "audio_filepath": str(output_path),
                    "text": text,
                    "duration": info.duration,
                    "language": item.get("language", lang),
                    "source": "synthetic",
                    "voice_id": voice_id,
                    "augmentation": None,
                })
    
    # Write manifest
    manifest_path = out / "manifest_clean.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Synthesized {len(manifest)} audio files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

### 3.2 Create placeholder voice bank README

Create `02_tts_synthesis/voice_bank/README.md`:

```markdown
# Voice Bank

Place reference speaker WAV files here for TTS voice cloning.

## Requirements
- 10-30 seconds of clean speech per speaker
- 16kHz+ sample rate
- Minimal background noise
- One WAV file per speaker

## Directory structure
- `singlish/` — Singlish speakers (aim for 20-50 speakers)
- `vietnamese/` — Vietnamese speakers (aim for 20-50 speakers, mix Northern/Southern)

## Quick start for prototyping
If you don't have voice samples yet, record yourself or a colleague reading
3-5 sentences in the target language/accent. Even 2-3 speakers is enough
to validate the pipeline.
```

---

## Phase 4: Audio Augmentation

### 4.1 Create `03_augmentation/augment.py`

```python
"""Apply audio augmentations to make synthetic speech more realistic."""
import json
import random
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from audiomentations import (
        Compose,
        AddBackgroundNoise,
        ApplyImpulseResponse,
        TimeStretch,
        PitchShift,
        Gain,
        Mp3Compression,
        BandPassFilter,
    )
except ImportError:
    raise ImportError("pip install audiomentations")


def build_augmentation_pipeline(noise_bank: str) -> Compose:
    """Build the augmentation pipeline. Falls back gracefully if noise files missing."""
    transforms = []
    
    noise_path = Path(noise_bank)
    
    # Background noise (most important augmentation)
    noise_dir = noise_path / "ambient"
    if noise_dir.exists() and list(noise_dir.glob("*.wav")):
        transforms.append(
            AddBackgroundNoise(sounds_path=str(noise_dir), min_snr_db=10, max_snr_db=25, p=0.7)
        )
    else:
        print(f"[WARN] No ambient noise files in {noise_dir}. Skipping noise augmentation.")
        print(f"  Download MUSAN: wget https://openslr.org/resources/17/musan.tar.gz")
    
    # Room impulse response (reverb)
    rir_dir = noise_path / "rir"
    if rir_dir.exists() and list(rir_dir.glob("*.wav")):
        transforms.append(ApplyImpulseResponse(ir_path=str(rir_dir), p=0.5))
    else:
        print(f"[WARN] No RIR files in {rir_dir}. Skipping reverb augmentation.")
        print(f"  Download: wget https://openslr.org/resources/28/rirs_noises.zip")
    
    # These always work (no external files needed)
    transforms.extend([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        Mp3Compression(min_bitrate=32, max_bitrate=64, p=0.3),
        BandPassFilter(min_center_freq=200, max_center_freq=4000, p=0.2),
    ])
    
    return Compose(transforms)


@click.command()
@click.option("--input-dir", required=True, help="Directory with clean WAVs")
@click.option("--output-dir", required=True, help="Output directory for augmented WAVs")
@click.option("--noise-bank", default="03_augmentation/noise_bank", help="Noise bank directory")
@click.option("--variants", default=2, help="Number of augmented variants per clean file")
@click.option("--sample-rate", default=16000, help="Target sample rate")
def main(input_dir: str, output_dir: str, noise_bank: str, variants: int, sample_rate: int):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all clean WAVs
    wav_files = list(in_path.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {in_path}")
        return
    
    augmenter = build_augmentation_pipeline(noise_bank)
    
    # Load clean manifest if exists
    clean_manifest_path = in_path / "manifest_clean.jsonl"
    clean_manifest = {}
    if clean_manifest_path.exists():
        with open(clean_manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                clean_manifest[Path(entry["audio_filepath"]).name] = entry
    
    augmented_manifest = []
    
    print(f"Augmenting {len(wav_files)} files × {variants} variants...")
    
    for wav_path in tqdm(wav_files):
        audio, sr = sf.read(str(wav_path), dtype="float32")
        
        if sr != sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
        
        for v in range(variants):
            aug_audio = augmenter(samples=audio, sample_rate=sr)
            
            aug_filename = f"{wav_path.stem}_aug{v:02d}.wav"
            aug_path = out_path / aug_filename
            sf.write(str(aug_path), aug_audio, sr)
            
            # Build manifest entry
            base_entry = clean_manifest.get(wav_path.name, {})
            augmented_manifest.append({
                "audio_filepath": str(aug_path),
                "text": base_entry.get("text", ""),
                "duration": len(aug_audio) / sr,
                "language": base_entry.get("language", ""),
                "source": "synthetic",
                "voice_id": base_entry.get("voice_id", ""),
                "augmentation": f"variant_{v}",
            })
    
    # Write augmented manifest
    manifest_path = out_path / "manifest_augmented.jsonl"
    with open(manifest_path, "w") as f:
        for entry in augmented_manifest:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Created {len(augmented_manifest)} augmented files. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

### 4.2 Create noise bank setup script `scripts/download_noise.sh`

```bash
#!/bin/bash
# Download noise corpora for augmentation
set -e

NOISE_DIR="03_augmentation/noise_bank"
mkdir -p "$NOISE_DIR/ambient" "$NOISE_DIR/rir"

echo "=== Downloading MUSAN noise corpus ==="
echo "This is ~11GB. If you want a smaller start, skip this and record your own noise."
echo "Press Ctrl+C to skip, or Enter to continue..."
read -r

wget -q --show-progress https://openslr.org/resources/17/musan.tar.gz -O /tmp/musan.tar.gz
tar -xzf /tmp/musan.tar.gz -C /tmp/
# Copy noise subset (skip music and speech to save space)
cp /tmp/musan/noise/*/*.wav "$NOISE_DIR/ambient/" 2>/dev/null || true
rm -rf /tmp/musan /tmp/musan.tar.gz

echo "=== Downloading Room Impulse Responses ==="
wget -q --show-progress https://openslr.org/resources/28/rirs_noises.zip -O /tmp/rirs.zip
unzip -q /tmp/rirs.zip -d /tmp/rirs/
find /tmp/rirs/ -name "*.wav" -exec cp {} "$NOISE_DIR/rir/" \;
rm -rf /tmp/rirs /tmp/rirs.zip

echo "Done. Noise bank at: $NOISE_DIR"
ls -la "$NOISE_DIR/ambient/" | head -5
ls -la "$NOISE_DIR/rir/" | head -5
```

---

## Phase 5: Quality Filtering

### 5.1 Create `04_quality_filter/filter.py`

```python
"""Three-layer quality filter: UTMOS + ASR roundtrip + acoustic sanity."""
import json
from pathlib import Path

import click
import numpy as np
import soundfile as sf
from tqdm import tqdm


def check_duration_sanity(
    audio_path: str, text: str, min_chars_per_sec: float = 4.0, max_chars_per_sec: float = 20.0
) -> tuple[bool, str]:
    """Check if audio duration is reasonable for the text length."""
    info = sf.info(audio_path)
    duration = info.duration
    char_count = len(text)
    
    if duration < 0.5:
        return False, "too_short"
    if duration > 60:
        return False, "too_long"
    
    chars_per_sec = char_count / duration if duration > 0 else 0
    if chars_per_sec < min_chars_per_sec or chars_per_sec > max_chars_per_sec:
        return False, f"rate_mismatch_{chars_per_sec:.1f}cps"
    
    # Check for excessive silence
    audio, sr = sf.read(audio_path, dtype="float32")
    energy = np.abs(audio)
    
    # Leading silence
    threshold = np.max(energy) * 0.02
    leading_silence = np.argmax(energy > threshold) / sr
    if leading_silence > 2.0:
        return False, f"leading_silence_{leading_silence:.1f}s"
    
    return True, "ok"


def score_utmos(audio_path: str) -> float:
    """Score audio naturalness using UTMOS. Returns 1.0-5.0."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoFeatureExtractor
        
        # Lazy load — this is slow on first call
        if not hasattr(score_utmos, "_model"):
            score_utmos._model = AutoModelForSequenceClassification.from_pretrained(
                "sarulab-speech/utmos22_strong", trust_remote_code=True
            )
            score_utmos._fe = AutoFeatureExtractor.from_pretrained(
                "sarulab-speech/utmos22_strong", trust_remote_code=True
            )
        
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        inputs = score_utmos._fe(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            score = score_utmos._model(**inputs).logits.squeeze().item()
        return float(np.clip(score, 1.0, 5.0))
    except Exception as e:
        print(f"[WARN] UTMOS scoring failed: {e}. Returning 3.0 as default.")
        return 3.0


def roundtrip_wer(audio_path: str, expected_text: str, language: str) -> float:
    """Transcribe audio with Whisper and compute WER against expected text."""
    try:
        import whisper
        import jiwer
        
        if not hasattr(roundtrip_wer, "_model"):
            print("Loading Whisper large-v3-turbo... (first call only)")
            roundtrip_wer._model = whisper.load_model("large-v3-turbo")
        
        result = roundtrip_wer._model.transcribe(
            audio_path, language=language, fp16=True
        )
        hypothesis = result["text"].strip()
        
        wer = jiwer.wer(expected_text.strip(), hypothesis)
        return float(wer)
    except Exception as e:
        print(f"[WARN] ASR roundtrip failed: {e}. Returning 1.0 (worst).")
        return 1.0


@click.command()
@click.option("--input-dir", required=True, help="Directory containing clean/ and augmented/ subdirs")
@click.option("--output", required=True, help="Output filtered manifest JSONL")
@click.option("--lang", required=True, help="Language code: en or vi")
@click.option("--utmos-threshold", default=3.5, help="Minimum UTMOS score")
@click.option("--wer-threshold", default=None, type=float, help="Max WER (auto: 0.15 vi, 0.25 en)")
@click.option("--skip-utmos", is_flag=True, help="Skip UTMOS scoring (faster, less filtering)")
@click.option("--skip-whisper", is_flag=True, help="Skip ASR roundtrip (faster, less filtering)")
def main(
    input_dir: str,
    output: str,
    lang: str,
    utmos_threshold: float,
    wer_threshold: float | None,
    skip_utmos: bool,
    skip_whisper: bool,
):
    if wer_threshold is None:
        wer_threshold = 0.15 if lang == "vi" else 0.25
    
    in_path = Path(input_dir)
    
    # Collect all manifests
    all_entries = []
    for manifest_file in in_path.rglob("manifest_*.jsonl"):
        with open(manifest_file) as f:
            for line in f:
                all_entries.append(json.loads(line))
    
    if not all_entries:
        print(f"[ERROR] No manifest files found in {in_path}")
        return
    
    print(f"Filtering {len(all_entries)} samples (UTMOS≥{utmos_threshold}, WER≤{wer_threshold})")
    
    passed = []
    rejected = {"utmos": 0, "wer": 0, "duration": 0, "missing": 0}
    
    for entry in tqdm(all_entries):
        audio_path = entry["audio_filepath"]
        text = entry.get("text", "")
        
        if not Path(audio_path).exists():
            rejected["missing"] += 1
            continue
        
        # Layer 3: Duration sanity (fast, do first)
        ok, reason = check_duration_sanity(audio_path, text)
        if not ok:
            rejected["duration"] += 1
            continue
        
        # Layer 1: UTMOS
        if not skip_utmos:
            mos = score_utmos(audio_path)
            if mos < utmos_threshold:
                rejected["utmos"] += 1
                continue
            entry["utmos_score"] = mos
        
        # Layer 2: ASR roundtrip
        if not skip_whisper:
            wer = roundtrip_wer(audio_path, text, lang)
            if wer > wer_threshold:
                rejected["wer"] += 1
                continue
            entry["roundtrip_wer"] = wer
        
        passed.append(entry)
    
    # Write filtered manifest
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in passed:
            f.write(json.dumps(entry) + "\n")
    
    total = len(all_entries)
    print(f"\n=== Filter Results ===")
    print(f"  Total:    {total}")
    print(f"  Passed:   {len(passed)} ({len(passed)/total:.1%})")
    print(f"  Rejected: {sum(rejected.values())}")
    for reason, count in rejected.items():
        if count > 0:
            print(f"    - {reason}: {count}")
    print(f"  Output:   {out_path}")


if __name__ == "__main__":
    main()
```

---

## Phase 6: Dataset Export

### 6.1 Create `06_dataset_export/export_nemo_manifest.py`

```python
"""Export filtered data to NeMo-compatible manifest format."""
import json
from pathlib import Path
import click


@click.command()
@click.option("--input", "input_path", required=True, help="Filtered manifest JSONL")
@click.option("--output", "output_path", required=True, help="Output NeMo manifest JSON")
@click.option("--audio-base", default=None, help="Rebase audio paths relative to this dir")
def main(input_path: str, output_path: str, audio_base: str | None):
    """Convert internal manifest to NeMo training format.
    
    NeMo format: one JSON object per line with keys:
    - audio_filepath (relative or absolute path to WAV)
    - text (ground truth transcript)
    - duration (float, seconds)
    """
    entries = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line)
            
            nemo_entry = {
                "audio_filepath": entry["audio_filepath"],
                "text": entry["text"],
                "duration": entry["duration"],
            }
            
            if audio_base:
                nemo_entry["audio_filepath"] = str(
                    Path(entry["audio_filepath"]).relative_to(audio_base)
                )
            
            entries.append(nemo_entry)
    
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    total_hours = sum(e["duration"] for e in entries) / 3600
    print(f"Exported {len(entries)} samples ({total_hours:.1f} hours) to {out}")


if __name__ == "__main__":
    main()
```

---

## Phase 7: README

### 7.1 Create `README.md`

```markdown
# VALSEA ASR Data Generation Pipeline

Generates synthetic training data for finetuning:
- **Blaze** (Vietnamese ASR)
- **MERaLiON** (Singlish ASR)

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

## Pipeline

```
Text Corpus (Claude API) → TTS (Fish Speech S2) → Augmentation → Quality Filter → NeMo Manifest
```

## Output

NeMo-format JSONL manifests in `outputs/{language}/train_manifest.json`, ready for consumption by the VALSEA monorepo finetuning scripts.

## Configuration

Edit `scripts/config.py` for thresholds, paths, and TTS settings.
```

---

## Execution Instructions for Claude Code

When you open this repo in Claude Code (VS Code), tell it:

> "Follow the build plan in BUILDPLAN.md. Start with Phase 1 to create all files and directories. Then I'll tell you which phase to execute next."

Or for maximum speed:

> "Read BUILDPLAN.md and create every file specified in Phases 1-7. Create the directory structure, all Python files, all config files, the Makefile, and README. Do not execute any code yet — just scaffold everything."

After scaffolding, run:
1. `pip install -e ".[dev]"` to validate the package installs
2. Add 2-3 voice WAV files to the voice_bank directories
3. `export ANTHROPIC_API_KEY=...`
4. `make prototype` to test end-to-end with 50 sentences
