# Use .venv if it exists, otherwise fall back to whatever 'python' resolves to.
# To create the venv: /usr/local/bin/python3.10 -m venv .venv
PYTHON := $(shell [ -f .venv/bin/python ] && echo .venv/bin/python || echo python3)
PIP    := $(PYTHON) -m pip

.PHONY: setup setup-prototype check-env generate-text synthesize augment filter export clean prototype prototype-edge

setup:
	$(PIP) install -e ".[quality,dev]"
	@echo "For local TTS: $(PIP) install -e '.[tts-local]'"
	@echo "For real data curation: $(PIP) install -e '.[real-data]'"

setup-prototype:
	$(PIP) install -e ".[prototype,dev]"
	@echo "Ready for: make prototype-edge"

check-env:
	$(PYTHON) scripts/check_env.py

generate-text:
	$(PYTHON)01_text_corpus/generate_singlish.py --output outputs/singlish/corpus.jsonl --count 1000
	$(PYTHON)01_text_corpus/generate_vietnamese.py --output outputs/vietnamese/corpus.jsonl --count 1000

synthesize:
	$(PYTHON)02_tts_synthesis/synthesize.py --corpus outputs/singlish/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/singlish --output-dir outputs/singlish/clean --lang en
	$(PYTHON)02_tts_synthesis/synthesize.py --corpus outputs/vietnamese/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/vietnamese --output-dir outputs/vietnamese/clean --lang vi

augment:
	$(PYTHON)03_augmentation/augment.py --input-dir outputs/singlish/clean --output-dir outputs/singlish/augmented --noise-bank 03_augmentation/noise_bank
	$(PYTHON)03_augmentation/augment.py --input-dir outputs/vietnamese/clean --output-dir outputs/vietnamese/augmented --noise-bank 03_augmentation/noise_bank

filter:
	$(PYTHON)04_quality_filter/filter.py --input-dir outputs/singlish --output outputs/singlish/manifest_filtered.jsonl --lang en
	$(PYTHON)04_quality_filter/filter.py --input-dir outputs/vietnamese --output outputs/vietnamese/manifest_filtered.jsonl --lang vi

export:
	$(PYTHON)06_dataset_export/export_nemo_manifest.py --input outputs/singlish/manifest_filtered.jsonl --output outputs/singlish/train_manifest.json
	$(PYTHON)06_dataset_export/export_nemo_manifest.py --input outputs/vietnamese/manifest_filtered.jsonl --output outputs/vietnamese/train_manifest.json

# Quick prototype: generate 50 sentences, synthesize, filter — end to end
prototype:
	$(PYTHON)01_text_corpus/generate_singlish.py --output outputs/singlish/corpus.jsonl --count 50
	$(PYTHON)02_tts_synthesis/synthesize.py --corpus outputs/singlish/corpus.jsonl --voice-bank 02_tts_synthesis/voice_bank/singlish --output-dir outputs/singlish/clean --lang en --voices-per-sentence 1
	$(PYTHON)03_augmentation/augment.py --input-dir outputs/singlish/clean --output-dir outputs/singlish/augmented --noise-bank 03_augmentation/noise_bank --variants 1
	$(PYTHON)04_quality_filter/filter.py --input-dir outputs/singlish --output outputs/singlish/manifest_filtered.jsonl --lang en
	@echo "Done. Check outputs/singlish/manifest_filtered.jsonl"

# Zero-infrastructure prototype: edge-tts (no docker, no voice bank), skip heavy quality filters
# Runs ~10 Singlish sentences end-to-end. Needs: OPENAI_API_KEY + pip install -e '.[prototype,dev]'
prototype-edge:
	@$(PYTHON)scripts/check_env.py
	$(PYTHON)01_text_corpus/generate_singlish.py --output outputs/singlish/corpus.jsonl --count 10 --batch-size 10
	$(PYTHON)02_tts_synthesis/synthesize.py \
		--corpus outputs/singlish/corpus.jsonl \
		--output-dir outputs/singlish/clean \
		--lang en \
		--backend edge \
		--voices-per-sentence 2
	$(PYTHON)03_augmentation/augment.py \
		--input-dir outputs/singlish/clean \
		--output-dir outputs/singlish/augmented \
		--noise-bank 03_augmentation/noise_bank \
		--variants 1
	$(PYTHON)04_quality_filter/filter.py \
		--input-dir outputs/singlish \
		--output outputs/singlish/manifest_filtered.jsonl \
		--lang en \
		--skip-utmos \
		--skip-whisper
	$(PYTHON)06_dataset_export/export_nemo_manifest.py \
		--input outputs/singlish/manifest_filtered.jsonl \
		--output outputs/singlish/train_manifest.json
	@echo ""
	@echo "=== Prototype complete ==="
	@echo "NeMo manifest: outputs/singlish/train_manifest.json"
	@$(PYTHON)-c "import json; rows=[json.loads(l) for l in open('outputs/singlish/train_manifest.json')]; print(f'  {len(rows)} samples, {sum(r[\"duration\"] for r in rows)/60:.1f} min total')"

clean:
	rm -rf outputs/singlish/clean/* outputs/singlish/augmented/* outputs/vietnamese/clean/* outputs/vietnamese/augmented/*
