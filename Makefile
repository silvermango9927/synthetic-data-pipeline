.PHONY: setup generate-text synthesize augment filter export clean prototype

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
