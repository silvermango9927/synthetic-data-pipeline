#!/usr/bin/env bash
# Zero-infrastructure prototype: Claude text → edge-tts → augment → filter → NeMo manifest
# Requirements: OPENAI_API_KEY set, .venv created and populated
#   /usr/local/bin/python3.10 -m venv .venv
#   .venv/bin/pip install -e ".[prototype,dev]"
#   export OPENAI_API_KEY=sk-...
#   bash scripts/run_prototype.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Resolve python — prefer .venv if present
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo ""
echo "=============================="
echo " VALSEA ASR — Prototype Run"
echo "=============================="
echo "Python: $($PYTHON --version)"
echo "Dir:    $REPO_DIR"
echo ""

# ── Preflight ───────────────────────────────────────────────────────────────
echo ">>> Step 0: Preflight check"
if ! "$PYTHON" scripts/check_env.py; then
    echo ""
    echo "[ABORT] Fix the FAILs above before running the prototype."
    exit 1
fi

# ── Step 1: Generate text corpus ─────────────────────────────────────────────
echo ">>> Step 1: Generate 10 Singlish sentences via Claude API"
"$PYTHON" data_generation/01_text_corpus/generate_singlish.py \
    --output outputs/singlish/corpus.jsonl \
    --count 10 \
    --batch-size 10

echo ""
echo "    Generated sentences:"
"$PYTHON" -c "
import json
rows = [json.loads(l) for l in open('outputs/singlish/corpus.jsonl')]
for i, r in enumerate(rows, 1):
    print(f'    {i:2d}. {r[\"text\"]}')
"

# ── Step 2: TTS synthesis ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Synthesize audio (edge-tts: en-SG-LunaNeural + en-SG-WayneNeural)"
"$PYTHON" data_generation/02_tts_synthesis/synthesize.py \
    --corpus outputs/singlish/corpus.jsonl \
    --output-dir outputs/singlish/clean \
    --lang en \
    --backend edge \
    --voices-per-sentence 2

WAV_COUNT=$(ls outputs/singlish/clean/*.wav 2>/dev/null | wc -l | tr -d ' ')
echo "    WAV files created: $WAV_COUNT"

# ── Step 3: Augmentation ──────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Augmentation (1 variant per file)"
"$PYTHON" data_generation/03_augmentation/augment.py \
    --input-dir outputs/singlish/clean \
    --output-dir outputs/singlish/augmented \
    --noise-bank data_generation/03_augmentation/noise_bank \
    --variants 1

AUG_COUNT=$(ls outputs/singlish/augmented/*.wav 2>/dev/null | wc -l | tr -d ' ')
echo "    Augmented WAV files: $AUG_COUNT"

# ── Step 4: Quality filter ────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: Quality filter (duration sanity only — skip UTMOS + Whisper)"
"$PYTHON" data_generation/04_quality_filter/filter.py \
    --input-dir outputs/singlish \
    --output outputs/singlish/manifest_filtered.jsonl \
    --lang en \
    --skip-utmos \
    --skip-whisper

# ── Step 5: Export ────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 5: Export to NeMo manifest"
"$PYTHON" data_generation/06_dataset_export/export_nemo_manifest.py \
    --input outputs/singlish/manifest_filtered.jsonl \
    --output outputs/singlish/train_manifest.json

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo " Prototype Complete"
echo "=============================="
"$PYTHON" -c "
import json
rows = [json.loads(l) for l in open('outputs/singlish/train_manifest.json')]
total_min = sum(r['duration'] for r in rows) / 60
print(f'  Samples : {len(rows)}')
print(f'  Duration: {total_min:.1f} min')
print(f'  Manifest: outputs/singlish/train_manifest.json')
print()
print('  First 3 entries:')
for r in rows[:3]:
    print(f'    [{r[\"duration\"]:.1f}s] {r[\"text\"][:60]}')
"
echo ""
