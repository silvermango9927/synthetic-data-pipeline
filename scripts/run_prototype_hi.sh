#!/usr/bin/env bash
# Hindi prototype: OpenAI text → Sarvam TTS → augment → filter → NeMo manifest
# Two length buckets per run: short (~5s) and long (~30s).
#
# Requirements:
#   - OPENAI_API_KEY and SARVAM_API_KEY in .env (or exported)
#   - .venv populated:
#       /usr/local/bin/python3.10 -m venv .venv
#       .venv/bin/pip install -e ".[prototype,tts-cloud]"
#   - Run: bash scripts/run_prototype_hi.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

LANG_CODE="hi"
LANG_DIR="outputs/hindi"
COUNT_SHORT="${COUNT_SHORT:-50}"
COUNT_LONG="${COUNT_LONG:-50}"

echo ""
echo "=============================="
echo " VALSEA ASR — Hindi (hi) Prototype"
echo "=============================="
echo "Python: $($PYTHON --version)"
echo "Dir:    $REPO_DIR"
echo "Counts: short=$COUNT_SHORT, long=$COUNT_LONG"
echo ""

run_bucket () {
    local LENGTH="$1"          # short | long
    local COUNT="$2"
    local OUT="$LANG_DIR/$LENGTH"
    echo ""
    echo "=============================="
    echo " bucket: $LENGTH ($COUNT samples)"
    echo "=============================="

    # ── Step 1: Generate text corpus ─────────────────────────────────────────
    echo ">>> [1/5] Generate $COUNT Hindi sentences ($LENGTH-form)"
    "$PYTHON" 01_text_corpus/generate_hindi.py \
        --output "$OUT/corpus.jsonl" \
        --count "$COUNT" \
        --batch-size 10 \
        --length-target "$LENGTH"

    echo ""
    echo "    First 3 samples:"
    "$PYTHON" -c "
import json
rows = [json.loads(l) for l in open('$OUT/corpus.jsonl')]
for i, r in enumerate(rows[:3], 1):
    print(f'    {i}. {r[\"text\"][:80]}')"

    # ── Step 2: TTS synthesis ────────────────────────────────────────────────
    echo ""
    echo ">>> [2/5] Synthesize via Sarvam TTS (bulbul:v3, 2 voices/sentence)"
    "$PYTHON" 02_tts_synthesis/synthesize.py \
        --corpus "$OUT/corpus.jsonl" \
        --output-dir "$OUT/clean" \
        --lang "$LANG_CODE" \
        --backend sarvam \
        --voices-per-sentence 2

    WAV_COUNT=$(ls "$OUT/clean"/*.wav 2>/dev/null | wc -l | tr -d ' ')
    echo "    WAV files created: $WAV_COUNT"

    # ── Step 3: Augmentation ─────────────────────────────────────────────────
    echo ""
    echo ">>> [3/5] Augmentation (1 variant per file)"
    "$PYTHON" 03_augmentation/augment.py \
        --input-dir "$OUT/clean" \
        --output-dir "$OUT/augmented" \
        --noise-bank 03_augmentation/noise_bank \
        --variants 1

    AUG_COUNT=$(ls "$OUT/augmented"/*.wav 2>/dev/null | wc -l | tr -d ' ')
    echo "    Augmented WAV files: $AUG_COUNT"

    # ── Step 4: Quality filter ───────────────────────────────────────────────
    echo ""
    echo ">>> [4/5] Quality filter (duration sanity only — skip UTMOS + Whisper)"
    "$PYTHON" 04_quality_filter/filter.py \
        --input-dir "$OUT" \
        --output "$OUT/manifest_filtered.jsonl" \
        --lang "$LANG_CODE" \
        --skip-utmos \
        --skip-whisper

    # ── Step 5: Export ───────────────────────────────────────────────────────
    echo ""
    echo ">>> [5/5] Export to NeMo manifest"
    "$PYTHON" 06_dataset_export/export_nemo_manifest.py \
        --input "$OUT/manifest_filtered.jsonl" \
        --output "$OUT/train_manifest.json"

    echo ""
    echo "=== $LENGTH bucket complete ==="
    "$PYTHON" -c "
import json
rows = [json.loads(l) for l in open('$OUT/train_manifest.json')]
total_min = sum(r['duration'] for r in rows) / 60
print(f'  Samples : {len(rows)}')
print(f'  Duration: {total_min:.1f} min')
print(f'  Manifest: $OUT/train_manifest.json')
print()
print('  First 3 entries:')
for r in rows[:3]:
    print(f'    [{r[\"duration\"]:.1f}s] {r[\"text\"][:60]}')"
}

run_bucket short "$COUNT_SHORT"
run_bucket long  "$COUNT_LONG"

echo ""
echo "=============================="
echo " Hindi prototype complete"
echo "=============================="
echo "  outputs/hindi/short/train_manifest.json"
echo "  outputs/hindi/long/train_manifest.json"
echo ""
