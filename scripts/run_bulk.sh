#!/usr/bin/env bash
# Bulk synthetic ASR data generation for Chinese + Hindi.
#
# Target per (lang × bucket × aug-state): ~3 hr of 16 kHz mono WAV.
# Aug is 1:1 derived from clean (augmentation --variants 1).
#
# Tunables (env vars):
#   COUNT_SHORT       sentences per language for short bucket (default: 1500)
#   COUNT_LONG        sentences per language for long bucket  (default: 250)
#   VOICES_PER_SENT   voices per sentence                     (default: 2)
#   WORKERS_ZH        concurrent TTS calls for Chinese (edge) (default: 6)
#   WORKERS_HI        concurrent TTS calls for Hindi  (Sarvam)(default: 4)
#   TTS_BACKEND_ZH    edge | minimax | qwen                   (default: edge)
#   TTS_BACKEND_HI    sarvam | edge                           (default: sarvam)
#   HF_REPO_ZH        HF dataset repo for Chinese             (default: skip push)
#   HF_REPO_HI        HF dataset repo for Hindi               (default: skip push)
#   SKIP_HF           if set, never push to HF
#
# Resumability: every stage is idempotent. If Sarvam dies mid-run, top up and
# re-run this script — the resume logic skips WAVs that already exist.
#
# Usage:
#   bash scripts/run_bulk.sh
#   # or with a HF push:
#   HF_REPO_ZH=valsea/synthetic-asr-zh HF_REPO_HI=valsea/synthetic-asr-hi \
#     bash scripts/run_bulk.sh
#   # Sarvam-credit-out fallback for Hindi:
#   TTS_BACKEND_HI=edge bash scripts/run_bulk.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# ---- Tunables ----
COUNT_SHORT="${COUNT_SHORT:-1500}"
COUNT_LONG="${COUNT_LONG:-250}"
VOICES_PER_SENT="${VOICES_PER_SENT:-2}"
WORKERS_ZH="${WORKERS_ZH:-6}"
WORKERS_HI="${WORKERS_HI:-4}"
TTS_BACKEND_ZH="${TTS_BACKEND_ZH:-edge}"
TTS_BACKEND_HI="${TTS_BACKEND_HI:-sarvam}"

LOG_DIR="logs/bulk_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo ""
echo "================================================="
echo " VALSEA bulk synthetic ASR data generation"
echo "================================================="
echo "  Counts:    short=$COUNT_SHORT,  long=$COUNT_LONG  (per language)"
echo "  Voices/sent: $VOICES_PER_SENT"
echo "  Workers:   zh=$WORKERS_ZH,  hi=$WORKERS_HI"
echo "  Backends:  zh=$TTS_BACKEND_ZH,  hi=$TTS_BACKEND_HI"
echo "  Logs:      $LOG_DIR"
echo "  Python:    $($PYTHON --version)"
echo ""

# ---- Preflight ----
echo ">>> [0/6] Preflight"
"$PYTHON" -c "
import os, sys
from dotenv import load_dotenv
load_dotenv()
need = ['OPENAI_API_KEY']
if '$TTS_BACKEND_HI' == 'sarvam': need.append('SARVAM_API_KEY')
if '$TTS_BACKEND_ZH' == 'minimax': need.append('MINIMAX_API_KEY')
if '$TTS_BACKEND_ZH' == 'qwen':    need.append('DASHSCOPE_API_KEY')
missing = [k for k in need if not os.environ.get(k)]
if missing:
    print(f'[ABORT] Missing env vars: {missing}')
    sys.exit(1)
print('  env vars OK:', need)
# audiomentations check
try:
    import audiomentations
    print(f'  audiomentations {audiomentations.__version__} OK')
except ImportError:
    print('  [WARN] audiomentations not installed — aug will fall back to pass-through copies.')
"

run_lang_bucket () {
    local LANG_CODE="$1"   # zh | hi
    local LENGTH="$2"      # short | long
    local COUNT="$3"
    local BACKEND="$4"
    local WORKERS="$5"

    local DIRNAME
    if [[ "$LANG_CODE" == "zh" ]]; then DIRNAME="chinese"; else DIRNAME="hindi"; fi
    local OUT="outputs/$DIRNAME/$LENGTH"
    local STAGE_LOG="$LOG_DIR/${LANG_CODE}_${LENGTH}"
    mkdir -p "$OUT" "$STAGE_LOG"

    echo ""
    echo "-------------------------------------------------"
    echo " $LANG_CODE / $LENGTH  (count=$COUNT, backend=$BACKEND)"
    echo "-------------------------------------------------"

    # ── 1. Text corpus ─────────────────────────────────────────────────
    if [[ -f "$OUT/corpus.jsonl" ]] && \
       [[ $(wc -l < "$OUT/corpus.jsonl") -ge "$COUNT" ]]; then
        echo "  [1/4] corpus.jsonl already has >=${COUNT} sentences, skipping text gen"
    else
        echo "  [1/4] text gen: $COUNT sentences ($LENGTH)"
        if [[ "$LANG_CODE" == "zh" ]]; then
            "$PYTHON" data_generation/01_text_corpus/generate_chinese.py \
                --output "$OUT/corpus.jsonl" \
                --count "$COUNT" --batch-size 10 \
                --length-target "$LENGTH" \
                2>&1 | tee "$STAGE_LOG/01_text.log"
        else
            "$PYTHON" data_generation/01_text_corpus/generate_hindi.py \
                --output "$OUT/corpus.jsonl" \
                --count "$COUNT" --batch-size 10 \
                --length-target "$LENGTH" \
                2>&1 | tee "$STAGE_LOG/01_text.log"
        fi
    fi

    # ── 2. TTS synthesis (resumable, parallel) ─────────────────────────
    echo "  [2/4] tts: backend=$BACKEND, workers=$WORKERS, voices/sent=$VOICES_PER_SENT"
    "$PYTHON" data_generation/02_tts_synthesis/synthesize.py \
        --corpus "$OUT/corpus.jsonl" \
        --output-dir "$OUT/clean" \
        --lang "$LANG_CODE" \
        --backend "$BACKEND" \
        --voices-per-sentence "$VOICES_PER_SENT" \
        --workers "$WORKERS" \
        --resume \
        2>&1 | tee "$STAGE_LOG/02_tts.log"

    # ── 3. Augmentation (1 variant per clean clip) ─────────────────────
    echo "  [3/4] augmentation (1 variant per clean)"
    "$PYTHON" data_generation/03_augmentation/augment.py \
        --input-dir "$OUT/clean" \
        --output-dir "$OUT/augmented" \
        --noise-bank data_generation/03_augmentation/noise_bank \
        --variants 1 \
        2>&1 | tee "$STAGE_LOG/03_aug.log"

    # ── 4. Filter (duration sanity only) + export ──────────────────────
    echo "  [4/4] filter (duration only) + export"
    "$PYTHON" data_generation/04_quality_filter/filter.py \
        --input-dir "$OUT" \
        --output "$OUT/manifest_filtered.jsonl" \
        --lang "$LANG_CODE" \
        --skip-utmos \
        --skip-whisper \
        2>&1 | tee "$STAGE_LOG/04_filter.log"

    "$PYTHON" data_generation/06_dataset_export/export_nemo_manifest.py \
        --input "$OUT/manifest_filtered.jsonl" \
        --output "$OUT/train_manifest.json" \
        2>&1 | tee "$STAGE_LOG/05_export.log"

    "$PYTHON" -c "
import json
rows = [json.loads(l) for l in open('$OUT/train_manifest.json')]
total_min = sum(r['duration'] for r in rows) / 60
print(f'  >>> $LANG_CODE/$LENGTH complete: {len(rows)} entries, {total_min:.1f} min audio')
"
}

# ---- Generate (text → TTS → aug → filter → export) per (lang, length) ----
echo ""
echo ">>> [1–5/6] Per-(lang, length) generation"
run_lang_bucket zh short "$COUNT_SHORT" "$TTS_BACKEND_ZH" "$WORKERS_ZH"
run_lang_bucket zh long  "$COUNT_LONG"  "$TTS_BACKEND_ZH" "$WORKERS_ZH"
run_lang_bucket hi short "$COUNT_SHORT" "$TTS_BACKEND_HI" "$WORKERS_HI"
run_lang_bucket hi long  "$COUNT_LONG"  "$TTS_BACKEND_HI" "$WORKERS_HI"

# ---- HF push ----
echo ""
echo ">>> [6/6] HuggingFace dataset push"
if [[ -n "${SKIP_HF:-}" ]]; then
    echo "  SKIP_HF set, skipping push"
elif [[ -z "${HF_REPO_ZH:-}" && -z "${HF_REPO_HI:-}" ]]; then
    echo "  HF_REPO_ZH and HF_REPO_HI not set, skipping push"
    echo "  To push later: HF_REPO_ZH=<org>/synthetic-asr-zh \\"
    echo "                 HF_REPO_HI=<org>/synthetic-asr-hi \\"
    echo "                 $PYTHON data_generation/07_hf_push/push_to_hf.py --lang zh --repo-id <...>"
else
    if [[ -n "${HF_REPO_ZH:-}" ]]; then
        echo "  pushing zh to $HF_REPO_ZH"
        "$PYTHON" data_generation/07_hf_push/push_to_hf.py --lang zh --repo-id "$HF_REPO_ZH" 2>&1 | tee "$LOG_DIR/06_hf_zh.log"
    fi
    if [[ -n "${HF_REPO_HI:-}" ]]; then
        echo "  pushing hi to $HF_REPO_HI"
        "$PYTHON" data_generation/07_hf_push/push_to_hf.py --lang hi --repo-id "$HF_REPO_HI" 2>&1 | tee "$LOG_DIR/06_hf_hi.log"
    fi
fi

echo ""
echo "================================================="
echo " Bulk run complete"
echo "================================================="
"$PYTHON" -c "
import json, glob, os
total_sec = 0
for p in glob.glob('outputs/{chinese,hindi}/{short,long}/train_manifest.json', recursive=False):
    pass  # glob doesn't expand braces; do it manually
for lang in ('chinese', 'hindi'):
    for bucket in ('short', 'long'):
        p = f'outputs/{lang}/{bucket}/train_manifest.json'
        if os.path.exists(p):
            rows = [json.loads(l) for l in open(p)]
            sec = sum(r['duration'] for r in rows)
            total_sec += sec
            print(f'  {lang}/{bucket}: {len(rows):4d} entries, {sec/60:6.1f} min ({sec/3600:.2f} hr)')
print(f'  TOTAL: {total_sec/3600:.2f} hr')
"
echo ""
echo "Logs:  $LOG_DIR"
