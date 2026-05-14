#!/usr/bin/env bash
# Smoke test for the bulk pipeline before the multi-hour production run.
#
# Runs the full bulk pipeline at ~1% scale (20 short + 5 long sentences per
# language) with HF push disabled. Verifies that every stage wires together
# end-to-end and that the new resume / worker / deterministic-rotation /
# HF-staging code paths don't error.
#
# Wall-clock: ~5–10 min.
#
# ─── Modes ───────────────────────────────────────────────────────────────────
#   default            Smoke test runs against the existing outputs/ tree.
#                      Resume logic will skip existing WAVs, so the test
#                      mostly exercises the wiring on already-synthesised data.
#
#   ISOLATE=1          Hermetic mode. Renames outputs/ → outputs.preserved_<ts>
#                      before the run, runs against a fresh empty outputs/,
#                      then renames the smoke result to outputs.smoke_<ts> and
#                      restores the preserved tree. Use this to actually
#                      exercise text gen + TTS from scratch.
#
# ─── Usage ───────────────────────────────────────────────────────────────────
#   bash scripts/smoke_test_bulk.sh             # against existing outputs/
#   ISOLATE=1 bash scripts/smoke_test_bulk.sh   # hermetic, ~5 min, ~$0.50
#
# ─── What to look for ────────────────────────────────────────────────────────
#   PASS lines for: env preflight, text gen, TTS, augmentation, filter+export,
#   HF staging dry-run. Any FAIL aborts and prints the offending log path.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON=".venv/bin/python"
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/smoke_$TS"
mkdir -p "$LOG_DIR"

# ── Counts: ~1% of the bulk target ───────────────────────────────────────────
export COUNT_SHORT=20
export COUNT_LONG=5
export VOICES_PER_SENT=2
export WORKERS_ZH=4
export WORKERS_HI=2
export SKIP_HF=1   # never push during smoke

# Default backends (override via env to test other paths).
export TTS_BACKEND_ZH="${TTS_BACKEND_ZH:-edge}"
export TTS_BACKEND_HI="${TTS_BACKEND_HI:-sarvam}"

PRESERVED_DIR=""
SMOKE_DIR=""

cleanup () {
    local rc=$?
    if [[ -n "$PRESERVED_DIR" && -d "$PRESERVED_DIR" ]]; then
        # Always try to put the user's real outputs back, even on failure.
        if [[ -d outputs ]]; then
            SMOKE_DIR="outputs.smoke_$TS"
            mv outputs "$SMOKE_DIR"
            echo ""
            echo "  Smoke outputs moved to: $SMOKE_DIR"
            echo "  Delete with:  rm -rf $SMOKE_DIR"
        fi
        mv "$PRESERVED_DIR" outputs
        echo "  Original outputs/ restored from $PRESERVED_DIR"
    fi
    if [[ $rc -ne 0 ]]; then
        echo ""
        echo "  FAIL — see logs in $LOG_DIR"
    fi
    return $rc
}
trap cleanup EXIT

echo "================================================="
echo " VALSEA bulk-pipeline smoke test"
echo "================================================="
echo "  Mode:      ${ISOLATE:+ISOLATED }${ISOLATE:-against-existing-outputs}"
echo "  Counts:    short=$COUNT_SHORT, long=$COUNT_LONG (per language)"
echo "  Backends:  zh=$TTS_BACKEND_ZH, hi=$TTS_BACKEND_HI"
echo "  Logs:      $LOG_DIR"
echo ""

# ── Preflight ────────────────────────────────────────────────────────────────
echo ">>> [1/7] Preflight: venv + deps + env"
"$PYTHON" -c "
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
import os
need = ['OPENAI_API_KEY']
if '$TTS_BACKEND_HI' == 'sarvam': need.append('SARVAM_API_KEY')
missing = [k for k in need if not os.environ.get(k)]
assert not missing, f'Missing env vars: {missing}'
import audiomentations, pydub, edge_tts, huggingface_hub, soundfile
print(f'  audiomentations={audiomentations.__version__}, hf_hub={huggingface_hub.__version__}')
print('  PASS preflight')
" 2>&1 | tee "$LOG_DIR/01_preflight.log"

# ── Optional: relocate real outputs/ for a hermetic run ─────────────────────
if [[ "${ISOLATE:-}" == "1" ]]; then
    if [[ -d outputs ]]; then
        PRESERVED_DIR="outputs.preserved_$TS"
        mv outputs "$PRESERVED_DIR"
        echo "  ISOLATE: moved outputs/ → $PRESERVED_DIR (will be restored on exit)"
    fi
    mkdir -p outputs
fi

# ── Run the bulk script at smoke scale ───────────────────────────────────────
echo ""
echo ">>> [2/7] Running scripts/run_bulk.sh at smoke scale (~5 min)"
COUNT_SHORT=$COUNT_SHORT COUNT_LONG=$COUNT_LONG \
VOICES_PER_SENT=$VOICES_PER_SENT \
WORKERS_ZH=$WORKERS_ZH WORKERS_HI=$WORKERS_HI \
TTS_BACKEND_ZH=$TTS_BACKEND_ZH TTS_BACKEND_HI=$TTS_BACKEND_HI \
SKIP_HF=1 \
bash scripts/run_bulk.sh 2>&1 | tee "$LOG_DIR/02_bulk.log"

# ── Verifications ────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/7] Verify: clean WAV counts per (lang, bucket)"
"$PYTHON" - <<EOF 2>&1 | tee "$LOG_DIR/03_verify_clean.log"
from pathlib import Path
expected = {('chinese','short'): 20*2, ('chinese','long'): 5*2,
            ('hindi','short'): 20*2,   ('hindi','long'): 5*2}
fail = False
for (lang, bucket), exp in expected.items():
    d = Path(f'outputs/{lang}/{bucket}/clean')
    wavs = list(d.glob('synth_*.wav'))
    # Resume may carry old WAVs; we only assert the floor of exp.
    if len(wavs) < exp:
        print(f'  FAIL {lang}/{bucket}: {len(wavs)} clean WAVs, expected >= {exp}')
        fail = True
    else:
        print(f'  PASS {lang}/{bucket}: {len(wavs)} clean WAVs (>= {exp})')
assert not fail
EOF

echo ""
echo ">>> [4/7] Verify: augmented manifests have real transforms (not passthrough)"
"$PYTHON" - <<'EOF' 2>&1 | tee "$LOG_DIR/04_verify_aug.log"
import json
from pathlib import Path
fail = False
for lang in ('chinese', 'hindi'):
    for bucket in ('short', 'long'):
        m = Path(f'outputs/{lang}/{bucket}/augmented/manifest_augmented.jsonl')
        if not m.exists():
            print(f'  FAIL {lang}/{bucket}: no manifest_augmented.jsonl')
            fail = True
            continue
        rows = [json.loads(l) for l in open(m)]
        passthrough = sum(1 for r in rows if r.get('augmentation') == 'passthrough')
        if passthrough:
            print(f'  FAIL {lang}/{bucket}: {passthrough}/{len(rows)} rows are passthrough — audiomentations not active')
            fail = True
        else:
            print(f'  PASS {lang}/{bucket}: {len(rows)} aug rows, 0 passthrough')
assert not fail
EOF

echo ""
echo ">>> [5/7] Verify: train_manifest.json is valid NeMo (audio_filepath + text + duration)"
"$PYTHON" - <<'EOF' 2>&1 | tee "$LOG_DIR/05_verify_export.log"
import json
from pathlib import Path
fail = False
for lang in ('chinese', 'hindi'):
    for bucket in ('short', 'long'):
        p = Path(f'outputs/{lang}/{bucket}/train_manifest.json')
        if not p.exists():
            print(f'  FAIL {lang}/{bucket}: no train_manifest.json')
            fail = True
            continue
        rows = [json.loads(l) for l in open(p)]
        if not rows:
            print(f'  FAIL {lang}/{bucket}: train_manifest.json is empty')
            fail = True
            continue
        bad = [r for r in rows if not all(k in r for k in ('audio_filepath','text','duration'))]
        if bad:
            print(f'  FAIL {lang}/{bucket}: {len(bad)} rows missing NeMo keys')
            fail = True
            continue
        total_sec = sum(r['duration'] for r in rows)
        print(f'  PASS {lang}/{bucket}: {len(rows)} rows, {total_sec/60:.1f} min audio')
assert not fail
EOF

echo ""
echo ">>> [6/7] Verify: HF staging tree builds + load_dataset accepts the schema"
for LANG in zh hi; do
    "$PYTHON" 07_hf_push/push_to_hf.py \
        --lang "$LANG" \
        --repo-id "smoke/test-$LANG" \
        --dry-run 2>&1 | tee "$LOG_DIR/06_hf_${LANG}.log"
done

"$PYTHON" - <<'EOF' 2>&1 | tee "$LOG_DIR/06_load_dataset.log"
from datasets import load_dataset
fail = False
for lang_dir, lang in (('chinese','zh'), ('hindi','hi')):
    staging = f'outputs/{lang_dir}/_hf_staging'
    for cfg in ('short_clean','short_augmented','long_clean','long_augmented'):
        try:
            ds = load_dataset(staging, cfg)
            assert 'train' in ds and 'val' in ds
            feat = ds['train'].features['audio']
            assert 'Audio' in type(feat).__name__, f'audio not Audio() feature: {feat}'
            print(f'  PASS {lang}/{cfg}: train={len(ds["train"])}, val={len(ds["val"])}')
        except Exception as e:
            print(f'  FAIL {lang}/{cfg}: {e}')
            fail = True
assert not fail
EOF

echo ""
echo ">>> [7/7] Smoke test PASSED"
echo "================================================="
echo "  All stages wired correctly."
echo "  Ready for the multi-hour bulk run:"
echo ""
echo "    HF_REPO_ZH=valsea/synthetic-asr-zh \\"
echo "    HF_REPO_HI=valsea/synthetic-asr-hi \\"
echo "      nohup bash scripts/run_bulk.sh > bulk_run.log 2>&1 &"
echo "    tail -f bulk_run.log"
echo "================================================="
