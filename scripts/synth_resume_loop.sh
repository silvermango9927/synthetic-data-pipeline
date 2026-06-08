#!/usr/bin/env bash
# Resume-retry wrapper around data_generation/02_tts_synthesis/synthesize.py.
#
# synthesize.py is idempotent (`--resume` skips WAVs already on disk), but its
# ThreadPoolExecutor path can take a transient native crash on macOS at bulk
# scale (SIGBUS / "leaked semaphore" at shutdown). This wrapper simply re-invokes
# `--resume` until the clean dir reaches TARGET WAVs, so a crash mid-run just
# means the next attempt picks up where it left off. It stops early if two
# consecutive attempts make no progress (genuine stall, not a transient crash).
#
# Usage:
#   bash scripts/synth_resume_loop.sh <lang> <dirname> <target_wavs>
#   # e.g. 4300 unique 1-voice Chinese long clips via edge-tts:
#   bash scripts/synth_resume_loop.sh zh chinese 4300
#
# Tunables (env vars):
#   WORKERS       concurrent TTS calls           (default: 4)
#   BACKEND       edge|sarvam|minimax|qwen|...   (default: edge)
#   LENGTH        short|long bucket              (default: long)
#   VOICES        voices per sentence            (default: 1)
#   MAX_ATTEMPTS  max resume attempts            (default: 12)
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <lang> <dirname> <target_wavs>   (e.g. $0 zh chinese 4300)" >&2
    exit 2
fi

LANGCODE="$1"; DIRNAME="$2"; TARGET="$3"
WORKERS="${WORKERS:-4}"
BACKEND="${BACKEND:-edge}"
LENGTH="${LENGTH:-long}"
VOICES="${VOICES:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"

CLEAN="outputs/$DIRNAME/$LENGTH/clean"
CORPUS="outputs/$DIRNAME/$LENGTH/corpus.jsonl"
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

count_wavs() { ls "$CLEAN"/*.wav 2>/dev/null | wc -l | tr -d ' '; }

prev=-1
stagnant=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    done=$(count_wavs)
    echo ">>> attempt $attempt/$MAX_ATTEMPTS: $done/$TARGET wavs present"
    if [[ "$done" -ge "$TARGET" ]]; then echo ">>> target reached"; break; fi
    if [[ "$done" -le "$prev" ]]; then
        stagnant=$((stagnant + 1))
        echo ">>> WARN no progress since last attempt ($prev -> $done); stagnant=$stagnant"
        [[ "$stagnant" -ge 2 ]] && { echo ">>> stalled (no progress in 2 attempts); giving up"; break; }
    else
        stagnant=0
    fi
    prev=$done
    "$PY" data_generation/02_tts_synthesis/synthesize.py \
        --corpus "$CORPUS" --output-dir "$CLEAN" \
        --lang "$LANGCODE" --backend "$BACKEND" \
        --voices-per-sentence "$VOICES" --workers "$WORKERS" --resume \
        || echo ">>> synthesize exited non-zero (will resume on next attempt)"
done

echo ">>> DONE: $(count_wavs)/$TARGET wavs in $CLEAN"
