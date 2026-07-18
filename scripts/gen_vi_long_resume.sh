#!/usr/bin/env bash
# Single-process, resume-safe long-form Vietnamese text generation.
#
# The OpenAI account is TPM-capped (30k tok/min for gpt-4.1); each request
# reserves its max_tokens against that budget, so CONCURRENCY causes 429s while
# a single serial stream stays comfortably under the ceiling. generate_vietnamese.py
# now writes incrementally and resumes from its own output, so if a run dies
# (rate limit exhausts retries, machine hiccup) we just re-invoke and it picks up.
# This wrapper re-invokes until the corpus holds TARGET unique sentences, stopping
# early if two consecutive attempts add nothing (genuine stall).
#
# Env: TARGET (default 6300), BATCH (10), MAX_ATTEMPTS (20)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=.venv/bin/python
[[ -x "$PY" ]] || PY=python3
CORPUS=outputs/vietnamese/long/corpus.jsonl
TARGET="${TARGET:-6300}"
BATCH="${BATCH:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
mkdir -p "$(dirname "$CORPUS")"

count_lines() { [[ -f "$CORPUS" ]] && wc -l < "$CORPUS" | tr -d ' ' || echo 0; }

prev=-1
stagnant=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    have=$(count_lines)
    echo ">>> attempt $attempt/$MAX_ATTEMPTS: $have/$TARGET unique sentences"
    if [[ "$have" -ge "$TARGET" ]]; then echo ">>> target reached"; break; fi
    if [[ "$have" -le "$prev" ]]; then
        stagnant=$((stagnant + 1))
        echo ">>> WARN no progress ($prev -> $have); stagnant=$stagnant"
        [[ "$stagnant" -ge 2 ]] && { echo ">>> stalled; giving up"; break; }
    else
        stagnant=0
    fi
    prev=$have
    "$PY" data_generation/01_text_corpus/generate_vietnamese.py \
        --output "$CORPUS" --count "$TARGET" --batch-size "$BATCH" \
        --length-target long --skip-reference \
        || echo ">>> generate exited non-zero (will resume on next attempt)"
done

echo ">>> DONE: $(count_lines)/$TARGET unique sentences in $CORPUS"
