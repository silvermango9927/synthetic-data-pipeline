#!/usr/bin/env python
"""Top up a text corpus to a target count of UNIQUE transcripts.

Wraps ``data_generation/01_text_corpus/generate_{chinese,hindi}.py``. The text
generators overwrite their output and only de-duplicate *within* a single run.
This driver lets us grow a corpus across rounds while guaranteeing **nothing is
repeated**: every transcript already in ``--corpus`` is preserved (reused as the
de-dup seed), and each newly generated sentence is appended only if its exact
text is not already present.

Generation runs in rounds until the corpus holds ``>= --target`` unique
transcripts (or ``--max-rounds`` is hit, or two consecutive rounds add nothing).
The merged corpus is rewritten after every round, so the process is resumable:
re-running with the same args picks up from the existing file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

GEN = {
    "zh": "data_generation/01_text_corpus/generate_chinese.py",
    "hi": "data_generation/01_text_corpus/generate_hindi.py",
    "vi": "data_generation/01_text_corpus/generate_vietnamese.py",
}


def load_texts(path: Path) -> list[dict]:
    rows: list[dict] = []
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=("zh", "hi", "vi"))
    ap.add_argument("--corpus", required=True, help="Corpus JSONL to grow in place")
    ap.add_argument("--target", type=int, required=True, help="Target unique transcript count")
    ap.add_argument("--length-target", default="long", choices=("short", "long"))
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument(
        "--margin",
        type=float,
        default=1.10,
        help="Overshoot factor on each round's request, to absorb dedup collisions",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--skip-reference",
        action="store_true",
        help="Pass --skip-reference to the generator (faster, no HF reference download)",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    corpus.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    uniq: list[dict] = []
    for r in load_texts(corpus):
        t = r.get("text")
        if t and t not in seen:
            seen.add(t)
            uniq.append({"text": t, "language": args.lang, "length_target": args.length_target})
    print(f"[topup:{args.lang}] start: {len(uniq)} unique (seed) / target {args.target}", flush=True)

    rounds = 0
    stagnant = 0
    while len(uniq) < args.target and rounds < args.max_rounds:
        rounds += 1
        need = args.target - len(uniq)
        req = int(need * args.margin) + args.batch_size
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "gen.jsonl"
            cmd = [
                args.python, GEN[args.lang],
                "--output", str(tmp),
                "--count", str(req),
                "--batch-size", str(args.batch_size),
                "--length-target", args.length_target,
            ]
            if args.skip_reference:
                cmd.append("--skip-reference")
            print(f"[topup:{args.lang}] round {rounds}: requesting {req} (need {need})", flush=True)
            subprocess.run(cmd, check=True)
            new_rows = load_texts(tmp)

        added = 0
        for r in new_rows:
            t = r.get("text")
            if t and t not in seen:
                seen.add(t)
                uniq.append({"text": t, "language": args.lang, "length_target": args.length_target})
                added += 1
                if len(uniq) >= args.target:
                    break

        with corpus.open("w") as f:
            for r in uniq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[topup:{args.lang}] round {rounds}: +{added} -> {len(uniq)} unique", flush=True)

        stagnant = stagnant + 1 if added == 0 else 0
        if stagnant >= 2:
            print(f"[topup:{args.lang}] no new uniques in 2 rounds; stopping early", flush=True)
            break

    status = "OK" if len(uniq) >= args.target else "SHORT"
    print(f"[topup:{args.lang}] done [{status}]: {len(uniq)} unique transcripts in {corpus}", flush=True)


if __name__ == "__main__":
    main()
