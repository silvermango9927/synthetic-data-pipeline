#!/usr/bin/env python
"""Robust (resumable, batched-commit) re-upload of an already-staged _hf_staging tree.

push_to_hf.py uses HfApi.upload_folder, which commits the whole tree in ONE commit.
For datasets with thousands of files that commit can 504 (Gateway Time-out) even
though the LFS blobs uploaded fine. upload_large_folder splits the work across many
small commits and is resumable, so re-running picks up where it left off.

Usage:
    python scripts/hf_upload_large_retry.py --repo-id silvermango9927/synthetic-asr-zh \
        --folder outputs/chinese/_hf_staging
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        num_workers=args.num_workers,
        ignore_patterns=[".DS_Store", "*.tmp"],
        print_report=True,
    )
    print(f"DONE: https://huggingface.co/datasets/{args.repo_id}/tree/main")


if __name__ == "__main__":
    main()
