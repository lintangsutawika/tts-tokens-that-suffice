#!/usr/bin/env python3
"""
Build a summarizer training dataset from base trajectory files (*:0.json).

Each output example is a single trajectory (messages + metadata). Splitting
into prefix/suffix is done at training time.

Usage:
    python scripts/data/make_summarizer_dataset.py \\
        --input ../data-trajectories/outputs_preference_Qwen3.6-35B-A3B-FP8 \\
        --output datasets/summarizer_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_BASE_TRAJ_RE = re.compile(r'^[0-9a-f-]{36}:0\.json$')

DONE_STATUSES = {"Submitted", "LimitsExceeded"}


def process_file(path: Path, min_steps: int) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    messages = data.get("messages", [])
    if not messages:
        return None

    last = messages[-1]
    if last.get("role") != "exit":
        return None
    status = last.get("extra", {}).get("exit_status") or last.get("content", "")
    if status not in DONE_STATUSES:
        return None

    num_calls = data.get("num_calls", 0)
    if num_calls < min_steps:
        return None

    return {
        "uid": data.get("uid"),
        "instance_id": data.get("instance_id"),
        "is_correct": data.get("is_correct", 0.0),
        "num_calls": num_calls,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="Directory containing *:0.json trajectory files")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--min-steps", type=int, default=50, help="Min num_calls to include a trajectory (default: 50)")
    parser.add_argument("--limit", type=int, default=None, help="Max trajectories to write")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if _BASE_TRAJ_RE.match(f.name))
    print(f"Found {len(files)} base trajectory files", file=sys.stderr)

    random.seed(args.seed)
    random.shuffle(files)

    n_written = n_skipped = 0
    with output_path.open("w") as out:
        for path in files:
            if args.limit and n_written >= args.limit:
                break
            example = process_file(path, args.min_steps)
            if example is None:
                n_skipped += 1
                continue
            out.write(json.dumps(example) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"  {n_written} written, {n_skipped} skipped...", file=sys.stderr)

    print(f"Done: {n_written} examples written to {output_path} ({n_skipped} skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
