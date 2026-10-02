#!/usr/bin/env python3
"""Aggregate per-trial n_compressions into a harbor run-level result.json.

The SummarizingAgent writes one compaction record per trigger (and a running
``count.json`` with n_compressions) into ``compressions_dir/<trial_iid>/`` under
the bind-mounted src/ (see harbor_eval.sh). Those counts survive container
teardown; this script sums them and records the total in the run's
``result.json`` under ``stats["n_compressions"]`` plus a per-trial breakdown.

Usage:
    record_n_compressions.py <run_dir> <compressions_root>
      <run_dir>          harbor run dir, e.g. jobs/verified-...-run-0 (result.json lives here)
      <compressions_root> dir of per-run compaction output, e.g.
                         /home/aci18914wh/tts-tokens-that-suffice/src/.compressions/<run_name>
                         (the run_name scope the agent wrote under)

Writes result.json in place (adds stats.n_compressions). Idempotent. Non-fatal: a
missing root is reported and stats stays unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(run_dir: str, compressions_root: str = "") -> int:
    # Compaction counts land in each trial's agent dir (harbor syncs the container
    # /logs/agent back): jobs/<run>/<trial>/agent/count.json
    run = Path(run_dir)
    rp = run / "result.json"
    if not rp.exists():
        print(f"no result.json at {rp}; nothing to do", file=sys.stderr)
        return 2

    per_trial: dict[str, int] = {}
    total = 0
    n_files = 0
    count_paths = sorted(run.glob("*/agent/count.json"))
    if not count_paths and compressions_root:
        root = Path(compressions_root)
        if root.is_dir():
            count_paths = sorted(root.glob("*/count.json"))
    for count_path in count_paths:
        try:
            n = int(json.loads(count_path.read_text()).get("n_compressions", 0))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"skip {count_path}: {e}", file=sys.stderr)
            continue
        per_trial[count_path.parent.name] = n
        total += n
        n_files += 1

    with rp.open("r+") as f:
        d = json.load(f)
        stats = d.setdefault("stats", {})
        stats["n_compressions"] = total
        stats["n_compressions_per_trial"] = per_trial
        f.seek(0)
        json.dump(d, f, indent=2)
        f.truncate()

    print(f"recorded n_compressions={total} across {n_files} trials -> {rp}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))