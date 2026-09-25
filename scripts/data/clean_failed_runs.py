#!/usr/bin/env python3
"""Remove eval runs that failed with a transient/server error so they re-run.

The downstream SWE-bench eval (`tts.eval_swebench`) skips instances already
present in `preds.json`. Runs that died from `InternalServerError` (deliberator
hiccup) or `RepeatedFormatError` (model stuck in a no-tool-call loop) are not
real attempts and should be retried — but the skip logic keeps them. This script
finds those trajectories, deletes their instance directory, and drops their entry
from `preds.json`, so the next eval run re-attempts them.

Usage:
  # dry run over every outputs/<mode>-<model>/ dir (default statuses):
  python scripts/data/clean_failed_runs.py --dry-run
  # actually remove:
  python scripts/data/clean_failed_runs.py
  # specific dirs / statuses:
  python scripts/data/clean_failed_runs.py outputs/base-Qwen3.6-35B-A3B \
      --status InternalServerError --status RepeatedFormatError
"""
import argparse
import json
import shutil
from pathlib import Path

DEFAULT_STATUSES = ["InternalServerError", "RepeatedFormatError"]


def find_output_dirs(args_dirs):
    if args_dirs:
        return [Path(d) for d in args_dirs]
    # Any dir containing a preds.json.
    return sorted(p.parent for p in Path("outputs").glob("*/preds.json"))


def exit_status_of(traj_path: Path):
    try:
        j = json.loads(traj_path.read_text())
    except Exception as e:
        print(f"    ! could not read {traj_path}: {e}")
        return None
    return j.get("info", {}).get("exit_status")


def clean_dir(out_dir: Path, statuses, dry_run: bool):
    preds_path = out_dir / "preds.json"
    if not preds_path.exists():
        print(f"[skip] {out_dir}: no preds.json")
        return 0
    preds = json.loads(preds_path.read_text())

    removed = 0
    changed = False
    # Each instance lives in out_dir/<instance_id>/<instance_id>.traj.json
    for inst_dir in sorted(out_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        iid = inst_dir.name
        traj = inst_dir / f"{iid}.traj.json"
        if not traj.exists():
            continue
        status = exit_status_of(traj)
        if status not in statuses:
            continue
        print(f"  {'[dry] ' if dry_run else ''}remove {iid} (exit_status={status})")
        removed += 1
        if dry_run:
            continue
        shutil.rmtree(inst_dir, ignore_errors=True)
        if iid in preds:
            del preds[iid]
            changed = True

    if changed and not dry_run:
        preds_path.write_text(json.dumps(preds, indent=2))

    print(f"[{out_dir.name}] {'would remove' if dry_run else 'removed'} {removed} run(s)")
    return removed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="*", help="Output dirs (default: all outputs/*/ with preds.json)")
    ap.add_argument("--status", action="append", dest="statuses",
                    help="exit_status to remove (repeatable). Default: %s" % ", ".join(DEFAULT_STATUSES))
    ap.add_argument("--dry-run", action="store_true", help="Report only; make no changes")
    args = ap.parse_args()

    statuses = set(args.statuses or DEFAULT_STATUSES)
    print(f"Removing runs with exit_status in: {sorted(statuses)}\n")

    total = 0
    for out_dir in find_output_dirs(args.dirs):
        total += clean_dir(out_dir, statuses, args.dry_run)
    print(f"\nTotal: {'would remove' if args.dry_run else 'removed'} {total} run(s)")


if __name__ == "__main__":
    main()
