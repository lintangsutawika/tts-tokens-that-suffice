#!/usr/bin/env python3
"""Poll a harbor job directory and stream its progress to Weights & Biases.

Harbor has NO native experiment-tracker integration (its only `wandb` is a sandbox
PROVIDER, not metrics logging), and its rich `Live` progress bar (harbor/job.py) is
TTY-gated -- invisible when stdout is a pipe/file, as in a Slurm/PBS batch job. This
poller sidesteps both: it reads the per-trial result.json files off the shared
filesystem (so it works WHILE the run is in progress, on the login node or as a
background process in the sbatch) and logs discrete progress metrics to a wandb run.
Being file-based, it survives PBS stdout spooling and needs no harbor hooks.

Metrics logged each tick:
  progress/{total,completed,running,errored,solved,frac_done,mean_reward}
  errors/<ExceptionType> = count      (one series per error type seen)

Usage:
  python scripts/eval/wandb_monitor.py <job_dir> [--interval 60] \
      [--project harbor-eval] [--name <run>] [--entity <team>] [--once]

Env: WANDB_API_KEY (required by wandb). WANDB_PROJECT / WANDB_ENTITY are honored as
defaults. Install wandb into the venv first: `uv pip install wandb` (or pip).

Exits when every trial has a result.json OR the job-level result.json has finished_at
(or immediately with --once). Safe to run many times / resume (resume="allow").
"""
import argparse
import glob
import json
import os
import time


def planned_total(job_dir, n_dirs):
    """The job's PLANNED trial count, for the frac_done denominator. Harbor creates
    trial dirs lazily (bounded by n_concurrent), so counting existing dirs overstates
    progress early on. Prefer the job result.json's n_total_trials, then the lock's
    trial count; fall back to the dir count (>= dirs, never less)."""
    jr = os.path.join(job_dir, "result.json")
    if os.path.exists(jr):
        try:
            n = json.load(open(jr)).get("n_total_trials")
            if n:
                return max(int(n), n_dirs)
        except Exception:
            pass
    lk = os.path.join(job_dir, "lock.json")
    if os.path.exists(lk):
        try:
            n = len(json.load(open(lk)).get("trials") or [])
            if n:
                return max(n, n_dirs)
        except Exception:
            pass
    return n_dirs


def scan(job_dir):
    """Aggregate progress from per-trial result.json files. Tolerant of
    half-written files (a trial mid-flush) -- those are just skipped this tick."""
    trials = [d for d in glob.glob(os.path.join(job_dir, "*")) if os.path.isdir(d)]
    total = planned_total(job_dir, len(trials))
    completed = errored = 0
    rewards = []
    errs = {}
    for d in trials:
        p = os.path.join(d, "result.json")
        if not os.path.exists(p):
            continue
        try:
            t = json.load(open(p))
        except Exception:
            continue  # half-written or unreadable; count next tick
        completed += 1
        ei = t.get("exception_info")
        if ei:
            et = (ei.get("exception_type") or "UnknownError") if isinstance(ei, dict) else "UnknownError"
            errs[et] = errs.get(et, 0) + 1
            errored += 1
        # reward lives at verifier_result.rewards.reward (nested "rewards" dict)
        vr = t.get("verifier_result") or {}
        rw = vr.get("rewards") if isinstance(vr, dict) else None
        r = rw.get("reward") if isinstance(rw, dict) else None
        if r is not None:
            rewards.append(r)
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    return {
        "total": total,
        "completed": completed,
        "running": total - completed,
        "errored": errored,
        "solved": len(rewards),
        "mean_reward": mean_reward,
        "errors": errs,
    }


def job_finished(job_dir):
    jr = os.path.join(job_dir, "result.json")
    if not os.path.exists(jr):
        return False
    try:
        return bool(json.load(open(jr)).get("finished_at"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", help="path to the harbor job dir (e.g. jobs/verified-...-run-0)")
    ap.add_argument("--interval", type=float, default=60, help="poll seconds (default 60)")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "harbor-eval"))
    ap.add_argument("--name", default=None, help="wandb run name (default: job dir basename)")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--once", action="store_true", help="log one snapshot and exit")
    args = ap.parse_args()

    import wandb  # imported here so --help works without wandb installed

    name = args.name or os.path.basename(os.path.normpath(args.job_dir))
    wandb.init(
        project=args.project,
        entity=args.entity,
        name=name,
        config={"job_dir": os.path.abspath(args.job_dir)},
        resume="allow",
    )
    try:
        while True:
            s = scan(args.job_dir)
            log = {f"progress/{k}": v for k, v in s.items() if k != "errors" and v is not None}
            if s["total"]:
                log["progress/frac_done"] = s["completed"] / s["total"]
            for et, n in s["errors"].items():
                log[f"errors/{et}"] = n
            wandb.log(log)
            done = (s["total"] > 0 and s["completed"] >= s["total"]) or job_finished(args.job_dir)
            if args.once or done:
                break
            time.sleep(args.interval)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
