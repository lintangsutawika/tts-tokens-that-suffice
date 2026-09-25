#!/usr/bin/env python3
"""Measure how long Harbor takes to PREPARE trial instances, separate from agent
execution.

"Prepare" = the two phases that run in every Modal sandbox before the agent makes
its first model call:
  * environment_setup - create the sandbox from the per-instance task image
  * agent_setup       - `apt-get install build-essential ...` + `uv tool install
                        mini-swe-agent --with litellm ...` (harbor's mini-swe-agent
                        adapter installs the agent into the otherwise agent-less
                        task image)

Because agent_setup happens BEFORE any model call, its timing is independent of
vLLM/the relay -- so you can benchmark prep with a dummy deliberator, no GPU. This
is the number the image pre-bake is meant to collapse; run this before and after
to prove it.

Usage:
    python tests/measure_prep.py jobs/<job-name>

Reads every <job>/*/result.json. Reports per-phase duration distribution and the
wall-clock span to prepare all instances (first env_setup start -> last agent_setup
finish), which is what N-way concurrency actually costs you.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from statistics import median


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def phase_dur(phase: object) -> float | None:
    """Seconds between started_at and finished_at, or None if incomplete."""
    if not isinstance(phase, dict):
        return None
    a, b = parse_ts(phase.get("started_at")), parse_ts(phase.get("finished_at"))
    return (b - a).total_seconds() if a and b else None


def pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    if f + 1 >= len(xs):
        return xs[f]
    return xs[f] + (xs[f + 1] - xs[f]) * (k - f)


def stats(xs: list[float], label: str) -> None:
    if not xs:
        print(f"  {label}: (none)")
        return
    print(
        f"  {label}: n={len(xs)}  min={min(xs):.1f}  med={median(xs):.1f}  "
        f"p90={pctl(xs, 0.9):.1f}  max={max(xs):.1f}  (s)"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    job = sys.argv[1].rstrip("/")
    results = sorted(glob.glob(os.path.join(job, "*", "result.json")))
    if not results:
        print(f"No <trial>/result.json under {job}. Has the run reached agent_setup yet?")
        return 1

    env_d: list[float] = []
    setup_d: list[float] = []
    starts: list[datetime] = []
    setup_ends: list[datetime] = []
    censored = 0
    rows: list[tuple[str, float | None, float | None, str | None]] = []

    for r in results:
        c = json.load(open(r))
        es, as_ = c.get("environment_setup"), c.get("agent_setup")
        ed, sd = phase_dur(es), phase_dur(as_)
        exc = (c.get("exception_info") or {}).get("exception_type")
        name = os.path.basename(os.path.dirname(r))
        if ed is not None:
            env_d.append(ed)
        if sd is not None:
            setup_d.append(sd)
            fe = parse_ts((as_ or {}).get("finished_at"))
            if fe:
                setup_ends.append(fe)
        elif exc == "AgentSetupTimeoutError":
            censored += 1  # setup was killed at the timeout: a lower bound, not a real duration
        s0 = parse_ts((es or {}).get("started_at"))
        if s0:
            starts.append(s0)
        rows.append((name, ed, sd, exc))

    print(f"Job: {job}")
    print(
        f"Trials with result.json: {len(results)}  "
        f"(agent_setup timed out / censored: {censored})"
    )
    stats(env_d, "environment_setup (sandbox create)                 ")
    stats(setup_d, "agent_setup      (apt build-essential + uv install)")
    if starts and setup_ends:
        wall = (max(setup_ends) - min(starts)).total_seconds()
        print(
            f"  WALL-CLOCK to prepare {len(setup_ends)} instances: {wall:.1f}s  "
            f"(first env_setup start -> last agent_setup finish)"
        )
    print()
    print("Per-instance, slowest agent_setup first:")
    for name, ed, sd, exc in sorted(rows, key=lambda x: -(x[2] or 0)):
        e = f"{ed:6.1f}" if ed is not None else "   n/a"
        s = f"{sd:7.1f}" if sd is not None else "    n/a"
        print(f"  {name:48.48s}  env={e}  setup={s}  {exc or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
