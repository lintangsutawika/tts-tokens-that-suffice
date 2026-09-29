#!/usr/bin/env python3
"""One-shot validation of the theme summarizer's tool-call output.

Runs SubtaskLitellmSummarizer against a REAL model on a slice of a real trajectory
and prints the emit_subtasks tool call it returns, so you can confirm the model
(a) emits the emit_subtasks tool (not prose) and (b) returns valid, contiguous,
non-overlapping step ranges with exactly one in_progress theme.

Env (loaded from the environment or .env if python-dotenv is present):
    TEST_MODEL          litellm model string (e.g. Qwen/Qwen3.5-9B or K2-Horizon-7B)
    TEST_LLM_BASE_URL   OpenAI-compatible base URL (endpoint the model serves)
    TEST_LLM_API_KEY    api key (may be dummy)

If TEST_MODEL is unset, tries the .env keys and otherwise prints usage.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> int:
    model = os.getenv("TEST_MODEL") or os.getenv("MODEL")
    base = os.getenv("TEST_LLM_BASE_URL") or os.getenv("LITELLM_PROXY_API_BASE")
    key = os.getenv("TEST_LLM_API_KEY") or os.getenv("LITELLM_PROXY_API_KEY") or "test"

    if not model or not base:
        print(
            "Usage: set TEST_MODEL, TEST_LLM_BASE_URL (TEST_LLM_API_KEY optional) "
            "in env or .env, then run this script.",
            file=sys.stderr,
        )
        return 2

    # A real trajectory slice as probe input (hand-picked from an available job).
    probe = (
        "/home/aci18914wh/tts-tokens-that-suffice/jobs/verified-IFM--K2-Horizon-7B"
        "-run-1/django__django-16938__iWkyR6W/agent/mini-swe-agent.trajectory.json"
    )
    if not Path(probe).exists():
        print(f"probe trajectory not found: {probe}", file=sys.stderr)
        return 2
    msgs = json.loads(Path(probe).read_text())["messages"]

    from tts.data.agent_trajectory import messages_to_steps
    from tts.summarization.subtask_based import SubtaskLitellmSummarizer, _parse_subtasks

    # Summarize a bounded middle (say first ~40 steps) to keep the call cheap.
    middle = msgs[4:80]
    steps = messages_to_steps(middle)
    print(f"model={model} base={base} n_steps={len(steps)}")

    s = SubtaskLitellmSummarizer(model, api_base=base, api_key=key, max_tokens=2048)
    raw = s.summarize(steps)
    print("--- raw tool-call arguments ---")
    print(raw[:4000])

    try:
        themes = _parse_subtasks(raw)
        print("--- parsed themes (%d) ---" % len(themes))
        for t in themes:
            print(
                f"  {t.get('name','?')}: steps[{t.get('start_idx')}..{t.get('end_idx')}] "
                f"in_progress={t.get('in_progress')} summary_len={len(t.get('summary') or '')}"
            )
        # sanity: exactly one in_progress, contiguous coverage
        in_prog = [t for t in themes if t.get("in_progress")]
        print(f"in_progress count: {len(in_prog)} (want exactly 1, the last)")
        return 0
    except Exception as e:
        print(f"PARSE FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())