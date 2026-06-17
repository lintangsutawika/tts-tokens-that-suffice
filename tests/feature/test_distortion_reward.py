#!/usr/bin/env python3
"""
tests/feature/test_distortion_reward.py

End-to-end feature test: trajectory cutting → prompt rendering
→ summary generation → KL distortion reward scoring.

Steps:
  1. Load example.json (a real coding-agent conversation trajectory)
  2. Cut messages at index :cut_index using tts.utils.trajectory utilities
  3. Render the summarization prompt exactly as SummarizationAgent.chat_completions does
  4. Generate a summary via a vLLM-compatible server (litellm)
  5. Score the continuation against both the full context (x) and the summary (z)
     to compute the per-token KL distortion reward:
         r = (1/|y|) Σ_t [log p(y_t|y<t, z) − log p(y_t|y<t, x)]

Requires a running OpenAI-compatible server (vLLM):
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000

Usage:
    python tests/feature/test_distortion_reward.py
    python tests/feature/test_distortion_reward.py --save tests/fixtures/summaries.json
    python tests/feature/test_distortion_reward.py \\
        --base-url http://localhost:8000 --model Qwen/Qwen3-4B-Instruct-2507 \\
        --cut-index 15 --save /tmp/summaries.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Path setup — allows running as a script without installing the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tts.utils.logprob import score_completion
from tts.utils.summary_utils import format_continuation, parse_messages, render_template
from tts.utils.trajectory import chunk_trajectory_by_assistant

# ---------------------------------------------------------------------------
# Inline DEFAULT_INSTANCE_TEMPLATE from summarization_agent.py
# (imported inline to avoid broken rllm/era package-level imports in that file)
# ---------------------------------------------------------------------------

DEFAULT_INSTANCE_TEMPLATE = """\
Here are the past events so far.
<TASK>
{{ task }}
</TASK>
{% if previous_summary %}{% for summary in previous_summary %}
<PREVIOUS SUMMARY>
{{ summary }}
</PREVIOUS SUMMARY>
{% endfor %}{% endif %}{% for event in events %}
<EVENT>
{{ event }}
</EVENT>
{% endfor %}
Summarize the above to assist the agent in understanding its current state.
Suggest next steps clearly and concisely.
A good summary enables the agent to effectively choose its next actions.
Focus on key decisions, actions taken, and their outcomes.

Your summary will be directly provided to the agent to help it decide its next actions."""

FIXTURE = Path(__file__).parent / "example.json"


# ---------------------------------------------------------------------------
# Step 1 helpers: trajectory cutting
# ---------------------------------------------------------------------------

def cut_trajectory(messages: list[dict], cut_index: int) -> tuple[list[dict], list[dict]]:
    """
    Split messages at cut_index using flat list slicing.

    Returns (partial, continuation) where partial = messages[:cut_index]
    and continuation = messages[cut_index:].
    """
    partial = messages[:cut_index]
    continuation = messages[cut_index:]

    chunks = chunk_trajectory_by_assistant(partial)
    print(f"[1] Cut at index {cut_index}:")
    print(f"    partial     : {len(partial)} messages, {len(chunks)} assistant turn(s)")
    print(f"    continuation: {len(continuation)} messages")
    return partial, continuation


# ---------------------------------------------------------------------------
# Step 2 helpers: prompt rendering (matching SummarizationAgent.chat_completions)
# ---------------------------------------------------------------------------

def _extract_task(messages: list[dict]) -> str:
    """
    Extract the task description from the trajectory.

    The task is the content of the <pr_description> block inside the first user
    message, if present; otherwise the full first user message is used.
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            m = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
            if m:
                return m.group(1).strip()
            return content.strip()
    return ""


def build_summary_prompt(
    partial_messages: list[dict],
    previous_summary: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build the summarizer chat message list exactly as SummarizationAgent does.

    Returns a single-element list [{"role": "user", "content": <rendered>}].
    """
    task = _extract_task(partial_messages)
    events = parse_messages(partial_messages)

    template_vars = {
        "task": task,
        "events": events,
        "previous_summary": previous_summary,
    }
    user_content = render_template(DEFAULT_INSTANCE_TEMPLATE, template_vars=template_vars)
    return [{"role": "user", "content": user_content}]


def show_rendered_prompt(chat_messages: list[dict]) -> None:
    """Print the rendered summary prompt for inspection."""
    content = chat_messages[0]["content"]
    lines = content.splitlines()
    print(f"[2] Rendered summary prompt ({len(lines)} lines, {len(content)} chars):")
    preview_lines = min(30, len(lines))
    for line in lines[:preview_lines]:
        print(f"    {line}")
    if len(lines) > preview_lines:
        print(f"    ... ({len(lines) - preview_lines} more lines)")


# ---------------------------------------------------------------------------
# Step 3: summary generation
# ---------------------------------------------------------------------------

def generate_summary(
    model: str,
    api_base: str,
    chat_messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.6,
) -> str:
    """Generate a trajectory summary via litellm chat completions."""
    response = litellm.completion(
        model=model,
        messages=chat_messages,
        api_base=api_base,
        api_key="dummy",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Step 4: distortion reward scoring
# ---------------------------------------------------------------------------

def _build_x_scoring_messages(partial_messages: list[dict]) -> list[dict]:
    """x-context as a chat message list: full agent history so far."""
    task = _extract_task(partial_messages)
    events = parse_messages(partial_messages)
    event_block = "\n\n".join(f"[Event {i+1}]\n{e}" for i, e in enumerate(events))
    return [
        {"role": "system", "content": f"Task:\n{task}"},
        {"role": "user", "content": f"Agent actions so far:\n{event_block}\n\nContinue:"},
    ]


def _build_z_scoring_messages(summary: str, task: str) -> list[dict]:
    """z-context as a chat message list: compressed summary only."""
    return [
        {"role": "system", "content": f"Task:\n{task}"},
        {"role": "user", "content": f"Agent progress summary:\n{summary}\n\nContinue:"},
    ]


def compute_distortion_reward(
    base_url: str,
    model: str,
    partial_messages: list[dict],
    summary: str,
    continuation_messages: list[dict],
) -> dict:
    """
    Compute the KL-distortion reward components.

    r(x, z) = (1/|y|) Σ_t [log p(y_t | y<t, z) − log p(y_t | y<t, x)]

    Returns a dict with fidelity, per-context mean logprobs, and token count.
    """
    y_text = format_continuation(continuation_messages)
    task = _extract_task(partial_messages)

    x_messages = _build_x_scoring_messages(partial_messages)
    z_messages = _build_z_scoring_messages(summary, task)

    print(f"    Scoring y ({len(y_text)} chars) against x-context")
    lp_x = score_completion(x_messages, y_text, model, base_url)

    print(f"    Scoring y against z-context (summary {len(summary)} chars)")
    lp_z = score_completion(z_messages, y_text, model, base_url)

    if lp_x is None or lp_z is None:
        return {
            "error": "Backend does not support /v1/completions with echo=True. "
                     "Enable vLLM with --enable-prefix-caching or use a compatible server.",
            "fidelity": None,
        }

    n = min(len(lp_x), len(lp_z))
    if n == 0:
        return {"error": "No completion tokens scored", "fidelity": None}

    fidelity = sum(lp_z[t] - lp_x[t] for t in range(n)) / n
    return {
        "fidelity": fidelity,
        "n_tokens": n,
        "lp_x_mean": sum(lp_x[:n]) / n,
        "lp_z_mean": sum(lp_z[:n]) / n,
        "lp_x_total": sum(lp_x[:n]),
        "lp_z_total": sum(lp_z[:n]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name")
    p.add_argument("--cut-index", type=int, default=15, help="Message slice point (default: 15)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max summary tokens")
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    p.add_argument("--fixture", default=str(FIXTURE), help="Path to trajectory JSON file")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Save generated summaries (and reward results) to a JSONL file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_url: str = args.base_url.rstrip("/")

    # --- Load fixture ---
    fixture_path = Path(args.fixture)
    print(f"\n=== Loading trajectory: {fixture_path} ===")
    with open(fixture_path) as f:
        messages: list[dict] = json.load(f)
    print(f"    Total messages: {len(messages)}")

    # -----------------------------------------------------------------------
    # Step 1: Cut trajectory
    # -----------------------------------------------------------------------
    print()
    partial, continuation = cut_trajectory(messages, args.cut_index)

    # -----------------------------------------------------------------------
    # Step 2: Render summary prompt
    # -----------------------------------------------------------------------
    print()
    chat_messages = build_summary_prompt(partial)
    show_rendered_prompt(chat_messages)

    # -----------------------------------------------------------------------
    # Step 3: Generate summary
    # -----------------------------------------------------------------------
    print()
    print("[3] Generating summary via litellm ...")
    try:
        summary = generate_summary(
            model=args.model,
            api_base=base_url,
            chat_messages=chat_messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"    ERROR: Cannot reach server at {base_url}: {exc}")
        print("    Start the server with: vllm serve <model> --port 8000")
        sys.exit(1)

    print(f"    Summary ({len(summary)} chars):")
    for line in summary.splitlines():
        print(f"      {line}")

    # -----------------------------------------------------------------------
    # Step 4: Distortion reward
    # -----------------------------------------------------------------------
    print()
    print("[4] Computing distortion reward ...")
    result = compute_distortion_reward(
        base_url=base_url,
        model=args.model,
        partial_messages=partial,
        summary=summary,
        continuation_messages=continuation,
    )

    print()
    print("=== Distortion Reward Results ===")
    if "error" in result:
        print(f"    ERROR: {result['error']}")
    else:
        print(f"    fidelity (r before λ|z|) : {result['fidelity']:.4f}")
        print(f"    n_tokens (|y|)            : {result['n_tokens']}")
        print(f"    mean log p(y|x)  (teacher): {result['lp_x_mean']:.4f}")
        print(f"    mean log p(y|z)  (student): {result['lp_z_mean']:.4f}")
        print(f"    KL estimate D(p(y|x)‖p(y|z)): {-result['fidelity']:.4f} nats/token")
        if result["fidelity"] >= -0.5:
            print("    ✓ Summary preserves most predictive information (low distortion)")
        else:
            print("    ✗ Summary loses significant predictive information (high distortion)")

    # -----------------------------------------------------------------------
    # Optionally save
    # -----------------------------------------------------------------------
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "fixture": str(fixture_path),
            "cut_index": args.cut_index,
            "model": args.model,
            "summary": summary,
            "distortion_reward": result,
            "n_partial_messages": len(partial),
            "n_continuation_messages": len(continuation),
        }
        with open(save_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\n    Saved to {save_path}")


if __name__ == "__main__":
    main()
