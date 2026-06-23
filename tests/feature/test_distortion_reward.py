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
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Path setup — allows running as a script without installing the package
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tts.summarization.utils import parse_messages, render_template, format_continuation
from tts.utils.trajectory import chunk_trajectory_by_assistant
from tts.reward.utils import (
    build_x_scoring_messages,
    build_z_scoring_messages,
    compute_distortion,
    last_n_turns,
)

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

def _run_self_test(
    base_url: str,
    model: str,
    partial_messages: list[dict],
    continuation_messages: list[dict],
    tokenizer,
) -> dict:
    """Score with x == z (no summary). KL should be ≈ 0."""
    from tts.utils.logprob import score_completion
    from tts.summarization.utils import format_continuation

    y_text = format_continuation(partial_messages, continuation_messages, tokenizer)
    kl_per_token = score_completion(
        partial_messages, partial_messages, y_text, model, base_url, tokenizer
    )
    if kl_per_token is None:
        return {"error": "score_completion failed", "fidelity": None}
    n = len(kl_per_token)
    if n == 0:
        return {"error": "No completion tokens scored", "fidelity": None}
    distortion = sum(kl_per_token) / n
    return {"distortion": distortion, "fidelity": -distortion, "n_tokens": n}


def _run_distortion_reward(
    base_url: str,
    model: str,
    partial_messages: list[dict],
    summary: str,
    continuation_messages: list[dict],
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
) -> dict:
    return compute_distortion(
        partial_messages=partial_messages,
        summary=summary,
        continuation_messages=continuation_messages,
        model=model,
        api_base=base_url,
        tokenizer=tokenizer,
        max_size=max_size,
        keep_first=keep_first,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name passed to the API")
    p.add_argument("--tokenizer", default=None, metavar="PATH",
                   help="HuggingFace tokenizer path/name (defaults to --model with litellm_proxy/ stripped)")
    p.add_argument("--cut-index", type=int, default=15, help="Message slice point (default: 15)")
    p.add_argument("--max-tokens", type=int, default=512, help="Max summary tokens")
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    p.add_argument("--fixture", default=str(FIXTURE), help="Path to trajectory JSON file")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="Save generated summaries (and reward results) to a JSONL file")
    p.add_argument("--summary-file", default=None, metavar="PATH",
                   help="Load pre-computed summary from a .txt file; skips generation step")
    p.add_argument("--max-size", type=int, default=20,
                   help="Target message budget for z-context; tail = max_size//2 - keep_first msgs (default: 20)")
    p.add_argument("--keep-first", type=int, default=4,
                   help="Messages to keep verbatim from the start in z-context (default: 4)")
    p.add_argument("--save-z", default=None, metavar="PATH",
                   help="Save the full z-context message list to a JSON file")
    p.add_argument("--save-continuation", default=None, metavar="PATH",
                   help="Save the continuation message list to a JSON file")
    p.add_argument("--self-test", action="store_true",
                   help="Use partial as both x and z (no summary); KL should be ≈0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_url: str = args.base_url.rstrip("/")

    # --- Load tokenizer ---
    tokenizer_name = args.tokenizer or args.model.removeprefix("litellm_proxy/")
    print(f"\n=== Loading tokenizer: {tokenizer_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

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

    if args.self_test:
        # -----------------------------------------------------------------------
        # Self-test: x == z (no summary). KL should be ≈ 0.
        # -----------------------------------------------------------------------
        print()
        print("[self-test] Using partial as both x and z — skipping summary generation.")
        print(f"            x messages: {len(partial)}  z messages: {len(partial)}")
        self_test_result = _run_self_test(
            base_url=base_url,
            model=args.model,
            partial_messages=partial,
            continuation_messages=continuation,
            tokenizer=tokenizer,
        )
        print()
        print("=== Self-Test Results (expect distortion ≈ 0) ===")
        if "error" in self_test_result:
            print(f"    ERROR: {self_test_result['error']}")
        else:
            print(f"    distortion D_KL(p(y|x)‖p(y|x)): {self_test_result['distortion']:.6f} nats/token")
            print(f"    fidelity (−distortion)           : {self_test_result['fidelity']:.6f}")
            print(f"    n_tokens (|y|)                   : {self_test_result['n_tokens']}")
            if self_test_result["distortion"] < 0.01:
                print("    ✓ PASS: distortion is effectively 0")
            else:
                print("    ✗ FAIL: distortion should be 0 when x == z")
        return

    # -----------------------------------------------------------------------
    # Step 2: Render summary prompt
    # Keep first keep_first messages and last keep tail turns verbatim; summarize middle.
    # -----------------------------------------------------------------------
    print()
    tail_messages = args.max_size // 2 - args.keep_first
    keep_turns = tail_messages // 2
    last_turns = last_n_turns(partial, keep_turns)
    middle = partial[args.keep_first : len(partial) - len(last_turns)] if last_turns else partial[args.keep_first:]
    print(f"    max_size={args.max_size}  keep_first={args.keep_first}  tail={tail_messages} msgs ({keep_turns} turns)")
    print(f"    Summarizing {len(middle)} middle messages (skipping first {args.keep_first} + last {len(last_turns)} verbatim)")
    chat_messages = build_summary_prompt(middle)
    show_rendered_prompt(chat_messages)

    # -----------------------------------------------------------------------
    # Step 3: Load or generate summary
    # -----------------------------------------------------------------------
    print()
    if args.summary_file:
        summary_path = Path(args.summary_file)
        print(f"[3] Loading summary from {summary_path} ...")
        summary = summary_path.read_text()
    else:
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
    result = _run_distortion_reward(
        base_url=base_url,
        model=args.model,
        partial_messages=partial,
        summary=summary,
        continuation_messages=continuation,
        tokenizer=tokenizer,
        max_size=args.max_size,
        keep_first=args.keep_first,
    )

    print()
    print("=== Distortion Reward Results ===")
    if "error" in result:
        print(f"    ERROR: {result['error']}")
    else:
        print(f"    x messages (before)              : {result['n_x_messages']}")
        print(f"    z messages (after)               : {result['n_z_messages']}  (compression {result['n_x_messages']}→{result['n_z_messages']}, max_size={args.max_size})")
        print(f"    distortion D_KL(p(y|x)‖p(y|z)): {result['distortion']:.4f} nats/token")
        print(f"    fidelity (−distortion)           : {result['fidelity']:.4f}")
        print(f"    n_tokens (|y|)                   : {result['n_tokens']}")
        if result["distortion"] <= 0.5:
            print("    ✓ Summary preserves most predictive information (low distortion)")
        else:
            print("    ✗ Summary loses significant predictive information (high distortion)")

    # -----------------------------------------------------------------------
    # Optionally save z messages
    # -----------------------------------------------------------------------
    if args.save_z:
        z_messages = build_z_scoring_messages(summary, partial, max_size=args.max_size, keep_first=args.keep_first)
        save_z_path = Path(args.save_z)
        save_z_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_z_path, "w") as f:
            json.dump(z_messages, f, indent=2)
        print(f"\n    z messages ({len(z_messages)}) saved to {save_z_path}")

    if args.save_continuation:
        save_cont_path = Path(args.save_continuation)
        save_cont_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_cont_path, "w") as f:
            json.dump(continuation, f, indent=2)
        print(f"\n    continuation ({len(continuation)} messages) saved to {save_cont_path}")

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
