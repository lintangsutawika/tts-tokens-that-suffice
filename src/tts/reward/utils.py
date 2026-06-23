"""
tts.reward.utils — shared utilities for reward computation.
"""

from __future__ import annotations

from tts.utils.logprob import score_completion
from tts.summarization.utils import format_continuation


def last_n_turns(messages: list[dict], n: int) -> list[dict]:
    """
    Return messages for the last n complete (assistant, tool) turns.

    Always ends on a tool message so the context is never cut mid-turn.
    If fewer than n complete turns exist, returns all messages.
    """
    if n <= 0:
        return []
    tool_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            tool_count += 1
            if tool_count == n:
                j = i - 1
                while j >= 0 and messages[j].get("role") != "tool":
                    j -= 1
                return messages[j + 1:]
    return messages


def build_x_scoring_messages(partial_messages: list[dict]) -> list[dict]:
    """x-context: the full original message history."""
    return partial_messages


def build_z_scoring_messages(
    summary: str,
    partial_messages: list[dict],
    max_size: int = 20,
    keep_first: int = 4,
) -> list[dict]:
    """
    z-context: [first keep_first msgs verbatim, user(summary), tail turns verbatim].

    max_size:   target total message budget for the z-context (before summary insertion).
                Tail size = max_size // 2 - keep_first messages = that many // 2 turns.
    keep_first: messages preserved verbatim from the start (sys, user task, first turn(s)).
    """
    first = partial_messages[:keep_first]
    tail_messages = max_size // 2 - keep_first
    keep_turns = tail_messages // 2
    last_turns = last_n_turns(partial_messages, keep_turns)
    summary_msg = {"role": "user", "content": f"<summary>\n{summary}\n</summary>"}
    return first + [summary_msg] + last_turns


def compute_distortion(
    partial_messages: list[dict],
    summary: str,
    continuation_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
) -> dict:
    """
    Compute the KL-distortion reward components.

    distortion(x, z) = (1/|y|) Σ_t KL_t
    where KL_t = Σ_{v ∈ top_k(x) ∩ top_k(z)} p_x(v) · (log p_x(v) − log p_z(v))

    max_size and keep_first control the z-context structure; see build_z_scoring_messages.

    Returns a dict with distortion, fidelity, and token count.
    Returns {"error": ...} on failure.
    """
    x_messages = build_x_scoring_messages(partial_messages)
    z_messages = build_z_scoring_messages(
        summary, partial_messages, max_size=max_size, keep_first=keep_first
    )
    y_text = format_continuation(x_messages, continuation_messages, tokenizer)

    kl_per_token = score_completion(x_messages, z_messages, y_text, model, api_base, tokenizer)

    if kl_per_token is None:
        return {"error": "score_completion failed", "fidelity": None}

    n = len(kl_per_token)
    if n == 0:
        return {"error": "No completion tokens scored", "fidelity": None}

    distortion = sum(kl_per_token) / n
    fidelity = -distortion
    return {
        "fidelity": fidelity,
        "distortion": distortion,
        "n_tokens": n,
        "n_x_messages": len(x_messages),
        "n_z_messages": len(z_messages),
    }
