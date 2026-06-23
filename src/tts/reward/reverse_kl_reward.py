"""
tts.reward.reverse_kl_reward — KL-distortion fidelity reward.

    r(x, z) = (1/|y|) Σ_t [log p(y_t | y<t, z) − log p(y_t | y<t, x)] − λ·|z|

x  = partial trajectory steps (the "seen" context)
z  = generated summary (the compression)
y  = continuation steps (what the agent does next)
λ  = length-penalty coefficient (distortion_lambda)
"""

from __future__ import annotations

from tts.reward.utils import compute_distortion


def distortion_reward(
    partial_messages: list[dict],
    summary: str,
    continuation_messages: list[dict],
    model: str,
    api_base: str,
    tokenizer,
    max_size: int = 20,
    keep_first: int = 4,
    lambda_len: float = 0.0,
) -> float:
    """
    KL-distortion fidelity reward from "Tokens That Suffice".

    max_size and keep_first control the z-context compression structure.
    Returns fidelity (negative distortion) minus a length penalty on the summary.
    Falls back to 0.0 on error.
    """
    result = compute_distortion(
        partial_messages=partial_messages,
        summary=summary,
        continuation_messages=continuation_messages,
        model=model,
        api_base=api_base,
        tokenizer=tokenizer,
        max_size=max_size,
        keep_first=keep_first,
    )
    if "error" in result or result["fidelity"] is None:
        return 0.0
    fidelity = result["fidelity"]
    length_penalty = lambda_len * len(summary.split())
    return fidelity - length_penalty
