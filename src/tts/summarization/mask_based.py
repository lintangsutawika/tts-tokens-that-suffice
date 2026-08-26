"""
Mask-based compaction: drop what the environment printed, keep what the agent did.

Tool output dominates an agent context — measured on North SWE-bench
trajectories it is ~73% of the continuation tokens — while the agent's own
actions are comparatively tiny. Eliding just the tool payloads therefore buys
most of the compression of a summary while requiring no model call at all,
which makes it the natural control for "is the learned summarizer earning its
keep?".

Unlike model-based compaction this is deterministic and free.
"""

from __future__ import annotations

from .base import CompactionResult, split_head_tail

ENV_MASK_PLACEHOLDER = "[OUTPUT]"
THINKING_MASK_PLACEHOLDER = "[THINKING]"


def mask_env_messages(
    messages: list[dict],
    placeholder: str = ENV_MASK_PLACEHOLDER,
    mask_output: bool = True,
    mask_thinking: bool = False,
    thinking_placeholder: str = THINKING_MASK_PLACEHOLDER,
) -> list[dict]:
    """
    Elide the two things an agent turn carries that it no longer needs verbatim.

    mask_output   — replace tool-result content. What the environment printed
                    back; the bulk of the context.
    mask_thinking — replace assistant reasoning_content. What the agent was
                    thinking at the time, as opposed to what it did.

    Messages are kept rather than deleted either way, so the (assistant, tool)
    alternation the chat template expects survives and the agent can still see
    *that* a command ran even when the output is gone. An empty placeholder
    drops the text outright.
    """
    out = []
    for m in messages:
        if mask_output and m.get("role") == "tool":
            out.append({**m, "content": placeholder})
        elif mask_thinking and m.get("role") == "assistant" and m.get("reasoning_content"):
            out.append({**m, "reasoning_content": thinking_placeholder})
        else:
            out.append(m)
    return out


def build_maskenv_scoring_messages(
    partial_messages: list[dict],
    keep_first: int = 4,
    keep_last_turns: int = 3,
    placeholder: str = ENV_MASK_PLACEHOLDER,
    mask_output: bool = True,
    mask_thinking: bool = False,
    thinking_placeholder: str = THINKING_MASK_PLACEHOLDER,
) -> list[dict]:
    """
    Compacted context with environment output and/or reasoning elided.

    Same head/tail skeleton as build_z_scoring_messages, so the two strategies
    are directly comparable — they differ only in what replaces the middle.
    """
    head, middle, tail = split_head_tail(partial_messages, keep_first, keep_last_turns)
    masked = mask_env_messages(
        middle,
        placeholder=placeholder,
        mask_output=mask_output,
        mask_thinking=mask_thinking,
        thinking_placeholder=thinking_placeholder,
    )
    return head + masked + tail


class MaskBasedSummarizer:
    """
    Compactor that elides environment output and/or agent reasoning.

    The two flags separate "what the agent saw" from "what the agent thought",
    so their costs can be measured independently rather than as one blob.
    Satisfies base.Compactor.
    """

    def __init__(
        self,
        placeholder: str = ENV_MASK_PLACEHOLDER,
        mask_output: bool = True,
        mask_thinking: bool = False,
        thinking_placeholder: str = THINKING_MASK_PLACEHOLDER,
    ):
        if not mask_output and not mask_thinking:
            raise ValueError("mask_output and mask_thinking are both False: nothing to mask")
        self.placeholder = placeholder
        self.mask_output = mask_output
        self.mask_thinking = mask_thinking
        self.thinking_placeholder = thinking_placeholder

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 3,
    ) -> CompactionResult:
        head, middle, tail = split_head_tail(messages, keep_first, keep_last_turns)
        masked = mask_env_messages(
            middle,
            placeholder=self.placeholder,
            mask_output=self.mask_output,
            mask_thinking=self.mask_thinking,
            thinking_placeholder=self.thinking_placeholder,
        )
        return CompactionResult(
            messages=head + masked + tail,
            kind="mask",
            summary=None,
            metadata={
                "mask_output": self.mask_output,
                "mask_thinking": self.mask_thinking,
                "n_output_masked": sum(
                    1 for m in middle if self.mask_output and m.get("role") == "tool"
                ),
                "n_thinking_masked": sum(
                    1
                    for m in middle
                    if self.mask_thinking
                    and m.get("role") == "assistant"
                    and m.get("reasoning_content")
                ),
                "placeholder": self.placeholder,
            },
        )
