"""
Mask-based compaction: drop what the environment printed, keep what the agent did.

Tool output dominates an agent context — measured on North SWE-bench
trajectories it is ~73% of the continuation tokens — while the agent's own
actions are comparatively tiny. Eliding just the tool payloads therefore buys
most of the compression of a summary while requiring no model call at all,
which makes it the natural control for "is the learned summarizer earning its
keep?".

The strategy adapts the Observation Masking mechanism of Lindenbauer et al.,
"The Complexity Trap" (arXiv:2508.21433): keep recent observations verbatim,
replace older tool results with a descriptive "(N lines omitted)" placeholder,
and (optionally) keep bug-context outputs by tag. Unlike the paper's
progressive per-turn masking, this runs at the agent's compaction firepoint
(hard context window) and reuses the same head/middle/tail skeleton as the
other strategies — the middle's observations are masked, the tail stays
verbatim, so it is directly comparable with sectioned/cliff/subtask.

Unlike model-based compaction this is deterministic and free.
"""

from __future__ import annotations

import re

from .base import CompactionResult, split_head_tail

ENV_MASK_PLACEHOLDER = "[OUTPUT]"
THINKING_MASK_PLACEHOLDER = "[THINKING]"
DESCRIPTIVE_PLACEHOLDER = "Old environment output: ({lines} lines omitted)"

# Tool outputs carrying debugging context worth keeping even when old.
_KEEP_OUTPUT_RE = re.compile(
    r"Error|Traceback|AssertionError|Exception|FAILED|FAIL|failed|"
    r"\bdiff\b|^\s*(\+\+\+|---)|=>|unexpected|TypeError|KeyError|ValueError",
    re.IGNORECASE,
)


def _content_lines(messages: list[dict], i: int) -> int:
    """Number of lines in a tool message's content (for the placeholder)."""
    c = messages[i].get("content", "")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    return len(str(c).splitlines())


def should_keep_tool_output(content: str) -> bool:
    """Rule-based tag decision: keep outputs carrying bug/error context.

    Mirrors the paper's ``keep_output`` tag without an external model: outputs
    containing error/traceback/diff markers are retained (they are cheap and
    carry the pointer to the bug), everything else is maskable.
    """
    if isinstance(content, list):
        content = " ".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return bool(_KEEP_OUTPUT_RE.search(str(content)))


def mask_env_messages(
    messages: list[dict],
    placeholder: str = ENV_MASK_PLACEHOLDER,
    mask_output: bool = True,
    mask_thinking: bool = False,
    thinking_placeholder: str = THINKING_MASK_PLACEHOLDER,
    keep_n: int = 10,
    tagged_keep: bool = True,
    long_output_chars: int = 5000,
    descriptive_placeholder: bool = True,
) -> list[dict]:
    """
    Elide the two things an agent turn carries that it no longer needs verbatim.

    mask_output   — replace tool-result content. What the environment printed
                    back; the bulk of the context.
    mask_thinking — replace assistant reasoning_content. What the agent was
                    thinking at the time, as opposed to what it did.
    keep_n        — the last `keep_n` tool observations are kept verbatim
                    (paper's ``n=10`` recency window scaled into the middle).
    tagged_keep   — keep bug-context outputs (errors/diffs) verbatim even when
                    older than keep_n; drop obvious noise.
    long_output_chars — outputs longer than this are masked regardless of
                    tagged_keep (a huge cat/head dump is almost always re-runnable).

    Messages are kept rather than deleted either way, so the (assistant, tool)
    alternation the chat template expects survives and the agent can still see
    *that* a command ran even when the output is gone. The default placeholder
    is descriptive ("N lines omitted") rather than the flat ``[OUTPUT]``.
    An empty placeholder drops the text outright.
    """
    # indices of tool messages in the window
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # keep the last keep_n verbatim (recency window)
    keep_from = len(tool_idx) - keep_n if keep_n else len(tool_idx)
    recent_tool_idx = set(tool_idx[keep_from:])
    out = []
    for i, m in enumerate(messages):
        if mask_thinking and m.get("role") == "assistant" and m.get("reasoning_content"):
            out.append({**m, "reasoning_content": thinking_placeholder})
        elif mask_output and m.get("role") == "tool":
            if i in recent_tool_idx:
                out.append(m)  # recent window: verbatim
            else:
                content = m.get("content", "")
                if tagged_keep and should_keep_tool_output(content):
                    # keep bug-context output verbatim even if old
                    if long_output_chars and len(str(content)) > long_output_chars:
                        out.append({**m, "content": _masked_content(m, descriptive_placeholder, placeholder)})
                    else:
                        out.append(m)
                else:
                    out.append({**m, "content": _masked_content(m, descriptive_placeholder, placeholder)})
        else:
            out.append(m)
    return out


def _masked_content(m: dict, descriptive: bool, placeholder: str) -> str:
    """The placeholder text for a masked tool output."""
    if not descriptive:
        return placeholder
    lines = len(str(m.get("content", "")).splitlines())
    return DESCRIPTIVE_PLACEHOLDER.format(lines=lines)


def build_maskenv_scoring_messages(
    partial_messages: list[dict],
    keep_first: int = 4,
    keep_last_turns: int = 3,
    placeholder: str = ENV_MASK_PLACEHOLDER,
    mask_output: bool = True,
    mask_thinking: bool = False,
    thinking_placeholder: str = THINKING_MASK_PLACEHOLDER,
    keep_n: int = 10,
    tagged_keep: bool = True,
    long_output_chars: int = 5000,
    descriptive_placeholder: bool = True,
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
        keep_n=keep_n,
        tagged_keep=tagged_keep,
        long_output_chars=long_output_chars,
        descriptive_placeholder=descriptive_placeholder,
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
        keep_n: int = 10,
        tagged_keep: bool = True,
        long_output_chars: int = 5000,
        descriptive_placeholder: bool = True,
    ):
        if not mask_output and not mask_thinking:
            raise ValueError("mask_output and mask_thinking are both False: nothing to mask")
        self.placeholder = placeholder
        self.mask_output = mask_output
        self.mask_thinking = mask_thinking
        self.thinking_placeholder = thinking_placeholder
        self.keep_n = keep_n
        self.tagged_keep = tagged_keep
        self.long_output_chars = long_output_chars
        self.descriptive_placeholder = descriptive_placeholder

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
            keep_n=self.keep_n,
            tagged_keep=self.tagged_keep,
            long_output_chars=self.long_output_chars,
            descriptive_placeholder=self.descriptive_placeholder,
        )
        n_output_masked = sum(
            1
            for m in masked
            if self.mask_output and m.get("role") == "tool" and m.get("content", "").startswith("Old environment output:")
        )
        return CompactionResult(
            messages=head + masked + tail,
            kind="mask",
            summary=None,
            metadata={
                "mask_output": self.mask_output,
                "mask_thinking": self.mask_thinking,
                "keep_n": self.keep_n,
                "tagged_keep": self.tagged_keep,
                "long_output_chars": self.long_output_chars,
                "descriptive_placeholder": self.descriptive_placeholder,
                "n_output_masked": n_output_masked,
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