"""
Truncation-based compaction: drop the middle outright.

The floor baseline. Head and tail survive, everything between them is
discarded with nothing put in its place — no summary, no placeholder, no
record that the turns existed.

Worth keeping distinct from mask_based even though both are free and
deterministic: masking preserves *what the agent did* and discards only what
the environment said back, while truncation loses both. The gap between them
is what "remembering your own actions" is worth; the gap between truncation
and model_based is what compaction is worth at all.
"""

from __future__ import annotations

from .base import CompactionResult, split_head_tail


def build_truncation_scoring_messages(
    partial_messages: list[dict],
    keep_first: int = 4,
    keep_last_turns: int = 3,
) -> list[dict]:
    """
    Compacted context with the middle removed.

    Same head/tail skeleton as the other strategies, so all three are directly
    comparable — they differ only in what replaces the middle.
    """
    head, _middle, tail = split_head_tail(partial_messages, keep_first, keep_last_turns)
    return head + tail


class TruncationBasedSummarizer:
    """Compactor that discards the middle. Satisfies base.Compactor."""

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 3,
    ) -> CompactionResult:
        head, middle, tail = split_head_tail(messages, keep_first, keep_last_turns)
        return CompactionResult(
            messages=head + tail,
            kind="truncation",
            summary=None,
            metadata={"n_dropped": len(middle)},
        )
