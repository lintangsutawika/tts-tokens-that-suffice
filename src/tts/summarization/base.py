"""
The compaction interface shared by every summarization strategy.

A compactor rewrites an agent's message history into a shorter one that the
agent can keep working from. Strategies differ in what they do to the middle:

  * model-based  (model_based.py) — replace it with a generated summary
  * mask-based   (mask_based.py)  — keep the agent's actions, elide what the
                                    environment printed back

Both preserve the same skeleton: `keep_first` messages of head and
`keep_last_turns` complete (assistant, tool) turns of tail, verbatim.

Note the interface is a *message-list transformation*, not "produce a summary
string". Mask-based compaction has no summary to return, so a string-valued
interface cannot express it; the generated summary is carried on the result as
metadata instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


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


def split_head_tail(
    messages: list[dict], keep_first: int, keep_last_turns: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split messages into (head, middle, tail).

    Head and tail survive compaction verbatim; only the middle is a strategy's
    to rewrite. Returned as three lists so a strategy never has to re-derive
    the boundaries and risk disagreeing with another strategy about them.
    """
    head = messages[:keep_first]
    tail = last_n_turns(messages[keep_first:], keep_last_turns)
    end = len(messages) - len(tail) if tail else len(messages)
    return head, messages[keep_first:end], tail


@dataclass
class CompactionResult:
    """What a compactor produced, plus enough detail to audit it after the fact."""

    messages: list[dict]
    kind: str  # "summary" | "mask" | "truncation" | "summary_failed"
    summary: str | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Compactor(Protocol):
    """Anything that can shrink a message history on demand."""

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 3,
    ) -> CompactionResult:
        ...
