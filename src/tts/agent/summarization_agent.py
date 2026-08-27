"""
An agent that compacts its own context when it grows past a budget.

SummarizingAgent decides *when* to compact; a Compactor decides *how*. The two
were previously fused inside eval_swebench, which meant adding a strategy meant
editing the agent and a `mode` string had to encode both the trigger and the
transformation. Now any tts.summarization strategy drops in unchanged:

    from tts.summarization import MaskBasedSummarizer
    agent = SummarizingAgent(model, env, compactor=MaskBasedSummarizer())

Passing compactor=None disables compaction (the `full` baseline: run the
complete context and let the model's own limit bite).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from minisweagent.agents.default import DefaultAgent

from tts.summarization.base import Compactor

logger = logging.getLogger(__name__)


def message_text(m: dict) -> str:
    """Rough text of a message for token counting (content + tool-call args)."""
    parts = [m.get("content") or ""]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", "")
        parts.append(fn.get("name", ""))
        parts.append(args if isinstance(args, str) else json.dumps(args))
    return "\n".join(p for p in parts if p)


class SummarizingAgent(DefaultAgent):
    """DefaultAgent that compacts its own context when it grows past a budget.

    When the rendered context exceeds `compress_at_tokens` (or `compress_at_turns`
    complete turns, if set), the history is handed to `compactor`. The first
    `keep_first` messages and the last `keep_last_turns` complete (assistant,
    tool) turns are always preserved verbatim — the compactor only rewrites what
    lies between them.
    """

    def __init__(
        self,
        *args,
        compactor: Compactor | None = None,
        tokenizer=None,
        compress_at_tokens: int = 24000,
        compress_at_turns: int = 0,
        keep_first: int = 4,
        keep_last_turns: int = 3,
        progress_manager=None,
        instance_id: str = "",
        compressions_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.compactor = compactor
        # Directory to write one file per compaction event as it is triggered.
        self.compressions_dir = compressions_dir
        self.tokenizer = tokenizer
        self.compress_at_tokens = compress_at_tokens
        # If > 0, trigger on complete-turn count instead of tokens (matches how
        # training/OpenHands measure history — by event/turn count, not tokens).
        self.compress_at_turns = compress_at_turns
        self.keep_first = keep_first
        self.keep_last_turns = keep_last_turns
        self._pm = progress_manager
        self._iid = instance_id
        self.n_compressions = 0
        # One record per compaction event; `kind` comes from the compactor
        # ("summary" | "mask" | "truncation" | "summary_failed").
        self.compressions: list[dict] = []
        # Input context token length fed to the deliberator at each model call
        # (post-compression) — a per-turn series for plotting context growth.
        self.context_tokens: list[int] = []

    # -- trigger -----------------------------------------------------------

    def _tokens_of(self, messages: list[dict]) -> int:
        text = "\n".join(message_text(m) for m in messages)
        if self.tokenizer is None:
            return len(text) // 4  # crude fallback
        return len(self.tokenizer.encode(text))

    def _context_tokens(self) -> int:
        return self._tokens_of(self.messages)

    def _context_turns(self) -> int:
        # One complete (assistant, tool) turn ends on a tool message.
        return sum(1 for m in self.messages if m.get("role") == "tool")

    def _over_budget(self) -> bool:
        if self.compress_at_turns > 0:
            return self._context_turns() >= self.compress_at_turns
        return self._context_tokens() >= self.compress_at_tokens

    def _should_compact(self) -> bool:
        if self.compactor is None:
            return False
        # Need at least keep_first + a summary slot + one tail turn to bother.
        if len(self.messages) <= self.keep_first + 2:
            return False
        return self._over_budget()

    # -- compaction --------------------------------------------------------

    def _maybe_compress(self) -> None:
        if not self._should_compact():
            return

        n_before = len(self.messages)
        tokens_before = self._tokens_of(self.messages)

        result = self.compactor.compact(
            self.messages,
            keep_first=self.keep_first,
            keep_last_turns=self.keep_last_turns,
        )
        if result.kind == "summary_failed":
            logger.warning(
                f"{self._iid}: summarization failed "
                f"({result.metadata.get('error')}); fell back to truncation"
            )

        record = {
            "index": len(self.compressions),  # 0-based order this compaction fired
            "kind": result.kind,
            "n_calls_at": self.n_calls,
            "n_msgs_before": n_before,
            "n_msgs_after": len(result.messages),
            "tokens_before": tokens_before,
            "tokens_after": self._tokens_of(result.messages),
            "summary": result.summary,
            "metadata": result.metadata,
            # The partial trajectory fed to the compactor (the pre-compression
            # context). Saved so compactions can be inspected against their input.
            "input_messages": [dict(m) for m in self.messages],
        }
        self.compressions.append(record)
        self.n_compressions = len(self.compressions)
        self._save_compaction(record)
        logger.info(
            f"{self._iid}: compaction #{self.n_compressions} kind={record['kind']} "
            f"({n_before} msgs / {tokens_before} tok -> "
            f"{len(result.messages)} msgs / {record['tokens_after']} tok)"
        )
        self.messages = result.messages

    def _save_compaction(self, record: dict) -> None:
        """Write one compaction event to its own file as it is triggered."""
        if self.compressions_dir is None:
            return
        self.compressions_dir.mkdir(parents=True, exist_ok=True)
        path = self.compressions_dir / f"compaction_{record['index']:03d}_{record['kind']}.json"
        path.write_text(json.dumps(record, indent=2))

    # -- agent loop hooks --------------------------------------------------

    def query(self) -> dict:
        self._maybe_compress()
        # Record the input context length for this turn (the messages the
        # deliberator is about to be queried with, after any compression).
        self.context_tokens.append(self._tokens_of(self.messages))
        return super().query()

    def step(self) -> list[dict]:
        if self._pm is not None:
            try:
                self._pm.update_instance_status(
                    self._iid,
                    f"Step {self.n_calls + 1:3d} (${self.cost:.2f}, {self.n_compressions}c)",
                )
            except KeyError:
                pass
        return super().step()
