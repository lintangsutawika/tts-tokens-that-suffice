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

from tts.summarization import make_compactor
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
        compactor: "Compactor | str | None" = None,
        tokenizer=None,
        compress_at_tokens: int = 24000,
        compress_at_turns: int = 0,
        keep_first: int = 4,
        keep_last_turns: int = 3,
        # Used only when `compactor` is a NAME (see below): the model-based
        # summarizer's backend. summarizer_model defaults to the agent's own model;
        # api_base/api_key empty => litellm resolves them from env (the agent's
        # endpoint). Ignored when `compactor` is already a Compactor instance.
        summarizer_model: str | None = None,
        summarizer_api_base: str = "",
        summarizer_api_key: str = "",
        summarizer_max_tokens: int = 512,
        summarizer_style: str = "sectioned",
        mask_output: bool = True,
        mask_thinking: bool = False,
        progress_manager=None,
        instance_id: str = "",
        compressions_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Accept a compactor NAME as well as an instance: a config file (e.g. the
        # mini-swe-agent yaml under harbor) can only carry scalars, so build the
        # compactor from the string here. An instance passes through unchanged.
        if isinstance(compactor, str):
            compactor = make_compactor(
                compactor,
                summarizer_model=summarizer_model or self._agent_model_name(),
                summarizer_api_base=summarizer_api_base,
                summarizer_api_key=summarizer_api_key,
                summarizer_max_tokens=summarizer_max_tokens,
                summarizer_style=summarizer_style,
                mask_output=mask_output,
                mask_thinking=mask_thinking,
            )
        self.compactor = compactor
        # Directory to write one file per compaction event as it is triggered.
        # compressions_dir may arrive as a str from the yaml config; coerce to
        # Path once so every `... / <name>` below works (str/str raises TypeError).
        self.compressions_dir = Path(compressions_dir) if compressions_dir else None
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
        # Actual input-prompt tokens the server reported on the last model call
        # (None until the first response). Drives _context_tokens() when set.
        self.last_prompt_tokens: int | None = None
        # Persist an explicit n_compressions=0 so a run with no compactions is
        # distinguishable from one with no count (only when compressions_dir set).
        if self.compressions_dir is not None:
            try:
                self._write_compression_count(0)
            except OSError:
                pass

    def _agent_model_name(self) -> str:
        """The litellm model string this agent uses (default for the summarizer)."""
        # Resolve the agent's own litellm model string across the model wrappers
        # used by mini-swe-agent / harbor (LitellmModel.config.model_name,
        # a bare .model_name, .name, or the repr). Raise loudly if we cannot
        # resolve it — the subtask/model compactors REQUIRE summarizer_model, and
        # a silent "" here leaves compactor=None (always False _should_compact).
        cfg = getattr(self.model, "config", None)
        name = (
            getattr(cfg, "model_name", None)
            or getattr(self.model, "model_name", None)
            or getattr(self.model, "name", None)
            or ""
        )
        if not name:
            s = str(self.model).strip()
            name = s if s and s.lower() not in ("<object>", "none", "") else ""
        if not name:
            raise ValueError(
                "could not resolve the agent's model name for the summarizer; "
                "set summarizer_model explicitly in the config"
            )
        return name

    # -- trigger -----------------------------------------------------------

    def _tokens_of(self, messages: list[dict]) -> int:
        text = "\n".join(message_text(m) for m in messages)
        if self.tokenizer is None:
            # Conservative estimate that must NOT undercount: chars//4 undercounts
            # code/JSON/tool-heavy content (real ratio can be ~3 chars/token) and
            # misses the chat-template/role overhead per message, so compaction
            # could fire too late and hit the serve's max_model_len (seen as
            # ContextWindowExceededError on long agent channels). Use chars//3 for
            # the text PLUS a small per-message overhead for role/format tokens.
            return len(text) // 3 + 4 * len(messages)
        return len(self.tokenizer.encode(text))

    def _context_tokens(self) -> int:
        # Prefer the ACTUAL input-token count the server reported on the last
        # model call (message.extra.response.usage.prompt_tokens), avoiding the
        # chars//4 estimate which undercounts code/tool-heavy contexts and
        # delayed compaction (ContextWindowExceededError). Falls back to the
        # estimate only before the first response has arrived.
        if self.last_prompt_tokens is not None:
            return self.last_prompt_tokens
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

    def _compression_subdir(self) -> Path:
        # compressions_dir points at the container's per-trial /logs/agent
        # (synced back to jobs/<run>/<trial>/agent/), so count.json lands
        # directly in that dir -- no extra per-trial subdir needed.
        return self.compressions_dir

    def _write_compression_count(self, n: int) -> None:
        """Persist the running n_compressions for this trial (best-effort)."""
        sub = self._compression_subdir()
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "count.json").write_text(json.dumps({"n_compressions": n}, indent=2))

    def _save_compaction(self, record: dict) -> None:
        """Write one compaction event to its own file as it is triggered."""
        if self.compressions_dir is None:
            return
        self.compressions_dir.mkdir(parents=True, exist_ok=True)
        path = self.compressions_dir / f"compaction_{record['index']:03d}.json"
        path.write_text(json.dumps(record, indent=2))
        # Persist the running count (n_compressions) so it survives container
        # teardown and can be aggregated into result.json. Per-trial subdir so
        # concurrent trials never collide under N_CONCURRENT. Best-effort.
        try:
            self._compression_subdir().mkdir(parents=True, exist_ok=True)
            self._write_compression_count(self.n_compressions)
        except OSError:
            pass

    # -- agent loop hooks --------------------------------------------------

    def query(self) -> dict:
        self._maybe_compress()
        # Record the input context length for this turn (the messages the
        # deliberator is about to be queried with, after any compression).
        self.context_tokens.append(self._tokens_of(self.messages))
        msg = super().query()
        # Capture the actual input-token count from the server's usage so the
        # NEXT turn's compaction trigger uses the real context, not an estimate.
        try:
            usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                self.last_prompt_tokens = int(usage["prompt_tokens"])
        except Exception:
            pass
        return msg

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
