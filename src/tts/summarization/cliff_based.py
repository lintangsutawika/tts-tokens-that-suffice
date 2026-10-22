"""CliffCompaction as a tts Compactor.

The rule-based context-compaction strategy from
`CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents`
(Nguyen et al., 2026 — arXiv:2609.26779, MIT). Uses the ``cliffcompaction``
PyPI package's ``compact()`` + OpenAI-chat dialect; this class adapts it to
the ``tts.summarization.base.Compactor`` interface so it drops into
``SummarizingAgent`` exactly like the mask/model/truncation strategies.

Unlike ``ModelBasedSummarizer`` this needs no model call: the middle of the
history is reduced to a single mechanical summary message (assistant text +
thinking + tool-call signatures; tool results kept iff <= result_max_chars,
dropped otherwise). It is the deterministic, cost-free counterpart to the
learned summarizer, and a distinct baseline from both ``mask`` (which keeps
the agent's actions and only elides environment output) and ``truncation``
(which drops the middle outright).

The agent's own token trigger decides *when* to compact; this strategy
decides *how*. ``keep_recent`` (CliffCompaction, in assistant-step turns)
maps onto the agent's ``keep_last_turns``; see the note below.
"""

from dataclasses import replace

from cliffcompaction.cliff import compact
from cliffcompaction.config import Config as CliffConfig
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.dialects.openai_chat import DIALECT

from .base import CompactionResult, split_head_tail


class CliffCompactor:
    """Compactor that replaces the middle of the history with a rule-based
    CliffCompaction summary. Satisfies base.Compactor.

    Parameters mirror CliffCompaction's ``cliff.compact`` config:

      result_max_chars : tool results longer than this are dropped from the
          summary (default 500, upstream default).
      thought_max_chars : cap on assistant ``content`` kept per summarized
          turn; 0 = unlimited (upstream default).
      thinking_max_chars : cap on ``reasoning_content``/``reasoning`` text kept
          per summarized turn; 0 = unlimited (upstream default).
      cmd_max_chars : cap on a tool-call signature's serialized arguments.
      human_max_chars : sanity cap on user/system text kept in the summary.
      keep_thinking : keep reasoning text in summaries (else drop it).

    ``keep_first``/``keep_last_turns`` come from the agent at compress time.
    CliffCompaction groups by *assistant-step* turns; this adapter runs its
    own grouping over the agent's ``split_head_tail`` middle so the head/tail
    skeleton matches the other strategies exactly, then applies ``compact``.
    """

    def __init__(
        self,
        *,
        result_max_chars: int = 500,
        thought_max_chars: int = 300,
        thinking_max_chars: int = 300,
        cmd_max_chars: int = 150,
        human_max_chars: int = 20000,
        keep_thinking: bool = True,
        # Match upstream CliffCompaction's MINIMAL head/tail: keep only the
        # pre-first-assistant head (system+user task) verbatim and let cliff's
        # OWN keep_recent decide the trailing turns, instead of this harness's
        # keep_first=4 / keep_last_turns=3 split_head_tail reservation. The
        # harness-wide head/tail convention is bigger than upstream's minimal
        # head, so on small windows (8k) the verbatim floor alone can exceed the
        # trigger; minimal_head_tail reduces it to upstream's leaner design.
        # Default False keeps the shared convention (comparability with the
        # mask/model/truncation strategies). When True, compact() ignores the
        # agent's keep_first/keep_last_turns and delegates head+tail entirely to
        # the upstream package (keep_recent = keep_last_turns).
        minimal_head_tail: bool = False,
    ):
        # With the harness head/tail convention (minimal_head_tail=False) the
        # agent already reserved the tail via split_head_tail, so every middle
        # turn should be compactable -> force keep_recent=0 (upstream's default
        # 3 would otherwise refuse to compact a short middle and no-op us).
        # With minimal_head_tail=True we delegate tail retention to upstream, so
        # keep_recent carries the agent's keep_last_turns (patched per-compact).
        self._minimal_head_tail = minimal_head_tail
        self._cfg = CliffConfig(
            result_max_chars=result_max_chars,
            thought_max_chars=thought_max_chars,
            thinking_max_chars=thinking_max_chars,
            cmd_max_chars=cmd_max_chars,
            human_max_chars=human_max_chars,
            keep_thinking=keep_thinking,
            keep_recent=0,
        )

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 3,
    ) -> CompactionResult:
        if self._minimal_head_tail:
            # Upstream fan: delegate head AND tail retention entirely to the
            # cliffcompaction package. upstream computes its own minimal head
            # (pre-first-assistant = system+user) and keeps its own keep_recent
            # trailing turns verbatim, so the verbatim floor is minimial.
            cfg = replace(self._cfg, keep_recent=keep_last_turns)
            head, middle, tail = [], [], []
            ctx = compact(messages, DIALECT, cfg)
        else:
            # Shared-convention path: reserve keep_first / keep_last_turns via
            # split_head_tail (matching mask/model/truncation), then compact the
            # middle with keep_recent=0 (everything in the middle eligible).
            head, middle, tail = split_head_tail(messages, keep_first, keep_last_turns)
            if not middle:
                # Nothing to compact — mirror truncation's no-op shape.
                return CompactionResult(
                    messages=list(messages),
                    kind="summary",
                    summary=None,
                    metadata={"n_dropped": 0, "cliff": True},
                )
            ctx = compact(middle, DIALECT, self._cfg)
        if ctx is None:
            # compact() found nothing to gain (e.g. every middle turn fits or
            # the summary would not shrink it). Keep the full list unchanged.
            return CompactionResult(
                messages=list(messages),
                kind="summary",
                summary=None,
                metadata={"n_dropped": 0, "cliff": True, "reason": "no_gain"},
            )
        if ctx.summary.get("role") != "user":
            raise TypeError(
                "unexpected summary message role "
                f"{ctx.summary.get('role')!r} from cliff.compact"
            )
        summary_text = ctx.summary.get("content") or ""
        if self._minimal_head_tail:
            return CompactionResult(
                messages=ctx.messages,  # upstream already built [head, summary, kept]
                kind="summary",
                summary=summary_text,
                metadata={
                    "cliff": True,
                    "minimal_head_tail": True,
                    "summary_len": len(summary_text),
                    "result_max_chars": self._cfg.result_max_chars,
                    "thought_max_chars": self._cfg.thought_max_chars,
                    "thinking_max_chars": self._cfg.thinking_max_chars,
                    "keep_thinking": self._cfg.keep_thinking,
                },
            )
        return CompactionResult(
            messages=[*head, ctx.summary, *tail],
            kind="summary",
            summary=summary_text,
            metadata={
                "cliff": True,
                "minimal_head_tail": False,
                "head_len": len(head),
                "middle_len": len(middle),
                "summary_len": len(summary_text),
                "result_max_chars": self._cfg.result_max_chars,
                "thought_max_chars": self._cfg.thought_max_chars,
                "thinking_max_chars": self._cfg.thinking_max_chars,
                "keep_thinking": self._cfg.keep_thinking,
            },
        )

    # -- introspection for plotting/tuning ---------------------------------

    @property
    def summary_header(self) -> str:
        """The marker prefix of a Cliff summary (used to recognize prior
        summaries on re-compaction)."""
        return SUMMARY_HEADER