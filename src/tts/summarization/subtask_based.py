"""Theme-based compaction: model segments the middle into per-subtask summaries.

Unlike ``ModelBasedSummarizer`` — which replaces the whole middle with ONE
generated summary — this strategy asks the model to (a) identify the distinct
subtasks/themes the agent has been working on, (b) produce a summary for EACH
completed theme, and (c) mark the most recent, still-in-progress theme so its
steps are kept verbatim (never summarized).

This is a better fit for long-horizon coding agents than a single summary: the
agent's current active subtask stays fully readable, while only genuinely
finished work is compacted — and compacted per-theme rather than as one blob.

Contract (Contract A, model-driven):
  The model is given the trajectory MIDDLE rendered as numbered ``[STEP i]``
  events (see ``format_trajectory_text_numbered``) and returns a JSON object:
      {"subtasks": [
          {"name", "start_idx", "end_idx", "in_progress", "summary"},
          ...
      ]}
  where indices are 0-based over `messages_to_steps(middle)`, ranges are
  contiguous/non-overlapping and span every step, exactly one theme (the last)
  has in_progress=true with summary=null. The compactor maps those step ranges
  back to the original messages, keeps the in-progress theme verbatim, and
  replaces each completed theme with its summary message.

Falls back to the base model-based behavior (whole-middle single summary) or
plain truncation on any failure, so an outage degrades rather than kills the run.
"""

from __future__ import annotations

import json
import logging

import litellm

from tts.data.agent_trajectory import (
    SUBTASK_SYSTEM_PROMPT,
    TrajectoryStep,
    format_trajectory_text_numbered,
    get_summary_prompt,
    messages_to_steps,
)

from .base import CompactionResult, split_head_tail


# Marker prefix on each inserted subtask-summary message so re-compaction can
# keep PRIOR per-theme summaries verbatim (they are already self-contained
# subtask summaries) and only summarize the NEW context accumulated since the
# last compaction -- not re-segment the old summaries.
SUBTASK_SUMMARY_HEADER = "<subtask-summary> "


def is_subtask_summary(msg: dict) -> bool:
    c = msg.get("content")
    return isinstance(c, str) and c.startswith(SUBTASK_SUMMARY_HEADER)

logger = logging.getLogger(__name__)


class SubtaskParsingError(ValueError):
    """The theme summarizer returned something we could not parse into themes."""


def _parse_subtasks(raw: str) -> list[dict]:
    """Parse the model's JSON theme output. Tolerates stray markdown fences or
    surrounding prose by extracting the first JSON object. Returns the theme
    list; raises SubtaskParsingError if unusable."""
    text = raw.strip()
    # strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # grab the first {...} block as a fallback
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise SubtaskParsingError("no JSON object found in theme output")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise SubtaskParsingError(f"theme output not JSON: {e}") from e
    themes = data.get("subtasks") if isinstance(data, dict) else None
    if not isinstance(themes, list) or not themes:
        raise SubtaskParsingError("theme output missing non-empty 'themes' list")
    return themes


def _message_for_step(steps: list[TrajectoryStep], idx: int, middle: list[dict]) -> dict:
    """Map a step index (over messages_to_steps(middle)) back to the original
    message. messages_to_steps skips system + the task user message, so step i
    corresponds to a later message; we find it by replaying the same skip."""
    # Reconstruct the step->message mapping exactly as messages_to_steps does.
    step_idx = 0
    seen_task = False
    for m in middle:
        role = m.get("role", "")
        if role == "system":
            continue
        if role == "user":
            if not seen_task:
                seen_task = True
                continue
        if role == "assistant" or role == "tool" or (role == "user" and seen_task):
            if step_idx == idx:
                return m
            step_idx += 1
    raise IndexError(f"step index {idx} out of range (n_steps={step_idx})")


class SubtaskModelBasedSummarizer:
    """Compactor that partitions the middle into per-theme summaries, keeping the
    in-progress theme verbatim. Satisfies base.Compactor.

    `summarizer` : a callable ``summarize(steps) -> str`` (litellm/tinker backend)
                   returning the JSON themes output under SUBTASK_SYSTEM_PROMPT.
    """

    def __init__(self, summarizer):
        self.summarizer = summarizer

    def compact(
        self,
        messages: list[dict],
        keep_first: int = 4,
        keep_last_turns: int = 0,
    ) -> CompactionResult:
        # No tail is kept verbatim: the only the head (keep_first) is preserved
        # alongside the per-theme summaries. keep_last_turns is accepted for
        # Compactor-interface compatibility but intentionally not used.
        head, _middle, _tail = split_head_tail(messages, keep_first, 0)
        middle = _middle
        if not middle:
            return CompactionResult(
                messages=list(messages), kind="summary", summary=None,
                metadata={"subtasks": [], "n_compressed": 0},
            )

        # Re-compaction: PRIOR subtask-summary messages are already self-contained
        # per-theme summaries -- keep them verbatim and only segment the NEW
        # context accumulated since the last compaction (never re-segment the old
        # summaries). Partition the middle accordingly; preserve order.
        prior_summaries = [m for m in middle if is_subtask_summary(m)]
        fresh = [m for m in middle if not is_subtask_summary(m)]

        steps = messages_to_steps(fresh)
        if not steps:
            # Nothing new to segment -- keep everything (prior summaries + fresh)
            # verbatim.
            return CompactionResult(
                messages=list(messages), kind="summary", summary=None,
                metadata={"subtasks": [], "n_compressed": 0},
            )
        try:
            raw = self.summarizer.summarize(steps)
            themes = _parse_subtasks(raw)
        except Exception as exc:
            logger.warning(f"theme summarization failed ({exc}); keeping middle verbatim")
            return CompactionResult(
                messages=list(messages), kind="summary_failed", summary=None,
                metadata={"error": str(exc)},
            )

        # Validate / normalize themes: they mock the JSON contract but be lenient
        # about exact bounds so a slightly-off model still works.
        themes = sorted(themes, key=lambda t: t.get("start_idx", 0))
        n_steps = len(steps)
        # Build block per theme: summary messages for completed themes; verbatim
        # step messages for the in-progress one.
        blocks: list[dict] = []
        compressed_msgs = 0
        kept_progress_msgs = 0
        meta_subtasks = []
        for t in themes:
            start = int(t.get("start_idx", 0))
            end = int(t.get("end_idx", start))
            # clamp to valid range
            start = max(0, start)
            end = min(n_steps - 1, end) if end >= 0 else start
            if end < start:
                start, end = end, start
            in_progress = bool(t.get("in_progress", False))
            summary = t.get("summary")
            meta_subtasks.append({
                "name": t.get("name", "?"),
                "start_idx": start, "end_idx": end,
                "in_progress": in_progress,
                "n_steps": end - start + 1,
                "compressed": (not in_progress) and bool(summary),
            })
            if in_progress or not summary:
                # keep the theme's steps verbatim
                for i in range(start, end + 1):
                    blocks.append(_message_for_step(steps, i, fresh))
                    kept_progress_msgs += 1
            else:
                blocks.append({"role": "user",
                               "content": SUBTASK_SUMMARY_HEADER + str(summary)})
                compressed_msgs += (end - start + 1)

        # If the model did not mark any theme in_progress, fall back to keeping the
        # last theme verbatim so the current work is always readable.
        if not any(t.get("in_progress", False) for t in themes) and themes:
            logger.info("no theme marked in_progress; keeping last theme verbatim")
            last = themes[-1]
            blocks = [b for b in blocks if b is not None]
            # safe: re-keep last theme's steps
            start = int(last.get("start_idx", 0))
            end = int(last.get("end_idx", start))
            start, end = max(0, start), min(n_steps - 1, end)
            blocks.extend(_message_for_step(steps, i, fresh) for i in range(start, end + 1))

        new_messages = [*head, *prior_summaries, *blocks]
        summary_text = "\n\n".join(
            t.get("summary") for t in themes if isinstance(t.get("summary"), str)
        )
        return CompactionResult(
            messages=new_messages,
            kind="summary",
            summary=summary_text or None,
            metadata={
                "theme": True,
                "n_subtasks": len(themes),
                "n_compressed_msgs": compressed_msgs,
                "n_kept_progress_msgs": kept_progress_msgs,
                "subtasks": meta_subtasks,
            },
        )

class SubtaskLitellmSummarizer:
    """litellm backend returning the JSON per-theme output as a TOOL CALL.

    The model emits the themes via the ``emit_subtasks`` tool (tool_choice forced),
    so the JSON arrives in ``tool_calls[0].function.arguments`` — far more reliable
    than parsing a JSON blob from free-form prose. Greedy, thinking disabled to
    match qwen3_disable_thinking. `model` is the litellm model string; api_base/
    api_key optional (default = env).
    """

    SUBTASKS_TOOL = {
        "type": "function",
        "function": {
            "name": "emit_subtasks",
            "description": (
                "Report the subtasks (themes) you identified. One entry per "
                "contiguous theme; completed themes carry a summary, the "
                "in-progress (last) theme has summary=null and in_progress=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "start_idx": {"type": "integer"},
                                "end_idx": {"type": "integer"},
                                "in_progress": {"type": "boolean"},
                                "summary": {"type": ["string", "null"]},
                            },
                            "required": [
                                "name", "start_idx", "end_idx",
                                "in_progress", "summary",
                            ],
                        },
                    }
                },
                "required": ["subtasks"],
            },
        },
    }

    def __init__(self, model: str, api_base: str = "", api_key: str = "",
                 max_tokens: int = 1024, system_prompt: str = SUBTASK_SYSTEM_PROMPT):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def summarize(self, steps: list[TrajectoryStep]) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": format_trajectory_text_numbered(steps)},
            ],
            tools=[self.SUBTASKS_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_subtasks"}},
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            # Fallback: backend returned plain content instead of a tool call.
            return msg.content or ""
        return tcs[0].function.arguments or ""
