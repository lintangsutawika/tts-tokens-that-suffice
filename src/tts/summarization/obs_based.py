"""
Per-observation LLM summarization compactor.

Keeps the agent's ASSISTANT messages (reasoning + tool-call framing) VERBATIM —
the react loop that drives decisions — and summarizes each TOOL/observation
(potentially long) into a short "[Compacted Obs] ..." string via a lightweight
LLM call. Because each observation is short-ish, summarizing it individually is
cheap, unlike full-context summarizers (subtask/sectioned) that rephrase the
whole middle (including the small, high-precision assistant core).

Rationale (from measured token composition): observations are ~76-86% of context
at 32k but have low per-unit value once summarized; assistant reasoning is
<12% and high-precision, so it must stay verbatim. This compactor drops the
observation bulk while preserving the react loop — the failure mode subtask/
sectioned hit (rephrasing the assistant core -> precision loss -> NonZero collapse).

Trigger: the agent's token threshold (compress_at_tokens) via SummarizingAgent.
keep_first/keep_last_turns split the head/tail verbatim; only the middle's
observations are LLM-summarized. Robust: a summarizer failure degrades to keeping
the observation as-is (never crashes), and empty/short observations are kept as-is.
"""

from __future__ import annotations

import logging
import litellm
from litellm.exceptions import APIError

from .base import CompactionResult, split_head_tail

logger = logging.getLogger("tts.obs")

OBS_PREFIX = "[Compacted Obs] "

_OBS_SUMMARY_SYSTEM = (
    "You are a context-compaction assistant. Given a single tool/environment "
    "observation (the output an agent's shell command produced), write a concise "
    "2-3 sentence summary that preserves the ACTIONABLE facts the agent needs: "
    "file paths, error messages, return codes, key values, and what the command "
    "accomplished. Do not add commentary. Output ONLY the summary text, no prefix."
)


class ObsSummarySummarizer:
    """lightweight per-observation LLM summarizer, one call per observation."""

    def __init__(self, model: str, api_base: str = "", api_key: str = "",
                 max_tokens: int = 96, temperature: float = 0.0):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    def summarize(self, obs_text: str) -> str:
        """Summarize one observation; on failure returns '' (caller keeps original)."""
        if not obs_text or len(obs_text.strip()) < 20:
            # tiny observation, no point summarizing
            return ""
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": _OBS_SUMMARY_SYSTEM},
                {"role": "user", "content": obs_text[:12000]},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        try:
            resp = litellm.completion(**kwargs)
            text = resp.choices[0].message.content.strip()
            if not text:
                return ""
            return OBS_PREFIX + text
        except Exception as e:  # never crash the agent on a summarizer failure
            logger.warning("obs summarizer failed: %s", e)
            return ""


def _obs_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        return " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
    return str(c)


class ObservationSummaryCompactor:
    """keep assistant verbatim, LLM-summarize each observation in the middle."""

    def __init__(self, summarizer: ObsSummarySummarizer, keep_obs_prefix: bool = True,
                 min_obs_chars: int = 120):
        self.summarizer = summarizer
        self.keep_obs_prefix = keep_obs_prefix
        self.min_obs_chars = min_obs_chars  # observations shorter than this stay as-is
        self._n_summarized = 0

    def compact(self, messages: list[dict], keep_first: int = 4, keep_last_turns: int = 3) -> CompactionResult:
        head, middle, tail = split_head_tail(messages, keep_first, keep_last_turns)
        summarized = 0
        new_middle = []
        for m in middle:
            role = m.get("role")
            if role == "assistant":
                # keep the agent's react-loop reasoning verbatim
                new_middle.append(m)
            elif role == "tool":
                txt = _obs_text(m)
                if len(txt) >= self.min_obs_chars:
                    summ = self.summarizer.summarize(txt)
                    if summ:
                        new_middle.append({**m, "content": summ})
                        summarized += 1
                        continue
                new_middle.append(m)  # skip/failure -> keep original
            else:
                new_middle.append(m)
        self._n_summarized = summarized
        return CompactionResult(
            messages=head + new_middle + tail,
            kind="obs",
            summary=None,
            metadata={
                "n_observations": sum(1 for m in middle if m.get("role") == "tool"),
                "n_summarized": summarized,
                "min_obs_chars": self.min_obs_chars,
            },
        )