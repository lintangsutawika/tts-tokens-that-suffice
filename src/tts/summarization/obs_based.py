"""
Per-observation LLM summarization compactor.

Keeps the agent's ASSISTANT messages (reasoning + tool-call framing) VERBATIM —
the react loop that drives decisions — and summarizes each TOOL/observation
(potentially long) into a short "[Compacted Obs] ..." string via a lightweight
LLM call. Because each observation is short-ish, summarizing it individually is
cheap, unlike full-context summarizers (subtask/sectioned) that rephrase the
whole middle (including the small, high-precision assistant core).

To avoid N sequential LLM round-trips when many observations accumulate (which
blows the agent timeout at larger windows -- 32k), observations are BATChED:
one litellm call summarizes K observations at once, returning per-index
summaries. This cuts ~K round-trips to 1.

Rationale (from measured token composition): observations are ~76-86% of context
at 32k but have low per-unit value once summarized; assistant reasoning is
<12% and high-precision, so it must stay verbatim.
"""

from __future__ import annotations

import logging
import litellm

from .base import CompactionResult, split_head_tail

logger = logging.getLogger("tts.obs")

OBS_PREFIX = "[Compacted Obs] "

_OBS_BATCH_SYSTEM = (
    "You are a context-compaction assistant. You are given a numbered list of "
    "tool/environment observations (outputs of an agent's shell commands). For "
    "EACH observation write a concise 1-2 sentence summary that preserves the "
    "ACTIONABLE facts: file paths, error messages, return codes, key values, and "
    "what the command accomplished. Output exactly one line per observation, "
    "formatted as '<N>: <summary>' where <N> is the observation index. Do not "
    "skip any; do not add commentary."
)


class ObsSummarySummarizer:
    """lightweight per-observation LLM summarizer, one batched call per group."""

    def __init__(self, model: str, api_base: str = "", api_key: str = "",
                 max_tokens: int = 96, temperature: float = 0.0, batch_size: int = 20):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size  # obs per LLM call

    def _call(self, messages: list[dict], max_tokens: int) -> str:
        kwargs = dict(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=self.temperature,
        )
        if self.api_base: kwargs["api_base"] = self.api_base
        if self.api_key: kwargs["api_key"] = self.api_key
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        text = msg.content if msg.content else (getattr(msg, "reasoning_content", None) or "")
        if not isinstance(text, str): text = str(text)
        return text.strip()

    def summarize_batch(self, obs_list: list[tuple[int, str]]) -> dict[int, str]:
        """Summarize many observations in one (or few) LLM calls.

        obs_list: list of (index, obs_text). Returns {index: summary}.
        Batches by self.batch_size to bound context/max_tokens per call.
        On a call failure, the affected indices are omitted (caller keeps orig).
        """
        out: dict[int, str] = {}
        for start in range(0, len(obs_list), self.batch_size):
            chunk = obs_list[start:start + self.batch_size]
            lines = "\n".join(f"{idx}: {txt[:4000]}" for idx, txt in chunk)
            user = f"Observation list:\n{lines}\n\nWrite '<N>: <summary>' for each."
            try:
                text = self._call([
                    {"role": "system", "content": _OBS_BATCH_SYSTEM},
                    {"role": "user", "content": user},
                ], max_tokens=self.max_tokens * max(1, len(chunk)))
            except Exception as e:
                logger.warning("obs batch summarizer failed (%d obs): %s", len(chunk), e)
                continue
            # parse "<N>: <summary>" lines
            for line in text.splitlines():
                line = line.strip()
                if ":" in line:
                    num, _, rest = line.partition(":")
                    try:
                        idx = int(num.strip())
                    except ValueError:
                        continue
                    if idx in dict(chunk) and rest.strip():
                        out[idx] = OBS_PREFIX + rest.strip()
        # fall back: any index without a summary -> individual call (best effort)
        missing = [ (idx, txt) for idx, txt in obs_list if idx not in out]
        for idx, txt in missing:
            s = self.summarize(txt)
            if s:
                out[idx] = s
        return out

    def summarize(self, obs_text: str) -> str:
        """Summarize one observation; on failure returns '' (caller keeps original)."""
        if not obs_text or len(obs_text.strip()) < 20:
            return ""
        try:
            text = self._call([
                {"role": "system", "content": _OBS_BATCH_SYSTEM},
                {"role": "user", "content": f"1: {obs_text[:12000]}\n\nWrite '1: <summary>'."},
            ], max_tokens=self.max_tokens)
            return OBS_PREFIX + text.split(":", 1)[-1].strip() if text else ""
        except Exception as e:
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
        # collect observations to summarize (index-aligned)
        obs_to_summarize = []
        obs_pos = []  # (position in new_middle) for each collectible obs
        new_middle = []
        for m in middle:
            if m.get("role") == "tool":
                txt = _obs_text(m)
                if len(txt) >= self.min_obs_chars:
                    obs_to_summarize.append((len(obs_to_summarize), txt))
                    obs_pos.append(len(new_middle))
                new_middle.append(m)
            else:
                new_middle.append(m)
        if obs_to_summarize:
            summaries = self.summarizer.summarize_batch(obs_to_summarize)
            for pos, (idx, _txt) in zip(obs_pos, obs_to_summarize):
                if idx in summaries:
                    new_middle[pos] = {**new_middle[pos], "content": summaries[idx]}
                    self._n_summarized += 1
        return CompactionResult(
            messages=head + new_middle + tail,
            kind="obs",
            summary=None,
            metadata={
                "n_observations": sum(1 for m in middle if m.get("role") == "tool"),
                "n_summarized": self._n_summarized,
                "min_obs_chars": self.min_obs_chars,
            },
        )