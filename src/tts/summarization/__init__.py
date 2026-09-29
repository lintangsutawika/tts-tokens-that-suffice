"""Types of summarization strategies
  * ModelBasedSummarizer      — replace the middle with a generated summary
  * MaskBasedSummarizer       — keep the agent's actions, elide environment output
  * TruncationBasedSummarizer — drop the middle outright
  * CliffCompactor            — rule-based CliffCompaction summary (no model call)
  * ThemeModelBasedSummarizer — model segments the middle into per-subtask
    (theme) summaries; the in-progress theme is kept verbatim (not summarized)

`make_compactor(name, ...)` builds one from a config string — used when the
compactor is selected from a config file (e.g. the mini-swe-agent yaml under
harbor) that can only carry scalars, not a constructed object.
"""
from .cliff_based import CliffCompactor
from .mask_based import MaskBasedSummarizer
from .model_based import ModelBasedSummarizer
from .theme_based import ThemeModelBasedSummarizer
from .truncation_based import TruncationBasedSummarizer

__all__ = [
    "CliffCompactor",
    "MaskBasedSummarizer",
    "ModelBasedSummarizer",
    "ThemeModelBasedSummarizer",
    "TruncationBasedSummarizer",
    "make_compactor",
]


def make_compactor(
    name: str,
    *,
    summarizer_model: str | None = None,
    summarizer_api_base: str = "",
    summarizer_api_key: str = "",
    summarizer_max_tokens: int = 512,
    summarizer_style: str = "sectioned",
    mask_output: bool = True,
    mask_thinking: bool = False,
):
    """Build a compactor from a config string.

    name: "mask" | "truncation" | "model" | "theme" | "cliff" | "none". For
    "model" and "theme", `summarizer_model` must be set (callers default it to
    the agent's own model); leaving api_base/api_key empty means "use the
    agent's endpoint" (litellm resolves them from the environment);
    `summarizer_style` picks the summary prompt ("sectioned" | "unconstrained"
    | "theme"). "cliff" builds the rule-based CliffCompaction compactor (no
    model call). Returns None for "none"/"off".
    """
    name = (name or "mask").strip().lower()
    if name in ("none", "off", ""):
        return None
    if name == "mask":
        return MaskBasedSummarizer(mask_output=mask_output, mask_thinking=mask_thinking)
    if name in ("truncation", "truncate"):
        return TruncationBasedSummarizer()
    if name in ("model", "summary"):
        from tts.data.agent_trajectory import get_summary_prompt

        from .model_based import LitellmSummarizer

        if not summarizer_model:
            raise ValueError("compactor='model' requires summarizer_model")
        return ModelBasedSummarizer(
            LitellmSummarizer(
                summarizer_model,
                api_base=summarizer_api_base,
                api_key=summarizer_api_key,
                max_tokens=summarizer_max_tokens,
                system_prompt=get_summary_prompt(summarizer_style),
            )
        )
    if name in ("theme", "theme_model", "subtask"):
        from .theme_based import ThemeLitellmSummarizer

        if not summarizer_model:
            raise ValueError("compactor='theme' requires summarizer_model")
        return ThemeModelBasedSummarizer(
            ThemeLitellmSummarizer(
                summarizer_model,
                api_base=summarizer_api_base,
                api_key=summarizer_api_key,
                max_tokens=max(summarizer_max_tokens, 1024),
            )
        )
    if name in ("cliff", "cliffcompaction"):
        return CliffCompactor()
    raise ValueError(
        f"unknown compactor {name!r} (mask|truncation|model|theme|cliff|none)"
    )