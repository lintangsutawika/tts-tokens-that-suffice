"""Types of summarization strategies
  * ModelBasedSummarizer      — replace the middle with a generated summary
  * MaskBasedSummarizer       — keep the agent's actions, elide environment output
  * TruncationBasedSummarizer — drop the middle outright
  * CliffCompactor            — rule-based CliffCompaction summary (no model call)
  * SubtaskModelBasedSummarizer — model segments the middle into per-subtask
    (theme) summaries; the in-progress theme is kept verbatim (not summarized)

`make_compactor(name, ...)` builds one from a config string — used when the
compactor is selected from a config file (e.g. the mini-swe-agent yaml under
harbor) that can only carry scalars, not a constructed object.
"""
from .cliff_based import CliffCompactor
from .mask_based import MaskBasedSummarizer
from .obs_based import ObservationSummaryCompactor, ObsSummarySummarizer
from .model_based import ModelBasedSummarizer
from .subtask_based import SubtaskModelBasedSummarizer
from .truncation_based import TruncationBasedSummarizer

__all__ = [
    "CliffCompactor",
    "MaskBasedSummarizer",
    "ModelBasedSummarizer",
    "SubtaskModelBasedSummarizer",
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
    mask_keep_n: int = 0,
    mask_tagged_keep: bool = True,
    mask_long_output_chars: int = 5000,
    mask_descriptive_placeholder: bool = True,
    # mask FLUSH mode: drop the OLDER masked tool messages (the placeholder ones)
    # so they don't accumulate past the context budget. Stock mask keeps every
    # message -> context grows unbounded (~95% overflow). flush=True removes prior
    # masked turns (like cliff discards its prior summary), bounding the context.
    mask_flush: bool = False,
    # cliff: when True, delegate head+tail retention entirely to upstream
    # CliffCompaction (pre-first-assistant head + its own keep_recent) instead
    # of the harness keep_first/keep_last_turns split_head_tail convention.
    # Reduces the verbatim floor on small windows (8k overflow).
    cliff_minimal_head_tail: bool = False,
    # model/sectioned: keep ONLY the leading user+system messages verbatim
    # (instead of the shared keep_first=4). Mirrors cliff-match's minimal head;
    # the summarizer still covers the full history (z = ONE summary).
    head_user_system: bool = False,
    # model/sectioned FLUSH mode: discard the prior summary on each recompaction
    # and re-summarize from RAW turns (retained across compactions), avoiding
    # summary-of-summary drift. Default False = recursive single-summary.
    model_flush: bool = False,
):
    """Build a compactor from a config string.

    name: "mask" | "truncation" | "model" | "subtask" | "cliff" | "none". For
    "model" and "subtask", `summarizer_model` must be set (callers default it to
    the agent's own model); leaving api_base/api_key empty means "use the
    agent's endpoint" (litellm resolves them from the environment);
    `summarizer_style` picks the summary prompt ("sectioned" | "unconstrained"
    | "subtask"). "cliff" builds the rule-based CliffCompaction compactor (no
    model call). Returns None for "none"/"off".
    """
    name = (name or "mask").strip().lower()
    if name in ("none", "off", ""):
        return None
    if name == "mask":
        return MaskBasedSummarizer(
            mask_output=mask_output,
            mask_thinking=mask_thinking,
            keep_n=mask_keep_n,
            tagged_keep=mask_tagged_keep,
            long_output_chars=mask_long_output_chars,
            descriptive_placeholder=mask_descriptive_placeholder,
            flush=mask_flush,
        )
    if name in ("truncation", "truncate"):
        return TruncationBasedSummarizer()
    if name in ("model", "summary"):
        from tts.data.agent_trajectory import get_summary_prompt

        from .model_based import LitellmSummarizer

        if not summarizer_model:
            raise ValueError("compactor='model' requires summarizer_model")
        from .model_based import ModelBasedSummarizer
        return ModelBasedSummarizer(
            LitellmSummarizer(
                summarizer_model,
                api_base=summarizer_api_base,
                api_key=summarizer_api_key,
                max_tokens=summarizer_max_tokens,
                system_prompt=get_summary_prompt(summarizer_style),
            ),
            head_user_system=head_user_system,
            flush=model_flush,
        )
    if name in ("subtask", "theme_model"):
        from .subtask_based import SubtaskLitellmSummarizer

        if not summarizer_model:
            raise ValueError("compactor='subtask' requires summarizer_model")
        return SubtaskModelBasedSummarizer(
            SubtaskLitellmSummarizer(
                summarizer_model,
                api_base=summarizer_api_base,
                api_key=summarizer_api_key,
                max_tokens=max(summarizer_max_tokens, 1024),
            )
        )
    if name in ("cliff", "cliffcompaction"):
        return CliffCompactor(minimal_head_tail=cliff_minimal_head_tail)
    if name in ("obs", "observation"):
        # per-observation LLM summarizer: keeps the agent's assistant turns
        # verbatim, summarizes each tool/observation into "[Compacted Obs] ...".
        from .obs_based import ObsSummarySummarizer, ObservationSummaryCompactor
        if not summarizer_model:
            raise ValueError("compactor='obs' requires summarizer_model")
        return ObservationSummaryCompactor(
            ObsSummarySummarizer(
                summarizer_model,
                api_base=summarizer_api_base,
                api_key=summarizer_api_key,
                max_tokens=min(summarizer_max_tokens, 128),
            )
        )
    raise ValueError(
        f"unknown compactor {name!r} (mask|truncation|model|theme|cliff|none)"
    )