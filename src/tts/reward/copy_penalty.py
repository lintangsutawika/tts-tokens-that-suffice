"""
Anti-copy penalty: make transcribing x into z unprofitable.

Fidelity alone is maximized at z = x — a summary that copies its input verbatim
scores perfectly — so a pure-fidelity reward has transcription as its global
optimum. This module is the counterweight.

Kept separate from the fidelity computation because the two answer different
questions ("is z sufficient?" vs "is z actually a summary?") and are tuned
independently; see distortion_reward for how they compose.
"""

from __future__ import annotations

from tts.reward.utils import messages_text

# Markers of the raw trajectory format the summarizer is shown (format_trajectory_text
# renders steps as <EVENT type='...'>), plus tool-call syntax. A summary containing
# these is transcribing its input rather than summarizing it.
_COPY_MARKERS = ("<EVENT", "</EVENT>", "<tool_call>", "</tool_call>")


def has_copy_markers(summary: str) -> bool:
    """True if the summary echoes the raw <EVENT>/tool-call scaffolding of its input."""
    return any(m.lower() in summary.lower() for m in _COPY_MARKERS)


def ngram_overlap(summary: str, source: str, n: int = 8) -> float:
    """
    Fraction of the summary's word n-grams that appear verbatim in the source.

    ~0 for genuine paraphrase, ~1 for a transcription. n=8 is long enough that
    incidental matches (identifiers, file paths, boilerplate) stay rare, so the
    score only climbs when spans are copied wholesale.
    """
    s_words, x_words = summary.split(), source.split()
    if len(s_words) < n or len(x_words) < n:
        return 0.0
    source_grams = {tuple(x_words[i : i + n]) for i in range(len(x_words) - n + 1)}
    s_grams = [tuple(s_words[i : i + n]) for i in range(len(s_words) - n + 1)]
    if not s_grams:
        return 0.0
    return sum(g in source_grams for g in s_grams) / len(s_grams)


def copy_penalty(
    summary: str,
    x_messages: list[dict],
    tokenizer,
    lambda_len: float = 0.0,
    lambda_copy: float = 0.0,
    copy_threshold: float = 0.3,
    marker_penalty: float = 0.0,
) -> tuple[float, dict]:
    """
    Penalty that makes copying x into z unprofitable.

    Fidelity alone is maximized at z = x (distortion 0), so a pure-fidelity reward
    has verbatim transcription as its global optimum. Three terms push back:

      * length  — lambda_len * |z|/|x| in TOKENS (relative, so it scales with the
                  context rather than punishing long summaries of long inputs).
      * copy    — lambda_copy * how far the verbatim n-gram overlap exceeds
                  copy_threshold, ramped to 1.0 at full overlap.
      * marker  — flat marker_penalty when z echoes <EVENT>/tool-call scaffolding.

    Returns (penalty, components) so the caller can log which term is biting.
    """
    x_text = messages_text(x_messages)
    n_z = len(tokenizer.encode(summary)) if summary else 0
    n_x = len(tokenizer.encode(x_text)) if x_text else 0

    len_ratio = n_z / n_x if n_x else 0.0
    len_pen = lambda_len * len_ratio

    overlap = ngram_overlap(summary, x_text)
    excess = max(0.0, overlap - copy_threshold)
    denom = 1.0 - copy_threshold
    copy_pen = lambda_copy * (excess / denom if denom > 0 else 0.0)

    marker_pen = marker_penalty if has_copy_markers(summary) else 0.0

    total = len_pen + copy_pen + marker_pen
    return total, {
        "n_z_tokens": n_z,
        "n_x_tokens": n_x,
        "len_ratio": len_ratio,
        "overlap": overlap,
        "len_penalty": len_pen,
        "copy_penalty": copy_pen,
        "marker_penalty": marker_pen,
        "total_penalty": total,
    }
