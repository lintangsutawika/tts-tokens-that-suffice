"""Types of summarization strategies

  * ModelBasedSummarizer      — replace the middle with a generated summary
  * MaskBasedSummarizer       — keep the agent's actions, elide environment output
  * TruncationBasedSummarizer — drop the middle outright

"""

from .mask_based import MaskBasedSummarizer
from .model_based import ModelBasedSummarizer
from .truncation_based import TruncationBasedSummarizer

__all__ = ["MaskBasedSummarizer", "ModelBasedSummarizer", "TruncationBasedSummarizer"]
