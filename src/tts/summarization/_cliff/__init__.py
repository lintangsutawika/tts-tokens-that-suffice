"""Rule-based context compaction (CliffCompaction), vendored from
https://github.com/nguyenvuthientrang/cliffcompaction (MIT).

Only the message-reduction core is vendored — `compact()`/`group_turns()`
and the OpenAI-chat dialect — with the proxy-only hash-chain/engine/daemon
machinery excluded. The distillation is a mechanism for a long-horizon
agent to shrink its own history without a model call.
"""

from .cliff_core import CompactResult, compact, group_turns
from .config import Config
from .dialects_base import SUMMARY_HEADER, Dialect, truncate
from .openai_chat import DIALECT, summarize_message

__all__ = [
    "CompactResult",
    "Config",
    "DIALECT",
    "Dialect",
    "SUMMARY_HEADER",
    "compact",
    "group_turns",
    "summarize_message",
    "truncate",
]
