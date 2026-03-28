"""agentmemo — Agent Knowledge Layer. Extract. Store. Retrieve. Any backend."""

from __future__ import annotations

from agentmemo.knowledge import KnowledgeBase
from agentmemo.types import ConversationTurn, Fact, MemoryType

__version__ = "0.1.0"

__all__ = [
    "ConversationTurn",
    "Fact",
    "KnowledgeBase",
    "MemoryType",
    "__version__",
]
