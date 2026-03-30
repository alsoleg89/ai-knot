"""ai-knot — Agent Knowledge Layer. Extract. Store. Retrieve. Any backend."""

from __future__ import annotations

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import ConversationTurn, Fact, MemoryType, SnapshotDiff

__version__ = "0.4.0"

__all__ = [
    "ConversationTurn",
    "Fact",
    "KnowledgeBase",
    "MemoryType",
    "SnapshotDiff",
    "__version__",
]
