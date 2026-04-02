"""ai-knot — Agent Knowledge Layer. Extract. Store. Retrieve. Any backend."""

from __future__ import annotations

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.types import ConversationTurn, Fact, MemoryType, SnapshotDiff

__version__ = "0.8.0"

__all__ = [
    "ConversationTurn",
    "Fact",
    "KnowledgeBase",
    "MemoryType",
    "SharedMemoryPool",
    "SnapshotDiff",
    "__version__",
]
