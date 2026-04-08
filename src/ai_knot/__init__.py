"""ai-knot — Agent Knowledge Layer. Extract. Store. Retrieve. Any backend."""

from __future__ import annotations

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.types import (
    CONFLICT_POLICIES,
    ConflictPolicy,
    ConversationTurn,
    Evidence,
    EvidenceStore,
    Fact,
    MemoryOp,
    MemoryType,
    Provenance,
    SnapshotDiff,
)

__version__ = "0.9.2"

__all__ = [
    "CONFLICT_POLICIES",
    "ConflictPolicy",
    "ConversationTurn",
    "Evidence",
    "EvidenceStore",
    "Fact",
    "KnowledgeBase",
    "MemoryOp",
    "MemoryType",
    "Provenance",
    "SharedMemoryPool",
    "SnapshotDiff",
    "__version__",
]
