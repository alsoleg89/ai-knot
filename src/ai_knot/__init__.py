"""ai-knot — deterministic, self-hosted long-term memory for AI agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from ai_knot.always_on.memory import AlwaysOnMemory


__version__ = "0.12.0"

__all__ = [
    "AlwaysOnMemory",
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


def __getattr__(name: str) -> Any:
    if name == "AlwaysOnMemory":
        from ai_knot.always_on.memory import AlwaysOnMemory

        return AlwaysOnMemory
    raise AttributeError(name)
