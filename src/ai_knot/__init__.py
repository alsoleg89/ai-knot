"""ai-knot — Agent Knowledge Layer. Extract. Store. Retrieve. Any backend."""

from __future__ import annotations

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.query_types import (
    AnswerContract,
    AnswerItem,
    AnswerSpace,
    AtomicClaim,
    ClaimKind,
    RawEpisode,
    TimeAxis,
    TruthMode,
)
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

__version__ = "0.9.4"

__all__ = [
    "CONFLICT_POLICIES",
    "AnswerContract",
    "AnswerItem",
    "AnswerSpace",
    "AtomicClaim",
    "ClaimKind",
    "ConflictPolicy",
    "ConversationTurn",
    "Evidence",
    "EvidenceStore",
    "Fact",
    "KnowledgeBase",
    "MemoryOp",
    "MemoryType",
    "Provenance",
    "RawEpisode",
    "SharedMemoryPool",
    "SnapshotDiff",
    "TimeAxis",
    "TruthMode",
    "__version__",
]
