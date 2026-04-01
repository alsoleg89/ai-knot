"""Core data types for ai_knot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class MemoryType(StrEnum):
    """Classification of stored knowledge.

    SEMANTIC — facts about the world or user ("works at Sber").
    PROCEDURAL — how the user wants things done ("always use type hints").
    EPISODIC — specific past events ("deploy failed last Tuesday").
    """

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


@dataclass
class Fact:
    """A single unit of knowledge extracted from or added to an agent's memory.

    Attributes:
        content: The human-readable knowledge string.
        type: Classification — semantic, procedural, or episodic.
        importance: How critical this fact is (0.0-1.0). Higher = remembered longer.
        retention_score: Current memory strength after Ebbinghaus decay (0.0-1.0).
        access_count: Number of times this fact was retrieved via recall().
        tags: Optional labels for organization.
        id: Unique 8-char hex identifier.
        created_at: When the fact was first stored (UTC).
        last_accessed: When the fact was last retrieved (UTC).
        source_snippets: Raw text excerpts that support this fact (v0.5).
        source_spans: Location references for each snippet (v0.5).
        supported: Whether the fact has been verified against sources (v0.5).
        support_confidence: Confidence score for source support (0.0-1.0) (v0.5).
        verification_source: How the fact was verified (v0.5).
    """

    content: str
    type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.8
    retention_score: float = 1.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Evidence and verification fields (v0.5)
    source_snippets: list[str] = field(default_factory=list)
    source_spans: list[str] = field(default_factory=list)
    supported: bool = True
    support_confidence: float = 1.0
    verification_source: str = "manual"
    # Spacing effect: hours between consecutive accesses (v0.6)
    access_intervals: list[float] = field(default_factory=list)
    # Multi-agent provenance fields (v0.6)
    origin_agent_id: str = ""
    visibility: str = "private"


@dataclass
class SnapshotDiff:
    """Difference between two named snapshots (or a snapshot and current state).

    Attributes:
        snapshot_a: Name of the first snapshot (or "current").
        snapshot_b: Name of the second snapshot (or "current").
        added: Facts present in snapshot_b but absent in snapshot_a.
        removed: Facts present in snapshot_a but absent in snapshot_b.
    """

    snapshot_a: str
    snapshot_b: str
    added: list[Fact]
    removed: list[Fact]


@dataclass
class ConversationTurn:
    """A single message in a conversation.

    Attributes:
        role: "user" or "assistant".
        content: The message text.
    """

    role: str
    content: str
