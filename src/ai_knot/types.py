"""Core data types for ai_knot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4


class MESIState(StrEnum):
    """MESI cache coherence state for shared pool facts (v1.0).

    EXCLUSIVE  — single private owner, no coordination needed.
    SHARED     — new fact published to pool (no prior for this entity+attribute).
    MODIFIED   — replaced an existing active fact via entity-addressed CAS.
    INVALID    — fact has been superseded; valid_until is set.
    """

    EXCLUSIVE = "E"
    SHARED = "S"
    MODIFIED = "M"
    INVALID = "I"


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
        canonical_surface: Normalised form for BM25 indexing and deduplication (v1.1).
        witness_surface: Verbatim quote from the source that grounds this fact (v1.1).
        prompt_surface: Compact form injected into LLM prompts to conserve tokens (v1.1).
        slot_key: Deterministic slot address ``"{entity}::{attribute}"`` (v1.1).
        value_text: The extracted value for this slot, e.g. ``"95000"`` (v1.1).
        qualifiers: Temporal or conditional modifiers, e.g. ``{"since": "2024-01"}`` (v1.1).
        state_confidence: Confidence that this fact reflects the current state (v1.1).
        topic_channel: Domain label for shared pool routing, e.g. ``"devops"`` (v1.2).
            Empty string = no topic filter (visible in all channels).
        visibility_scope: How widely the fact propagates — ``"global"`` (all agents)
            or ``"local"`` (only the owning agent, v1.2).
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
    # Verbatim source preservation (v0.8): exact original phrase before LLM normalisation.
    # When non-empty, recall() surfaces this instead of the normalised content.
    source_verbatim: str = ""
    # Temporal validity (v0.9): mono-temporal model (Allen's Interval Logic).
    # valid_from: when this version of the fact became true (insertion time by default).
    # valid_until: when this fact was superseded; None = currently valid.
    valid_from: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    # Structured addressing (v0.9): entity+attribute for deterministic dedup.
    # Empty strings = general statement (falls back to cosine-based dedup).
    entity: str = ""  # "Alex Chen"
    attribute: str = ""  # "salary"
    # MESI cache coherence (v1.0): for shared pool invalidation.
    # version: monotonic counter incremented on every update.
    # mesi_state: E=Exclusive, S=Shared, M=Modified, I=Invalid.
    version: int = 0
    mesi_state: MESIState = MESIState.EXCLUSIVE
    canonical_surface: str = ""
    witness_surface: str = ""
    prompt_surface: str = ""
    # slot_key: always "{entity}::{attribute}" when both are non-empty, else "".
    # Stored explicitly so storage queries can filter/group by slot without recomputing.
    slot_key: str = ""
    value_text: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    state_confidence: float = 1.0
    topic_channel: str = ""
    visibility_scope: str = "global"

    def is_active(self, at: datetime | None = None) -> bool:
        """Return True if this fact is valid at *at* (default: now UTC)."""
        t = at or datetime.now(UTC)
        return self.valid_until is None or self.valid_until > t


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
class SlotDelta:
    """Lightweight record of a slot state change in the shared pool (v1.1).

    Used by ``SharedMemoryPool.sync_slot_deltas()`` to transmit only the
    minimal information needed for a receiving agent to update its view —
    far cheaper than copying full ``Fact`` objects across agents.

    Attributes:
        slot_key: Deterministic slot address ``"{entity}::{attribute}"``,
            or ``""`` for unslotted facts.
        version: Monotonic version counter at the time of the change.
        op: Type of change — ``"new"`` (first publish), ``"supersede"``
            (replaces a prior value), or ``"invalidate"`` (fact closed).
        fact_id: ID of the new active fact (or the closed fact for "invalidate").
        content: Full fact content — used when the receiver needs the text.
        prompt_surface: Compact surface for prompt injection; may be empty.
    """

    slot_key: str
    version: int
    op: str  # "new" | "supersede" | "invalidate"
    fact_id: str
    content: str
    prompt_surface: str = ""


@dataclass
class ConversationTurn:
    """A single message in a conversation.

    Attributes:
        role: "user" or "assistant".
        content: The message text.
    """

    role: str
    content: str
