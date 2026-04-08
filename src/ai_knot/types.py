"""Core data types for ai_knot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
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


class MemoryOp(StrEnum):
    """LLM-signalled intent for a newly extracted fact (v1.3).

    ADD    — insert as a new fact (default).
    UPDATE — conversation explicitly corrects an existing value; behaves like a
             slot supersede even when structural resolution would reinforce.
    DELETE — conversation explicitly removes knowledge (e.g. "I no longer work
             at Acme"); close the matched slot without inserting a replacement.
    NOOP   — conversation merely confirms existing known information; skip
             entirely (no insert, no mutation).
    """

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


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
    # Lightweight claim fingerprint for conflict detection in the shared pool (v1.4).
    # Populated at publish time for unslotted facts; empty string = no claim extracted.
    claim_key: str = ""
    # Memory tier (v1.5): internal semantic lever for multi-agent promotion.
    # "private" — visible only to owning agent (default).
    # "pool" — promoted to shared pool via publish().
    # "org" — promoted to organization-wide knowledge (future).
    # Not exposed in public recall()/learn() signatures.
    memory_tier: str = "private"
    # Extraction intent (v1.3): set by Extractor._parse_fact(); never persisted.
    # ADD = insert (default); UPDATE = force supersede; DELETE = close without insert;
    # NOOP = skip entirely. Storage backends do not serialize this field.
    op: MemoryOp = MemoryOp.ADD

    def is_active(self, at: datetime | None = None) -> bool:
        """Return True if this fact is valid at *at* (default: now UTC).

        A fact is active when ``valid_from <= at < valid_until`` (open upper bound).
        Future-dated facts (``valid_from > at``) are not yet active.
        """
        t = at or datetime.now(UTC)
        if self.valid_from > t:
            return False
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


# ---------------------------------------------------------------------------
# Provenance (Phase 2): lineage tracking for facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable provenance record for a fact.

    Tracks the origin and transformation history of a fact through the
    learn/publish/promote pipeline.

    Attributes:
        origin_agent: Agent that first created the fact.
        origin_turn: Conversation turn index that produced it (0-based), or -1.
        published_by: Agent that published the fact to the pool (empty if private).
        promoted_by: Agent that promoted the fact to a higher tier (empty if not promoted).
        supersedes_id: ID of the fact this one replaced via CAS (empty if no predecessor).
        consolidation_ids: IDs of episodic facts consolidated into this one (empty list if none).
    """

    origin_agent: str = ""
    origin_turn: int = -1
    published_by: str = ""
    promoted_by: str = ""
    supersedes_id: str = ""
    consolidation_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# ConflictPolicy protocol (Phase 2): per-type conflict resolution strategy
# ---------------------------------------------------------------------------


@runtime_checkable
class ConflictPolicy(Protocol):
    """Strategy for resolving conflicts between new and existing facts.

    Each MemoryType can have a different policy governing how conflicts
    are detected and resolved during the learn() resolve phase.
    """

    @staticmethod
    def should_supersede(new_fact: Fact, existing: Fact) -> bool:
        """Return True if *new_fact* should supersede *existing*."""
        ...

    @staticmethod
    def decay_immune(fact: Fact) -> bool:
        """Return True if *fact* is immune to retention decay."""
        ...

    @staticmethod
    def ttl_seconds(fact: Fact) -> float | None:
        """Return a TTL in seconds for *fact*, or None for no expiry."""
        ...


class SlotStateMachinePolicy:
    """Conflict policy for SEMANTIC facts — existing slot CAS logic.

    Semantic facts use deterministic slot_key-based CAS: same slot + same value
    reinforces confidence; same slot + different value supersedes; no slot = branch.
    All semantic facts participate in retention decay normally.
    """

    @staticmethod
    def should_supersede(new_fact: Fact, existing: Fact) -> bool:
        if not new_fact.slot_key or new_fact.slot_key != existing.slot_key:
            return False
        return new_fact.value_text != existing.value_text

    @staticmethod
    def decay_immune(fact: Fact) -> bool:
        return False

    @staticmethod
    def ttl_seconds(fact: Fact) -> float | None:
        return None


class ProcedureStabilityPolicy:
    """Conflict policy for PROCEDURAL facts — stability-first.

    Procedural facts represent user preferences and workflows. They are
    immune to retention decay (pinned) and require explicit supersession
    or DELETE to be removed. This prevents learned procedures from being
    forgotten due to disuse.
    """

    @staticmethod
    def should_supersede(new_fact: Fact, existing: Fact) -> bool:
        if not new_fact.slot_key or new_fact.slot_key != existing.slot_key:
            return False
        return new_fact.value_text != existing.value_text

    @staticmethod
    def decay_immune(fact: Fact) -> bool:
        return True

    @staticmethod
    def ttl_seconds(fact: Fact) -> float | None:
        return None


class EpisodicTimelinePolicy:
    """Conflict policy for EPISODIC facts — TTL + consolidation path.

    Episodic facts represent specific events. They have a default TTL of 7 days
    and are never superseded by slot CAS — instead, multiple episodes with the
    same slot coexist on a timeline. High-importance episodes (>= 0.9) are
    exempt from TTL.
    """

    _DEFAULT_TTL: float = 7 * 24 * 3600.0  # 7 days

    @staticmethod
    def should_supersede(new_fact: Fact, existing: Fact) -> bool:
        # Episodic facts don't supersede — they coexist on a timeline.
        return False

    @staticmethod
    def decay_immune(fact: Fact) -> bool:
        return False

    @staticmethod
    def ttl_seconds(fact: Fact) -> float | None:
        if fact.importance >= 0.9:
            return None  # High-importance episodes are kept indefinitely.
        return EpisodicTimelinePolicy._DEFAULT_TTL


# Registry mapping MemoryType to its conflict policy.
CONFLICT_POLICIES: dict[MemoryType, ConflictPolicy] = {
    MemoryType.SEMANTIC: SlotStateMachinePolicy(),
    MemoryType.PROCEDURAL: ProcedureStabilityPolicy(),
    MemoryType.EPISODIC: EpisodicTimelinePolicy(),
}


# ---------------------------------------------------------------------------
# Evidence storage protocol (Phase 5): physical separation seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """Extracted evidence record linked to a Fact by ``fact_id``.

    Holds the raw source material that supports a canonical fact.
    When evidence is stored inline (default), these fields live directly
    on the Fact.  When physically separated, they live in an external store.

    Attributes:
        fact_id: ID of the canonical Fact this evidence supports.
        snippets: Raw source text snippets.
        spans: Character spans in the original document.
        verbatim: Verbatim source quote.
        witness_surface: Surface form as observed by the witness agent.
        support_confidence: How strongly this evidence supports the fact (0.0-1.0).
        verification_source: How the evidence was verified ("manual", "llm", etc.).
    """

    fact_id: str
    snippets: list[str] = field(default_factory=list)
    spans: list[str] = field(default_factory=list)
    verbatim: str = ""
    witness_surface: str = ""
    support_confidence: float = 1.0
    verification_source: str = "manual"


@runtime_checkable
class EvidenceStore(Protocol):
    """Protocol for evidence storage backends (Phase 5).

    Default implementation reads evidence inline from the Fact fields.
    Future implementations may use a separate table/file for evidence,
    enabling independent scaling and TTL for raw source material.
    """

    def get_evidence(self, fact_id: str) -> Evidence | None:
        """Retrieve evidence for a single fact."""
        ...

    def save_evidence(self, evidence: list[Evidence]) -> None:
        """Persist a batch of evidence records."""
        ...

    def delete_evidence(self, fact_ids: list[str]) -> None:
        """Remove evidence for the given fact IDs."""
        ...


class InlineEvidenceStore:
    """Default evidence store — reads evidence inline from Fact fields.

    This is a zero-cost adapter: it extracts evidence from the Fact's
    own fields (source_snippets, source_verbatim, etc.) without any
    separate storage.  This is the Phase 5 "no-op" implementation that
    preserves current behaviour while providing the protocol seam.
    """

    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def set_facts(self, facts: list[Fact]) -> None:
        """Load facts for inline evidence extraction."""
        self._facts = {f.id: f for f in facts}

    def get_evidence(self, fact_id: str) -> Evidence | None:
        """Extract evidence from inline Fact fields."""
        fact = self._facts.get(fact_id)
        if fact is None:
            return None
        return Evidence(
            fact_id=fact.id,
            snippets=fact.source_snippets,
            spans=fact.source_spans,
            verbatim=fact.source_verbatim,
            witness_surface=fact.witness_surface,
            support_confidence=fact.support_confidence,
            verification_source=fact.verification_source,
        )

    def save_evidence(self, evidence: list[Evidence]) -> None:
        """No-op for inline store — evidence is saved with the Fact."""

    def delete_evidence(self, fact_ids: list[str]) -> None:
        """No-op for inline store — evidence is deleted with the Fact."""
