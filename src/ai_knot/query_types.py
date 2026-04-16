"""Core data types for the contract-first answer engine (v2 query planes).

Track A — single-agent dataclasses only.
``PublishManifest`` lives in ``ai_knot.multi_agent.manifest_types`` (Track B).

ClaimKind deterministic core: STATE / RELATION / EVENT / DURATION / TRANSITION.
DESCRIPTOR / INTENT are defined here for table compatibility but produced *only*
by the optional LLM enrichment pass (``ai_knot.enrichment``), never by the
deterministic materializer (``ai_knot.materialization``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ClaimKind(StrEnum):
    """Semantic kind of an AtomicClaim.

    Deterministic core (produced by ``materialization.materialize_episode``):
      STATE      — entity has a property at a point in time.
      RELATION   — entity-to-entity relationship.
      EVENT      — something happened (has implicit or explicit event_time).
      DURATION   — something lasted for a period.
      TRANSITION — entity changed from one state/location/role to another.

    Enrichment-only (produced only by ``enrichment.enrich_claims_with_llm``):
      DESCRIPTOR — descriptive or characterizing claim (no strict triple form).
      INTENT     — forward-looking intention or plan.
    """

    # ---- deterministic core (5) ------------------------------------------------
    STATE = "state"
    RELATION = "relation"
    EVENT = "event"
    DURATION = "duration"
    TRANSITION = "transition"
    # ---- enrichment-only (2) ---------------------------------------------------
    DESCRIPTOR = "descriptor"
    INTENT = "intent"


#: The five kinds that the deterministic materializer is allowed to emit.
DETERMINISTIC_CLAIM_KINDS: frozenset[ClaimKind] = frozenset(
    {
        ClaimKind.STATE,
        ClaimKind.RELATION,
        ClaimKind.EVENT,
        ClaimKind.DURATION,
        ClaimKind.TRANSITION,
    }
)


class BundleKind(StrEnum):
    """Coarse retrieval unit type."""

    ENTITY_TOPIC = "entity_topic"
    STATE_TIMELINE = "state_timeline"
    EVENT_NEIGHBORHOOD = "event_neighborhood"
    RELATION_SUPPORT = "relation_support"


class AnswerSpace(StrEnum):
    """Shape of the expected answer."""

    BOOL = "bool"
    ENTITY = "entity"
    SET = "set"
    SCALAR = "scalar"
    DESCRIPTION = "description"


class TruthMode(StrEnum):
    """How truth is established for this answer."""

    DIRECT = "direct"
    RECONSTRUCT = "reconstruct"
    RANKED = "ranked"
    HYPOTHESIS = "hypothesis"
    NARRATIVE = "narrative"


class TimeAxis(StrEnum):
    """Temporal dimension the answer operates on."""

    EVENT = "event"
    CURRENT = "current"
    INTERVAL = "interval"
    NONE = "none"


class EvidenceRegime(StrEnum):
    """How evidence is collected and compared."""

    SINGLE = "single"
    AGGREGATE = "aggregate"
    SUPPORT_VS_CONTRA = "support_vs_contra"


# ---------------------------------------------------------------------------
# Raw episode (source of truth)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RawEpisode:
    """A single immutable conversation turn — source of truth for all derived planes.

    IDs are deterministic: ``sha256(agent_id + session_id + turn_id)[:16]``.
    """

    id: str
    agent_id: str
    session_id: str
    turn_id: str
    speaker: str
    observed_at: datetime
    raw_text: str
    session_date: datetime | None = None
    source_meta: dict[str, Any] = field(default_factory=dict)
    parent_episode_id: str | None = None


def make_episode_id(agent_id: str, session_id: str, turn_id: str) -> str:
    """Deterministic episode ID from (agent, session, turn)."""
    import hashlib

    key = f"{agent_id}|{session_id}|{turn_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Atomic claim
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AtomicClaim:
    """A single normalized claim derived from one RawEpisode.

    ``slot_key = f"{subject}::{relation}"`` when both are non-empty; else "".
    ``source_spans`` — char spans ``(start, end)`` in the original raw_text.
    ``materialization_version`` — version of the materializer that produced this claim.
    """

    id: str
    agent_id: str
    kind: ClaimKind
    subject: str
    relation: str
    value_text: str
    value_tokens: tuple[str, ...]
    qualifiers: dict[str, str]
    polarity: str  # "support" | "contra" | "neutral"
    event_time: datetime | None
    observed_at: datetime
    valid_from: datetime
    valid_until: datetime | None
    confidence: float
    salience: float
    source_episode_id: str
    source_spans: tuple[tuple[int, int], ...]
    materialization_version: int
    materialized_at: datetime
    slot_key: str  # "{subject}::{relation}" or ""
    version: int  # MESI compat
    origin_agent_id: str


def make_claim_id() -> str:
    """Random 16-char hex claim ID."""
    return uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Support bundle
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SupportBundle:
    """Coarse retrieval unit grouping related AtomicClaims for a topic.

    ``member_claim_ids`` — IDs of constituent AtomicClaims (stored separately
    in bundle_members table for efficient membership queries).
    ``bundle_score`` — aggregate salience * confidence; used for ranking,
    never as truth.
    """

    id: str
    agent_id: str
    kind: BundleKind
    topic: str  # normalised entity or entity::relation string
    member_claim_ids: tuple[str, ...]
    score_formula: str
    bundle_score: float
    built_from_materialization_version: int
    built_at: datetime


def make_bundle_id() -> str:
    return uuid4().hex[:16]


def stable_bundle_id(kind: BundleKind, topic: str) -> str:
    """Deterministic id for a persisted bundle keyed by (kind, topic).

    Schema PK is (agent_id, id); hashing (kind, topic) is sufficient to
    prevent cross-topic collisions because topics are agent-scoped.
    """
    import hashlib

    kind_val = kind.value if hasattr(kind, "value") else str(kind)
    return hashlib.sha1(f"{kind_val}:{topic}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dirty invalidation key (three-level granularity)
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DirtyKey:
    """Three-level cache invalidation key for support bundles.

    Resolution rules (finest to coarsest):
      SUBJECT_RELATION  subject + relation → invalidate STATE_TIMELINE / RELATION_SUPPORT
                        bundles whose topic == "{subject}::{relation}".
      SUBJECT           subject only       → invalidate ENTITY_TOPIC / EVENT_NEIGHBORHOOD
                        bundles whose topic == subject.
      BUNDLE_KIND_TOPIC bundle_kind + topic → invalidate a specific bundle directly.

    When ``subject`` and ``relation`` are both set → SUBJECT_RELATION level.
    When only ``subject`` is set                   → SUBJECT level.
    When ``bundle_kind`` and ``topic`` are set     → BUNDLE_KIND_TOPIC level.
    """

    subject: str | None = None
    relation: str | None = None
    bundle_kind: BundleKind | None = None
    topic: str | None = None

    @classmethod
    def for_slot(cls, subject: str, relation: str) -> DirtyKey:
        """Finest-grain key: invalidates only state/relation bundles for this slot."""
        return cls(subject=subject, relation=relation)

    @classmethod
    def for_subject(cls, subject: str) -> DirtyKey:
        """Middle-grain key: invalidates entity-topic and event-neighborhood bundles."""
        return cls(subject=subject)

    @classmethod
    def for_bundle(cls, kind: BundleKind, topic: str) -> DirtyKey:
        """Coarsest key: invalidates a specific bundle directly."""
        return cls(bundle_kind=kind, topic=topic)


# ---------------------------------------------------------------------------
# Query frame and contract
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class QueryFrame:
    """Geometric representation of a question (not tied to surface words)."""

    focus_entities: tuple[str, ...]
    target_kind: str  # state|relation|location|identity|event|set|scalar|description
    answer_space: AnswerSpace
    temporal_scope: str  # current|historical|interval|none
    epistemic_mode: TruthMode
    locality: str  # point|entity_scope|event_neighborhood|cross_entity
    evidence_regime: EvidenceRegime
    focus_relation: str | None


@dataclass(slots=True, frozen=True)
class AnswerContract:
    """Specification derived from QueryFrame that operators must satisfy."""

    answer_space: AnswerSpace
    truth_mode: TruthMode
    time_axis: TimeAxis
    locality: str
    evidence_regime: EvidenceRegime
    uncertainty_threshold: float = 0.25


# ---------------------------------------------------------------------------
# Evidence profile
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class EvidenceProfile:
    """Summary of the evidence landscape for a query."""

    n_support: int
    n_contra: int
    n_ambiguous: int
    density_per_entity: float
    temporal_span: tuple[datetime, datetime] | None
    coverage_ratio: float  # claims expanded / bundles retrieved
    has_explicit_event_time: bool
    # Retrieval quality signals (with defaults for backward compatibility)
    slot_bundle_hits: int = 0  # bundles with "entity::relation" topic
    explicit_time_hits: int = 0  # claims with qualifiers['date_token']
    fallback_used: bool = False  # BM25 fallback was needed
    episode_fallback_used: bool = False  # raw-episode search fallback was needed
    question_tokens: tuple[str, ...] = ()  # tokenized question for relevance scoring
    focus_entities: tuple[str, ...] = ()  # from QueryFrame
    focus_relation: str | None = None  # from QueryFrame
    has_temporal_anchor: bool = False  # any claim has qualifiers["time_anchor"]=="session_date"


# ---------------------------------------------------------------------------
# Answer items and trace
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AnswerItem:
    """A single answer value with provenance."""

    value: str
    confidence: float
    source_claim_ids: tuple[str, ...]
    source_episode_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class AnswerTrace:
    """Full audit trail for one query execution."""

    question: str
    frame: QueryFrame
    contract: AnswerContract
    retrieved_bundle_ids: tuple[str, ...]
    expanded_claim_ids: tuple[str, ...]
    evidence_profile: EvidenceProfile
    strategy: str  # operator name chosen
    decision_notes: tuple[str, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "frame": {
                "focus_entities": list(self.frame.focus_entities),
                "target_kind": self.frame.target_kind,
                "answer_space": self.frame.answer_space,
                "temporal_scope": self.frame.temporal_scope,
                "epistemic_mode": self.frame.epistemic_mode,
                "locality": self.frame.locality,
                "evidence_regime": self.frame.evidence_regime,
                "focus_relation": self.frame.focus_relation,
            },
            "contract": {
                "answer_space": self.contract.answer_space,
                "truth_mode": self.contract.truth_mode,
                "time_axis": self.contract.time_axis,
                "locality": self.contract.locality,
                "evidence_regime": self.contract.evidence_regime,
                "uncertainty_threshold": self.contract.uncertainty_threshold,
            },
            "retrieved_bundle_ids": list(self.retrieved_bundle_ids),
            "expanded_claim_ids": list(self.expanded_claim_ids),
            "evidence_profile": {
                "n_support": self.evidence_profile.n_support,
                "n_contra": self.evidence_profile.n_contra,
                "n_ambiguous": self.evidence_profile.n_ambiguous,
                "density_per_entity": self.evidence_profile.density_per_entity,
                "temporal_span": (
                    [
                        self.evidence_profile.temporal_span[0].isoformat(),
                        self.evidence_profile.temporal_span[1].isoformat(),
                    ]
                    if self.evidence_profile.temporal_span
                    else None
                ),
                "coverage_ratio": self.evidence_profile.coverage_ratio,
                "has_explicit_event_time": self.evidence_profile.has_explicit_event_time,
                "slot_bundle_hits": self.evidence_profile.slot_bundle_hits,
                "explicit_time_hits": self.evidence_profile.explicit_time_hits,
                "fallback_used": self.evidence_profile.fallback_used,
                "episode_fallback_used": self.evidence_profile.episode_fallback_used,
                "question_tokens": list(self.evidence_profile.question_tokens),
                "focus_entities": list(self.evidence_profile.focus_entities),
                "focus_relation": self.evidence_profile.focus_relation,
                "has_temporal_anchor": self.evidence_profile.has_temporal_anchor,
            },
            "strategy": self.strategy,
            "decision_notes": list(self.decision_notes),
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Final answer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QueryAnswer:
    """Structured answer returned by ``KnowledgeBase.query()``."""

    text: str
    items: tuple[AnswerItem, ...]
    confidence: float
    trace: AnswerTrace
    evidence_text: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_text": self.evidence_text,
            "confidence": self.confidence,
            "items": [
                {
                    "value": item.value,
                    "confidence": item.confidence,
                    "source_claim_ids": list(item.source_claim_ids),
                    "source_episode_ids": list(item.source_episode_ids),
                }
                for item in self.items
            ],
            "trace": self.trace.to_dict(),
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Rebuild report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RebuildReport:
    """Result of ``KnowledgeBase.rebuild_materialized()``."""

    skipped: bool = False
    n_episodes: int = 0
    n_claims: int = 0
    n_bundles_cleared: int = 0
    materialization_version: int = 0
    duration_s: float = 0.0
    error: str | None = None
