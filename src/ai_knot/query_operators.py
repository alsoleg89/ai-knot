"""Deterministic query operators — pure functions, no storage access.

Each operator receives:
    claims:  list[AtomicClaim]
    bundles: list[SupportBundle]
    contract: AnswerContract
    profile:  EvidenceProfile
    now:      datetime

Returns:
    (items: list[AnswerItem], confidence: float, decision_notes: list[str])

Operators never call LLM or access storage.
The ``narrative_cluster_render`` operator accepts an optional renderer callable
so that KnowledgeBase.query() can inject an LLM renderer without breaking
operator purity.

Enforced by: scripts/check_query_runtime_isolation.py
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from datetime import UTC, datetime

from ai_knot.query_types import (
    AnswerContract,
    AnswerItem,
    AnswerSpace,
    AtomicClaim,
    ClaimKind,
    EvidenceProfile,
    EvidenceRegime,
    SupportBundle,
    TimeAxis,
)

# ---------------------------------------------------------------------------
# Type alias for operator return
# ---------------------------------------------------------------------------

OperatorResult = tuple[list[AnswerItem], float, list[str]]

# ---------------------------------------------------------------------------
# Public operator registry
# ---------------------------------------------------------------------------

# Populated at bottom of file.
OPERATORS: dict[str, Callable[..., OperatorResult]] = {}


# ---------------------------------------------------------------------------
# Strategy chooser (also used by query_runtime)
# ---------------------------------------------------------------------------


def choose_strategy(
    frame: QueryFrame,  # type: ignore[name-defined]  # noqa: F821
    contract: AnswerContract,
    profile: EvidenceProfile,
) -> str:
    """Select the operator name given geometric query shape and evidence profile.

    Rules (in order of priority):
    1. SET → set_collect (always, regardless of surface form)
    2. TIME axis → time_resolve
    3. Direct single strong evidence, no contra, and slot-level retrieval → exact_state
    4. BOOL with distributed / conflicting evidence → bounded_hypothesis_test
    5. Multiple support candidates → candidate_rank
    6. Fallback → narrative_cluster_render
    """

    # Rule 1: aggregation — always set_collect.
    if contract.answer_space is AnswerSpace.SET:
        return "set_collect"

    # Rule 2: temporal question — time_resolve.
    if contract.time_axis in (TimeAxis.EVENT, TimeAxis.INTERVAL):
        return "time_resolve"

    # Fallback-only with no slot bundle on a description question: candidate_rank
    # or narrative (don't exact_state on generic unrelated claims).
    if (
        profile.fallback_used
        and profile.slot_bundle_hits == 0
        and contract.answer_space is AnswerSpace.DESCRIPTION
    ):
        if profile.n_support >= 1:
            return "candidate_rank"
        return "narrative_cluster_render"

    # Rule 3: single strong direct evidence without contradiction.
    # Require a slot-level bundle hit OR a non-DESCRIPTION answer space to
    # avoid routing generic description questions with arbitrary claims to
    # exact_state.
    if (
        profile.n_support >= 1
        and profile.n_contra == 0
        and contract.evidence_regime is EvidenceRegime.SINGLE
        and (
            profile.slot_bundle_hits >= 1
            or contract.answer_space in (AnswerSpace.BOOL, AnswerSpace.SCALAR, AnswerSpace.ENTITY)
        )
    ):
        return "exact_state"

    # Rule 4: BOOL with distributed / conflicting support.
    if frame.answer_space is AnswerSpace.BOOL:
        return "bounded_hypothesis_test"

    # Rule 5: multiple candidates to rank.
    if profile.n_support >= 1:
        return "candidate_rank"

    # Rule 6: fallback narrative.
    return "narrative_cluster_render"


# ---------------------------------------------------------------------------
# Operator: exact_state
# ---------------------------------------------------------------------------


def exact_state(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Return the single best state/value for the queried slot.

    Composite score = slot_match + question_relevance + confidence*salience + recency.
    Slot_match: +1.0 when claim.slot_key == "entity::relation" from the query frame.
    Question_relevance: Jaccard overlap of claim value_tokens with question_tokens.
    """
    notes: list[str] = []

    state_claims = [
        c
        for c in claims
        if c.kind in (ClaimKind.STATE, ClaimKind.RELATION, ClaimKind.TRANSITION)
        and c.polarity == "support"
    ]
    if not state_claims:
        # Fall back to any active claim.
        state_claims = [c for c in claims if c.polarity == "support"]

    if not state_claims:
        return [], 0.0, ["no supporting claims found"]

    # Build expected slot keys from frame for slot_match scoring.
    slot_keys: set[str] = set()
    if profile.focus_relation:
        for ent in profile.focus_entities:
            slot_keys.add(f"{ent}::{profile.focus_relation}")

    q_tokens = set(profile.question_tokens)

    def _score(c: AtomicClaim) -> float:
        slot_match = 1.0 if (slot_keys and c.slot_key in slot_keys) else 0.0
        q_relevance = 0.0
        if q_tokens and c.value_tokens:
            v_tokens = set(c.value_tokens)
            union = q_tokens | v_tokens
            q_relevance = len(q_tokens & v_tokens) / len(union) if union else 0.0
        base = c.confidence * c.salience
        recency = _recency_bonus(c, now)
        return slot_match + q_relevance + base + recency

    best = max(state_claims, key=_score)
    notes.append(
        f"selected claim {best.id!r} (kind={best.kind}, conf={best.confidence:.2f}, "
        f"slot_match={best.slot_key in slot_keys})"
    )

    item = AnswerItem(
        value=best.value_text,
        confidence=best.confidence * best.salience,
        source_claim_ids=(best.id,),
        source_episode_ids=(best.source_episode_id,),
    )
    return [item], best.confidence * best.salience, notes


# ---------------------------------------------------------------------------
# Operator: set_collect
# ---------------------------------------------------------------------------


def set_collect(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Collect all distinct values for a set-type question.

    De-duplicates by (subject, relation, normalized_value_text).
    Does NOT collapse different values under the same slot_key — each
    distinct value is a separate AnswerItem.
    """
    notes: list[str] = []
    seen: dict[str, AnswerItem] = {}

    for c in claims:
        if c.polarity == "contra":
            continue
        norm = _normalize_value(c.value_text)
        dedup_key = f"{c.subject}|{c.relation}|{norm}"
        if dedup_key not in seen or c.confidence > seen[dedup_key].confidence:
            seen[dedup_key] = AnswerItem(
                value=c.value_text,
                confidence=c.confidence * c.salience,
                source_claim_ids=(c.id,),
                source_episode_ids=(c.source_episode_id,),
            )

    items = sorted(seen.values(), key=lambda x: x.confidence, reverse=True)
    notes.append(f"collected {len(items)} distinct values from {len(claims)} claims")

    if not items:
        return [], 0.0, notes
    confidence = sum(i.confidence for i in items) / len(items)
    return items, confidence, notes


# ---------------------------------------------------------------------------
# Operator: time_resolve
# ---------------------------------------------------------------------------


def time_resolve(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Resolve a temporal question by finding the most accurate event time.

    Preference order:
    1. event_time explicitly set on the claim (e.g., extracted date).
    2. valid_from for STATE/TRANSITION claims.
    3. observed_at as last resort.
    """
    notes: list[str] = []

    event_claims = sorted(
        (c for c in claims if c.polarity == "support"),
        key=lambda c: _temporal_sort_key(c, contract),
    )

    if not event_claims:
        return [], 0.0, ["no temporal claims found"]

    # For INTERVAL, return the full range.
    if contract.time_axis is TimeAxis.INTERVAL:
        earliest = _get_time(event_claims[0], contract)
        latest = _get_time(event_claims[-1], contract)
        if earliest is None or latest is None:
            val = (
                f"{event_claims[0].observed_at.date().isoformat()} – "
                f"{event_claims[-1].observed_at.date().isoformat()}"
            )
        else:
            val = f"{earliest.date().isoformat()} – {latest.date().isoformat()}"
        item = AnswerItem(
            value=val,
            confidence=0.8,
            source_claim_ids=tuple(c.id for c in event_claims[:5]),
            source_episode_ids=tuple(dict.fromkeys(c.source_episode_id for c in event_claims[:5])),
        )
        notes.append(f"resolved temporal interval from {len(event_claims)} claims")
        return [item], 0.8, notes

    # For EVENT: require at least one claim with an explicit event_time or
    # EVENT/TRANSITION/DURATION kind to avoid returning session dates.
    if contract.time_axis is TimeAxis.EVENT:
        explicit_event_claims = [
            c
            for c in event_claims
            if (c.qualifiers.get("date_token") and c.event_time is not None)
            or c.kind in (ClaimKind.EVENT, ClaimKind.TRANSITION, ClaimKind.DURATION)
        ]
        if not explicit_event_claims:
            notes.append("no explicit event time found — refusing to return session date")
            return [], 0.0, notes
        best = explicit_event_claims[-1]  # most recent event
    else:
        # CURRENT: most recent state.
        best = event_claims[0]

    t = _get_time(best, contract)
    notes.append(
        f"resolved time from claim {best.id!r} event_time={best.event_time} "
        f"valid_from={best.valid_from}"
    )
    item = AnswerItem(
        value=t.date().isoformat() if t else best.value_text,
        confidence=best.confidence * best.salience,
        source_claim_ids=(best.id,),
        source_episode_ids=(best.source_episode_id,),
    )
    return [item], best.confidence * best.salience, notes


def _temporal_sort_key(c: AtomicClaim, contract: AnswerContract) -> datetime:
    t = _get_time(c, contract)
    return t or datetime.min.replace(tzinfo=UTC)


def _get_time(c: AtomicClaim, contract: AnswerContract) -> datetime | None:
    """Return the most specific available time for a claim."""
    # Explicit event_time in qualifiers wins.
    if c.qualifiers.get("date_token") and c.event_time:
        return c.event_time
    return c.event_time or c.valid_from or c.observed_at


# ---------------------------------------------------------------------------
# Operator: candidate_rank
# ---------------------------------------------------------------------------


def candidate_rank(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Rank supporting claims by a composite score and return top candidates."""
    notes: list[str] = []

    support = [c for c in claims if c.polarity == "support"]
    contra_ids = {c.id for c in claims if c.polarity == "contra"}

    if not support:
        return [], 0.0, ["no supporting claims"]

    # Build slot keys from actually retrieved bundles, filtered by focus_entities.
    slot_keys: set[str] = set()
    focus_ents = set(profile.focus_entities) if profile.focus_entities else None
    for b in bundles:
        if "::" not in b.topic:
            continue
        entity_part = b.topic.split("::", 1)[0]
        if focus_ents is None or entity_part in focus_ents:
            slot_keys.add(b.topic)
    q_tokens = set(profile.question_tokens)

    # Score: slot_match + question_relevance + salience*confidence + recency - contra penalty.
    scored: list[tuple[float, AtomicClaim]] = []
    for c in support:
        slot_match = 1.0 if (slot_keys and c.slot_key in slot_keys) else 0.0
        q_relevance = 0.0
        if q_tokens and c.value_tokens:
            v_tokens = set(c.value_tokens)
            union = q_tokens | v_tokens
            q_relevance = len(q_tokens & v_tokens) / len(union) if union else 0.0
        base = c.salience * c.confidence
        recency = _recency_bonus(c, now)
        contra_pen = 0.1 if c.id in contra_ids else 0.0
        score = slot_match + q_relevance + base + recency - contra_pen
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:10]

    items = [
        AnswerItem(
            value=c.value_text,
            confidence=min(score, 1.0),
            source_claim_ids=(c.id,),
            source_episode_ids=(c.source_episode_id,),
        )
        for score, c in top
    ]
    notes.append(f"ranked {len(top)} candidates from {len(support)} support claims")
    return items, items[0].confidence if items else 0.0, notes


def _recency_bonus(c: AtomicClaim, now: datetime) -> float:
    ref = c.valid_from or c.observed_at
    if not ref:
        return 0.0
    age_days = (now - ref).total_seconds() / 86400.0
    return max(0.0, 0.05 * math.exp(-age_days / 90.0))  # half-life ~90 days


# ---------------------------------------------------------------------------
# Operator: bounded_hypothesis_test
# ---------------------------------------------------------------------------


def bounded_hypothesis_test(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Yes/no/uncertain answer using support-minus-contra scoring.

    Returns "yes", "no", or "uncertain" as the value.
    Sigmoid of |score| gives confidence.
    """
    notes: list[str] = []

    n_support = profile.n_support
    n_contra = profile.n_contra
    n_ambiguous = profile.n_ambiguous

    score = n_support - n_contra - 0.5 * n_ambiguous
    threshold = contract.uncertainty_threshold

    if abs(score) < threshold:
        verdict = "uncertain"
        conf = 1.0 - abs(score) / max(threshold, 1e-6)
    elif score > 0:
        verdict = "yes"
        conf = _sigmoid(score)
    else:
        verdict = "no"
        conf = _sigmoid(-score)

    # If we have focus entities/relation, check that at least one support/contra
    # claim is actually relevant to the subject being asked about.  Generic
    # claims unrelated to the query subject should yield "uncertain".
    if profile.focus_entities or profile.focus_relation:
        relevant_claims = [
            c
            for c in claims
            if c.polarity in ("support", "contra")
            and (not profile.focus_entities or any(c.subject == e for e in profile.focus_entities))
        ]
        if not relevant_claims:
            notes.append("no claims matched focus subject — returning uncertain")
            item = AnswerItem(
                value="uncertain",
                confidence=0.0,
                source_claim_ids=(),
                source_episode_ids=(),
            )
            return [item], 0.0, notes

    notes.append(
        f"hypothesis: support={n_support} contra={n_contra} ambig={n_ambiguous} "
        f"score={score:.2f} threshold={threshold:.2f} → {verdict}"
    )
    src_ids = tuple(c.id for c in claims[:5] if c.polarity in ("support", "contra"))
    ep_ids = tuple(dict.fromkeys(c.source_episode_id for c in claims[:5]))
    item = AnswerItem(
        value=verdict,
        confidence=min(conf, 1.0),
        source_claim_ids=src_ids,
        source_episode_ids=ep_ids,
    )
    return [item], min(conf, 1.0), notes


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Operator: narrative_cluster_render
# ---------------------------------------------------------------------------


def narrative_cluster_render(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    profile: EvidenceProfile,
    now: datetime,
    renderer: Callable[[str], str] | None = None,
) -> OperatorResult:
    """Group claims into a narrative text response.

    Deterministic fallback: joins value_texts sorted by event_time.
    Optional renderer callable (injected by KnowledgeBase.query) may call LLM.
    """
    notes: list[str] = []

    support = sorted(
        (c for c in claims if c.polarity == "support"),
        key=lambda c: c.event_time or c.valid_from or c.observed_at,
    )

    if not support:
        return [], 0.0, ["no claims for narrative"]

    # Deterministic text: join value_texts.
    fragments = [c.value_text for c in support[:20]]
    det_text = "; ".join(fragments)

    if renderer is not None:
        try:
            rendered = renderer(det_text)
        except Exception as exc:
            rendered = det_text
            notes.append(f"renderer failed ({exc}), using deterministic join")
    else:
        rendered = det_text

    notes.append(f"narrative from {len(support)} claims")
    src_ids = tuple(c.id for c in support[:10])
    ep_ids = tuple(dict.fromkeys(c.source_episode_id for c in support[:10]))
    item = AnswerItem(
        value=rendered,
        confidence=0.6,
        source_claim_ids=src_ids,
        source_episode_ids=ep_ids,
    )
    return [item], 0.6, notes


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------


def _normalize_value(text: str) -> str:
    """Lowercase and strip punctuation for de-duplication."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# ---------------------------------------------------------------------------
# Register operators
# ---------------------------------------------------------------------------

OPERATORS = {
    "exact_state": exact_state,
    "set_collect": set_collect,
    "time_resolve": time_resolve,
    "candidate_rank": candidate_rank,
    "bounded_hypothesis_test": bounded_hypothesis_test,
    "narrative_cluster_render": narrative_cluster_render,
}


# Late import to avoid circular dependency (query_contract imports from query_types only).
def _import_query_frame() -> type:
    from ai_knot.query_types import QueryFrame

    return QueryFrame
