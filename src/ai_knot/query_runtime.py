"""Query runtime orchestrator — contract-first answer engine.

Design invariants:
  * No direct storage access; all retrieval goes through support_retrieval.
  * No business logic; operators and retrieval are delegated.
  * No kb object passed in; receives agent_id + storage + optional renderer.

Enforced by: scripts/check_query_runtime_isolation.py
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import ai_knot.support_retrieval as _sr
from ai_knot.query_contract import analyze_query, derive_answer_contract
from ai_knot.query_operators import OPERATORS, choose_strategy
from ai_knot.query_types import (
    AnswerContract,
    AnswerItem,
    AnswerTrace,
    AtomicClaim,
    EvidenceProfile,
    QueryAnswer,
    QueryFrame,
    SupportBundle,
    TimeAxis,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute_query(
    storage: object,
    agent_id: str,
    question: str,
    *,
    top_k: int = 60,
    now: datetime | None = None,
    renderer: Callable[[str], str] | None = None,
) -> QueryAnswer:
    """Run the full contract-first query pipeline.

    Args:
        storage:   Backend that implements RawEpisodeStore / ClaimStore / BundleStore.
        agent_id:  Agent namespace.
        question:  Natural-language question.
        top_k:     Maximum bundles to retrieve.
        now:       Override current time (for testing).
        renderer:  Optional callable for narrative rendering (LLM or stub).

    Returns:
        A fully populated QueryAnswer with trace.
    """
    t0 = time.monotonic()
    now = now or datetime.now(UTC)

    # 1. Analyze query geometry.
    frame = analyze_query(question)
    contract = derive_answer_contract(frame)

    # 2. Drain pending dirty keys before retrieval.
    _drain_dirty_keys(storage, agent_id)

    # 3. Retrieve support bundles.
    topics = _sr.topics_for_entities(
        frame.focus_entities, contract, focus_relation=frame.focus_relation
    )
    kinds = _sr.bundle_kinds_for_contract(contract, focus_relation=frame.focus_relation)
    bundles, fallback_used = _sr.retrieve_bundles(
        storage,
        agent_id,
        topics=topics,
        kinds=kinds,
        question=question,
        top_k=top_k,
    )

    # 4. Expand claims from bundles.
    active_only = contract.time_axis in (TimeAxis.CURRENT, TimeAxis.NONE)
    claims = _sr.expand_claims(storage, agent_id, bundles, active_only=active_only)

    # 5. Build evidence profile.
    profile = _build_evidence_profile(claims, bundles, contract, frame, question, fallback_used)

    # 6. Choose strategy.
    strategy = choose_strategy(frame, contract, profile)

    # 7. Execute operator.
    operator_fn = OPERATORS[strategy]
    answer_items, confidence, decision_notes = operator_fn(
        claims, bundles, contract, profile, now, renderer
    )

    # 8. Render text.
    text = _render_text(answer_items, contract)

    # 9. Build trace.
    latency_ms = (time.monotonic() - t0) * 1000.0
    trace = AnswerTrace(
        question=question,
        frame=frame,
        contract=contract,
        retrieved_bundle_ids=tuple(b.id for b in bundles),
        expanded_claim_ids=tuple(c.id for c in claims),
        evidence_profile=profile,
        strategy=strategy,
        decision_notes=tuple(decision_notes),
        latency_ms=latency_ms,
    )

    return QueryAnswer(
        text=text,
        items=tuple(answer_items),
        confidence=confidence,
        trace=trace,
    )


# ---------------------------------------------------------------------------
# Evidence profile builder
# ---------------------------------------------------------------------------


def _build_evidence_profile(
    claims: list[AtomicClaim],
    bundles: list[SupportBundle],
    contract: AnswerContract,
    frame: QueryFrame,
    question: str = "",
    fallback_used: bool = False,
) -> EvidenceProfile:
    """Summarize the evidence landscape for the query."""
    from ai_knot.tokenizer import tokenize as _tokenize

    n_support = sum(1 for c in claims if c.polarity == "support")
    n_contra = sum(1 for c in claims if c.polarity == "contra")
    n_ambiguous = len(claims) - n_support - n_contra

    # Density: support per focus entity.
    n_entities = max(1, len(frame.focus_entities))
    density = n_support / n_entities

    # Temporal span.
    times = sorted(t for c in claims for t in [c.event_time, c.valid_from] if t is not None)
    temporal_span = (times[0], times[-1]) if len(times) >= 2 else None

    # Coverage ratio.
    n_bundle_members = sum(len(b.member_claim_ids) for b in bundles)
    coverage = len(claims) / max(1, n_bundle_members)

    has_explicit_event_time = any(
        c.event_time is not None and c.qualifiers.get("date_token") for c in claims
    )

    # Slot bundle hits: bundles whose topic is "entity::relation" form.
    slot_bundle_hits = sum(1 for b in bundles if "::" in b.topic)

    # Explicit time hits: claims with a date_token qualifier.
    explicit_time_hits = sum(
        1 for c in claims if c.qualifiers.get("date_token") and c.event_time is not None
    )

    # Question tokens for relevance scoring in operators.
    question_tokens = tuple(_tokenize(question)) if question else ()

    return EvidenceProfile(
        n_support=n_support,
        n_contra=n_contra,
        n_ambiguous=n_ambiguous,
        density_per_entity=density,
        temporal_span=temporal_span,
        coverage_ratio=coverage,
        has_explicit_event_time=has_explicit_event_time,
        slot_bundle_hits=slot_bundle_hits,
        explicit_time_hits=explicit_time_hits,
        fallback_used=fallback_used,
        question_tokens=question_tokens,
        focus_entities=frame.focus_entities,
        focus_relation=frame.focus_relation,
    )


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


def _render_text(items: list[AnswerItem], contract: AnswerContract) -> str:
    """Convert AnswerItems to a single text string."""
    if not items:
        return "No answer found."

    from ai_knot.query_types import AnswerSpace

    if contract.answer_space is AnswerSpace.SET:
        values = [i.value for i in items]
        if len(values) == 1:
            return values[0]
        last = values[-1]
        rest = ", ".join(values[:-1])
        return f"{rest} and {last}"

    if contract.answer_space is AnswerSpace.BOOL:
        top = items[0]
        return top.value  # "yes" / "no" / "uncertain"

    # Default: join all values.
    if len(items) == 1:
        return items[0].value
    return "; ".join(i.value for i in items[:5])


# ---------------------------------------------------------------------------
# Dirty key drainage
# ---------------------------------------------------------------------------


def _drain_dirty_keys(storage: object, agent_id: str) -> None:
    """Read pending dirty_keys_json from meta and invalidate matching bundles."""
    if not hasattr(storage, "load_materialization_meta"):
        return
    meta = storage.load_materialization_meta(agent_id)
    dirty_json = meta.get("dirty_keys_json", "[]")
    if dirty_json in ("[]", "", "null", None):
        return

    invalidated = _sr.apply_pending_dirty_keys(storage, agent_id, dirty_json)

    if invalidated > 0 and hasattr(storage, "save_materialization_meta"):
        # Clear dirty keys after draining.
        storage.save_materialization_meta(
            agent_id,
            schema_version=meta.get("schema_version", 2),
            materialization_version=meta.get("materialization_version", 0),
            last_rebuild_at=None,
            dirty_keys_json="[]",
            rebuild_status=meta.get("rebuild_status", "ready"),
        )
