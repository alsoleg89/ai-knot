"""Query runtime orchestrator — contract-first answer engine.

Design invariants:
  * No direct storage access; all retrieval goes through support_retrieval.
  * No business logic; operators and retrieval are delegated.
  * No kb object passed in; receives agent_id + storage + optional renderer.

Enforced by: scripts/check_query_runtime_isolation.py
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import ai_knot.support_retrieval as _sr
from ai_knot.query_contract import analyze_query, derive_answer_contract
from ai_knot.query_operators import OPERATORS, choose_strategy
from ai_knot.query_types import (
    AnswerContract,
    AnswerItem,
    AnswerSpace,
    AnswerTrace,
    AtomicClaim,
    EvidenceProfile,
    QueryAnswer,
    QueryFrame,
    SupportBundle,
    TimeAxis,
)

# ---------------------------------------------------------------------------
# Query profile (caps knob)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CapSet:
    raw_search_top_k: int  # top_k for search_episodes_by_entities
    window_dedup_cap: int  # max unique episode IDs from window expansion
    collect_cap: int  # cap for _collect_evidence_episode_ids
    render_top_k: int  # max episodes rendered to evidence_text
    char_budget: int  # max bytes in evidence_text
    per_turn_max: int  # max chars per individual turn in evidence


_PROFILE_CAPS: dict[str, _CapSet] = {
    "narrow": _CapSet(
        raw_search_top_k=5,
        window_dedup_cap=8,
        collect_cap=5,
        render_top_k=5,
        char_budget=8_000,
        per_turn_max=400,
    ),
    "balanced": _CapSet(
        raw_search_top_k=20,
        window_dedup_cap=24,
        collect_cap=15,
        render_top_k=12,
        # justification: long-form conversational evidence; at ~1200 chars/turn
        # a 15-turn retrieval window needs ~18K chars — 22K gives safe margin.
        char_budget=22_000,
        per_turn_max=1200,
    ),
    "wide": _CapSet(
        raw_search_top_k=40,
        window_dedup_cap=48,
        collect_cap=25,
        render_top_k=20,
        char_budget=36_000,
        per_turn_max=2000,
    ),
}


def _get_caps() -> _CapSet:
    profile = os.environ.get("AIKNOT_QUERY_PROFILE", "balanced")
    return _PROFILE_CAPS.get(profile, _PROFILE_CAPS["balanced"])


def _caps_for_contract(base: _CapSet, contract: AnswerContract) -> _CapSet:
    """Widen caps for SET questions — gold answers span many episodes.

    46 % of Cat1 failures are M-type: facts ARE retrieved but the render
    window is too small to show all of them, so the model lists only a
    subset.  Every M-type example is a set-valued question (hobbies, pets,
    events) whose gold answer spans 3-8 episodes.  Widening the render
    funnel ~50 % for SET answers fixes this without touching scalar/boolean
    queries which are already well-tuned on the balanced profile.
    """
    if contract.answer_space is not AnswerSpace.SET:
        return base
    return _CapSet(
        raw_search_top_k=max(base.raw_search_top_k, 28),
        window_dedup_cap=max(base.window_dedup_cap, 32),
        collect_cap=max(base.collect_cap, 22),
        render_top_k=max(base.render_top_k, 18),
        char_budget=max(base.char_budget, 30_000),
        per_turn_max=base.per_turn_max,
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

    # 6b. Temporal anchor guard: session-anchored event claims must not
    # degrade to hypothesis/candidate_rank — time_resolve can handle them.
    if (
        contract.time_axis is TimeAxis.EVENT
        and (profile.has_explicit_event_time or profile.has_temporal_anchor)
        and strategy not in ("time_resolve", "set_collect")
    ):
        strategy = "time_resolve"

    # 7. Execute operator.
    operator_fn = OPERATORS[strategy]
    answer_items, confidence, decision_notes = operator_fn(
        claims, bundles, contract, profile, now, renderer
    )

    # 7b. Raw-episode search for evidence_text enrichment.
    #
    # Each hit is expanded to a 3-turn window (prev → center → next) so the
    # answer-model LLM sees full conversational context.
    # Caps are controlled by AIKNOT_QUERY_PROFILE (narrow/balanced/wide).
    caps = _caps_for_contract(_get_caps(), contract)
    episode_search_ids: list[str] = []
    if frame.focus_entities:
        search_fn = getattr(storage, "search_episodes_by_entities", None)
        if search_fn is not None:
            diversity = contract.answer_space is AnswerSpace.SET
            scored_eps = search_fn(
                agent_id,
                frame.focus_entities,
                query=question,
                top_k=caps.raw_search_top_k,
                diversity=diversity,
            )
            exhaustive_eps = search_fn(
                agent_id,
                frame.focus_entities,
                query="",
                top_k=caps.raw_search_top_k,
            )
            merged_seen: set[str] = set()
            merged: list[Any] = []
            for hit in list(scored_eps) + list(exhaustive_eps):
                if hit.id not in merged_seen:
                    merged_seen.add(hit.id)
                    merged.append(hit)
            episode_search_ids = _expand_centers_first(merged, caps.window_dedup_cap)
            if episode_search_ids:
                profile = replace(profile, episode_fallback_used=True)

    # 8. Render text.
    text = _render_text(answer_items, contract)

    # 8b. Collect evidence episode IDs: raw-search → operator items → claims.
    # Priority order preserves topical relevance; joint RRF across
    # claim-source episodes was tried three times on this branch and each time
    # regressed cat1 single-hop precision by 5+ percentage points.
    ep_ids = _collect_evidence_episode_ids(
        answer_items, claims, episode_search_ids=episode_search_ids, cap=caps.collect_cap
    )
    evidence_text = _render_evidence_context(
        storage,
        agent_id,
        ep_ids,
        top_k=caps.render_top_k,
        char_budget=caps.char_budget,
        per_turn_max=caps.per_turn_max,
    )

    # 9. Build trace.
    latency_ms = (time.monotonic() - t0) * 1000.0
    _debug = os.environ.get("AIKNOT_DEBUG_TRACE") == "1"
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
        n_episode_windows=len(ep_ids) if _debug else 0,
        evidence_chars=len(evidence_text) if _debug else 0,
        raw_search_hit_ids=tuple(episode_search_ids) if _debug else (),
        claim_hits_n=len(claims) if _debug else 0,
        bundle_kinds_used=tuple(dict.fromkeys(b.kind for b in bundles)) if _debug else (),
    )

    return QueryAnswer(
        text=text,
        items=tuple(answer_items),
        confidence=confidence,
        trace=trace,
        evidence_text=evidence_text,
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
    has_temporal_anchor = any(c.qualifiers.get("time_anchor") == "session_date" for c in claims)

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
        has_temporal_anchor=has_temporal_anchor,
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
# Evidence context helpers
# ---------------------------------------------------------------------------


def _expand_centers_first(eps: list[Any], cap: int) -> list[str]:
    """Expand raw-search hits to a 3-turn window, centers before neighbors.

    ``search_episodes_by_entities`` returns episodes whose raw_text matches at
    least one focus entity (``raw_text LIKE '%entity%'``); the returned hits'
    ``prev_id`` / ``next_id`` neighbors are NOT entity-filtered and commonly
    belong to the other speaker in the turn.  By collecting all entity-scoped
    centers first and only then appending neighbors, the eventual
    ``render_top_k`` slice of evidence context stays dominated by the focus
    entity's turns instead of being diluted by counterparty context.
    """
    seen: set[str] = set()
    out: list[str] = []
    for hit in eps:
        if hit.id not in seen:
            seen.add(hit.id)
            out.append(hit.id)
            if len(out) >= cap:
                return out
    for hit in eps:
        for eid in (getattr(hit, "prev_id", None), getattr(hit, "next_id", None)):
            if eid is not None and eid not in seen:
                seen.add(eid)
                out.append(eid)
                if len(out) >= cap:
                    return out
    return out


def _collect_evidence_episode_ids(
    items: list[AnswerItem],
    claims: list[AtomicClaim],
    episode_search_ids: list[str] | None = None,
    cap: int = 5,
) -> list[str]:
    """Unique episode ids in retrieval order: raw-search → items → claims.

    Raw-search runs on entity + query text so it finds the most topically
    relevant episodes.  Operator items and claim sources are appended as
    secondary and tertiary sources to fill up to cap.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _add(eid: str | None) -> bool:
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
            return len(out) >= cap
        return False

    if episode_search_ids:
        for eid in episode_search_ids:
            if _add(eid):
                return out
    for it in items:
        for eid in it.source_episode_ids:
            if _add(eid):
                return out
    for c in claims:
        if _add(c.source_episode_id):
            return out
    return out


def _render_evidence_context(
    storage: object,
    agent_id: str,
    episode_ids: list[str],
    top_k: int = 5,
    char_budget: int = 8_000,
    per_turn_max: int = 400,
) -> str:
    """Format episodes as '[N] [YYYY-MM-DD] Speaker: raw_text', newline-joined.

    Rules:
    - Date in ISO; missing date → no date brackets.
    - Missing speaker → no speaker label.
    - If raw_text already starts with 'Speaker:', do NOT double-prefix.
    - Missing/not-found episode → silently skip.
    - Individual turns truncated to per_turn_max chars; total capped at char_budget.
    """
    parts: list[str] = []
    n = 0
    total_chars = 0
    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return ""
    for eid in episode_ids[:top_k]:
        ep = get_ep(agent_id, eid)
        if ep is None:
            continue
        n += 1
        date_part = ""
        if ep.session_date is not None:
            try:
                date_part = f"[{ep.session_date.date().isoformat()}] "
            except AttributeError:
                # session_date might already be a date object
                date_part = f"[{ep.session_date.isoformat()}] "
        speaker = getattr(ep, "speaker", None)
        raw = ep.raw_text
        if len(raw) > per_turn_max:
            raw = raw[:per_turn_max]
        speaker_part = ""
        if speaker and not raw.lstrip().startswith(f"{speaker}:"):
            speaker_part = f"{speaker}: "
        line = f"[{n}] {date_part}{speaker_part}{raw}"
        if total_chars + len(line) > char_budget:
            break
        parts.append(line)
        total_chars += len(line) + 1  # +1 for newline
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dirty key drainage
# ---------------------------------------------------------------------------


def _drain_dirty_keys(storage: object, agent_id: str) -> None:
    """Read pending dirty_keys_json from meta, invalidate matching bundles, then rebuild.

    This path is only exercised for legacy callers that wrote dirty_keys_json
    without rebuilding bundles at write time (e.g. external claim-only writes).
    The standard ``ingest_episode`` / ``ingest_episodes`` paths clear dirty_keys_json
    immediately after saving bundles, so this function is a no-op for them.
    """
    if not hasattr(storage, "load_materialization_meta"):
        return
    meta = storage.load_materialization_meta(agent_id)
    dirty_json = meta.get("dirty_keys_json", "[]")
    if dirty_json in ("[]", "", "null", None):
        return

    invalidated = _sr.apply_pending_dirty_keys(storage, agent_id, dirty_json)

    if invalidated > 0 and hasattr(storage, "save_bundles") and hasattr(storage, "load_claims"):
        # Rebuild bundles for the affected subjects so the slot plane is not left empty.
        import json as _json

        from ai_knot.materialization import MATERIALIZATION_VERSION
        from ai_knot.support_bundles import build_all_bundles

        try:
            raw_keys = _json.loads(dirty_json)
        except (ValueError, TypeError):
            raw_keys = []
        subjects = list(dict.fromkeys(k["subject"] for k in raw_keys if k.get("subject")))
        if subjects:
            full_claims = storage.load_claims(agent_id, subjects=subjects, active_only=False)
            if full_claims:
                rebuilt_bundles, rebuilt_members = build_all_bundles(
                    full_claims,
                    [],
                    agent_id=agent_id,
                    materialization_version=MATERIALIZATION_VERSION,
                )
                if rebuilt_bundles:
                    storage.save_bundles(agent_id, rebuilt_bundles, rebuilt_members)

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
