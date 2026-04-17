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

    # 5. Raw-episode search for evidence_text enrichment (always runs).
    episode_search_ids: list[str] = []
    episode_fallback_used = False
    if frame.focus_entities:
        search_fn = getattr(storage, "search_episodes_by_entities", None)
        if search_fn is not None:
            eps = search_fn(agent_id, frame.focus_entities, query=question, top_k=60)
            seen: set[str] = set()
            window_ids: list[str] = []
            for hit in eps:
                for eid in (
                    getattr(hit, "prev_id", None),
                    hit.id,
                    getattr(hit, "next_id", None),
                ):
                    if eid is not None and eid not in seen:
                        seen.add(eid)
                        window_ids.append(eid)
                        if len(window_ids) >= 200:
                            break
                if len(window_ids) >= 200:
                    break
            episode_search_ids = window_ids
            episode_fallback_used = bool(window_ids)

    # 6. Build evidence profile.
    profile = _build_evidence_profile(
        claims,
        bundles,
        contract,
        frame,
        question,
        fallback_used,
        episode_fallback_used=episode_fallback_used,
    )

    # 7. Choose strategy.
    strategy = choose_strategy(frame, contract, profile)

    # 8. Execute operator.
    operator_fn = OPERATORS[strategy]
    answer_items, confidence, decision_notes = operator_fn(
        claims, bundles, contract, profile, now, renderer
    )

    # 9. Render text.
    text = _render_text(answer_items, contract)

    # 10. Build evidence_text (raw-search first, then items, then claims).
    ep_ids = _collect_evidence_episode_ids(answer_items, claims, episode_search_ids)
    ep_ids = _sort_episode_ids_by_date(storage, agent_id, ep_ids)
    evidence_text = _render_evidence_context(storage, agent_id, ep_ids)

    # 11. Build trace.
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
        evidence_text=evidence_text,
    )


# ---------------------------------------------------------------------------
# Evidence collection helpers
# ---------------------------------------------------------------------------


def _collect_evidence_episode_ids(
    items: list[AnswerItem],
    claims: list[AtomicClaim],
    episode_search_ids: list[str] | None = None,
    cap: int = 120,
) -> list[str]:
    """Collect episode IDs for evidence_text, raw-search results first."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(eid: str | None) -> bool:
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
        return len(out) >= cap

    # 1. Raw-search results FIRST (highest priority).
    for eid in episode_search_ids or ():
        if _add(eid):
            return out

    # 2. Episodes from answer items.
    for it in items:
        for eid in it.source_episode_ids:
            if _add(eid):
                return out

    # 3. Episodes from claims.
    for c in claims:
        if _add(c.source_episode_id):
            return out

    return out


def _sort_episode_ids_by_date(
    storage: object,
    agent_id: str,
    episode_ids: list[str],
) -> list[str]:
    """Sort episode IDs chronologically by session_date, then turn_id."""
    if not episode_ids:
        return episode_ids
    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return episode_ids
    dated: list[tuple[datetime | None, str, str]] = []
    for eid in episode_ids:
        ep = get_ep(agent_id, eid)
        if ep is None:
            dated.append((None, "", eid))
        else:
            dated.append((getattr(ep, "session_date", None), getattr(ep, "turn_id", ""), eid))
    dated.sort(key=lambda t: (t[0] or datetime.min, t[1]))
    return [eid for _, _, eid in dated]


def _raw_text_has_speaker_prefix(raw: str) -> bool:
    """Return True if raw text already starts with 'Speaker: ' prefix."""
    if ": " not in raw:
        return False
    prefix, _, _ = raw.partition(": ")
    # Speaker prefix is a single capitalized word with no spaces.
    return bool(" " not in prefix and prefix and prefix[0].isupper())


def _render_evidence_context(
    storage: object,
    agent_id: str,
    episode_ids: list[str],
) -> str:
    """Load raw episode texts and format as 3-turn window evidence context.

    For each episode ID, fetches the prev/center/next window and formats as
    ``[date] prev_text / center_text / next_text`` — mirroring the dated
    ingest format so the answer LLM sees comparable context volume.
    """
    if not episode_ids:
        return ""

    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return ""

    rendered: set[str] = set()
    lines: list[str] = []
    for eid in episode_ids:
        ep = get_ep(agent_id, eid)
        if ep is None or eid in rendered:
            continue
        rendered.add(eid)

        # Build 3-turn window: prev / center / next
        window_parts: list[str] = []
        for neighbor_id in (getattr(ep, "prev_id", None), eid, getattr(ep, "next_id", None)):
            if neighbor_id is None:
                continue
            rendered.add(neighbor_id)
            nep = get_ep(agent_id, neighbor_id) if neighbor_id != eid else ep
            if nep is None:
                continue
            raw = nep.raw_text or ""
            if not raw.strip():
                continue
            speaker = getattr(nep, "speaker", "") or ""
            if speaker and not _raw_text_has_speaker_prefix(raw):
                window_parts.append(f"{speaker}: {raw}")
            else:
                window_parts.append(raw)

        if not window_parts:
            continue

        date = getattr(ep, "session_date", None) or ""
        prefix = f"[{date}] " if date else ""
        lines.append(f"{prefix}{' / '.join(window_parts)}")

    result = "\n".join(lines)
    # Cap evidence context to ~8K tokens (~32K chars) to test memory
    # quality rather than raw-context recall.
    max_chars = 32_000
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


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
    *,
    episode_fallback_used: bool = False,
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
        episode_fallback_used=episode_fallback_used,
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
    """Read pending dirty_keys_json from meta, invalidate matching bundles, then rebuild.

    This path is exercised when ingest_episode wrote dirty_keys_json without
    rebuilding bundles at write time. After invalidation, bundles for the
    affected subjects are rebuilt so the slot plane is not left empty.
    """
    if not hasattr(storage, "load_materialization_meta"):
        return
    meta = storage.load_materialization_meta(agent_id)
    dirty_json = meta.get("dirty_keys_json", "[]")
    if dirty_json in ("[]", "", "null", None):
        return

    _sr.apply_pending_dirty_keys(storage, agent_id, dirty_json)

    # Always rebuild bundles when dirty_keys are present — including the first-time
    # build case where invalidated == 0 because no bundles existed yet.
    if hasattr(storage, "save_bundles") and hasattr(storage, "load_claims"):
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

    if hasattr(storage, "save_materialization_meta"):
        # Clear dirty keys after draining.
        storage.save_materialization_meta(
            agent_id,
            schema_version=meta.get("schema_version", 2),
            materialization_version=meta.get("materialization_version", 0),
            last_rebuild_at=None,
            dirty_keys_json="[]",
            rebuild_status=meta.get("rebuild_status", "ready"),
        )
