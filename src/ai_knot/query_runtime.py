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
            diversity = contract.answer_space is AnswerSpace.SET
            eps = search_fn(
                agent_id, frame.focus_entities, query=question, top_k=60, diversity=diversity
            )
            # Contract-driven window: EVENT/INTERVAL → 5-turn (±2); others → 3-turn (±1)
            if contract.time_axis in (TimeAxis.EVENT, TimeAxis.INTERVAL):
                window_ids = _expand_window_n_turns(eps, storage, agent_id, n=2)
                q_date = _extract_explicit_date_from_question(question)
                if q_date is not None:
                    window_ids = _sort_by_date_proximity(storage, agent_id, window_ids, q_date)
            else:
                window_ids = _expand_window_n_turns(eps, storage, agent_id, n=1)
            episode_search_ids = window_ids[:200]
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

    # 10. Build evidence_text: preference block first, then session-grouped main.
    ep_ids = _collect_evidence_episode_ids(answer_items, claims, episode_search_ids)
    ep_ids = _sort_episode_ids_by_date(storage, agent_id, ep_ids)

    # Preference block (cat3 speculation support) — separate retrieval.
    from ai_knot.preference_retrieval import retrieve_preference_episodes

    pref_eps = []
    if frame.focus_entities:
        pref_eps = retrieve_preference_episodes(storage, agent_id, frame.focus_entities, top_k=20)
    pref_ep_ids = [e.id for e in pref_eps if hasattr(e, "id")]
    pref_block = _render_preference_block(storage, agent_id, pref_ep_ids, frame.focus_entities)

    main_block = _render_evidence_context(storage, agent_id, ep_ids)
    evidence_text = (pref_block + "\n" + main_block).strip() if pref_block else main_block

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


def _extract_explicit_date_from_question(question: str) -> datetime | None:
    """Extract an explicit calendar date from the question text.

    Only extracts absolute dates (January 19, 2023-01-19, etc.).
    Relative phrases (yesterday, last Friday, next month) intentionally
    return None — we never resolve relative time expressions.
    """

    from ai_knot.materialization import _DATE_RE

    m = _DATE_RE.search(question)
    if not m:
        return None
    from ai_knot.materialization import _parse_date_str

    return _parse_date_str(m.group(1))


def _expand_window_n_turns(
    eps: list[Any],
    storage: object,
    agent_id: str,
    n: int,
) -> list[str]:
    """Build ±n-turn window around each episode, returning ordered unique IDs.

    n=1 → prev + center + next (3-turn, current default)
    n=2 → prev.prev + prev + center + next + next.next (5-turn)
    """
    get_ep = getattr(storage, "get_episode", None)
    seen: set[str] = set()
    result: list[str] = []

    def _add(eid: str | None) -> None:
        if eid and eid not in seen:
            seen.add(eid)
            result.append(eid)

    for hit in eps:
        center_id = hit.id
        # Walk backwards n steps
        back_ids: list[str] = []
        cur = hit
        for _ in range(n):
            pid = getattr(cur, "prev_id", None)
            if pid is None:
                break
            back_ids.append(pid)
            if get_ep is not None:
                cur = get_ep(agent_id, pid) or cur
            else:
                break
        for bid in reversed(back_ids):
            _add(bid)
        _add(center_id)
        # Walk forwards n steps
        cur = hit
        for _ in range(n):
            nid = getattr(cur, "next_id", None)
            if nid is None:
                break
            _add(nid)
            if get_ep is not None:
                cur = get_ep(agent_id, nid) or cur
            else:
                break

    return result


def _sort_by_date_proximity(
    storage: object,
    agent_id: str,
    episode_ids: list[str],
    target_date: datetime,
) -> list[str]:
    """Sort episode IDs by proximity of session_date to target_date."""
    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return episode_ids

    def _dist(eid: str) -> float:
        ep = get_ep(agent_id, eid)
        sd = getattr(ep, "session_date", None) if ep else None
        if sd is None:
            return float("inf")
        return float(abs((sd - target_date).total_seconds()))

    return sorted(episode_ids, key=_dist)


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


def _render_preference_block(
    storage: object,
    agent_id: str,
    ep_ids: list[str],
    focus_entities: tuple[str, ...],
) -> str:
    """Render preference/affect episodes as a separate evidence block.

    Appears BEFORE the session-grouped main evidence so cat3 speculative
    questions get the affect context prominently.
    """
    if not ep_ids:
        return ""
    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return ""

    entity_label = ", ".join(focus_entities) if focus_entities else "entity"
    header = f"## Preferences & feelings ({entity_label})"
    lines: list[str] = [header]
    seen: set[str] = set()
    for eid in ep_ids:
        if eid in seen:
            continue
        seen.add(eid)
        ep = get_ep(agent_id, eid)
        if ep is None or not (ep.raw_text or "").strip():
            continue
        sd = getattr(ep, "session_date", None)
        prefix = f"[{sd.date().isoformat()}] " if sd is not None else ""
        speaker = getattr(ep, "speaker", "") or ""
        raw = ep.raw_text or ""
        if speaker and not _raw_text_has_speaker_prefix(raw):
            lines.append(f"{prefix}{speaker}: {raw}")
        else:
            lines.append(f"{prefix}{raw}")

    if len(lines) <= 1:  # only the header, no content
        return ""
    lines.append("")
    return "\n".join(lines)


def _render_evidence_context(
    storage: object,
    agent_id: str,
    episode_ids: list[str],
) -> str:
    """Load raw episodes and format grouped by session with headers.

    Groups episodes by session_id, renders each session with a header line
    ## Session YYYY-MM-DD, then individual turns with [date] Speaker: text.
    """
    from collections import OrderedDict

    if not episode_ids:
        return ""

    get_ep = getattr(storage, "get_episode", None)
    if get_ep is None:
        return ""

    # Collect episodes, skip missing
    ep_pairs: list[tuple[str, Any]] = []
    seen_ids: set[str] = set()
    for eid in episode_ids:
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        ep = get_ep(agent_id, eid)
        if ep is not None:
            ep_pairs.append((eid, ep))

    if not ep_pairs:
        return ""

    # Group by session_id, preserving first-appearance order of sessions
    sessions: OrderedDict[str, list[tuple[str, Any]]] = OrderedDict()
    session_first_date: dict[str, datetime | None] = {}
    for eid, ep in ep_pairs:
        sid = getattr(ep, "session_id", None) or eid
        if sid not in sessions:
            sessions[sid] = []
            session_first_date[sid] = getattr(ep, "session_date", None)
        sessions[sid].append((eid, ep))

    # Keep retrieval relevance order (first-appearance = highest ranked)
    sorted_sids = list(sessions.keys())

    rendered_eps: set[str] = set()
    lines: list[str] = []
    for sid in sorted_sids:
        # Build session header
        sd: datetime | None = session_first_date[sid]
        hdr = f"## Session {sd.date().isoformat()}" if sd is not None else f"## Session {sid[:16]}"
        lines.append(hdr)

        for eid, ep in sessions[sid]:
            if eid in rendered_eps:
                continue
            rendered_eps.add(eid)

            # Build 3-turn window: prev / center / next
            window_parts: list[str] = []
            for neighbor_id in (
                getattr(ep, "prev_id", None),
                eid,
                getattr(ep, "next_id", None),
            ):
                if neighbor_id is None:
                    continue
                if neighbor_id in rendered_eps and neighbor_id != eid:
                    continue
                rendered_eps.add(neighbor_id)
                nep: Any = get_ep(agent_id, neighbor_id) if neighbor_id != eid else ep
                if nep is None:
                    continue
                raw: str = nep.raw_text or ""
                if not raw.strip():
                    continue
                speaker: str = getattr(nep, "speaker", "") or ""
                ep_sd: datetime | None = getattr(nep, "session_date", None)
                date_prefix = f"[{ep_sd.date().isoformat()}] " if ep_sd is not None else ""
                if speaker and not _raw_text_has_speaker_prefix(raw):
                    window_parts.append(f"{date_prefix}{speaker}: {raw}")
                else:
                    window_parts.append(f"{date_prefix}{raw}")

            for part in window_parts:
                lines.append(part)

        lines.append("")  # blank line between sessions

    return "\n".join(lines).rstrip()


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
