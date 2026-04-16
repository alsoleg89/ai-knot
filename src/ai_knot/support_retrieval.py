"""Retrieval abstraction for the query runtime — the sole physical retrieval path.

This module is the only place in the query runtime that accesses storage.
``query_runtime.py``, ``query_operators.py``, and ``query_contract.py`` must
import from here and nowhere else for retrieval.

Enforced by: scripts/check_query_runtime_isolation.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ai_knot.query_types import (
    AnswerContract,
    AtomicClaim,
    BundleKind,
    SupportBundle,
    TimeAxis,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Storage registry helpers — deferred to avoid circular imports
# ---------------------------------------------------------------------------


def _get_episode_store(storage: object) -> object:
    return storage  # SQLiteStorage implements all protocols


def _get_claim_store(storage: object) -> object:
    return storage


def _get_bundle_store(storage: object) -> object:
    return storage


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------


def retrieve_bundles(
    storage: object,
    agent_id: str,
    *,
    topics: list[str],
    kinds: list[BundleKind] | None = None,
    question: str = "",
    top_k: int = 60,
) -> tuple[list[SupportBundle], bool]:
    """Load support bundles matching any of the given topics.

    Falls back to synthesized single-claim bundles via BM25 only when the
    primary bundle plane has no slot bundle AND fewer than 2 entity-topic
    bundles (to avoid diluting a valid slot bundle with unrelated fallback
    claims).

    Args:
        storage:   Storage backend (must implement BundleStore + ClaimStore).
        agent_id:  Agent namespace.
        topics:    Entity strings or "entity::relation" slot strings.
        kinds:     Optional filter on BundleKind; None = all kinds.
        question:  Raw question string used for BM25 fallback.
        top_k:     Maximum bundles to return.

    Returns:
        (bundles, fallback_used) where fallback_used=True means BM25 was used.
    """
    bs = _get_bundle_store(storage)

    # 1. Primary: load by topic.
    bundles: list[SupportBundle] = []
    if topics and hasattr(bs, "load_bundles_by_topic"):
        bundles = bs.load_bundles_by_topic(agent_id, topics, kinds)

    # 2. Fallback: BM25 only when no slot bundle and primary is sparse.
    has_slot_bundle = any("::" in b.topic for b in bundles)
    fallback_used = False
    if not has_slot_bundle and len(bundles) < 2 and question:
        fallback = fallback_claim_search(storage, agent_id, question, top_k=top_k)
        if fallback:
            synth = _synthesize_bundles_for_claims(
                fallback, agent_id=agent_id, kind=BundleKind.ENTITY_TOPIC
            )
            bundles.extend(synth)
            fallback_used = True

    return bundles[:top_k], fallback_used


def expand_claims(
    storage: object,
    agent_id: str,
    bundles: list[SupportBundle],
    *,
    active_only: bool = True,
) -> list[AtomicClaim]:
    """Expand a list of bundles into their constituent AtomicClaims.

    Uses in-memory member_claim_ids first (works for synthetic fallback bundles
    that are never persisted), then augments from storage for bundles without
    inline members.  Preserves bundle member_rank ordering (no set de-dup).
    """
    if not bundles:
        return []

    # 1. Collect member IDs in bundle member_rank order — use ordered de-dup
    #    (dict.fromkeys preserves insertion order, eliminates duplicates).
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for b in bundles:
        for cid in b.member_claim_ids or ():
            if cid not in seen_ids:
                seen_ids.add(cid)
                ordered_ids.append(cid)

    # 2. Augment from storage for persisted bundles that have no inline members.
    bs = _get_bundle_store(storage)
    persisted_bundle_ids = [b.id for b in bundles if not b.member_claim_ids]
    if persisted_bundle_ids and hasattr(bs, "load_bundle_members"):
        member_map: dict[str, list[str]] = bs.load_bundle_members(agent_id, persisted_bundle_ids)
        for mids in member_map.values():
            for cid in mids:
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    ordered_ids.append(cid)

    if not ordered_ids:
        return []

    cs = _get_claim_store(storage)
    if not hasattr(cs, "load_claims"):
        return []

    return cast(
        list[AtomicClaim], cs.load_claims(agent_id, ids=ordered_ids, active_only=active_only)
    )


def fallback_claim_search(
    storage: object,
    agent_id: str,
    question: str,
    *,
    top_k: int = 60,
) -> list[AtomicClaim]:
    """BM25 search over atomic_claims.value_text when bundle plane is empty.

    Uses the lightweight built-in BM25 kernel.
    """
    cs = _get_claim_store(storage)
    if not hasattr(cs, "iter_value_text") or not hasattr(cs, "load_claims"):
        return []

    pairs: list[tuple[str, str]] = list(cs.iter_value_text(agent_id))
    if not pairs:
        return []

    scored = _bm25_score_pairs(question, pairs, top_k=top_k)
    if not scored:
        return []

    ids = [cid for cid, _ in scored]
    return cast(list[AtomicClaim], cs.load_claims(agent_id, ids=ids, active_only=True))


def apply_pending_dirty_keys(
    storage: object,
    agent_id: str,
    dirty_keys_json: str,
) -> int:
    """Drain any pending dirty keys from materialization_meta and invalidate bundles.

    Returns the number of bundles invalidated.

    Note: this function only DELETES matching bundles — it does NOT rebuild them.
    Callers must rebuild via ``build_all_bundles`` + ``save_bundles`` if the slot
    plane must remain populated after invalidation.  Prefer skipping this API
    when bundles were already rebuilt at ingest time (the standard ingest path
    clears dirty_keys_json immediately after saving bundles, making this a no-op).
    """
    import json

    from ai_knot.query_types import BundleKind, DirtyKey

    if dirty_keys_json in ("[]", "", "null"):
        return 0

    try:
        raw_keys = json.loads(dirty_keys_json)
    except (ValueError, TypeError):
        return 0

    if not raw_keys:
        return 0

    keys: list[DirtyKey] = []
    for k in raw_keys:
        keys.append(
            DirtyKey(
                subject=k.get("subject"),
                relation=k.get("relation"),
                bundle_kind=(BundleKind(k["bundle_kind"]) if k.get("bundle_kind") else None),
                topic=k.get("topic"),
            )
        )

    bs = _get_bundle_store(storage)
    if not hasattr(bs, "invalidate_by_keys"):
        return 0

    return cast(int, bs.invalidate_by_keys(agent_id, keys))


# Maps a query-extracted focus_relation (canonical lemma) to the set of
# compound relation names the materializer may have stored under that entity.
# Keeps the fan-out small and mechanical — no guessing, no LLM.
_RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "find": ("finds_satisfying",),
    "like": ("likes",),
    "love": ("likes",),
    "enjoy": ("likes",),
    "hate": ("dislikes",),
    "dislike": ("dislikes",),
    "drive": ("drives",),
    "move": ("moved_to",),
    "work": ("works_at", "works_as"),
    "pass": ("passed_away",),
}


def topics_for_entities(
    entities: tuple[str, ...],
    contract: AnswerContract | None = None,
    *,
    focus_relation: str | None = None,
) -> list[str]:
    """Convert focus entities to bundle topic strings.

    When focus_relation is provided, slot topics ("entity::relation") are
    prepended so that STATE_TIMELINE / RELATION_SUPPORT bundles are retrieved
    first.  Compound-relation aliases are also expanded so that a query lemma
    like "find" also searches for the stored relation "finds_satisfying".
    """
    topics: list[str] = []
    for entity in entities:
        if focus_relation:
            # Fan out to alias relations first (most specific match), then the
            # raw lemma form, then the bare entity topic.
            aliases = _RELATION_ALIASES.get(focus_relation, ())
            for alias in aliases:
                topics.append(f"{entity}::{alias}")
            topics.append(f"{entity}::{focus_relation}")
        topics.append(entity)
    return list(dict.fromkeys(topics))  # deduplicate preserving order


def bundle_kinds_for_contract(
    contract: AnswerContract | None,
    *,
    focus_relation: str | None = None,
) -> list[BundleKind] | None:
    """Map an AnswerContract to the preferred BundleKind list.

    Returns None (= all kinds) when no specific preference is inferrable.
    Temporal queries are checked first so EVENT/INTERVAL questions are not
    incorrectly routed to SET bundles.  When focus_relation is set, slot
    bundles (STATE_TIMELINE, RELATION_SUPPORT) are prioritized.
    """
    from ai_knot.query_types import AnswerSpace, TruthMode

    if contract is None:
        return None

    # Temporal takes precedence: EVENT/INTERVAL questions must include
    # EVENT_NEIGHBORHOOD first — even when a focus relation is present.
    if contract.time_axis in (TimeAxis.EVENT, TimeAxis.INTERVAL):
        if focus_relation:
            return [
                BundleKind.EVENT_NEIGHBORHOOD,
                BundleKind.STATE_TIMELINE,
                BundleKind.RELATION_SUPPORT,
            ]
        return [BundleKind.EVENT_NEIGHBORHOOD, BundleKind.STATE_TIMELINE]

    # Slot-first: when we have a focus relation, prefer STATE_TIMELINE and
    # RELATION_SUPPORT so the slot bundle is retrieved before entity-topic.
    if focus_relation:
        return [BundleKind.STATE_TIMELINE, BundleKind.RELATION_SUPPORT, BundleKind.ENTITY_TOPIC]

    if contract.answer_space is AnswerSpace.SET:
        # Aggregate over entity-topic bundles.
        return [BundleKind.ENTITY_TOPIC, BundleKind.STATE_TIMELINE]
    if contract.truth_mode is TruthMode.RECONSTRUCT:
        return [BundleKind.STATE_TIMELINE, BundleKind.ENTITY_TOPIC]
    return None  # all kinds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bm25_score_pairs(
    query: str,
    pairs: list[tuple[str, str]],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    """Very lightweight BM25-like scoring over (id, text) pairs."""
    from ai_knot.tokenizer import tokenize as _tokenize

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for doc_id, text in pairs:
        doc_tokens = _tokenize(text)
        if not doc_tokens:
            continue
        overlap = sum(1 for t in doc_tokens if t in q_tokens)
        if overlap > 0:
            tf = overlap / len(doc_tokens)
            scored.append((doc_id, tf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _synthesize_bundles_for_claims(
    claims: list[AtomicClaim],
    *,
    agent_id: str,
    kind: BundleKind,
) -> list[SupportBundle]:
    """Synthesize ephemeral bundles for fallback claims (not persisted)."""
    from datetime import UTC, datetime

    from ai_knot.query_types import make_bundle_id

    if not claims:
        return []

    now = datetime.now(UTC)
    bid = make_bundle_id()
    score = sum(c.salience * c.confidence for c in claims) / len(claims)

    return [
        SupportBundle(
            id=bid,
            agent_id=agent_id,
            kind=kind,
            topic="__fallback__",
            member_claim_ids=tuple(c.id for c in claims),
            score_formula="bm25_fallback",
            bundle_score=score,
            built_from_materialization_version=0,
            built_at=now,
        )
    ]
