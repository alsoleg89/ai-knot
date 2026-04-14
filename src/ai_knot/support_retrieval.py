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
) -> list[SupportBundle]:
    """Load support bundles matching any of the given topics.

    Falls back to synthesized single-claim bundles via BM25 when bundle
    plane is empty or sparse.

    Args:
        storage:   Storage backend (must implement BundleStore + ClaimStore).
        agent_id:  Agent namespace.
        topics:    Entity strings or "entity::relation" slot strings.
        kinds:     Optional filter on BundleKind; None = all kinds.
        question:  Raw question string used for BM25 fallback.
        top_k:     Maximum bundles to return.
    """
    bs = _get_bundle_store(storage)

    # 1. Primary: load by topic.
    bundles: list[SupportBundle] = []
    if topics and hasattr(bs, "load_bundles_by_topic"):
        bundles = bs.load_bundles_by_topic(agent_id, topics, kinds)

    # 2. Fallback: BM25 over claim value_text when bundle plane is sparse.
    if len(bundles) < 3 and question:
        fallback = fallback_claim_search(storage, agent_id, question, top_k=top_k)
        if fallback:
            synth = _synthesize_bundles_for_claims(
                fallback, agent_id=agent_id, kind=BundleKind.ENTITY_TOPIC
            )
            bundles.extend(synth)

    return bundles[:top_k]


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
    inline members.
    """
    if not bundles:
        return []

    # 1. Collect member IDs already available in-memory.
    claim_ids: set[str] = set()
    for b in bundles:
        claim_ids.update(b.member_claim_ids or ())

    # 2. Augment from storage for persisted bundles that have no inline members.
    bs = _get_bundle_store(storage)
    persisted_bundle_ids = [b.id for b in bundles if not b.member_claim_ids]
    if persisted_bundle_ids and hasattr(bs, "load_bundle_members"):
        member_map: dict[str, list[str]] = bs.load_bundle_members(agent_id, persisted_bundle_ids)
        for mids in member_map.values():
            claim_ids.update(mids)

    if not claim_ids:
        return []

    cs = _get_claim_store(storage)
    if not hasattr(cs, "load_claims"):
        return []

    return cast(
        list[AtomicClaim], cs.load_claims(agent_id, ids=sorted(claim_ids), active_only=active_only)
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


def topics_for_entities(
    entities: tuple[str, ...],
    contract: AnswerContract | None = None,
) -> list[str]:
    """Convert focus entities to bundle topic strings.

    For slot-level queries (RELATION/STATE focused), also adds "entity::relation".
    """
    topics: list[str] = list(entities)
    # Topic strings are just the entity strings for now.
    # Future: add "entity::relation" for targeted slot queries.
    return list(dict.fromkeys(topics))  # deduplicate preserving order


def bundle_kinds_for_contract(contract: AnswerContract | None) -> list[BundleKind] | None:
    """Map an AnswerContract to the preferred BundleKind list.

    Returns None (= all kinds) when no specific preference is inferrable.
    Temporal queries are checked first so EVENT/INTERVAL questions are not
    incorrectly routed to SET bundles.
    """
    from ai_knot.query_types import AnswerSpace, TruthMode

    if contract is None:
        return None

    # Temporal first: EVENT/INTERVAL questions need event-neighbourhood bundles,
    # not entity-topic ones (previously SET check here was always truthy due to
    # attribute-lookup bug on the enum value).
    if contract.time_axis in (TimeAxis.EVENT, TimeAxis.INTERVAL):
        return [BundleKind.EVENT_NEIGHBORHOOD, BundleKind.STATE_TIMELINE]
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
