"""Pool reranking and claim conflict resolution helpers."""

from __future__ import annotations

from collections.abc import Callable

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact, MESIState

# ---------------------------------------------------------------------------
# Claim normalization for unslotted facts in the shared pool
# ---------------------------------------------------------------------------

_CLAIM_ATTR_STEMS = frozenset(
    {
        "sla",
        "price",
        "cost",
        "limit",
        "rate",
        "version",
        "api",
        "timeout",
        "region",
        "uptim",
        "user",
        "tier",
        "window",
        "hour",
        "minut",
        "coverag",
        "rotat",
        "schedul",
        "migrat",
        "deploy",
        "review",
        "scan",
        "endpoint",
        "authen",
        "support",
        "call",
        "deprec",
    }
)


def _extract_claim_key(content: str) -> str:
    """Extract a lightweight claim fingerprint from free-text content.

    Tokenises the content and collects up to 2 entity-like tokens (those
    appearing before the first attribute keyword) plus the first attribute
    keyword.  Returns ``"{entity_tok}_{entity_tok}::{attr_stem}"`` or
    ``""`` if no clear claim structure is detected.
    """
    tokens = _tokenize(content)
    if len(tokens) < 3:
        return ""

    entity_tokens: list[str] = []
    attr_token = ""
    for t in tokens:
        if t in _CLAIM_ATTR_STEMS:
            attr_token = t
            break
        if len(entity_tokens) < 2:
            entity_tokens.append(t)

    if not entity_tokens or not attr_token:
        return ""
    return f"{'_'.join(entity_tokens)}::{attr_token}"


# ---------------------------------------------------------------------------
# Pool reranking
# ---------------------------------------------------------------------------


def _pool_rerank(
    pairs: list[tuple[Fact, float]],
    *,
    recency_weight: float = 0.05,
    freshness_weight: float = 0.03,
    slot_winner_weight: float = 0.10,
) -> list[tuple[Fact, float]]:
    """Rerank pool retrieval results with recency, freshness, and slot-winner boosts.

    Signals applied multiplicatively to the incoming score:
    1. Recency: newer facts (by ``created_at``) receive up to
       ``+recency_weight`` boost (linear rank-normalised).
    2. Freshness: facts in MODIFIED or SHARED MESI state receive
       ``+freshness_weight`` boost (active CAS winners).
    3. Slot winner: CAS-updated facts (MODIFIED + slot_key) receive
       ``+slot_winner_weight`` boost.  This is a slot property, not an
       intent property — the latest canonical version should rank first
       for any query that matches its slot.
    """
    if len(pairs) <= 1:
        return pairs

    sorted_by_time = sorted(pairs, key=lambda p: p[0].created_at)
    n = len(sorted_by_time) - 1
    recency_rank: dict[str, float] = {f.id: i / n for i, (f, _) in enumerate(sorted_by_time)}

    reranked: list[tuple[Fact, float]] = []
    for fact, score in pairs:
        boost = 1.0
        boost += recency_weight * recency_rank.get(fact.id, 0.0)
        if fact.mesi_state in (MESIState.MODIFIED, MESIState.SHARED):
            boost += freshness_weight
        if fact.slot_key and fact.mesi_state == MESIState.MODIFIED:
            boost += slot_winner_weight
        reranked.append((fact, score * boost))
    return reranked


# ---------------------------------------------------------------------------
# Claim conflict resolution
# ---------------------------------------------------------------------------


def _resolve_claim_conflicts(
    pairs: list[tuple[Fact, float]],
    get_trust: Callable[[str], float],
) -> list[tuple[Fact, float]]:
    """Among facts sharing a ``claim_key``, keep only the canonical winner.

    Winner selection (priority order):
    1. If any member has a ``slot_key``, it wins (CAS is authoritative).
    2. Otherwise: highest ``trust(origin_agent) × created_at`` wins.
    3. Tie-breaker: latest ``created_at``.

    Facts with an empty ``claim_key`` pass through unchanged.
    """
    if not pairs:
        return pairs

    clusters: dict[str, list[tuple[Fact, float]]] = {}
    unclustered: list[tuple[Fact, float]] = []

    for fact, score in pairs:
        if fact.claim_key:
            clusters.setdefault(fact.claim_key, []).append((fact, score))
        else:
            unclustered.append((fact, score))

    resolved = list(unclustered)
    for members in clusters.values():
        if len(members) == 1:
            resolved.append(members[0])
            continue

        # Prefer slotted facts (CAS is authoritative).
        slotted = [(f, s) for f, s in members if f.slot_key]
        if slotted:
            resolved.append(max(slotted, key=lambda x: x[1]))
            continue

        # Pick the winner by trust × recency.
        def _conflict_score(pair: tuple[Fact, float]) -> float:
            f = pair[0]
            trust = get_trust(f.origin_agent_id) if f.origin_agent_id else 1.0
            return trust * f.created_at.timestamp()

        resolved.append(max(members, key=_conflict_score))

    return resolved
