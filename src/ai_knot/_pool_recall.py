"""_PoolRecallMixin — recall, arecall, and embed_pool_facts for SharedMemoryPool."""

from __future__ import annotations

import logging
import math
import os
from datetime import UTC, datetime

from ai_knot._pool_helpers import _pool_rerank
from ai_knot._query_intent import (
    _CANONICAL_RESOLVER_INTENTS,
    _INTENT_DIVERSITY_CAP,
    _INTENT_RERANK_WEIGHTS,
    _INTENT_RRF_WEIGHTS,
    _V3_INTENT_MAP,
    _PoolQueryIntent,
    _RecallMeta,
)
from ai_knot.multi_agent.canonical import ClaimFamilyResolver
from ai_knot.multi_agent.models import ExplorationMode
from ai_knot.multi_agent.recall_service import SharedPoolRecallService
from ai_knot.multi_agent.router import QueryShapeRouter
from ai_knot.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from ai_knot.storage.base import StorageBackend, TemporalStorageCapable
from ai_knot.types import Fact

logger = logging.getLogger(__name__)

_SHARED_NAMESPACE = "__shared__"
_POOL_DEBUG = bool(os.environ.get("AI_KNOT_POOL_DEBUG", ""))
_POOL_RECALL_OVERFETCH = 3
_COVERAGE_SCORE_FLOOR: float = 0.01


class _PoolRecallMixin:
    """Mixin providing recall, arecall, and embed_pool_facts to SharedMemoryPool.

    Depends on the following attributes being set by SharedMemoryPool.__init__:
        _storage, _bm25, _dense, _retriever, _embedded_ids, _query_vector,
        _last_recall_meta, _fact_consumers, _used_count, _recall_service,
        _claim_resolver, _query_router, _known_version,
        _AUTO_PROMOTE_THRESHOLD, _TIER_BOOST
    """

    # Declared for type-checker awareness; set by SharedMemoryPool.__init__.
    _storage: StorageBackend
    _bm25: BM25Retriever
    _dense: DenseRetriever
    _retriever: HybridRetriever
    _embedded_ids: set[str]
    _query_vector: list[float] | None
    _last_recall_meta: _RecallMeta | None
    _fact_consumers: dict[str, set[str]]
    _used_count: dict[str, int]
    _recall_service: SharedPoolRecallService
    _claim_resolver: ClaimFamilyResolver
    _query_router: QueryShapeRouter
    _known_version: dict[str, int]
    _AUTO_PROMOTE_THRESHOLD: int
    _TIER_BOOST: dict[str, float]

    def get_trust(self, agent_id: str) -> float:  # provided by SharedMemoryPool
        raise NotImplementedError

    def recall(
        self,
        query: str,
        requesting_agent_id: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        topic_channel: str = "",
    ) -> list[tuple[Fact, float]]:
        """Search the shared pool with provenance discount.

        Applies temporal filter (only active facts) before retrieval.
        Facts originating from the requesting agent receive full score;
        facts from other agents are discounted by per-agent trust (Marsh 1994).

        Args:
            query: The search query.
            requesting_agent_id: Agent performing the query.
            top_k: Maximum results to return.
            now: Point-in-time for temporal filter (default: UTC now).
            topic_channel: If non-empty, only return facts with a matching
                ``topic_channel`` or no channel (empty = visible in all channels).

        Returns:
            List of (Fact, score) pairs sorted by relevance.
        """
        # Use index-accelerated fast path if available (SQLite/Postgres).
        if isinstance(self._storage, TemporalStorageCapable):
            active = self._storage.load_active(_SHARED_NAMESPACE)
        else:
            now_dt = now or datetime.now(UTC)
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            active = [f for f in all_shared if f.is_active(now_dt)]

        # Topic channel filter: include global facts (no channel) + matching channel.
        if topic_channel:
            active = [f for f in active if not f.topic_channel or f.topic_channel == topic_channel]

        # visibility_scope filter: hide local-only facts from foreign agents.
        active = [
            f
            for f in active
            if f.visibility_scope != "local" or f.origin_agent_id == requesting_agent_id
        ]

        if not active:
            self._last_recall_meta = _RecallMeta(
                intent=_PoolQueryIntent.GENERAL,
                total_active=0,
                returned=0,
                coverage=0.0,
                low_coverage=True,
            )
            return []

        agent_private = self._storage.load(requesting_agent_id)
        agent_fact_count = sum(1 for f in agent_private if f.is_active())

        # --- V3 query analysis: separate semantic intent from exploration mode ---
        v3_analysis = self._query_router.analyze(
            query,
            requesting_agent_id=requesting_agent_id,
            active_facts=active,
            requesting_agent_fact_count=agent_fact_count,
            topic_channel=topic_channel,
        )

        # --- Facet-aware MULTI_SOURCE path (V2 pipeline) ---
        # Delegate to SharedPoolRecallService for conjunctive queries.
        # Returns None if the query is not MULTI_SOURCE or decomposition fails,
        # in which case we fall through to the existing flat retrieval below.
        facet_result = self._recall_service.recall(
            query,
            requesting_agent_id=requesting_agent_id,
            active_facts=active,
            requesting_agent_fact_count=agent_fact_count,
            top_k=top_k,
            topic_channel=topic_channel,
            get_trust=self.get_trust,
        )
        if facet_result is not None:
            # Clear single-use query vector (facet path exits before flat path).
            self._query_vector = None
            # Track recall hits for trust accounting (same as flat path).
            for fact, _ in facet_result:
                if fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                    self._used_count[fact.origin_agent_id] = (
                        self._used_count.get(fact.origin_agent_id, 0) + 1
                    )
                consumers = self._fact_consumers.setdefault(fact.id, set())
                consumers.add(requesting_agent_id)
            self._last_recall_meta = _RecallMeta(
                intent=_PoolQueryIntent.MULTI_SOURCE,
                total_active=len(active),
                returned=len(facet_result),
                coverage=1.0 if facet_result else 0.0,
                low_coverage=not facet_result,
            )
            return facet_result

        # --- Flat retrieval path (existing logic) ---
        # Use V3 intent mapping for canonical resolver gating.
        # WIDE exploration mode replaces BROAD_DISCOVERY as the "empty KB" signal.
        # The canonical resolver runs for CANONICAL, GENERAL, and narrow WIDE queries —
        # but NOT for WIDE queries, because WIDE means the agent is onboarding and
        # the resolver would over-aggressively eliminate correct answers.
        _v3_is_wide = v3_analysis.exploration_mode == ExplorationMode.WIDE

        intent = _V3_INTENT_MAP.get(v3_analysis.intent, _PoolQueryIntent.GENERAL)
        # Wide exploration maps to BROAD_DISCOVERY for downstream diversity logic.
        if _v3_is_wide:
            intent = _PoolQueryIntent.BROAD_DISCOVERY

        rrf_override = _INTENT_RRF_WEIGHTS.get(intent)

        # Over-fetch so trust discount is applied before the top-k cutoff.
        # Without this, low-trust facts can displace better candidates by scoring
        # high in retrieval and then being down-ranked after the cut.
        overfetch_k = min(top_k * _POOL_RECALL_OVERFETCH, len(active))
        pairs = self._retriever.search(
            query,
            active,
            top_k=overfetch_k,
            query_vector=self._query_vector,
            rrf_weights=rrf_override,
        )
        # Clear single-use query vector after search.
        self._query_vector = None

        if _POOL_DEBUG:
            logger.debug(
                "pool_recall query=%r intent=%s active=%d overfetch=%d raw_top5=%s",
                query,
                intent.value,
                len(active),
                overfetch_k,
                [(f.id[:8], f.origin_agent_id, round(s, 4)) for f, s in pairs[:5]],
            )

        # Resolve claim conflicts before applying trust discount.
        # Running on raw BM25 scores ensures competing claims are identified by
        # content relevance, not trust level.  Winner selection inside the resolver
        # already uses get_trust() directly (trust × recency × score), so source
        # quality still governs which competing claim survives.
        # If this ran after trust discount, a correct fact from a low-trust agent
        # could be ranked below unrelated facts from high-trust agents, displacing
        # the correct answer out of top-k.
        if intent in _CANONICAL_RESOLVER_INTENTS or _v3_is_wide:
            pairs = self._claim_resolver.resolve(
                pairs,
                canonical_mode=True,
                get_trust=self.get_trust,
            )

        # Apply per-agent trust discount + tier boost before final cutoff.
        # For WIDE (empty-KB) queries, skip trust discount: the querier has no
        # interaction history with the pool and cannot know which agents are
        # trustworthy.  Applying trust would unfairly penalise agents that
        # happened not to appear in earlier (unrelated) queries.
        apply_trust = not _v3_is_wide
        discounted: list[tuple[Fact, float]] = []
        for fact, score in pairs:
            if apply_trust and fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                trust = self.get_trust(fact.origin_agent_id)
                score *= trust
            # Tier-aware scoring: org-tier facts get a small boost.
            score *= self._TIER_BOOST.get(fact.memory_tier, 1.0)
            discounted.append((fact, score))

        # Pool-specific reranking: recency + freshness boosts.
        rerank_params = _INTENT_RERANK_WEIGHTS.get(intent, (0.05, 0.03))
        discounted = _pool_rerank(
            discounted,
            recency_weight=rerank_params[0],
            freshness_weight=rerank_params[1],
        )

        if _POOL_DEBUG:
            logger.debug(
                "pool_recall trust_discounted_top5=%s",
                [
                    (f.id[:8], f.origin_agent_id, round(s, 4))
                    for f, s in sorted(discounted, key=lambda x: x[1], reverse=True)[:5]
                ],
            )

        # Stage 1 — Adaptive monopoly breaker: prevent single-agent dominance
        # when 3+ credible agents have published.  Computed from candidate
        # distribution, not from intent name.  ENTITY_LOOKUP exempt — one agent
        # may be authoritative for a specific entity (e.g. CAS-updated slot).
        # Only count agents with trust above the adversary floor — untrusted
        # agents (adversaries, trust≈0.1) must not inflate the publisher count.
        # Set to 0.2: below the Bayesian prior for fresh publishers (~0.3) but
        # above the hard floor for adversaries after CAS supersession (0.1).
        _TRUST_FLOOR_FOR_DIVERSITY = 0.2
        n_publishers = len(
            {
                f.origin_agent_id
                for f in active
                if f.origin_agent_id
                and self.get_trust(f.origin_agent_id) >= _TRUST_FLOOR_FOR_DIVERSITY
            }
        )
        if n_publishers >= 3 and intent != _PoolQueryIntent.ENTITY_LOOKUP:
            discounted.sort(key=lambda x: x[1], reverse=True)
            max_per_agent = max(1, top_k // n_publishers + 1)
            _agent_counts: dict[str, int] = {}
            _capped: list[tuple[Fact, float]] = []
            for fact, score in discounted:
                aid = fact.origin_agent_id or "__self__"
                cnt = _agent_counts.get(aid, 0)
                if cnt < max_per_agent:
                    _capped.append((fact, score))
                    _agent_counts[aid] = cnt + 1
            discounted = _capped

        # Stage 2 — Intent-specific floor for intents that structurally need
        # wider multi-agent coverage.  Skip when Stage 1 already applied the
        # adaptive cap — double-filtering crushes diversity further.
        _diversity_cap = _INTENT_DIVERSITY_CAP.get(intent)
        if _diversity_cap is not None and n_publishers < 3:
            discounted.sort(key=lambda x: x[1], reverse=True)
            agent_cap = math.ceil(top_k * _diversity_cap)
            agent_counts: dict[str, int] = {}
            capped: list[tuple[Fact, float]] = []
            for fact, score in discounted:
                aid = fact.origin_agent_id or "__self__"
                cnt = agent_counts.get(aid, 0)
                if cnt < agent_cap:
                    capped.append((fact, score))
                    agent_counts[aid] = cnt + 1
            discounted = capped

        discounted.sort(key=lambda x: x[1], reverse=True)
        top_results = discounted[:top_k]

        # Track recall hits only for facts actually returned — not over-fetched
        # candidates that were discarded after trust discount.
        # Cap at one credit per agent per recall() call to prevent trust inflation
        # from sessions where a single agent contributes multiple top-k results.
        auto_promote_ids: list[str] = []
        credited_agents: set[str] = set()
        for fact, _ in top_results:
            if (
                fact.origin_agent_id
                and fact.origin_agent_id != requesting_agent_id
                and fact.origin_agent_id not in credited_agents
            ):
                self._used_count[fact.origin_agent_id] = (
                    self._used_count.get(fact.origin_agent_id, 0) + 1
                )
                credited_agents.add(fact.origin_agent_id)
            # Track per-fact consumer agents for auto-promotion.
            consumers = self._fact_consumers.setdefault(fact.id, set())
            consumers.add(requesting_agent_id)
            if fact.memory_tier == "pool" and len(consumers) >= self._AUTO_PROMOTE_THRESHOLD:
                auto_promote_ids.append(fact.id)

        # Auto-promote facts consumed by enough distinct agents.
        if auto_promote_ids:
            shared = self._storage.load(_SHARED_NAMESPACE)
            promoted = 0
            for f in shared:
                if f.id in auto_promote_ids and f.memory_tier == "pool":
                    f.memory_tier = "org"
                    promoted += 1
            if promoted:
                self._storage.save(_SHARED_NAMESPACE, shared)
                logger.info("Auto-promoted %d facts to 'org' tier", promoted)

        # Compute coverage: fraction of returned results with meaningful scores.
        relevant_count = sum(1 for _, s in top_results if s >= _COVERAGE_SCORE_FLOOR)
        coverage = relevant_count / len(top_results) if top_results else 0.0
        self._last_recall_meta = _RecallMeta(
            intent=intent,
            total_active=len(active),
            returned=len(top_results),
            coverage=coverage,
            low_coverage=coverage < 0.5,
        )

        if _POOL_DEBUG:
            logger.debug(
                "pool_recall returned=%s coverage=%.2f intent=%s",
                [(f.id[:8], f.origin_agent_id, round(s, 4)) for f, s in top_results],
                coverage,
                intent.value,
            )

        return top_results

    async def embed_pool_facts(self) -> int:
        """Embed all active pool facts that haven't been embedded yet.

        Fetches facts from storage, embeds new ones via Ollama, and loads
        the vectors into the DenseRetriever.  Returns the number of newly
        embedded facts.  Returns 0 (no-op) if Ollama is unreachable.
        """
        from ai_knot.embedder import embed_texts

        if isinstance(self._storage, TemporalStorageCapable):
            active = self._storage.load_active(_SHARED_NAMESPACE)
        else:
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            active = [f for f in all_shared if f.is_active()]

        new_facts = [f for f in active if f.id not in self._embedded_ids]
        if not new_facts:
            return 0

        texts = [f.content for f in new_facts]
        vectors = await embed_texts(texts)
        if not vectors:
            return 0  # Ollama unavailable — BM25-only fallback.

        new_vectors = {f.id: vec for f, vec in zip(new_facts, vectors, strict=True)}
        self._dense.add_embeddings(new_vectors)
        self._embedded_ids.update(new_vectors.keys())
        return len(new_vectors)

    async def arecall(
        self,
        query: str,
        requesting_agent_id: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        topic_channel: str = "",
    ) -> list[tuple[Fact, float]]:
        """Async variant of :meth:`recall` with embedding-based hybrid retrieval.

        Embeds any new pool facts, embeds the query, then delegates to the
        synchronous ``recall()`` with the dense signal available.  Falls back
        to BM25-only if Ollama is unreachable.
        """
        from ai_knot.embedder import embed_texts

        # Embed new pool facts (incremental — skips already-embedded).
        await self.embed_pool_facts()

        # Embed the query.
        if self._dense.has_embeddings():
            qvecs = await embed_texts([query])
            self._query_vector = qvecs[0] if qvecs else None
        else:
            self._query_vector = None

        return self.recall(
            query, requesting_agent_id, top_k=top_k, now=now, topic_channel=topic_channel
        )
