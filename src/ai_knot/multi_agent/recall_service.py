"""Orchestration service for multi-source pool retrieval.

Replaces flat top-k retrieval with a facet-aware pipeline:
    route → decompose → per-facet retrieve → score → assemble

Used by ``SharedMemoryPool.recall()`` for MULTI_SOURCE queries.
Other intents continue to use the existing flat retrieval path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.bridge import BridgeRetriever
from ai_knot.multi_agent.expertise import AgentExpertiseIndex
from ai_knot.multi_agent.facets import ConjunctiveFacetPlanner
from ai_knot.multi_agent.models import (
    CandidateFact,
    QueryAnalysis,
    QueryFacet,
    RetrievalIntent,
    RoutedPoolQuery,
)
from ai_knot.multi_agent.router import QueryShapeRouter
from ai_knot.multi_agent.scoring import DiversityPolicy, NearMissDetector, SpecificityScorer
from ai_knot.retriever import BM25Retriever
from ai_knot.types import Fact

logger = logging.getLogger(__name__)

# Over-fetch multiplier per facet: retrieve more candidates than top_k
# so scoring and diversity filtering have enough material.
_FACET_OVERFETCH = 3

# Trust floor for counting credible publishers (same as knowledge.py).
_TRUST_FLOOR_FOR_DIVERSITY = 0.2

# Maximum expert agents to consider per facet when expertise routing is on.
# Higher = better recall but slower; 10 keeps BM25 fast at N=1000.
_EXPERTISE_TOP_N = 10

# Minimum pool size (distinct agents) before expertise routing kicks in.
# Below this threshold, BM25 over the full pool works fine and expertise
# narrowing risks cutting relevant agents from small pools.
_EXPERTISE_MIN_POOL_AGENTS = 30


class SharedPoolRecallService:
    """Orchestrate facet-aware pool retrieval for MULTI_SOURCE queries.

    Pipeline:
    1. Route query (intent classification).
    2. Decompose into facets (for MULTI_SOURCE only).
    3. Per-facet BM25 retrieval with overfetch.
    4. Apply specificity scoring and near-miss detection.
    5. Coverage-aware assembly with diversity caps.

    For non-MULTI_SOURCE intents, returns None to signal that the caller
    should use the existing flat retrieval path.
    """

    def __init__(self) -> None:
        self._router = QueryShapeRouter()
        self._planner = ConjunctiveFacetPlanner()
        self._specificity = SpecificityScorer()
        self._near_miss = NearMissDetector()
        self._assembler = CoverageAwareAssembler(diversity=DiversityPolicy())
        # PRF-enabled BM25 for the V2 facet path (PRF controlled externally via prf_expand).
        self._bm25 = BM25Retriever(skip_prf=False)
        # PRF-disabled BM25 for the V3 flat harvest — PRF on large pools can hurt recall
        # when the first-pass candidates are not cleanly relevant.
        self._bm25_flat = BM25Retriever(skip_prf=True)
        self._expertise = AgentExpertiseIndex()
        self._bridge = BridgeRetriever()

    def recall(
        self,
        query: str,
        *,
        requesting_agent_id: str,
        active_facts: list[Fact],
        requesting_agent_fact_count: int,
        top_k: int = 5,
        topic_channel: str = "",
        get_trust: Callable[[str], float] | None = None,
    ) -> list[tuple[Fact, float]] | None:
        """Run the multi-source retrieval pipeline.

        Returns:
            List of (Fact, score) pairs if MULTI_SOURCE intent was
            detected and facets were decomposed.  Returns ``None`` if
            the query should use the standard flat retrieval path.
        """
        # Step 1: Route.
        routed = self._router.route(
            query,
            requesting_agent_id=requesting_agent_id,
            active_facts=active_facts,
            requesting_agent_fact_count=requesting_agent_fact_count,
            topic_channel=topic_channel,
        )

        # Only intercept MULTI_SOURCE queries.
        if routed.intent != "multi_source":
            return None

        # Step 2: Decompose into facets.
        facets = self._planner.decompose(routed)
        routed = RoutedPoolQuery(
            raw_query=routed.raw_query,
            intent=routed.intent,
            facets=facets,
            topic_channel=routed.topic_channel,
            use_expertise_routing=routed.use_expertise_routing,
        )

        # If planner produced only 1 facet (decomposition failed), fall back.
        if len(facets) <= 1:
            return None

        logger.debug(
            "recall_service: MULTI_SOURCE decomposed into %d facets: %s",
            len(facets),
            [f.text for f in facets],
        )

        # Count trusted publishers for diversity cap.
        if get_trust:
            n_publishers = len(
                {
                    f.origin_agent_id
                    for f in active_facts
                    if f.origin_agent_id
                    and get_trust(f.origin_agent_id) >= _TRUST_FLOOR_FOR_DIVERSITY
                }
            )
        else:
            n_publishers = len({f.origin_agent_id for f in active_facts if f.origin_agent_id})

        # Step 2b: Build/refresh expertise index for route-before-retrieve.
        if get_trust and self._expertise.is_stale(active_facts):
            self._expertise.build(active_facts, get_trust)

        # Step 3: Per-facet retrieval with expertise routing.
        candidates_by_facet = self._retrieve_per_facet(
            facets,
            active_facts,
            top_k=top_k,
            get_trust=get_trust,
            requesting_agent_id=requesting_agent_id,
        )

        # Step 5: Coverage-aware assembly.
        result = self._assembler.assemble(
            candidates_by_facet=candidates_by_facet,
            top_k=top_k,
            n_publishers=max(n_publishers, 1),
        )

        logger.debug(
            "recall_service: assembled %d facts, coverage=%.2f, covered=%s, uncovered=%s",
            len(result.selected),
            result.coverage_score,
            result.covered_facets,
            result.uncovered_facets,
        )

        # Convert to (Fact, score) pairs.
        return [(c.fact, c.final_score) for c in result.selected]

    def _retrieve_per_facet(
        self,
        facets: tuple[QueryFacet, ...],
        active_facts: list[Fact],
        *,
        top_k: int,
        get_trust: Callable[[str], float] | None,
        requesting_agent_id: str,
    ) -> dict[str, list[CandidateFact]]:
        """Run BM25 retrieval for each facet independently.

        Each facet gets its own two-pass BM25 search over the full active pool:
        1. First pass: plain BM25 (no expansion).
        2. If the first pass yields >= 3 results with score > 0 and the
           results are not dominated by near-miss facts, compute per-facet
           PRF expansion and re-search with those terms.

        Results are scored for specificity and near-miss penalty, then
        trust-discounted.
        """
        overfetch_k = min(top_k * _FACET_OVERFETCH, len(active_facts))
        candidates_by_facet: dict[str, list[CandidateFact]] = {}

        # Count distinct agents — only use expertise routing for large pools.
        pool_agent_count = len({f.origin_agent_id for f in active_facts if f.origin_agent_id})
        use_expertise = self._expertise.built and pool_agent_count >= _EXPERTISE_MIN_POOL_AGENTS

        for facet in facets:
            # Expertise routing: narrow the search space to facts from
            # the top expert agents for this facet.  Falls back to the
            # full pool when the expertise index isn't built or pool is small.
            if use_expertise:
                expert_hits = self._expertise.top_agents_for_facet(
                    facet.tokens, top_n=_EXPERTISE_TOP_N
                )
                if expert_hits:
                    expert_ids = {h.agent_id for h in expert_hits}
                    facet_facts = [f for f in active_facts if f.origin_agent_id in expert_ids]
                    # Guard: if narrowing removed too many facts, fall back to full pool.
                    search_facts = facet_facts if len(facet_facts) >= top_k else active_facts
                else:
                    search_facts = active_facts
            else:
                search_facts = active_facts

            facet_overfetch = min(overfetch_k, len(search_facts))

            # --- First pass: plain BM25, no expansion ---
            first_pass = self._bm25.search(
                facet.text,
                search_facts,
                top_k=facet_overfetch,
            )

            # Determine whether per-facet PRF is worthwhile.
            scored_results = [(f, s) for f, s in first_pass if s > 0]
            expansion: dict[str, float] = {}

            if len(scored_results) >= 3:
                # Guard: skip PRF if initial results are dominated by
                # near-miss facts (>50% with penalty > 0.3).
                top3_facts = [f for f, _ in scored_results[:3]]
                near_miss_count = sum(1 for f in top3_facts if self._near_miss.penalty(f) > 0.3)
                if near_miss_count <= len(top3_facts) // 2:
                    expansion = self._bm25.prf_expand(facet.text, [f for f, _ in scored_results])

            # --- Second pass (or reuse first pass) ---
            if expansion:
                pairs = self._bm25.search(
                    facet.text,
                    search_facts,
                    top_k=facet_overfetch,
                    expansion_weights=expansion,
                )
            else:
                pairs = first_pass

            candidates: list[CandidateFact] = []
            for fact, score in pairs:
                # Apply trust discount.
                if (
                    get_trust
                    and fact.origin_agent_id
                    and fact.origin_agent_id != requesting_agent_id
                ):
                    score *= get_trust(fact.origin_agent_id)

                # Step 4: Score specificity and near-miss.
                spec = self._specificity.score(fact)
                nm_penalty = self._near_miss.penalty(fact)

                cand = CandidateFact(
                    fact=fact,
                    base_score=score,
                    facet_scores={facet.facet_id: score},
                    specificity_score=spec,
                    near_miss_penalty=nm_penalty,
                )
                candidates.append(cand)

            # Sort by final_score (includes specificity and near-miss adjustments).
            candidates.sort(key=lambda c: c.final_score, reverse=True)
            candidates_by_facet[facet.facet_id] = candidates

        # Cross-populate facet_scores: for each candidate, compute its
        # relevance to ALL facets (not just the one it was retrieved for).
        # This helps the assembler understand multi-facet coverage.
        all_candidates: dict[str, CandidateFact] = {}
        for facet_id, candidates in candidates_by_facet.items():
            for cand in candidates:
                fid = cand.fact.id
                if fid in all_candidates:
                    # Merge facet scores into existing candidate.
                    all_candidates[fid].facet_scores[facet_id] = cand.facet_scores.get(
                        facet_id, 0.0
                    )
                else:
                    all_candidates[fid] = cand

        # Rebuild per-facet lists using the merged candidates.
        merged_by_facet: dict[str, list[CandidateFact]] = {}
        for facet_id, candidates in candidates_by_facet.items():
            merged = [all_candidates[c.fact.id] for c in candidates]
            merged_by_facet[facet_id] = merged

        return merged_by_facet

    # ------------------------------------------------------------------
    # V3 pipeline
    # ------------------------------------------------------------------

    def recall_v3(
        self,
        query: str,
        *,
        analysis: QueryAnalysis,
        requesting_agent_id: str,
        active_facts: list[Fact],
        top_k: int = 5,
        get_trust: Callable[[str], float] | None = None,
    ) -> list[tuple[Fact, float]]:
        """V3 retrieval pipeline: flat harvest + optional bridge + RRF merge.

        Used for ASSEMBLY and INTEGRATION intents where the direct answer
        may require a concept bridge across two or more facts.

        Pipeline:
        1. Flat BM25 overfetch (3× top_k).
        2. Bridge harvest: extract high-IDF terms from top-N flat results,
           run second-hop BM25 per term, RRF-merge into bridge candidates.
        3. RRF merge of flat + bridge lists.
        4. Trust discount on other-agent facts.
        5. Top-k cutoff.

        Args:
            query: Raw query string.
            analysis: V3 QueryAnalysis (intent, exploration_mode).
            requesting_agent_id: Agent performing the query.
            active_facts: Active facts in the shared pool.
            top_k: Maximum results to return.
            get_trust: Optional trust function for provenance discount.

        Returns:
            List of (Fact, score) pairs.
        """
        overfetch = min(top_k * _FACET_OVERFETCH, len(active_facts))

        # Step 1: Flat BM25 baseline (PRF disabled — see _bm25_flat).
        flat = self._bm25_flat.search(query, active_facts, top_k=overfetch)

        # Step 2: Bridge harvest for ASSEMBLY / INTEGRATION.
        bridge: list[tuple[Fact, float]] = []
        if analysis.intent in (RetrievalIntent.ASSEMBLY, RetrievalIntent.INTEGRATION):
            terms = self._bridge.extract_bridge_terms(flat, top_n=3)
            if terms:
                bridge = self._bridge.second_hop(
                    terms,
                    active_facts,
                    top_k=min(top_k * 2, len(active_facts)),
                    exclude_ids={f.id for f, _ in flat},
                )
                logger.debug(
                    "recall_v3: bridge terms=%s found %d extra candidates",
                    terms,
                    len(bridge),
                )

        # Step 3: RRF merge of flat + bridge.
        merged = _rrf_merge(flat, bridge)

        # Step 4: Trust discount.
        if get_trust:
            discounted: list[tuple[Fact, float]] = []
            for fact, score in merged:
                if fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                    score = score * get_trust(fact.origin_agent_id)
                discounted.append((fact, score))
            discounted.sort(key=lambda x: x[1], reverse=True)
            merged = discounted

        return merged[:top_k]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_RRF_K = 60


def _rrf_merge(
    list_a: list[tuple[Fact, float]],
    list_b: list[tuple[Fact, float]],
) -> list[tuple[Fact, float]]:
    """Reciprocal Rank Fusion of two ranked (Fact, score) lists.

    Facts that appear in both lists accumulate rank contributions from each.
    list_a takes priority for facts appearing only in list_a.
    """
    scores: dict[str, float] = {}
    facts: dict[str, Fact] = {}

    for rank, (fact, _) in enumerate(list_a):
        scores[fact.id] = scores.get(fact.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        facts[fact.id] = fact

    for rank, (fact, _) in enumerate(list_b):
        scores[fact.id] = scores.get(fact.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        facts[fact.id] = fact

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(facts[fid], score) for fid, score in ranked]
