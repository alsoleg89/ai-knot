"""Coverage-aware fact assembly for multi-source queries.

Selects top-k facts from per-facet candidate lists using a greedy
set-cover algorithm that maximises:
- facet coverage (primary objective)
- source diversity (per-agent cap)
- specificity (prefer implementation facts over overviews)

This is the heart of the S26 fix: instead of one flat BM25 search
over the whole pool, each facet is searched independently and the
assembler picks the best combination.
"""

from __future__ import annotations

from ai_knot.multi_agent.models import AssemblyResult, CandidateFact
from ai_knot.multi_agent.scoring import DiversityPolicy


class CoverageAwareAssembler:
    """Greedy max-coverage assembler for multi-facet retrieval.

    Selection strategy (greedy set-cover):
    1. For each uncovered facet, pick the highest-scoring candidate.
    2. After covering all facets (or exhausting candidates), backfill
       remaining top-k slots with next-best candidates.
    3. Apply per-agent diversity cap throughout.
    """

    def __init__(self, *, diversity: DiversityPolicy | None = None) -> None:
        self._diversity = diversity or DiversityPolicy()

    def assemble(
        self,
        *,
        candidates_by_facet: dict[str, list[CandidateFact]],
        top_k: int,
        n_publishers: int = 3,
    ) -> AssemblyResult:
        """Select top-k facts maximising facet coverage and diversity.

        Args:
            candidates_by_facet: Per-facet candidate lists, pre-sorted by
                score (best first).  Key is facet_id.
            top_k: Maximum number of facts to select.
            n_publishers: Number of distinct trusted publishers in the pool
                (used for per-agent cap calculation).

        Returns:
            AssemblyResult with selected candidates and coverage metrics.
        """
        all_facet_ids = set(candidates_by_facet.keys())
        if not all_facet_ids:
            return AssemblyResult(
                uncovered_facets=set(),
                coverage_score=0.0,
            )

        agent_cap = self._diversity.per_agent_cap(top_k=top_k, n_publishers=n_publishers)

        selected: list[CandidateFact] = []
        selected_fact_ids: set[str] = set()
        agent_counts: dict[str, int] = {}
        covered_facets: set[str] = set()

        # Phase 1: Greedy coverage — pick best candidate for each uncovered facet.
        # Process facets in order of fewest available candidates first
        # (most constrained first heuristic).
        facet_order = sorted(
            all_facet_ids,
            key=lambda fid: len(candidates_by_facet.get(fid, [])),
        )

        for facet_id in facet_order:
            if facet_id in covered_facets:
                continue
            if len(selected) >= top_k:
                break

            candidates = candidates_by_facet.get(facet_id, [])
            # Sort by final_score so the best candidate is picked first,
            # not just the first in insertion order.
            sorted_cands = sorted(candidates, key=lambda c: c.final_score, reverse=True)
            for cand in sorted_cands:
                fid = cand.fact.id
                aid = cand.fact.origin_agent_id or "__self__"

                if fid in selected_fact_ids:
                    # Already selected via another facet — still counts as coverage.
                    covered_facets.add(facet_id)
                    break

                if agent_counts.get(aid, 0) >= agent_cap:
                    continue

                selected.append(cand)
                selected_fact_ids.add(fid)
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
                covered_facets.add(facet_id)
                break

        # Phase 2: Backfill remaining slots from all candidates, ranked by
        # final_score, respecting diversity cap.
        if len(selected) < top_k:
            # Collect all unused candidates across facets, deduplicated.
            all_remaining: list[CandidateFact] = []
            seen_ids: set[str] = set()
            for candidates in candidates_by_facet.values():
                for cand in candidates:
                    fid = cand.fact.id
                    if fid not in selected_fact_ids and fid not in seen_ids:
                        all_remaining.append(cand)
                        seen_ids.add(fid)

            all_remaining.sort(key=lambda c: c.final_score, reverse=True)

            for cand in all_remaining:
                if len(selected) >= top_k:
                    break
                aid = cand.fact.origin_agent_id or "__self__"
                if agent_counts.get(aid, 0) >= agent_cap:
                    continue
                selected.append(cand)
                selected_fact_ids.add(cand.fact.id)
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
                # Check if this backfill candidate covers any uncovered facets.
                for fid, score in cand.facet_scores.items():
                    if score > 0 and fid not in covered_facets:
                        covered_facets.add(fid)

        uncovered = all_facet_ids - covered_facets
        coverage = len(covered_facets) / len(all_facet_ids) if all_facet_ids else 0.0

        return AssemblyResult(
            selected=selected,
            covered_facets=covered_facets,
            uncovered_facets=uncovered,
            coverage_score=coverage,
        )
