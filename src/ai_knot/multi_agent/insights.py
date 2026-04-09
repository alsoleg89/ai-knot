"""Team-level insight store for reusable multi-agent knowledge.

Phase 1: in-memory only, derived from successful assemblies.
Behind a flag (disabled by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import uuid4

from ai_knot.multi_agent.models import AssemblyResult
from ai_knot.tokenizer import tokenize as _tokenize


@dataclass(slots=True)
class TeamInsight:
    """A reusable team-level insight derived from a successful assembly."""

    insight_id: str
    summary: str
    tokens: tuple[str, ...]  # stemmed tokens for retrieval
    supporting_fact_ids: tuple[str, ...]
    supporting_agents: tuple[str, ...]
    tags: tuple[str, ...] = ()
    reuse_count: int = 0


class TeamInsightStore:
    """In-memory store for reusable team-level insights.

    Phase 1: in-memory only, derived from successful assemblies.

    Use carefully:
    - Do NOT store every answer as an insight.
    - Only promote stable, high-coverage assemblies.
    """

    def __init__(self, *, min_coverage_to_promote: float = 0.8) -> None:
        self._insights: dict[str, TeamInsight] = {}
        self._min_coverage = min_coverage_to_promote

    def remember(self, insight: TeamInsight) -> None:
        """Store a team insight."""
        self._insights[insight.insight_id] = insight

    def retrieve(self, query: str, *, top_k: int = 5) -> list[TeamInsight]:
        """Find insights relevant to query using token overlap scoring.

        Simple scoring: Jaccard similarity between query tokens and insight
        tokens.  Boost by reuse_count (log scale).
        """
        if not self._insights:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, TeamInsight]] = []
        for insight in self._insights.values():
            insight_tokens = set(insight.tokens)
            if not insight_tokens:
                continue
            intersection = query_tokens & insight_tokens
            union = query_tokens | insight_tokens
            jaccard = len(intersection) / len(union)
            if jaccard <= 0.0:
                continue
            # Boost by reuse_count (log scale, +1 to avoid log(0)).
            boost = 1.0 + 0.1 * math.log1p(insight.reuse_count)
            scored.append((jaccard * boost, insight))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [insight for _, insight in scored[:top_k]]

    def promote_from_assembly(
        self, result: AssemblyResult, *, query: str = ""
    ) -> TeamInsight | None:
        """Promote a successful assembly result to a team insight.

        Only promotes if:
        - coverage_score >= min_coverage_to_promote
        - at least 2 distinct agents contributed
        - the assembly selected >= 2 facts

        Returns the created insight, or None if criteria not met.
        """
        if result.coverage_score < self._min_coverage:
            return None

        if len(result.selected) < 2:
            return None

        agents = {c.fact.origin_agent_id for c in result.selected}
        if len(agents) < 2:
            return None

        fact_ids = tuple(c.fact.id for c in result.selected)
        summary = " | ".join(c.fact.content for c in result.selected)
        tokens = tuple(_tokenize(summary + " " + query))

        insight = TeamInsight(
            insight_id=uuid4().hex[:8],
            summary=summary,
            tokens=tokens,
            supporting_fact_ids=fact_ids,
            supporting_agents=tuple(sorted(agents)),
        )
        self.remember(insight)
        return insight

    @property
    def count(self) -> int:
        """Number of stored insights."""
        return len(self._insights)

    def clear(self) -> None:
        """Remove all stored insights."""
        self._insights.clear()
