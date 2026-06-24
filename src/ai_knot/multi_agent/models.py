"""Shared dataclasses for the multi-agent retrieval pipeline.

These types are local to the multi_agent package — they do not
overload the core ``Fact`` dataclass with planner-only fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ai_knot.types import Fact

# ---------------------------------------------------------------------------
# V3 query classification types
# ---------------------------------------------------------------------------


class RetrievalIntent(StrEnum):
    """Semantic intent of the query — what the agent is asking for.

    Derived from query content, never from agent KB state.
    """

    CANONICAL = "canonical"  # Single authoritative answer (entity/slot lookup)
    INCIDENT = "incident"  # Timeline of events — prefer diversity + recency
    ASSEMBLY = "assembly"  # Synthesis from multiple domains
    INTEGRATION = "integration"  # Relay / cross-domain relationship
    GENERAL = "general"  # Default — balanced retrieval


class ExplorationMode(StrEnum):
    """How broadly the retrieval should search.

    Derived from agent state and pool context — never from query semantics.
    WIDE replaces the old BROAD_DISCOVERY intent, keeping state-based
    search-width separate from semantic meaning.
    """

    PRECISE = "precise"  # High-confidence entity/slot match
    BALANCED = "balanced"  # Most queries
    WIDE = "wide"  # Agent has empty KB + vague query → cast wide net


@dataclass(slots=True)
class QueryAnalysis:
    """Result of V3 query analysis: intent + exploration mode.

    Separates *what the query means* (intent) from *how broadly to search*
    (exploration_mode), so empty-KB state does not override query semantics.
    """

    raw_query: str
    intent: RetrievalIntent
    exploration_mode: ExplorationMode
    bridge_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class QueryFacet:
    """A single independent retrieval facet extracted from a conjunctive query.

    Attributes:
        facet_id: Short identifier (e.g. "f0", "f1").
        text: The facet query text used for BM25 retrieval.
        tokens: Stemmed tokens for the facet (via shared tokenizer).
        facet_type: Classification hint — "domain", "entity", "constraint",
            "time", or "general".
        weight: Relative importance of this facet (default 1.0).
    """

    facet_id: str
    text: str
    tokens: tuple[str, ...]
    facet_type: str = "general"
    weight: float = 1.0


@dataclass(slots=True)
class RoutedPoolQuery:
    """Query after intent classification and routing decisions.

    Attributes:
        raw_query: The original user query string.
        intent: Classified intent (ENTITY_LOOKUP, INCIDENT, BROAD_DISCOVERY,
            MULTI_SOURCE, GENERAL).
        facets: Decomposed facets (empty tuple for non-MULTI_SOURCE intents).
        topic_channel: Optional topic channel filter.
        use_expertise_routing: Whether to use agent expertise index.
        use_insight_boost: Whether to boost from team insight store.
        use_llm_expansion: Whether LLM query expansion is available.
    """

    raw_query: str
    intent: str
    facets: tuple[QueryFacet, ...] = ()
    topic_channel: str = ""
    use_expertise_routing: bool = False
    use_insight_boost: bool = False
    use_llm_expansion: bool = False


@dataclass(slots=True)
class CandidateFact:
    """A fact scored across multiple dimensions during assembly.

    Attributes:
        fact: The underlying Fact from the shared pool.
        base_score: Raw retrieval score (BM25/RRF after trust discount).
        facet_scores: Per-facet relevance scores {facet_id: score}.
        specificity_score: How implementation-specific vs overview-like (0-1).
        near_miss_penalty: Penalty for overview/near-miss facts (0-1, higher=worse).
        expertise_boost: Boost from agent expertise routing (0+).
    """

    fact: Fact
    base_score: float = 0.0
    facet_scores: dict[str, float] = field(default_factory=dict)
    specificity_score: float = 0.5
    near_miss_penalty: float = 0.0
    expertise_boost: float = 0.0

    @property
    def final_score(self) -> float:
        """Composite score used for assembly ranking."""
        return (
            self.base_score
            * (1.0 + self.expertise_boost)
            * (1.0 + 0.5 * self.specificity_score)
            * (1.0 - 0.5 * self.near_miss_penalty)
        )

    @property
    def best_facet_id(self) -> str:
        """Return the facet this candidate is most relevant to."""
        if not self.facet_scores:
            return ""
        return max(self.facet_scores, key=self.facet_scores.get)  # type: ignore[arg-type]


@dataclass(slots=True)
class AssemblyResult:
    """Result of coverage-aware fact assembly.

    Attributes:
        selected: Chosen candidates in final ranking order.
        covered_facets: Facet IDs with at least one selected candidate.
        uncovered_facets: Facet IDs with no selected candidate.
        coverage_score: Fraction of facets covered (0.0-1.0).
    """

    selected: list[CandidateFact] = field(default_factory=list)
    covered_facets: set[str] = field(default_factory=set)
    uncovered_facets: set[str] = field(default_factory=set)
    coverage_score: float = 0.0
