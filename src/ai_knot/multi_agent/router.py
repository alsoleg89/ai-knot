"""Query-shape routing for shared pool retrieval.

Classifies pool queries by intent and produces a ``RoutedPoolQuery``
with routing decisions (facet decomposition, expertise routing, etc.).

Intent classification rules are extracted from ``knowledge.py`` and
kept deterministic — no LLM calls.

V3 additions:
- ``classify_exploration_mode``: separates search-breadth (WIDE/BALANCED/PRECISE)
  from semantic intent. Empty-KB state sets exploration mode, not query meaning.
- ``classify_intent``: semantic intent derived from query content only.
- ``QueryShapeRouter.analyze``: returns ``QueryAnalysis`` for the V3 pipeline.
"""

from __future__ import annotations

import re

from ai_knot.multi_agent.models import (
    ExplorationMode,
    QueryAnalysis,
    RetrievalIntent,
    RoutedPoolQuery,
)
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# Intent constants — match the names used in knowledge.py _PoolQueryIntent.
ENTITY_LOOKUP = "entity_lookup"
INCIDENT = "incident"
BROAD_DISCOVERY = "broad_discovery"
MULTI_SOURCE = "multi_source"
GENERAL = "general"

_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")

_INCIDENT_STRONG_STEMS = frozenset(
    stem
    for kw in {"error", "outage", "incident", "alert", "failure", "timeout"}
    for stem in _tokenize(kw)
)
_INCIDENT_WEAK_STEMS = frozenset(stem for kw in {"deploy", "rollout"} for stem in _tokenize(kw))
_INCIDENT_STEMS = _INCIDENT_STRONG_STEMS | _INCIDENT_WEAK_STEMS

_MULTI_SOURCE_STEMS = frozenset(
    stem
    for kw in {"include", "pricing", "sla", "tier", "region", "integrate", "across", "compare"}
    for stem in _tokenize(kw)
)


def _is_incident_query(tokens: set[str], q_lower: str) -> bool:
    """Return True if the query signals an incident intent.

    A time pattern (e.g. "10:30") or a strong incident stem (error, outage…)
    is sufficient on its own.  Weak stems (deploy, rollout) only qualify when
    paired with one of the above — prevents "How do teams deploy?" → INCIDENT.
    """
    return bool(_TIME_PATTERN.search(q_lower) or tokens & _INCIDENT_STRONG_STEMS)


# Stopword token set used by exploration mode classification —
# single-character and purely functional words that do not carry domain meaning.
_STOP_SHORT: frozenset[str] = frozenset(
    stem
    for word in {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "it",
        "its",
        "this",
        "that",
        "and",
        "but",
        "or",
        "not",
        "no",
        "what",
        "who",
        "how",
        "when",
        "where",
        "which",
    }
    for stem in _tokenize(word)
)


def classify_exploration_mode(
    query: str,
    requesting_agent_fact_count: int,
    pool_publishers: int,
) -> ExplorationMode:
    """Classify how broadly retrieval should search based on agent/pool state.

    WIDE is assigned only when the querier has an empty KB, the pool has 3+
    publishers, AND the query itself lacks domain-specific vocabulary — a vague
    broad question from an agent with no private context.

    If the query contains domain vocabulary (incident/multi-source stems),
    entity-like tokens (>4 chars, alphabetic), or is long (>8 content tokens),
    it returns BALANCED even if the KB is empty — the query is specific enough
    to use normal intent-driven retrieval.
    """
    if requesting_agent_fact_count > 0:
        return ExplorationMode.BALANCED
    if pool_publishers < 3:
        return ExplorationMode.BALANCED

    # KB is empty and pool is diverse — check query specificity.
    # WIDE is assigned only when the query lacks clear domain vocabulary AND
    # is short (≤8 content tokens).  We deliberately do NOT gate on
    # entity-like tokens (long alpha tokens) because stemmed common words
    # such as "manag", "servic", "deploy" are >4 chars and would incorrectly
    # classify genuinely vague queries (e.g. "Who manages the OIDC service?")
    # as BALANCED — causing the canonical resolver to fire on empty-KB agents.
    tokens = set(_tokenize(query))
    has_domain_vocab = bool(tokens & (_MULTI_SOURCE_STEMS | _INCIDENT_STEMS))
    content_tokens = tokens - _STOP_SHORT
    if has_domain_vocab or len(content_tokens) > 8:
        return ExplorationMode.BALANCED
    return ExplorationMode.WIDE


def classify_intent(
    query: str,
    active_facts: list[Fact],
    exploration_mode: ExplorationMode,  # noqa: ARG001  (reserved for future use)
) -> RetrievalIntent:
    """Classify the semantic intent of a pool query from query content only.

    Intent is never derived from agent KB state — that information belongs
    in ExplorationMode.  This ensures that an empty-KB agent asking "Who owns
    the OIDC service?" gets CANONICAL intent, not BROAD_DISCOVERY.

    Priority:
    1. INCIDENT — time patterns or incident/error vocabulary.
    2. ASSEMBLY — multi-source aggregation stems or long conjunctive query.
    3. CANONICAL — query mentions a known pool entity.
    4. GENERAL — fallback.
    """
    tokens = set(_tokenize(query))
    q_lower = query.lower()

    if _is_incident_query(tokens, q_lower):
        return RetrievalIntent.INCIDENT

    if tokens & _MULTI_SOURCE_STEMS or ("and" in tokens and len(query.split()) > 6):
        return RetrievalIntent.ASSEMBLY

    for f in active_facts:
        if f.entity and len(f.entity) > 2 and f.entity.lower() in q_lower:
            return RetrievalIntent.CANONICAL

    return RetrievalIntent.GENERAL


def classify_pool_query(
    query: str,
    active_facts: list[Fact],
    *,
    requesting_agent_fact_count: int = -1,
    topic_channel: str = "",
) -> str:
    """Classify a pool query by retrieval intent using observable signals.

    Signals evaluated in priority order:
    1. INCIDENT — time patterns or incident/error stems.
    2. BROAD_DISCOVERY — agent has empty KB + diverse pool.
    3. ENTITY_LOOKUP — query mentions a known pool entity.
    4. MULTI_SOURCE — cross-domain aggregation stems or long conjunctive queries.
    5. GENERAL — fallback.

    Returns:
        One of: "entity_lookup", "incident", "broad_discovery",
        "multi_source", "general".
    """
    tokens = set(_tokenize(query))
    q_lower = query.lower()

    # Signal 1: Time patterns or incident/error vocabulary.
    if _is_incident_query(tokens, q_lower):
        return INCIDENT

    # Signal 1b: Strong conjunctive signal — comma-separated clauses with
    # aggregation vocabulary.  Takes priority over BROAD_DISCOVERY because
    # a clearly multi-facet query needs facet decomposition even when the
    # agent has no private facts (e.g. S26 querier agent).
    has_commas = "," in query
    has_agg_stem = bool(tokens & _MULTI_SOURCE_STEMS)
    if has_commas and has_agg_stem and len(query.split()) > 8:
        return MULTI_SOURCE

    # Signal 2: Agent state — zero private facts AND diverse pool.
    if requesting_agent_fact_count == 0 and not topic_channel:
        pool_publishers = len({f.origin_agent_id for f in active_facts if f.origin_agent_id})
        if pool_publishers >= 3:
            return BROAD_DISCOVERY

    # Signal 3: Query mentions a known pool entity.
    for f in active_facts:
        if f.entity and len(f.entity) > 2 and f.entity.lower() in q_lower:
            return ENTITY_LOOKUP

    # Signal 4: Cross-domain aggregation vocabulary or long conjunctive query.
    if has_agg_stem or ("and" in tokens and len(query.split()) > 6):
        return MULTI_SOURCE

    return GENERAL


class QueryShapeRouter:
    """Route pool queries to the appropriate retrieval strategy.

    Produces a ``RoutedPoolQuery`` with intent classification and
    routing flags.  Does not perform facet decomposition itself —
    that is handled by ``ConjunctiveFacetPlanner`` downstream.
    """

    def route(
        self,
        query: str,
        *,
        requesting_agent_id: str,
        active_facts: list[Fact],
        requesting_agent_fact_count: int,
        topic_channel: str = "",
    ) -> RoutedPoolQuery:
        """Classify query intent and set routing flags.

        Args:
            query: The raw query string.
            requesting_agent_id: Agent performing the query.
            active_facts: Currently active facts in the shared pool.
            requesting_agent_fact_count: Number of active facts in the
                requesting agent's private KB.
            topic_channel: Optional topic channel filter.

        Returns:
            A ``RoutedPoolQuery`` with intent and routing flags set.
        """
        intent = classify_pool_query(
            query,
            active_facts,
            requesting_agent_fact_count=requesting_agent_fact_count,
            topic_channel=topic_channel,
        )

        return RoutedPoolQuery(
            raw_query=query,
            intent=intent,
            topic_channel=topic_channel,
            # Facet decomposition is only useful for MULTI_SOURCE.
            # The facets tuple is populated downstream by ConjunctiveFacetPlanner.
            use_expertise_routing=intent == MULTI_SOURCE,
        )

    def analyze(
        self,
        query: str,
        *,
        requesting_agent_id: str,
        active_facts: list[Fact],
        requesting_agent_fact_count: int,
        topic_channel: str = "",
    ) -> QueryAnalysis:
        """V3 query analysis: separate semantic intent from exploration mode.

        Args:
            query: The raw query string.
            requesting_agent_id: Agent performing the query (unused here,
                reserved for future per-agent analysis).
            active_facts: Currently active facts in the shared pool.
            requesting_agent_fact_count: Number of active facts in the
                requesting agent's private KB.
            topic_channel: Optional topic channel filter.

        Returns:
            ``QueryAnalysis`` with intent and exploration_mode separated.
        """
        _ = requesting_agent_id  # reserved for future use
        pool_publishers = len({f.origin_agent_id for f in active_facts if f.origin_agent_id})
        mode = classify_exploration_mode(query, requesting_agent_fact_count, pool_publishers)
        intent = classify_intent(query, active_facts, mode)
        return QueryAnalysis(
            raw_query=query,
            intent=intent,
            exploration_mode=mode,
        )
