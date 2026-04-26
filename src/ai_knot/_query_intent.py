"""Pool query intent classification (rule-based, no LLM calls)."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ai_knot.multi_agent.models import RetrievalIntent
from ai_knot.multi_agent.router import (
    _MULTI_SOURCE_STEMS,
    _is_incident_query,
)
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact, MemoryType

if TYPE_CHECKING:
    from ai_knot._inverted_index import InvertedIndex

_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")

# Vocabulary signals for AGGREGATION intent: queries that need ALL mentions
# of a topic rather than a single best-matching fact.
_AGGREGATION_TOKENS = frozenset(
    {
        "list",
        "all",
        "every",
        "various",
        "different",
        "describe",
        "enumerate",
        "overview",
        "summary",
        "summariz",  # stem of "summarize"
        "summar",  # short stem variant
    }
)
_AGGREGATION_PHRASES = (
    "how many",
    "tell me about",
    "what are",
    "what does",
    "what did",
    "what has",
    "what have",
    "what do",
    "what were",
    "know about",
)


class _PoolQueryIntent(StrEnum):
    """Retrieval mode required by a shared-pool query.

    Names reflect what the retrieval system needs to do, not which scenario
    exercises the intent.  All routing decisions must be derivable from the
    query text, agent state, or candidate distribution — never from scenario
    metadata.
    """

    ENTITY_LOOKUP = "entity_lookup"  # Query targets a known entity — prefer canonical slot truth
    INCIDENT = "incident"  # Query is about events/timeline — prefer diversity + recency
    BROAD_DISCOVERY = "broad_discovery"  # Agent has thin local KB — cast wide net, flat weights
    MULTI_SOURCE = "multi_source"  # Query needs synthesis from multiple domains
    AGGREGATION = "aggregation"  # Query needs breadth — ALL mentions of a topic
    GENERAL = "general"  # Default — balanced retrieval


# Maps V3 RetrievalIntent → legacy _PoolQueryIntent for RRF weight selection.
_V3_INTENT_MAP: dict[RetrievalIntent, _PoolQueryIntent] = {
    RetrievalIntent.CANONICAL: _PoolQueryIntent.ENTITY_LOOKUP,
    RetrievalIntent.INCIDENT: _PoolQueryIntent.INCIDENT,
    RetrievalIntent.ASSEMBLY: _PoolQueryIntent.MULTI_SOURCE,
    RetrievalIntent.INTEGRATION: _PoolQueryIntent.MULTI_SOURCE,
    RetrievalIntent.GENERAL: _PoolQueryIntent.GENERAL,
}

# Intents that trigger canonical claim resolution before trust discount.
# WIDE (empty-KB) queries also run the resolver — conflict-signal gating inside
# ClaimFamilyResolver ensures only clusters with an explicit update marker are
# collapsed, so complementary facts are never accidentally eliminated.
_CANONICAL_RESOLVER_INTENTS = frozenset(
    {
        _PoolQueryIntent.ENTITY_LOOKUP,
        _PoolQueryIntent.AGGREGATION,
        _PoolQueryIntent.GENERAL,
    }
)

# Intent-aware RRF weights (BM25, slot-exact, trigram, importance, retention, recency).
# Default is (5.0, 3.0, 2.0, 1.5, 1.5, 1.0).
_INTENT_RRF_WEIGHTS: dict[_PoolQueryIntent, tuple[float, ...]] = {
    # Boost slot-exact for entity lookups — deterministic slot match > BM25 for known entities.
    _PoolQueryIntent.ENTITY_LOOKUP: (5.0, 8.0, 2.0, 1.5, 1.5, 1.0),
    # Boost recency for incidents — recent facts are more relevant.
    _PoolQueryIntent.INCIDENT: (5.0, 3.0, 2.0, 1.5, 1.5, 3.0),
    # Aggregation: lower BM25 (scoped search handles precision), boost importance,
    # suppress recency (want completeness not freshness).
    _PoolQueryIntent.AGGREGATION: (3.0, 8.0, 2.0, 2.5, 1.0, 0.5),
}

# Intent-aware pool rerank weights (recency_weight, freshness_weight).
_INTENT_RERANK_WEIGHTS: dict[_PoolQueryIntent, tuple[float, float]] = {
    _PoolQueryIntent.INCIDENT: (0.12, 0.05),
}

# Diversity cap per intent: maximum fraction of top-k from one agent.
_INTENT_DIVERSITY_CAP: dict[_PoolQueryIntent, float] = {
    _PoolQueryIntent.MULTI_SOURCE: 0.6,
    _PoolQueryIntent.BROAD_DISCOVERY: 0.4,
}


def _classify_pool_query(
    query: str,
    active_facts: list[Fact],
    *,
    requesting_agent_fact_count: int = -1,
    topic_channel: str = "",
) -> _PoolQueryIntent:
    """Classify a pool query by retrieval mode using observable signals.

    Signals evaluated in priority order:
    1. INCIDENT — time patterns or incident/error stems (content-based, highest priority).
    2. BROAD_DISCOVERY — agent has empty KB + diverse pool (agent-state-based).
    3. ENTITY_LOOKUP — query text mentions a known pool entity (length > 2).
    4. MULTI_SOURCE — cross-domain aggregation stems or long conjunctive queries.
    5. GENERAL — fallback.
    """
    tokens = set(_tokenize(query))
    q_lower = query.lower()

    # Signal 1: Time patterns or incident/error vocabulary — always takes priority.
    if _is_incident_query(tokens, q_lower):
        return _PoolQueryIntent.INCIDENT

    # Signal 1b: Strong conjunctive signal — comma-separated clauses with
    # aggregation vocabulary.  Takes priority over BROAD_DISCOVERY because
    # a clearly multi-facet query needs facet decomposition even when the
    # agent has no private facts (e.g. S26 querier agent).
    has_commas = "," in query
    has_agg_stem = bool(tokens & _MULTI_SOURCE_STEMS)
    if has_commas and has_agg_stem and len(query.split()) > 8:
        return _PoolQueryIntent.MULTI_SOURCE

    # Signal 2: Agent state — zero private facts AND diverse pool → broad discovery.
    # Skip for channel-scoped queries (narrow channel needs BM25 precision).
    # Require 3+ distinct publishers in the active pool — a thin pool (1-2 agents)
    # means the agent is a simple querier, not doing multi-source onboarding.
    if requesting_agent_fact_count == 0 and not topic_channel:
        pool_publishers = len({f.origin_agent_id for f in active_facts if f.origin_agent_id})
        if pool_publishers >= 3:
            return _PoolQueryIntent.BROAD_DISCOVERY

    # Signal 3: Entity-scoped queries.
    # 3a: Entity + aggregation vocabulary → AGGREGATION (breadth over depth).
    # 3b: Entity only → ENTITY_LOOKUP (precision).
    has_agg_signal = bool(tokens & _AGGREGATION_TOKENS) or any(
        p in q_lower for p in _AGGREGATION_PHRASES
    )
    for f in active_facts:
        if f.entity and len(f.entity) > 2 and f.entity.lower() in q_lower:
            if has_agg_signal:
                return _PoolQueryIntent.AGGREGATION
            return _PoolQueryIntent.ENTITY_LOOKUP

    # 3c: Aggregation signal without entity match — facts may lack structured
    # entity fields (e.g. add()-only ingestion).  Still apply breadth-oriented
    # RRF weights to improve coverage for enumeration questions.
    if has_agg_signal:
        return _PoolQueryIntent.AGGREGATION

    # Signal 4: Cross-domain aggregation vocabulary or long conjunctive query.
    if has_agg_stem or ("and" in tokens and len(query.split()) > 6):
        return _PoolQueryIntent.MULTI_SOURCE

    return _PoolQueryIntent.GENERAL


@dataclasses.dataclass(frozen=True, slots=True)
class _RecallMeta:
    """Internal metadata from the last pool recall (not part of public API).

    Exposes coverage and intent classification for downstream logic
    (e.g. coverage-aware abstention) without changing the public ``recall()``
    return type.
    """

    intent: _PoolQueryIntent
    total_active: int
    returned: int
    coverage: float  # fraction of returned results above _COVERAGE_SCORE_FLOOR
    low_coverage: bool  # True when coverage < 0.5


def _query_specificity(query: str, index: InvertedIndex) -> float:
    """Fraction of query tokens that are rare in this corpus (IDF > median).

    0.0 = broad/aggregation-like — few rare tokens (e.g. "what does Melanie do?").
    1.0 = narrow/point-like — many rare tokens (e.g. "when did Melanie sign up for pottery?").

    Purely corpus-derived via IDF statistics — no hardcoded vocabulary lists.
    Adapts to any language, domain, and corpus size.

    Args:
        query: The retrieval query.
        index: Inverted index built from the current corpus.

    Returns:
        Float in [0.0, 1.0].  Returns 0.5 when insufficient data
        (empty query or no corpus terms with positive IDF).
    """
    tokens = _tokenize(query)
    if not tokens:
        return 0.5
    idf_values = [v for t in tokens if (v := index.idf(t)) > 0]
    if not idf_values:
        return 0.5
    med = index.median_idf()
    n_rare = sum(1 for v in idf_values if v > med)
    return n_rare / len(idf_values)


# ---------------------------------------------------------------------------
# Single-agent recall intent classification (Phase E — Enterprise Router)
# ---------------------------------------------------------------------------

# Functional stopwords — copied from multi_agent.router._STOP_SHORT to avoid
# circular imports between single-agent and pool-routing modules.
_STOP_SHORT_RECALL: frozenset[str] = frozenset(
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

# Phrases that signal PROCEDURAL intent (instructions, rules, workflows).
_PROCEDURAL_PHRASES: tuple[str, ...] = (
    "how to",
    "steps to",
    "how do i",
)
_PROCEDURAL_TOKENS: frozenset[str] = frozenset(
    {"rule", "policy", "procedure", "guideline", "deploy", "instructions", "instruction"}
)

# Phrases/tokens that signal NAVIGATIONAL intent (file/log/artifact search).
_NAVIGATIONAL_PHRASES: tuple[str, ...] = ("meeting notes",)
_NAVIGATIONAL_TOKENS: frozenset[str] = frozenset(
    {"find", "show", "open", "file", "document", "log", "transcript", "report"}
)

# Phrases/tokens that signal EXPLORATORY intent (why/how/timeline/connections).
_EXPLORATORY_PHRASES: tuple[str, ...] = (
    "how does",
    "how did",
    "before",
    "after",
    "between",
    "during",
    "history",
    "related",
    "connection",
)
_EXPLORATORY_TOKENS: frozenset[str] = frozenset({"why"})

# Long query threshold for EXPLORATORY fallback.
_EXPLORATORY_LENGTH_THRESHOLD: int = 10


class RecallIntent(StrEnum):
    """Retrieval intent for single-agent (KnowledgeBase) recall pipeline.

    Distinct from _PoolQueryIntent (shared pool) — do not conflate.
    """

    FACTUAL = "factual"  # Single-hop point queries: "When did X sign up?"
    AGGREGATIONAL = "aggregational"  # Breadth queries: "List all X", "What activities?"
    EXPLORATORY = "exploratory"  # Temporal / relational: "Why", "How", timeline
    NAVIGATIONAL = "navigational"  # Artifact search: "Find transcript of meeting Y"
    PROCEDURAL = "procedural"  # Instructions / rules: "How to deploy?"
    BROAD_CONTEXT = "broad_context"  # Vague / short: "pricing?", "status?"


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Per-intent retrieval pipeline configuration for KnowledgeBase._execute_recall.

    Attributes:
        skip_prf: When True, skip Pseudo-Relevance Feedback expansion.
        rrf_weights: Six RRF signal weights in order:
            (BM25, slot-exact, trigram, importance, retention, recency).
        mmr_lambda: MMR relevance weight (0 = full diversity, 1 = no dedup).
        use_ddsa: Whether to run spreading-activation expansion (Stage 4a).
        sort_strategy: Final sort mode for recall() — 'relevance', 'chronological',
            or 'sandwich'.
        memory_type_filter: When set, exclude facts of all other MemoryTypes.
        field_weights_override: Optional BM25F field weight overrides passed to
            InvertedIndex.score(). Keys: 'content', 'tags', 'canonical', 'evidence'.
    """

    skip_prf: bool
    rrf_weights: tuple[float, float, float, float, float, float]
    mmr_lambda: float
    use_ddsa: bool
    sort_strategy: str  # 'relevance' | 'chronological' | 'sandwich'
    memory_type_filter: MemoryType | None = None
    field_weights_override: dict[str, float] | None = None
    lexical_expansion_max: int = 8  # max expansion terms for Lexical Bridge (0 = disabled)


# Registry mapping RecallIntent → PipelineConfig.
# RRF weight order: (BM25, slot-exact, trigram, importance, retention, recency).
_PIPELINE_CONFIGS: dict[RecallIntent, PipelineConfig] = {
    RecallIntent.FACTUAL: PipelineConfig(
        skip_prf=True,
        rrf_weights=(10.0, 5.0, 2.0, 0.5, 0.5, 0.0),
        mmr_lambda=0.85,
        use_ddsa=False,
        sort_strategy="relevance",
        lexical_expansion_max=8,
    ),
    RecallIntent.AGGREGATIONAL: PipelineConfig(
        skip_prf=False,
        rrf_weights=(3.0, 2.0, 2.0, 3.0, 2.0, 2.0),
        mmr_lambda=0.3,
        use_ddsa=False,
        sort_strategy="sandwich",
        lexical_expansion_max=12,
    ),
    RecallIntent.EXPLORATORY: PipelineConfig(
        skip_prf=False,
        rrf_weights=(5.0, 3.0, 2.0, 2.0, 2.0, 4.0),
        mmr_lambda=0.65,
        use_ddsa=True,
        sort_strategy="chronological",
        lexical_expansion_max=10,
    ),
    RecallIntent.NAVIGATIONAL: PipelineConfig(
        skip_prf=True,
        rrf_weights=(2.0, 1.0, 8.0, 0.0, 0.0, 5.0),
        mmr_lambda=0.9,
        use_ddsa=False,
        sort_strategy="relevance",
        field_weights_override={"tags": 5.0, "canonical": 3.0},
        lexical_expansion_max=0,  # NAVIGATIONAL never gets expansion
    ),
    RecallIntent.PROCEDURAL: PipelineConfig(
        skip_prf=False,
        rrf_weights=(8.0, 4.0, 2.0, 5.0, 0.0, 0.0),
        mmr_lambda=0.7,
        use_ddsa=False,
        sort_strategy="relevance",
        # memory_type_filter is intentionally None here: auto-filtering by MemoryType
        # from query intent alone is too aggressive and silently drops SEMANTIC facts
        # about deployments, policies etc. stored via kb.add().
        # Enterprise-only isolation should be enforced at the KnowledgeBase level, not here.
        lexical_expansion_max=6,
    ),
    RecallIntent.BROAD_CONTEXT: PipelineConfig(
        skip_prf=True,
        rrf_weights=(3.0, 1.0, 1.0, 6.0, 5.0, 2.0),
        mmr_lambda=0.5,
        use_ddsa=True,
        sort_strategy="relevance",
        lexical_expansion_max=10,
    ),
}


def classify_recall_intent(query: str) -> RecallIntent:
    """Classify a single-agent recall query into one of six retrieval intents.

    Rule-based, no LLM calls.  Priority order (first match wins):
    1. BROAD_CONTEXT — very short query (≤2 content tokens).
    2. PROCEDURAL — instruction / rule / workflow vocabulary.
    3. NAVIGATIONAL — artifact / file / log search vocabulary.
    4. AGGREGATIONAL — breadth signals (list, all, every, etc.).
    5. EXPLORATORY — relational / temporal vocabulary or long query (≥10 tokens).
    6. FACTUAL — fallback for all point-like queries.

    Args:
        query: The raw search query string.

    Returns:
        A :class:`RecallIntent` value.
    """
    q_lower = query.lower()
    tokens = _tokenize(q_lower)
    tok_set = set(tokens)
    content_tokens = tok_set - _STOP_SHORT_RECALL

    # 1. BROAD_CONTEXT — nearly empty queries (≤1 content token) lack routing signal.
    # Threshold is intentionally low: entity+attribute (e.g. "alice salary") = 2 tokens
    # and should route to FACTUAL, not BROAD_CONTEXT.
    if len(content_tokens) <= 1:
        return RecallIntent.BROAD_CONTEXT

    # 2. PROCEDURAL — instructions, rules, workflows, deployment.
    if any(p in q_lower for p in _PROCEDURAL_PHRASES) or bool(tok_set & _PROCEDURAL_TOKENS):
        return RecallIntent.PROCEDURAL

    # 3. NAVIGATIONAL — find/open/show specific artifacts.
    if any(p in q_lower for p in _NAVIGATIONAL_PHRASES) or bool(tok_set & _NAVIGATIONAL_TOKENS):
        return RecallIntent.NAVIGATIONAL

    # 4. AGGREGATIONAL — breadth queries (reuse pool-path vocabulary).
    has_agg_signal = bool(tok_set & _AGGREGATION_TOKENS) or any(
        p in q_lower for p in _AGGREGATION_PHRASES
    )
    if has_agg_signal:
        return RecallIntent.AGGREGATIONAL

    # 5. EXPLORATORY — relational / temporal / long queries.
    has_exploratory = any(p in q_lower for p in _EXPLORATORY_PHRASES) or bool(
        tok_set & _EXPLORATORY_TOKENS
    )
    if has_exploratory or len(tokens) >= _EXPLORATORY_LENGTH_THRESHOLD:
        return RecallIntent.EXPLORATORY

    # 6. FACTUAL — point queries, default.
    return RecallIntent.FACTUAL


def get_pipeline_config(intent: RecallIntent) -> PipelineConfig:
    """Return the :class:`PipelineConfig` for the given recall intent."""
    return _PIPELINE_CONFIGS[intent]
