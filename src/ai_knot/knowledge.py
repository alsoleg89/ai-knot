"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import math
import os
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from ai_knot.extractor import (
    Extractor,
    resolve_against_existing,
    resolve_by_slot,
    resolve_structured,
)
from ai_knot.forgetting import apply_decay
from ai_knot.multi_agent.canonical import ClaimFamilyResolver
from ai_knot.multi_agent.models import ExplorationMode, RetrievalIntent
from ai_knot.multi_agent.recall_service import SharedPoolRecallService
from ai_knot.multi_agent.router import (
    _MULTI_SOURCE_STEMS,
    QueryShapeRouter,
    _is_incident_query,
)
from ai_knot.providers import LLMProvider, create_provider
from ai_knot.query_expander import LLMQueryExpander
from ai_knot.retriever import BM25Retriever, DenseRetriever, HybridRetriever, TFIDFRetriever
from ai_knot.storage.base import (
    AtomicUpdateCapable,
    SnapshotCapable,
    StorageBackend,
    TemporalStorageCapable,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import (
    CONFLICT_POLICIES,
    ConversationTurn,
    Fact,
    MemoryOp,
    MemoryType,
    MESIState,
    SlotDelta,
    SnapshotDiff,
)

# LLM expansion token weight — higher than PRF (0.5) to reflect
# the semantic advantage of LLM-based query understanding.
_LLM_EXPANSION_WEIGHT: float = 0.6

_SHARED_NAMESPACE = "__shared__"
_PROVENANCE_DISCOUNT = 0.8
# Facts superseded by a different agent within this window count as "quick invalidations".
_QUICK_INV_WINDOW_S = 3600.0  # 1 hour
# Over-fetch multiplier for shared-pool recall: fetch N×top_k before trust discount so that
# high-scoring low-trust facts don't crowd out lower-scoring high-trust ones before discounting.
_POOL_RECALL_OVERFETCH = 3
_POOL_DEBUG = bool(os.environ.get("AI_KNOT_POOL_DEBUG", ""))
_LEARN_DEBUG = bool(os.environ.get("AI_KNOT_LEARN_DEBUG", ""))

logger = logging.getLogger(__name__)

# Minimum score threshold: results below this are considered non-relevant filler.
# Computed empirically — a BM25 match on a single common term scores ~0.02.
_COVERAGE_SCORE_FLOOR: float = 0.01

# ---------------------------------------------------------------------------
# Pool query intent classification (rule-based, no LLM calls)
# ---------------------------------------------------------------------------

_TIME_PATTERN = re.compile(r"\d{1,2}:\d{2}")


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

    # Signal 3: Query mentions a known pool entity (guard len > 2 to filter noise).
    for f in active_facts:
        if f.entity and len(f.entity) > 2 and f.entity.lower() in q_lower:
            return _PoolQueryIntent.ENTITY_LOOKUP

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


class KnowledgeBase:
    """Agent knowledge store with extraction, retrieval, and forgetting.

    Usage::

        kb = KnowledgeBase(agent_id="my_agent")
        kb.add("User prefers Python", importance=0.9)
        context = kb.recall("what language?")
        # → "[procedural] User prefers Python"

        # Configure provider once at init — no need to repeat credentials:
        kb = KnowledgeBase(agent_id="my_agent", provider="openai", api_key="sk-...")
        kb.learn(turns)
        kb.learn(more_turns)

    Args:
        agent_id: Unique identifier for this agent's memory namespace.
        storage: Storage backend (defaults to YAMLStorage in .ai_knot/).
        provider: Default LLM provider name or instance used by :meth:`learn`.
            When set, ``learn()`` calls do not need to repeat the provider name.
        api_key: Default API key for the provider. Falls back to environment
            variables when not set.
        model: Default model override for the provider.
        **provider_kwargs: Extra provider arguments stored as defaults for
            ``learn()`` (e.g. ``folder_id`` for Yandex, ``base_url`` for
            openai-compat).
    """

    def __init__(
        self,
        agent_id: str,
        storage: StorageBackend | None = None,
        *,
        provider: str | LLMProvider | None = None,
        api_key: str | None = None,
        model: str | None = None,
        decay_config: dict[str, float] | None = None,
        llm_recall: bool = False,
        rrf_weights: tuple[float, ...] | None = None,
        expansion_weight: float | None = None,
        episodic_ttl_hours: float = 72.0,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        **provider_kwargs: str,
    ) -> None:
        self._agent_id = agent_id
        self._storage: StorageBackend = storage or YAMLStorage()
        self._bm25 = TFIDFRetriever(
            rrf_weights=rrf_weights or (5.0, 3.0, 2.0, 1.5, 1.5, 1.0),
        )
        self._dense = DenseRetriever()
        self._hybrid = HybridRetriever(self._bm25, self._dense)
        self._retriever: TFIDFRetriever | HybridRetriever = self._bm25
        self._embed_url = embed_url
        self._embed_model = embed_model
        self._embedded_ids: set[str] = set()
        self._default_provider = provider
        self._default_api_key = api_key
        self._default_model = model
        self._decay_config = decay_config
        self._llm_recall = llm_recall
        self._expansion_weight = expansion_weight
        self._query_expander: LLMQueryExpander | None = None
        self._default_provider_kwargs: dict[str, str] = dict(provider_kwargs)
        self._episodic_ttl_hours = episodic_ttl_hours

    def add(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.8,
        tags: list[str] | tuple[str, ...] = (),
    ) -> Fact:
        """Add a fact manually to the knowledge base.

        Args:
            content: The knowledge string.
            type: Classification (semantic/procedural/episodic).
            importance: How critical (0.0-1.0).
            tags: Optional labels.

        Returns:
            The created Fact.
        """
        if not content.strip():
            raise ValueError("content must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

        fact = Fact(
            content=content,
            type=type,
            importance=importance,
            tags=list(tags),
        )
        facts = self._storage.load(self._agent_id)
        facts.append(fact)
        self._storage.save(self._agent_id, facts)
        logger.info("Added fact '%s' to agent '%s'", content[:50], self._agent_id)
        return fact

    def add_many(
        self,
        facts: Sequence[str | dict[str, Any]],
        *,
        type: MemoryType = MemoryType.SEMANTIC,
        importance: float = 0.8,
        tags: list[str] | tuple[str, ...] = (),
    ) -> list[Fact]:
        """Add multiple pre-extracted facts at once without an LLM call.

        Each item can be a plain string (content only) or a dict with any of the
        keys ``content`` (required), ``type``, ``importance``, ``tags``.  Dict
        values take precedence over the method-level defaults.

        Args:
            facts: Sequence of fact strings or dicts.
            type: Default memory type applied to string items and to dicts that
                do not specify a type.
            importance: Default importance applied to string items and to dicts
                that do not specify importance.
            tags: Default tags applied to string items and to dicts that do not
                specify tags.

        Returns:
            List of created Facts in the same order as the input.

        Raises:
            ValueError: If any fact has empty content or an invalid importance.
        """
        if not facts:
            return []

        # Build and validate all Fact objects before touching storage so that
        # a validation error on item N does not leave the first N-1 persisted.
        new_facts: list[Fact] = []
        for item in facts:
            if isinstance(item, str):
                content = item
                item_type = type
                item_importance = importance
                item_tags: list[str] = list(tags)
            else:
                raw_content = item.get("content")
                if not raw_content:
                    raise ValueError("each dict item must include a non-empty 'content' key")
                content = str(raw_content)
                raw_type = item.get("type")
                item_type = MemoryType(raw_type) if raw_type else type
                item_importance = float(item.get("importance", importance))
                raw_tags = item.get("tags", list(tags))
                item_tags = raw_tags if isinstance(raw_tags, (list, tuple)) else list(tags)  # type: ignore[assignment]

            if not content.strip():
                raise ValueError("content must not be empty")
            if not 0.0 <= item_importance <= 1.0:
                raise ValueError(f"importance must be between 0.0 and 1.0, got {item_importance}")
            new_facts.append(
                Fact(content=content, type=item_type, importance=item_importance, tags=item_tags)
            )

        # Single load + save: O(1) storage round-trips regardless of list length.
        existing = self._storage.load(self._agent_id)
        existing.extend(new_facts)
        self._storage.save(self._agent_id, existing)
        logger.info("Added %d facts to agent '%s'", len(new_facts), self._agent_id)
        return new_facts

    def add_episodic(
        self,
        content: str,
        *,
        importance: float = 0.3,
        tags: list[str] | tuple[str, ...] = (),
        ttl_hours: float | None = None,
    ) -> Fact:
        """Add a short-lived episodic fact (L1 hippocampus-like buffer).

        Episodic facts have a time-to-live: they expire after ``ttl_hours``
        (defaults to ``episodic_ttl_hours`` set at init, typically 72h).
        They are excluded from default recall() and recall_facts() results
        (which only return active semantic/procedural facts), but are visible
        to consolidate_episodic() for promotion to semantic memory.

        Use for: raw conversation snippets, session context, unverified claims
        that need consolidation before becoming durable knowledge.
        """
        if not content.strip():
            raise ValueError("content must not be empty")
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

        ttl = ttl_hours if ttl_hours is not None else self._episodic_ttl_hours
        now = datetime.now(UTC)
        fact = Fact(
            content=content,
            type=MemoryType.EPISODIC,
            importance=importance,
            tags=list(tags),
            valid_from=now,
            valid_until=now + timedelta(hours=ttl),
        )
        facts = self._storage.load(self._agent_id)
        facts.append(fact)
        self._storage.save(self._agent_id, facts)
        logger.info("Added episodic fact (TTL=%.0fh) to agent '%s'", ttl, self._agent_id)
        return fact

    def learn(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Extract and store facts from a conversation using an LLM.

        Resolution uses a three-phase pipeline:

        1. **Slot-based** (deterministic): facts with ``slot_key`` are matched
           against existing active facts by exact ``slot_key`` equality.
           Same slot + same value → *reinforce* (bump confidence, no insert).
           Same slot + new value → *supersede* (temporal close + versioned insert).
           No slot match → *branch* (insert as new).

        2. **Entity-addressed CAS** (fuzzy, backward compat): unslotted facts
           with ``entity`` + ``attribute`` are matched via ``resolve_structured``
           (Jaccard entity matching) to close superseded pre-Phase-1 facts.

        3. **Lexical dedup**: remaining unslotted facts are checked against active
           facts with ``resolve_against_existing`` (combined Jaccard + containment).

        Provider credentials default to those passed at :meth:`__init__` when
        not specified per-call.

        Args:
            turns: Conversation messages to extract knowledge from.
            api_key: LLM API key. Falls back to the value set at init, then to
                environment variables.
            provider: Provider name or a pre-configured ``LLMProvider`` instance.
                Falls back to the value set at init, then to ``"openai"``.
                Supported names: openai, anthropic, gigachat, yandex, qwen,
                openai-compat.
            model: Override the default model for this provider. Falls back to
                the value set at init.
            conflict_threshold: Jaccard similarity threshold for the lexical
                dedup pass (phase 3). Does not affect slot-based resolution.
            timeout: Per-request timeout in seconds for LLM calls. ``None``
                uses the provider's built-in default (30 s).
            batch_size: Maximum conversation turns sent per LLM call. Longer
                conversations are split into batches to prevent JSON truncation.
            **provider_kwargs: Extra args forwarded to the provider constructor
                (e.g. ``folder_id`` for Yandex, ``base_url`` for openai-compat).
                Merged with any defaults set at init, with per-call values taking
                precedence.

        Returns:
            List of inserted Facts (new inserts + versioned replacements).
            Reinforced facts (same value) are excluded from the return value.
        """
        if not turns:
            return []

        # Stage 1: extract raw candidate facts from LLM.
        candidates = self._extract_phase(
            turns,
            provider=provider,
            api_key=api_key,
            model=model,
            timeout=timeout,
            batch_size=batch_size,
            **provider_kwargs,
        )
        if not candidates:
            return []

        # Stage 2: candidate verification (dedup, future ATC checks).
        verified = self._candidate_phase(candidates)

        # Stage 3: resolve against existing knowledge.
        existing = self._storage.load(self._agent_id)
        to_insert = self._resolve_phase(verified, existing, conflict_threshold)

        # Stage 4: commit to storage.
        self._commit_phase(existing, to_insert)
        return to_insert

    # ------------------------------------------------------------------
    # learn() pipeline stages
    # ------------------------------------------------------------------

    def _extract_phase(
        self,
        turns: list[ConversationTurn],
        *,
        provider: str | LLMProvider | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Stage 1: extract raw candidate facts from an LLM."""
        resolved_provider = provider or self._default_provider or "openai"
        resolved_api_key = api_key or self._default_api_key
        resolved_model = model or self._default_model
        resolved_kwargs: dict[str, str] = {**self._default_provider_kwargs, **provider_kwargs}

        if isinstance(resolved_provider, str):
            if not resolved_api_key:
                import os

                resolved_api_key = os.environ.get(
                    {
                        "openai": "OPENAI_API_KEY",
                        "anthropic": "ANTHROPIC_API_KEY",
                        "gigachat": "GIGACHAT_API_KEY",
                        "yandex": "YANDEX_API_KEY",
                        "qwen": "QWEN_API_KEY",
                    }.get(resolved_provider, "LLM_API_KEY"),
                    "",
                )
            if not resolved_api_key:
                raise ValueError(
                    f"No API key for provider {resolved_provider!r}. "
                    "Pass api_key= or set the environment variable "
                    f"(e.g. OPENAI_API_KEY for openai, ANTHROPIC_API_KEY for anthropic)."
                )
            extractor = Extractor(
                resolved_provider,
                api_key=resolved_api_key,
                model=resolved_model,
                timeout=timeout,
                batch_size=batch_size,
                **resolved_kwargs,
            )
        else:
            extractor = Extractor(
                resolved_provider,
                model=resolved_model,
                timeout=timeout,
                batch_size=batch_size,
            )

        facts = extractor.extract(turns)
        if _LEARN_DEBUG:
            logger.debug(
                "learn/_extract_phase: %d candidates from %d turns",
                len(facts),
                len(turns),
            )
        return facts

    def _candidate_phase(self, candidates: list[Fact]) -> list[Fact]:
        """Stage 2: verify and deduplicate candidate facts.

        Currently a pass-through. Future: ATC (Assertion-Truth-Checking),
        source grounding, confidence calibration.
        """
        if _LEARN_DEBUG:
            logger.debug("learn/_candidate_phase: %d candidates (pass-through)", len(candidates))
        return candidates

    def _resolve_phase(
        self,
        verified: list[Fact],
        existing: list[Fact],
        conflict_threshold: float,
    ) -> list[Fact]:
        """Stage 3: resolve new facts against existing knowledge.

        Three-phase resolution:
        1. Slot-based CAS (deterministic, exact slot_key match).
        2. Entity-addressed CAS (fuzzy, for pre-Phase-1 facts).
        3. Lexical dedup (Jaccard + containment).

        Returns:
            Facts to insert (new + versioned replacements).
            Mutates ``existing`` in place (closing superseded facts).
        """
        now_close = datetime.now(UTC)
        active_existing = [f for f in existing if f.is_active(now_close)]

        to_insert: list[Fact] = []
        handled_ids: set[str] = set()
        n_reinforce = n_supersede = n_branch = n_delete = n_noop = 0

        # Phase 1: slot-based resolution (deterministic, exact slot_key match).
        # Consults the per-type ConflictPolicy to decide supersession behaviour.
        slotted_facts = [f for f in verified if f.slot_key]
        for new_fact in slotted_facts:
            if new_fact.op == MemoryOp.NOOP:
                n_noop += 1
                continue

            policy = CONFLICT_POLICIES.get(new_fact.type, CONFLICT_POLICIES[MemoryType.SEMANTIC])

            if new_fact.op == MemoryOp.DELETE:
                unhandled = [f for f in active_existing if f.id not in handled_ids]
                _, matched = resolve_by_slot(new_fact, unhandled)
                if matched is not None:
                    matched.valid_until = now_close
                    handled_ids.add(matched.id)
                n_delete += 1
                continue

            slot_op, matched = resolve_by_slot(new_fact, active_existing)
            if new_fact.op == MemoryOp.UPDATE and slot_op == "reinforce":
                slot_op = "supersede"

            # Policy override: if the policy says don't supersede, branch instead.
            # Explicit UPDATE ops bypass the policy (user intent is authoritative).
            if (
                slot_op == "supersede"
                and matched is not None
                and new_fact.op != MemoryOp.UPDATE
                and not policy.should_supersede(new_fact, matched)
            ):
                slot_op = "branch"

            if slot_op == "reinforce":
                assert matched is not None
                matched.state_confidence = min(1.0, matched.state_confidence + 0.05)
                matched.importance = min(1.0, matched.importance + 0.02)
                matched.last_accessed = now_close
                # Accumulate evidence snippets from the new observation.
                if new_fact.source_snippets:
                    existing_snips = set(matched.source_snippets)
                    for s in new_fact.source_snippets:
                        if s not in existing_snips:
                            matched.source_snippets.append(s)
                            existing_snips.add(s)
                    matched.source_snippets = matched.source_snippets[:5]
                handled_ids.add(matched.id)
                n_reinforce += 1
            elif slot_op == "supersede":
                assert matched is not None
                matched.valid_until = now_close
                handled_ids.add(matched.id)
                new_fact.importance = min(1.0, matched.importance + 0.05)
                new_fact.version = matched.version + 1
                # Carry over evidence trail from the old fact.
                if matched.source_snippets:
                    existing_snips = set(new_fact.source_snippets)
                    carried = [s for s in matched.source_snippets if s not in existing_snips]
                    new_fact.source_snippets = (new_fact.source_snippets + carried)[:5]
                to_insert.append(new_fact)
                n_supersede += 1
            else:  # branch — new slot, insert as-is
                to_insert.append(new_fact)
                n_branch += 1

        # Phase 2: entity-addressed CAS for unslotted facts with entity+attribute.
        unslotted_facts = [f for f in verified if not f.slot_key and f.op != MemoryOp.NOOP]
        unslotted_with_entity = [f for f in unslotted_facts if f.entity and f.attribute]
        entity_candidates = [f for f in active_existing if f.id not in handled_ids]
        for new_fact in unslotted_with_entity:
            available = [f for f in entity_candidates if f.id not in handled_ids]
            matched_fact = resolve_structured(new_fact, available)
            if matched_fact is not None:
                matched_fact.valid_until = now_close
                handled_ids.add(matched_fact.id)
                # Carry over evidence trail from the old entity fact.
                if matched_fact.source_snippets and new_fact.op != MemoryOp.DELETE:
                    existing_snips = set(new_fact.source_snippets)
                    carried = [s for s in matched_fact.source_snippets if s not in existing_snips]
                    new_fact.source_snippets = (new_fact.source_snippets + carried)[:5]
            if new_fact.op == MemoryOp.DELETE:
                n_delete += 1

        # Phase 3: lexical dedup for remaining unslotted facts.
        remaining_active = [f for f in entity_candidates if f.id not in handled_ids]
        unslotted_to_insert = [f for f in unslotted_facts if f.op != MemoryOp.DELETE]
        unslotted_inserted, _ = resolve_against_existing(
            unslotted_to_insert, remaining_active, threshold=conflict_threshold
        )
        to_insert.extend(unslotted_inserted)

        if _LEARN_DEBUG:
            logger.debug(
                "learn/_resolve_phase: slot=%d(r=%d s=%d b=%d d=%d n=%d) "
                "entity=%d lexical=%d→%d insert=%d",
                len(slotted_facts),
                n_reinforce,
                n_supersede,
                n_branch,
                n_delete,
                n_noop,
                len(unslotted_with_entity),
                len(unslotted_to_insert),
                len(unslotted_inserted),
                len(to_insert),
            )
        return to_insert

    def _commit_phase(self, existing: list[Fact], to_insert: list[Fact]) -> None:
        """Stage 4: persist resolved facts to storage."""
        self._storage.save(self._agent_id, existing + to_insert)
        if _LEARN_DEBUG:
            logger.debug(
                "learn/_commit_phase: saved %d total facts (%d new) for '%s'",
                len(existing) + len(to_insert),
                len(to_insert),
                self._agent_id,
            )
        logger.info(
            "Learned %d facts for agent '%s'",
            len(to_insert),
            self._agent_id,
        )

    async def alearn(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Async variant of :meth:`learn` — non-blocking for asyncio applications.

        Runs ``learn()`` in a thread-pool executor so the event loop is never
        blocked during the LLM HTTP call.  All parameters are identical to
        :meth:`learn`.

        Example::

            # FastAPI handler — does not block the event loop
            facts = await kb.alearn(turns, provider="openai", api_key="sk-...")

            # Concurrent extraction for multiple agents
            results = await asyncio.gather(
                kb_a.alearn(turns_a, ...),
                kb_b.alearn(turns_b, ...),
            )
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.learn(
                turns,
                api_key=api_key,
                provider=provider,
                model=model,
                conflict_threshold=conflict_threshold,
                timeout=timeout,
                batch_size=batch_size,
                **provider_kwargs,
            ),
        )

    async def learn_async(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str | LLMProvider | None = None,
        model: str | None = None,
        conflict_threshold: float = 0.7,
        timeout: float | None = None,
        batch_size: int = 20,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        semantic_threshold: float = 0.80,
        **provider_kwargs: str,
    ) -> list[Fact]:
        """Like :meth:`alearn` but adds a semantic deduplication pass after extraction.

        After the standard LLM extraction + lexical dedup (``resolve_against_existing``),
        a second pass embeds remaining new facts alongside existing facts and merges
        any pair with cosine similarity ≥ ``semantic_threshold``.  This detects
        topic-evolution updates (e.g. "I use Airflow" followed by "I switched to
        Prefect") that share almost no tokens but are semantically near-duplicate.

        On merge the *newer* fact's content replaces the existing one (temporal
        consolidation).  Importance is bumped by ``+0.05`` (clamped to 1.0).

        Graceful degradation: if the Ollama embedding endpoint is unreachable the
        semantic pass is silently skipped and behaviour is identical to
        :meth:`alearn`.

        Args:
            turns: Conversation messages to extract knowledge from.
            embed_url: Base URL of the Ollama server for embeddings.
            embed_model: Embedding model name (must support /v1/embeddings).
            semantic_threshold: Cosine similarity above which two facts are
                considered the same topic (0.0–1.0, default 0.82).
            All other args: same as :meth:`learn`.
        """
        from ai_knot.embedder import cosine, embed_texts

        loop = asyncio.get_running_loop()
        new_facts: list[Fact] = await loop.run_in_executor(
            None,
            lambda: self.learn(
                turns,
                api_key=api_key,
                provider=provider,
                model=model,
                conflict_threshold=conflict_threshold,
                timeout=timeout,
                batch_size=batch_size,
                **provider_kwargs,
            ),
        )

        if not new_facts:
            return new_facts

        existing = await loop.run_in_executor(None, lambda: self._storage.load(self._agent_id))
        if not existing:
            return new_facts

        # Build a set of already-inserted new fact IDs to avoid comparing against
        # themselves (learn() already saved them to storage).
        new_ids = {f.id for f in new_facts}
        prior_facts = [f for f in existing if f.id not in new_ids]
        if not prior_facts:
            return new_facts

        new_texts = [f.content for f in new_facts]
        all_embeddings = await embed_texts(
            new_texts + [f.content for f in prior_facts], base_url=embed_url, model=embed_model
        )

        if not all_embeddings:
            # Ollama unavailable — graceful degradation.
            return new_facts

        new_embs = all_embeddings[: len(new_facts)]
        prior_embs = all_embeddings[len(new_facts) :]

        # For each new fact find its nearest prior fact.  If above threshold,
        # close the old fact (set valid_until = now) and keep the new fact as
        # the current version — proper temporal versioning instead of mutation.
        closed_ids: set[str] = set()
        updated_prior: list[Fact] = []
        for _new_fact, new_emb in zip(new_facts, new_embs, strict=True):
            updated_prior_ids = {p.id for p in updated_prior}
            best_score = 0.0
            best_prior: Fact | None = None
            for prior_fact, prior_emb in zip(prior_facts, prior_embs, strict=True):
                if prior_fact.id in updated_prior_ids:
                    continue  # already merged into this slot — don't double-merge
                score = cosine(new_emb, prior_emb)
                if score > best_score:
                    best_score = score
                    best_prior = prior_fact

            if best_prior is not None and best_score >= semantic_threshold:
                # Temporal close: mark prior fact as superseded instead of mutating content.
                best_prior.valid_until = datetime.now(UTC)
                best_prior.importance = min(1.0, best_prior.importance + 0.05)
                updated_prior.append(best_prior)
                closed_ids.add(best_prior.id)

        if closed_ids:
            await loop.run_in_executor(None, lambda: self._storage.save(self._agent_id, existing))
            logger.info(
                "Temporal consolidation closed %d prior fact(s) for agent '%s'",
                len(closed_ids),
                self._agent_id,
            )

        return new_facts

    async def consolidate_episodic(
        self,
        *,
        older_than_hours: float = 24.0,
        embed_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        semantic_threshold: float = 0.80,
    ) -> int:
        """Promote episodic facts to semantic memory (async "sleep consolidation").

        Finds episodic facts older than ``older_than_hours``, runs LLM extraction
        to produce structured semantic facts, deduplicates against existing
        semantic memory, and marks episodic facts as consolidated (valid_until=now).

        Inspired by CLS theory (McClelland et al. 1995): hippocampal → neocortical
        transfer happens offline (during "sleep"), not during encoding.

        Args:
            older_than_hours: Only consolidate episodic facts created more than
                this many hours ago (avoids consolidating too-recent episodes).
            embed_url: Ollama base URL for semantic dedup embeddings.
            embed_model: Embedding model for dedup pass.
            semantic_threshold: Cosine threshold for semantic dedup.

        Returns:
            Number of new semantic facts created from consolidation.
        """
        if not self._default_provider:
            logger.warning(
                "consolidate_episodic() requires a provider; "
                "configure KnowledgeBase with provider= to enable this."
            )
            return 0

        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=older_than_hours)

        all_facts = self._storage.load(self._agent_id)
        to_consolidate = [
            f
            for f in all_facts
            if f.type == MemoryType.EPISODIC and f.is_active(now) and f.created_at <= cutoff
        ]

        if not to_consolidate:
            return 0

        # Run LLM extraction on the episodic batch (treat as conversation turns)
        turns = [
            ConversationTurn(role="user", content=f.source_verbatim or f.content)
            for f in to_consolidate
        ]
        new_semantic = await self.learn_async(
            turns,
            embed_url=embed_url,
            embed_model=embed_model,
            semantic_threshold=semantic_threshold,
        )

        # Reload after learn_async() (which internally saved new semantic facts)
        # to avoid overwriting them with the stale snapshot.
        all_facts = self._storage.load(self._agent_id)
        consolidated_ids = {f.id for f in to_consolidate}
        for fact in all_facts:
            if fact.id in consolidated_ids:
                fact.valid_until = now

        self._storage.save(self._agent_id, all_facts)
        logger.info(
            "Consolidated %d episodic facts → %d new semantic facts for agent '%s'",
            len(to_consolidate),
            len(new_semantic),
            self._agent_id,
        )
        return len(new_semantic)

    async def arecall(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> str:
        """Async variant of :meth:`recall` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recall(
                query, top_k=top_k, now=now, include_unsupported=include_unsupported
            ),
        )

    async def arecall_facts(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> list[Fact]:
        """Async variant of :meth:`recall_facts` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recall_facts(
                query, top_k=top_k, now=now, include_unsupported=include_unsupported
            ),
        )

    def _expand_query(self, query: str) -> tuple[str, dict[str, float] | None]:
        """Optionally expand a query using the configured LLM provider.

        Returns ``(original_query, expansion_weights)`` where expansion_weights
        maps new tokens to their BM25 weight (0.4).  When expansion is disabled
        or unavailable, returns ``(query, None)``.
        """
        if not self._llm_recall:
            return query, None
        if not self._default_provider:
            logger.warning(
                "llm_recall=True but no provider configured — "
                "query expansion skipped, returning original query"
            )
            return query, None
        if self._query_expander is None:
            provider = self._default_provider
            if isinstance(provider, str):
                provider = create_provider(
                    provider, self._default_api_key, **self._default_provider_kwargs
                )
            self._query_expander = LLMQueryExpander(provider, self._default_model)
        expanded_text = self._query_expander.expand(query)

        from ai_knot.tokenizer import tokenize

        original_tokens = set(tokenize(query))
        expanded_tokens = tokenize(expanded_text)
        expansion: dict[str, float] = {}
        for token in expanded_tokens:
            if token not in original_tokens:
                expansion[token] = self._expansion_weight or _LLM_EXPANSION_WEIGHT
        return query, expansion if expansion else None

    def _embed_for_recall(
        self,
        facts: list[Fact],
        query: str,
    ) -> list[float] | None:
        """Embed new facts and the query for hybrid retrieval.

        Returns the query vector, or ``None`` if embedding is unavailable
        (Ollama down, timeout, etc.) — callers fall back to BM25-only.
        """
        from ai_knot.embedder import embed_texts

        new_facts = [f for f in facts if f.id not in self._embedded_ids]
        texts_to_embed: list[str] = [f.content for f in new_facts] + [query]

        try:
            import contextlib

            loop: asyncio.AbstractEventLoop | None = None
            with contextlib.suppress(RuntimeError):
                loop = asyncio.get_running_loop()

            if loop is not None and loop.is_running():
                # Already inside an event loop (e.g. Jupyter, MCP server) —
                # can't nest asyncio.run().  Use a thread to avoid deadlock.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    vectors = pool.submit(
                        asyncio.run,
                        embed_texts(
                            texts_to_embed,
                            base_url=self._embed_url,
                            model=self._embed_model,
                        ),
                    ).result(timeout=60)
            else:
                vectors = asyncio.run(
                    embed_texts(
                        texts_to_embed,
                        base_url=self._embed_url,
                        model=self._embed_model,
                    )
                )
        except Exception:
            logger.debug("Embedding unavailable — falling back to BM25-only")
            return None

        if not vectors:
            return None

        # Store fact embeddings (all except the last which is the query).
        fact_vectors = vectors[:-1]
        for f, vec in zip(new_facts, fact_vectors, strict=True):
            self._dense.add_embeddings({f.id: vec})
            self._embedded_ids.add(f.id)

        return vectors[-1]  # query vector

    def _execute_recall(
        self,
        query: str,
        *,
        top_k: int,
        now: datetime | None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[tuple[Fact, float]]:
        """Core recall logic shared by recall(), recall_facts(), recall_facts_with_scores().

        Loads facts, applies decay, runs retrieval, updates access metadata, and
        persists.  Returns (Fact, score) pairs in relevance order.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to exclude from results (e.g. already-seen facts
                in a novelty-aware retrieval loop).  ``None`` means no exclusion.
            include_unsupported: When False (default), facts with ``supported=False``
                are excluded from retrieval. Pass True to include them (e.g. for
                manual review workflows).
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return []

        now_dt = now or datetime.now(UTC)
        # Temporal filter + exclude episodic (raw buffer) + verification gate.
        facts = [
            f
            for f in facts
            if f.is_active(now_dt)
            and f.type != MemoryType.EPISODIC
            and (include_unsupported or f.supported is not False)
        ]
        if not facts:
            return []

        facts = apply_decay(facts, type_exponents=self._decay_config, now=now_dt)
        expanded_query, expansion = self._expand_query(query)

        # Intent-aware RRF weights for private KB (same classifier as pool).
        intent = _classify_pool_query(query, facts)
        kb_rrf: dict[_PoolQueryIntent, tuple[float, ...]] = {
            _PoolQueryIntent.ENTITY_LOOKUP: (5.0, 8.0, 2.0, 1.5, 1.5, 1.0),
            _PoolQueryIntent.INCIDENT: (5.0, 3.0, 2.0, 1.5, 1.5, 3.0),
        }
        rrf_override = kb_rrf.get(intent)

        candidate_facts = [f for f in facts if f.id not in excluded_ids] if excluded_ids else facts

        # Embed facts + query for hybrid retrieval (best-effort).
        query_vector = self._embed_for_recall(candidate_facts, expanded_query)

        if query_vector is not None:
            pairs = self._hybrid.search(
                expanded_query,
                candidate_facts,
                top_k=top_k,
                query_vector=query_vector,
                expansion_weights=expansion,
                rrf_weights=rrf_override,
            )
        else:
            pairs = self._bm25.search(
                expanded_query,
                candidate_facts,
                top_k=top_k,
                expansion_weights=expansion,
                rrf_weights=rrf_override,
            )
        if not pairs:
            return []

        returned_ids = {f.id for f, _ in pairs}
        access_time = datetime.now(UTC)
        for fact in facts:
            if fact.id in returned_ids:
                interval = (access_time - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = access_time
        self._storage.save(self._agent_id, facts)
        return pairs

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        include_unsupported: bool = False,
    ) -> str:
        """Retrieve relevant facts as a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        pairs = self._execute_recall(
            query, top_k=top_k, now=now, include_unsupported=include_unsupported
        )
        if not pairs:
            return ""
        lines = [
            f"[{f.type.value}] {f.prompt_surface or f.source_verbatim or f.content}"
            for f, _ in pairs
        ]
        return "\n".join(lines)

    def list_facts(self) -> list[Fact]:
        """Return all stored facts for this agent.

        Returns:
            List of all Facts, in storage order.
        """
        return self._storage.load(self._agent_id)

    def recall_facts(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[Fact]:
        """Structured alternative to recall() — returns Fact objects.

        Use when you need IDs, types, importance scores, or other metadata.
        Use recall() when you only need a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        return [
            f
            for f, _ in self._execute_recall(
                query,
                top_k=top_k,
                now=now,
                excluded_ids=excluded_ids,
                include_unsupported=include_unsupported,
            )
        ]

    def recall_facts_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        excluded_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[tuple[Fact, float]]:
        """Like recall_facts() but also returns the relevance score for each fact.

        The score is a hybrid value (TF-IDF + retention + importance). Use it
        to rank or filter results in integration adapters.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Empty list if no facts stored or none match.
        """
        return self._execute_recall(
            query,
            top_k=top_k,
            now=now,
            excluded_ids=excluded_ids,
            include_unsupported=include_unsupported,
        )

    def recall_by_tag(self, tag: str, *, include_unsupported: bool = False) -> list[Fact]:
        """Return all facts that carry the given tag.

        Tags are assigned at add() time via the ``tags=`` parameter.

        Args:
            tag: The tag string to filter by.
            include_unsupported: Include facts with ``supported=False`` (default: False).

        Returns:
            List of Facts whose tags include ``tag`` (may be empty).
        """
        return [
            f
            for f in self._storage.load(self._agent_id)
            if tag in f.tags and (include_unsupported or f.supported is not False)
        ]

    def replace_facts(self, facts: list[Fact]) -> None:
        """Replace all stored facts with the given list (used for import).

        Args:
            facts: New facts to store; replaces any existing facts.
        """
        self._storage.save(self._agent_id, facts)
        logger.info("Replaced facts for agent '%s' (%d total)", self._agent_id, len(facts))

    def forget(self, fact_id: str) -> None:
        """Remove a specific fact by its ID.

        Args:
            fact_id: The 8-char hex ID of the fact to remove.
        """
        self._storage.delete(self._agent_id, fact_id)
        logger.info("Forgot fact '%s' from agent '%s'", fact_id, self._agent_id)

    def decay(self, *, now: datetime | None = None) -> None:
        """Apply Ebbinghaus forgetting curve to all stored facts.

        Args:
            now: Point-in-time for decay calculation (default: current UTC).
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return
        apply_decay(facts, type_exponents=self._decay_config, now=now)
        self._storage.save(self._agent_id, facts)
        logger.debug("Applied decay to %d facts for agent '%s'", len(facts), self._agent_id)

    def clear_all(self) -> None:
        """Remove all facts for this agent."""
        self._storage.save(self._agent_id, [])
        logger.info("Cleared all facts for agent '%s'", self._agent_id)

    def snapshot(self, name: str) -> None:
        """Save the current state of the knowledge base as a named snapshot.

        Args:
            name: Identifier for this snapshot (e.g. "before_v2_campaign").

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts = self._storage.load(self._agent_id)
        self._storage.save_snapshot(self._agent_id, name, facts)
        logger.info("Snapshot '%s' saved for agent '%s'", name, self._agent_id)

    def list_snapshots(self) -> list[str]:
        """Return names of all saved snapshots for this agent.

        Returns:
            List of snapshot names sorted by creation time (oldest first).

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        return self._storage.list_snapshots(self._agent_id)

    def restore(self, name: str) -> None:
        """Replace current facts with the contents of a named snapshot.

        Args:
            name: The snapshot to restore.

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
            KeyError: If the snapshot does not exist.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts = self._storage.load_snapshot(self._agent_id, name)
        self._storage.save(self._agent_id, facts)
        logger.info("Restored snapshot '%s' for agent '%s'", name, self._agent_id)

    def diff(self, a: str, b: str) -> SnapshotDiff:
        """Compute the difference between two snapshots.

        Use the special name ``"current"`` to refer to the live facts in storage.

        Args:
            a: Name of the first snapshot (or "current").
            b: Name of the second snapshot (or "current").

        Returns:
            A :class:`SnapshotDiff` with ``added`` and ``removed`` fact lists.

        Raises:
            NotImplementedError: If the storage backend does not support snapshots.
            KeyError: If a named snapshot does not exist.
        """
        if not isinstance(self._storage, SnapshotCapable):
            raise NotImplementedError(f"{type(self._storage).__name__} does not support snapshots.")
        facts_a = (
            self._storage.load(self._agent_id)
            if a == "current"
            else self._storage.load_snapshot(self._agent_id, a)
        )
        facts_b = (
            self._storage.load(self._agent_id)
            if b == "current"
            else self._storage.load_snapshot(self._agent_id, b)
        )
        ids_a = {f.id for f in facts_a}
        ids_b = {f.id for f in facts_b}
        removed = [f for f in facts_a if f.id not in ids_b]
        added = [f for f in facts_b if f.id not in ids_a]
        return SnapshotDiff(snapshot_a=a, snapshot_b=b, added=added, removed=removed)

    def stats(self) -> dict[str, Any]:
        """Return statistics about the knowledge base.

        Returns:
            Dict with total_facts, by_type counts, avg_importance, avg_retention.
        """
        _zero: dict[str, Any] = {
            "total_facts": 0,
            "by_type": {"semantic": 0, "procedural": 0, "episodic": 0},
            "avg_importance": 0.0,
            "avg_retention": 0.0,
        }
        facts = self._storage.load(self._agent_id)
        now_dt = datetime.now(UTC)
        active_facts = [f for f in facts if f.is_active(now_dt)]
        if not active_facts:
            return _zero

        type_counts = Counter(f.type.value for f in active_facts)
        return {
            "total_facts": len(active_facts),
            "by_type": {
                "semantic": type_counts.get("semantic", 0),
                "procedural": type_counts.get("procedural", 0),
                "episodic": type_counts.get("episodic", 0),
            },
            "avg_importance": sum(f.importance for f in active_facts) / len(active_facts),
            "avg_retention": sum(f.retention_score for f in active_facts) / len(active_facts),
        }


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


class SharedMemoryPool:
    """Shared memory pool for multi-agent knowledge exchange.

    Provides a shared namespace (``__shared__``) where agents can publish
    facts for cross-agent retrieval. Each published fact retains its
    ``origin_agent_id`` for provenance tracking.

    Inspired by CommNet (Sukhbaatar et al., 2016): a shared communication
    channel with selective read access. Facts from other agents receive a
    provenance discount reflecting per-agent trust (Marsh 1994).

    Trust is computed automatically from observed behaviour:
        ``trust = min(1, used / published) × (1 − quick_invalidation_rate)``

    where *published* is the number of facts an agent has contributed,
    *used* is the total number of recall hits across all queries, and
    *quick_invalidation_rate* is the fraction of the agent's facts that were
    superseded by a different agent within ``_QUICK_INV_WINDOW_S`` seconds —
    a signal that the original data was stale or low-quality.

    Usage::

        pool = SharedMemoryPool(storage=SQLiteStorage("mem.db"))
        pool.register("devops_agent")
        pool.register("coding_agent")

        # DevOps agent publishes a fact
        pool.publish("devops_agent", [fact_id], kb=devops_kb)

        # Coding agent queries the shared pool
        results = pool.recall("what database?", "coding_agent", top_k=5)

    Args:
        storage: Backend used to persist the shared namespace.
    """

    # Auto-promotion: promote facts used by >= N distinct agents to "org" tier.
    _AUTO_PROMOTE_THRESHOLD: int = 3
    # Pool TTL: expire pool facts unused for this many seconds (30 days).
    _POOL_TTL_SECONDS: float = 30 * 24 * 3600.0
    # Tier score boost: multiplicative bonus for higher-tier facts in recall.
    _TIER_BOOST: dict[str, float] = {"private": 1.0, "pool": 1.0, "org": 1.05}

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage: StorageBackend = storage or YAMLStorage()
        self._bm25 = BM25Retriever(skip_prf=True)
        self._dense = DenseRetriever()
        # Pool retrieval has a wider semantic gap than private KB recall:
        # queries use abstract terms while facts contain technical specifics.
        # Weight dense higher to bridge this gap (2:3 vs KnowledgeBase's 2:1).
        self._retriever = HybridRetriever(
            self._bm25, self._dense, bm25_weight=2.0, dense_weight=3.0
        )
        self._agents: set[str] = set()
        self._publish_count: dict[str, int] = {}
        self._used_count: dict[str, int] = {}
        self._quick_inv_count: dict[str, int] = {}
        # MESI: per-agent high-water mark of versions pulled from shared pool.
        self._known_version: dict[str, int] = {}
        # Serialise concurrent publish() calls on the same pool instance.
        self._publish_lock = threading.Lock()
        # Internal recall metadata (not part of public API).
        self._last_recall_meta: _RecallMeta | None = None
        # Per-fact usage tracking: fact_id -> set of agent_ids that recalled it.
        self._fact_consumers: dict[str, set[str]] = {}
        # Tracks which fact IDs have been embedded to avoid re-embedding.
        self._embedded_ids: set[str] = set()
        # Transient query vector set by arecall() for hybrid search.
        self._query_vector: list[float] | None = None
        self._recall_service = SharedPoolRecallService()
        self._claim_resolver = ClaimFamilyResolver()
        self._query_router = QueryShapeRouter()

    def register(self, agent_id: str) -> None:
        """Register an agent to participate in the shared pool.

        Args:
            agent_id: Unique identifier for the agent.
        """
        self._agents.add(agent_id)

    @property
    def agents(self) -> set[str]:
        """Return the set of registered agent IDs."""
        return set(self._agents)

    def get_trust(self, agent_id: str) -> float:
        """Return the current auto-computed trust score for an agent.

        Formula:
            ``trust = min(1, used / published) × (1 − quick_inv_rate)``

        Falls back to ``_PROVENANCE_DISCOUNT`` when the agent has not yet
        published any facts (no track record).

        Args:
            agent_id: The agent to query.

        Returns:
            Trust score in [0.1, 1.0].
        """
        published = self._publish_count.get(agent_id, 0)
        if published == 0:
            return _PROVENANCE_DISCOUNT
        used = self._used_count.get(agent_id, 0)
        quick_inv = self._quick_inv_count.get(agent_id, 0)
        # Use Bayesian prior: start at _PROVENANCE_DISCOUNT and adjust toward
        # observed quality as evidence accumulates.  Without this, agents with
        # published facts but no retrievals yet get trust=0.1 (the floor),
        # which is far below the old flat 0.8 default and distorts ranking.
        _PRIOR_WEIGHT = 3  # pseudocount — ~3 observations before prior fades
        quality = min(
            1.0, (used + _PRIOR_WEIGHT * _PROVENANCE_DISCOUNT) / (published + _PRIOR_WEIGHT)
        )
        inv_penalty = quick_inv / published
        return max(0.1, quality * (1.0 - inv_penalty))

    def publish(
        self,
        agent_id: str,
        fact_ids: list[str],
        *,
        kb: KnowledgeBase,
        utility_threshold: float = 0.0,
    ) -> list[Fact]:
        """Copy facts from an agent's private KB into the shared pool.

        Uses slot-addressed CAS: for each fact with a ``slot_key``, the
        existing active fact for that slot is closed (valid_until=now,
        mesi_state='I') and the new fact is inserted (mesi_state='M' if
        replacing, 'S' if new). Facts with no slot fall back to ID-based dedup.

        An optional publish gate filters facts by utility score before
        inserting:
            ``utility = state_confidence × importance``
        Only facts with ``utility >= utility_threshold`` are published.
        Threshold 0.0 (default) = no gating.

        Args:
            agent_id: The agent publishing the facts.
            fact_ids: IDs of facts to publish from the agent's KB.
            kb: The agent's KnowledgeBase instance.
            utility_threshold: Minimum utility score (0.0–1.0) to publish.
                Facts below the threshold are silently skipped.

        Returns:
            List of facts that were published (copies, not mutations of private KB).

        Raises:
            ValueError: If agent_id is not registered.

        Note:
            Thread-safe within a single process (protected by ``_publish_lock``).
            Cross-process atomicity is not guaranteed — concurrent writers from
            separate processes can interleave; use an external advisory lock if
            cross-process CAS semantics are required.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id!r} is not registered. Call register() first.")

        private_facts = kb.list_facts()
        id_set = set(fact_ids)
        to_publish = [
            f
            for f in private_facts
            if f.id in id_set and f.state_confidence * f.importance >= utility_threshold
        ]

        if not to_publish:
            return []

        with self._publish_lock:
            return self._publish_locked(agent_id, to_publish)

    def _publish_locked(self, agent_id: str, to_publish: list[Fact]) -> list[Fact]:
        """Execute publish while holding ``_publish_lock``.  Called only from :meth:`publish`."""
        now = datetime.now(UTC)
        published: list[Fact] = []
        quick_inv_updates: dict[str, int] = {}

        def _merge(shared: list[Fact]) -> list[Fact]:
            # Index active shared facts by slot_key for O(1) CAS lookup.
            # Falls back to (entity, attribute) for pre-Phase-3 facts without slot_key.
            active_by_slot: dict[str, Fact] = {}
            existing_ids: set[str] = set()
            for f in shared:
                existing_ids.add(f.id)
                if f.is_active(now):
                    if f.slot_key:
                        active_by_slot[f.slot_key] = f
                    elif f.entity and f.attribute:
                        legacy_key = f"{f.entity.lower().strip()}::{f.attribute.lower().strip()}"
                        active_by_slot.setdefault(legacy_key, f)

            # Global monotonic version counter for the pool.
            # Each published or superseded fact gets a strictly increasing version
            # so that load_since_version() and sync_slot_deltas() work correctly
            # when multiple facts are published in separate batches.
            next_version = max((f.version for f in shared), default=0) + 1

            for fact in to_publish:
                new_fact = copy.deepcopy(fact)
                new_fact.origin_agent_id = agent_id
                new_fact.visibility = "pool"
                new_fact.memory_tier = "pool"
                new_fact.valid_from = now
                new_fact.valid_until = None

                # For unslotted facts, extract a claim fingerprint for conflict detection.
                if not new_fact.slot_key and not new_fact.claim_key:
                    new_fact.claim_key = _extract_claim_key(new_fact.content)

                # Resolve CAS key: prefer slot_key, fall back to entity+attribute.
                cas_key = fact.slot_key or (
                    f"{fact.entity.lower().strip()}::{fact.attribute.lower().strip()}"
                    if fact.entity and fact.attribute
                    else ""
                )

                if cas_key and cas_key in active_by_slot:
                    old = active_by_slot[cas_key]
                    if old.id == fact.id:
                        # Same fact already published as the active version — no-op.
                        continue
                    # Slot-addressed CAS: close old version, insert new.
                    old.valid_until = now
                    old.mesi_state = MESIState.INVALID
                    new_fact.mesi_state = MESIState.MODIFIED
                    new_fact.version = next_version
                    next_version += 1
                    # Quick invalidation: a different agent superseded this slot within the window.
                    if old.origin_agent_id and old.origin_agent_id != agent_id:
                        age_s = (now - old.valid_from).total_seconds()
                        if age_s < _QUICK_INV_WINDOW_S:
                            quick_inv_updates[old.origin_agent_id] = (
                                quick_inv_updates.get(old.origin_agent_id, 0) + 1
                            )
                    # If the fact was previously published (INVALID), remove the stale
                    # copy so that re-activating it via CAS does not create a duplicate.
                    if new_fact.id in existing_ids:
                        shared[:] = [f for f in shared if f.id != new_fact.id]
                elif new_fact.id not in existing_ids:
                    new_fact.mesi_state = MESIState.SHARED
                    new_fact.version = next_version
                    next_version += 1
                else:
                    # ID already in pool and no slot key — skip duplicate.
                    continue

                shared.append(new_fact)
                published.append(new_fact)

            return shared

        # AtomicUpdateCapable backends (SQLite) protect the full load→merge→save
        # cycle with an EXCLUSIVE transaction, preventing cross-process lost updates.
        # Other backends fall back to the in-process lock only.
        if isinstance(self._storage, AtomicUpdateCapable):
            self._storage.atomic_update(_SHARED_NAMESPACE, _merge)
        else:
            current = self._storage.load(_SHARED_NAMESPACE)
            _merge(current)
            if published:
                if isinstance(self._storage, TemporalStorageCapable):
                    self._storage.save_atomic(_SHARED_NAMESPACE, current)
                else:
                    self._storage.save(_SHARED_NAMESPACE, current)

        # Apply trust-tracking side effects outside the storage transaction.
        for agt, count in quick_inv_updates.items():
            self._quick_inv_count[agt] = self._quick_inv_count.get(agt, 0) + count

        if published:
            self._publish_count[agent_id] = self._publish_count.get(agent_id, 0) + len(published)
            logger.info(
                "Agent '%s' published %d facts to shared pool",
                agent_id,
                len(published),
            )

        return published

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

    def promote(self, agent_id: str, fact_ids: list[str], *, tier: str = "pool") -> int:
        """Manually promote pool facts to a higher memory tier.

        Only facts currently in the shared pool (origin_agent_id == agent_id)
        are eligible.  Returns the number of facts actually promoted.

        Args:
            agent_id: The agent requesting the promotion.
            fact_ids: IDs of pool facts to promote.
            tier: Target tier — ``"pool"`` (default) or ``"org"`` (future).

        Raises:
            ValueError: If *tier* is not a valid tier name.
        """
        if tier not in ("pool", "org"):
            raise ValueError(f"Invalid tier {tier!r}; must be 'pool' or 'org'.")
        shared = self._storage.load(_SHARED_NAMESPACE)
        promoted = 0
        for fact in shared:
            if fact.id in fact_ids and fact.is_active() and fact.origin_agent_id == agent_id:
                fact.memory_tier = tier
                promoted += 1
        if promoted:
            self._storage.save(_SHARED_NAMESPACE, shared)
            logger.info(
                "Promoted %d facts from agent '%s' to tier '%s'",
                promoted,
                agent_id,
                tier,
            )
        return promoted

    def gc_pool(self, *, now: datetime | None = None) -> int:
        """Garbage-collect stale pool facts that exceed the pool TTL.

        Facts in the ``"org"`` tier are exempt from TTL (they are considered
        permanently promoted).  Only ``"pool"`` tier facts whose
        ``last_accessed`` is older than ``_POOL_TTL_SECONDS`` are expired.

        Returns:
            Number of facts expired.
        """
        now_dt = now or datetime.now(UTC)
        cutoff = now_dt - timedelta(seconds=self._POOL_TTL_SECONDS)
        shared = self._storage.load(_SHARED_NAMESPACE)
        expired = 0
        for fact in shared:
            if (
                fact.is_active(now_dt)
                and fact.memory_tier == "pool"
                and fact.last_accessed < cutoff
            ):
                fact.valid_until = now_dt
                expired += 1
        if expired:
            self._storage.save(_SHARED_NAMESPACE, shared)
            logger.info(
                "Pool GC: expired %d stale facts (TTL=%ds)", expired, self._POOL_TTL_SECONDS
            )
        return expired

    def sync_dirty(self, agent_id: str) -> list[Fact]:
        """Pull facts changed by other agents since the last sync (MESI lazy invalidation).

        Implements the Modified/Invalid state pull from MESI protocol.
        Token savings: ~95% vs broadcast when only a small subset of facts
        changes between syncs (arXiv 2603.15183).

        Uses ``TemporalStorageCapable.load_since_version()`` for index-accelerated
        queries on SQLite/Postgres; falls back to Python filtering on YAML.

        Args:
            agent_id: The agent requesting dirty facts.

        Returns:
            Facts changed by other agents since the last sync call for this agent.
        """
        since = self._known_version.get(agent_id, 0)

        if isinstance(self._storage, TemporalStorageCapable):
            dirty = self._storage.load_since_version(_SHARED_NAMESPACE, since, agent_id)
        else:
            now_dt = datetime.now(UTC)
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            dirty = [
                f
                for f in all_shared
                if f.version > since and f.origin_agent_id != agent_id and f.is_active(now_dt)
            ]

        if dirty:
            self._known_version[agent_id] = max(f.version for f in dirty)
        return dirty

    def sync_slot_deltas(self, agent_id: str) -> list[SlotDelta]:
        """Pull slot-level changes since the last sync as lightweight ``SlotDelta`` records.

        Semantically equivalent to :meth:`sync_dirty` but returns compact
        ``SlotDelta`` objects instead of full ``Fact`` copies.  Each delta
        carries only ``slot_key``, ``version``, ``op``, ``fact_id``,
        ``content``, and ``prompt_surface`` — roughly 10–30× less data than
        a full ``Fact`` for a typical memory entry.

        On SQLite/Postgres backends the query is index-accelerated; on YAML
        backends it falls back to Python-level filtering.

        Args:
            agent_id: The agent requesting delta records.

        Returns:
            ``SlotDelta`` records for slot changes by other agents since the
            last call for this agent.  The per-agent high-water mark is
            updated on each call.
        """
        since = self._known_version.get(agent_id, 0)

        if isinstance(self._storage, TemporalStorageCapable):
            deltas = self._storage.load_slot_deltas_since(_SHARED_NAMESPACE, since, agent_id)
        else:
            # Python-level fallback for YAML backends.
            all_shared = self._storage.load(_SHARED_NAMESPACE)
            deltas = []
            for f in all_shared:
                if f.version <= since or f.origin_agent_id == agent_id:
                    continue
                if f.valid_until is not None:
                    op = "invalidate"
                elif f.mesi_state == MESIState.MODIFIED:
                    op = "supersede"
                else:
                    op = "new"
                deltas.append(
                    SlotDelta(
                        slot_key=f.slot_key,
                        version=f.version,
                        op=op,
                        fact_id=f.id,
                        content=f.content,
                        prompt_surface=f.prompt_surface,
                    )
                )

        if deltas:
            self._known_version[agent_id] = max(d.version for d in deltas)
        return deltas

    def list_shared_facts(self) -> list[Fact]:
        """Return all facts in the shared pool.

        Returns:
            List of all shared Facts (including closed/invalidated).
        """
        return self._storage.load(_SHARED_NAMESPACE)
