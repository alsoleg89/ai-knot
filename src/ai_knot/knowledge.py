"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_knot.extractor import (
    Extractor,
    resolve_against_existing,
    resolve_by_slot,
    resolve_structured,
)
from ai_knot.forgetting import apply_decay
from ai_knot.providers import LLMProvider, create_provider
from ai_knot.query_expander import LLMQueryExpander
from ai_knot.retriever import TFIDFRetriever
from ai_knot.storage.base import (
    AtomicUpdateCapable,
    SnapshotCapable,
    StorageBackend,
    TemporalStorageCapable,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import (
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

logger = logging.getLogger(__name__)


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
        **provider_kwargs: str,
    ) -> None:
        self._agent_id = agent_id
        self._storage: StorageBackend = storage or YAMLStorage()
        self._retriever = TFIDFRetriever(
            rrf_weights=rrf_weights or (5.0, 3.0, 2.0, 1.5, 1.5, 1.0),
        )
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

        new_facts = extractor.extract(turns)

        if new_facts:
            existing = self._storage.load(self._agent_id)
            now_close = datetime.now(UTC)
            active_existing = [f for f in existing if f.is_active(now_close)]

            to_insert: list[Fact] = []
            # IDs of active facts already handled — excluded from later phases.
            handled_ids: set[str] = set()
            n_reinforce = n_supersede = n_branch = n_delete = n_noop = 0

            # Phase 1: slot-based resolution (deterministic, exact slot_key match).
            slotted_facts = [f for f in new_facts if f.slot_key]
            for new_fact in slotted_facts:
                if new_fact.op == MemoryOp.NOOP:
                    n_noop += 1
                    continue

                if new_fact.op == MemoryOp.DELETE:
                    # Exclude already-handled IDs so a second DELETE can't double-close.
                    unhandled = [f for f in active_existing if f.id not in handled_ids]
                    _, matched = resolve_by_slot(new_fact, unhandled)
                    if matched is not None:
                        matched.valid_until = now_close
                        handled_ids.add(matched.id)
                    n_delete += 1
                    continue

                slot_op, matched = resolve_by_slot(new_fact, active_existing)
                # UPDATE overrides structural "reinforce" when value unchanged but context differs.
                if new_fact.op == MemoryOp.UPDATE and slot_op == "reinforce":
                    slot_op = "supersede"

                if slot_op == "reinforce":
                    assert matched is not None
                    matched.state_confidence = min(1.0, matched.state_confidence + 0.05)
                    matched.importance = min(1.0, matched.importance + 0.02)
                    matched.last_accessed = now_close
                    handled_ids.add(matched.id)
                    n_reinforce += 1
                elif slot_op == "supersede":
                    assert matched is not None
                    matched.valid_until = now_close
                    handled_ids.add(matched.id)
                    new_fact.importance = min(1.0, matched.importance + 0.05)
                    new_fact.version = matched.version + 1
                    to_insert.append(new_fact)
                    n_supersede += 1
                else:  # branch — new slot, insert as-is
                    to_insert.append(new_fact)
                    n_branch += 1

            # Phase 2: entity-addressed CAS for unslotted facts with entity+attribute.
            # Closes pre-Phase-1 storage facts that carry entity/attribute but no slot_key.
            unslotted_facts = [f for f in new_facts if not f.slot_key and f.op != MemoryOp.NOOP]
            unslotted_with_entity = [f for f in unslotted_facts if f.entity and f.attribute]
            entity_candidates = [f for f in active_existing if f.id not in handled_ids]
            for new_fact in unslotted_with_entity:
                # Re-filter candidates each iteration so each existing fact is matched at most once.
                available = [f for f in entity_candidates if f.id not in handled_ids]
                matched_fact = resolve_structured(new_fact, available)
                if matched_fact is not None:
                    matched_fact.valid_until = now_close
                    handled_ids.add(matched_fact.id)
                if new_fact.op == MemoryOp.DELETE:
                    n_delete += 1

            # Phase 3: lexical dedup for remaining unslotted facts.
            remaining_active = [f for f in entity_candidates if f.id not in handled_ids]
            unslotted_to_insert = [f for f in unslotted_facts if f.op != MemoryOp.DELETE]
            unslotted_inserted, _ = resolve_against_existing(
                unslotted_to_insert, remaining_active, threshold=conflict_threshold
            )
            to_insert.extend(unslotted_inserted)

            self._storage.save(self._agent_id, existing + to_insert)
            logger.info(
                "Learned %d facts for agent '%s' "
                "(slot: %d reinforced, %d superseded, %d new, %d deleted, %d noop; "
                "lexical: %d merged)",
                len(to_insert),
                self._agent_id,
                n_reinforce,
                n_supersede,
                n_branch,
                n_delete,
                n_noop,
                len(unslotted_facts) - len(unslotted_inserted),
            )
            return to_insert
        return []

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

        candidate_facts = [f for f in facts if f.id not in excluded_ids] if excluded_ids else facts
        pairs = self._retriever.search(
            expanded_query, candidate_facts, top_k=top_k, expansion_weights=expansion
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

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage: StorageBackend = storage or YAMLStorage()
        self._retriever = TFIDFRetriever()
        self._agents: set[str] = set()
        self._publish_count: dict[str, int] = {}
        self._used_count: dict[str, int] = {}
        self._quick_inv_count: dict[str, int] = {}
        # MESI: per-agent high-water mark of versions pulled from shared pool.
        self._known_version: dict[str, int] = {}
        # Serialise concurrent publish() calls on the same pool instance.
        self._publish_lock = threading.Lock()

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
        quality = min(1.0, used / published)
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
                new_fact.valid_from = now
                new_fact.valid_until = None

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
            return []

        # Over-fetch so trust discount is applied before the top-k cutoff.
        # Without this, low-trust facts can displace better candidates by scoring
        # high in retrieval and then being down-ranked after the cut.
        overfetch_k = min(top_k * _POOL_RECALL_OVERFETCH, len(active))
        pairs = self._retriever.search(query, active, top_k=overfetch_k)

        # Apply per-agent trust discount before final cutoff.
        discounted: list[tuple[Fact, float]] = []
        for fact, score in pairs:
            if fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                trust = self.get_trust(fact.origin_agent_id)
                score *= trust
            discounted.append((fact, score))

        discounted.sort(key=lambda x: x[1], reverse=True)
        top_results = discounted[:top_k]

        # Track recall hits only for facts actually returned — not over-fetched
        # candidates that were discarded after trust discount.
        for fact, _ in top_results:
            if fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                self._used_count[fact.origin_agent_id] = (
                    self._used_count.get(fact.origin_agent_id, 0) + 1
                )

        return top_results

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
