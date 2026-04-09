"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_knot._query_intent import (
    _classify_pool_query,
    _PoolQueryIntent,
)
from ai_knot.forgetting import apply_decay
from ai_knot.learning import _LearningMixin
from ai_knot.providers import LLMProvider, create_provider
from ai_knot.query_expander import LLMQueryExpander
from ai_knot.retriever import DenseRetriever, HybridRetriever, TFIDFRetriever
from ai_knot.storage.base import (
    SnapshotCapable,
    StorageBackend,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import (
    Fact,
    MemoryType,
    SnapshotDiff,
)

# LLM expansion token weight — higher than PRF (0.5) to reflect
# the semantic advantage of LLM-based query understanding.
_LLM_EXPANSION_WEIGHT: float = 0.6

_LEARN_DEBUG = bool(os.environ.get("AI_KNOT_LEARN_DEBUG", ""))

logger = logging.getLogger(__name__)

# Patch the _LEARN_DEBUG flag into the learning module so pipeline stages log correctly.
import ai_knot.learning as _learning_mod  # noqa: E402

_learning_mod._LEARN_DEBUG = _LEARN_DEBUG


class KnowledgeBase(_LearningMixin):
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
        all_facts = self._storage.load(self._agent_id)
        if not all_facts:
            return []

        now_dt = now or datetime.now(UTC)
        # Temporal filter + exclude episodic (raw buffer) + verification gate.
        facts = [
            f
            for f in all_facts
            if f.is_active(now_dt)
            and f.type != MemoryType.EPISODIC
            and (include_unsupported or f.supported is not False)
        ]
        if not facts:
            return []

        facts = apply_decay(facts, type_exponents=self._decay_config, now=now_dt)
        expanded_query, expansion = self._expand_query(query)

        # Intent-aware RRF weights for private KB (same classifier as pool).
        # Only apply intent overrides when no custom RRF weights were supplied
        # by the caller — custom weights represent an explicit user choice that
        # should not be silently replaced by intent heuristics.
        rrf_override: tuple[float, ...] | None = None
        if self._bm25._rrf_weights == (5.0, 3.0, 2.0, 1.5, 1.5, 1.0):
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
        # Update access metadata on the *original* (unfiltered) fact objects so
        # that the subsequent save persists ALL facts, not just the filtered
        # working set.  ``facts`` entries are the same objects as in
        # ``all_facts``, so mutating them here propagates automatically.
        for fact in facts:
            if fact.id in returned_ids:
                interval = (access_time - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = access_time
        self._storage.save(self._agent_id, all_facts)
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
        seen: set[str] = set()
        lines: list[str] = []
        for f, _ in pairs:
            text = f.prompt_surface or f.source_verbatim or f.content
            if text not in seen:
                seen.add(text)
                lines.append(f"[{len(lines) + 1}] {text}")
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


# Backward-compat re-export: callers that do
#   from ai_knot.knowledge import SharedMemoryPool
# continue to work without changes.
from ai_knot.pool import SharedMemoryPool as SharedMemoryPool  # noqa: E402, F401
