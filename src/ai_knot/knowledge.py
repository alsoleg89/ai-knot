"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_knot.extractor import Extractor, resolve_against_existing, resolve_structured
from ai_knot.forgetting import apply_decay
from ai_knot.providers import LLMProvider, create_provider
from ai_knot.query_expander import LLMQueryExpander
from ai_knot.retriever import TFIDFRetriever
from ai_knot.storage.base import SnapshotCapable, StorageBackend, TemporalStorageCapable
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType, MESIState, SnapshotDiff

# LLM expansion token weight — higher than PRF (0.5) to reflect
# the semantic advantage of LLM-based query understanding.
_LLM_EXPANSION_WEIGHT: float = 0.6

_SHARED_NAMESPACE = "__shared__"
_PROVENANCE_DISCOUNT = 0.8

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
            rrf_weights=rrf_weights or (5.0, 2.0, 2.0, 1.0),
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

        Existing facts that are similar to newly extracted ones (Jaccard >=
        ``conflict_threshold``) are updated in place (importance bumped, access
        time refreshed) instead of being duplicated.

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
            conflict_threshold: Jaccard similarity threshold above which a new
                fact is treated as a duplicate of an existing one (0.0–1.0).
            timeout: Per-request timeout in seconds for LLM calls. ``None``
                uses the provider's built-in default (30 s).
            batch_size: Maximum conversation turns sent per LLM call. Longer
                conversations are split into batches to prevent JSON truncation.
            **provider_kwargs: Extra args forwarded to the provider constructor
                (e.g. ``folder_id`` for Yandex, ``base_url`` for openai-compat).
                Merged with any defaults set at init, with per-call values taking
                precedence.

        Returns:
            List of genuinely new Facts that were inserted (excludes updates).
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

            # Phase 1: entity-addressed close (temporal versioning for same entity+attribute)
            now_close = datetime.now(UTC)
            for new_fact in new_facts:
                matched_fact = resolve_structured(new_fact, existing)
                if matched_fact is not None:
                    matched_fact.valid_until = now_close

            # Phase 2: lexical dedup against active facts only
            active_existing = [f for f in existing if f.is_active(now_close)]
            to_insert, _ = resolve_against_existing(
                new_facts, active_existing, threshold=conflict_threshold
            )
            self._storage.save(self._agent_id, existing + to_insert)
            logger.info(
                "Learned %d new facts (%d merged) for agent '%s'",
                len(to_insert),
                len(new_facts) - len(to_insert),
                self._agent_id,
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

    async def arecall(self, query: str, *, top_k: int = 5, now: datetime | None = None) -> str:
        """Async variant of :meth:`recall` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.recall(query, top_k=top_k, now=now))

    async def arecall_facts(
        self, query: str, *, top_k: int = 5, now: datetime | None = None
    ) -> list[Fact]:
        """Async variant of :meth:`recall_facts` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.recall_facts(query, top_k=top_k, now=now)
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
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return []

        now_dt = now or datetime.now(UTC)
        # Temporal filter + exclude episodic (raw buffer) in one pass.
        facts = [f for f in facts if f.is_active(now_dt) and f.type != MemoryType.EPISODIC]
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

    def recall(self, query: str, *, top_k: int = 5, now: datetime | None = None) -> str:
        """Retrieve relevant facts as a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        pairs = self._execute_recall(query, top_k=top_k, now=now)
        if not pairs:
            return ""
        lines = [f"[{f.type.value}] {f.source_verbatim or f.content}" for f, _ in pairs]
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
    ) -> list[Fact]:
        """Structured alternative to recall() — returns Fact objects.

        Use when you need IDs, types, importance scores, or other metadata.
        Use recall() when you only need a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        return [
            f
            for f, _ in self._execute_recall(query, top_k=top_k, now=now, excluded_ids=excluded_ids)
        ]

    def recall_facts_with_scores(
        self,
        query: str,
        *,
        top_k: int = 5,
        now: datetime | None = None,
        excluded_ids: set[str] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Like recall_facts() but also returns the relevance score for each fact.

        The score is a hybrid value (TF-IDF + retention + importance). Use it
        to rank or filter results in integration adapters.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Point-in-time for decay calculation (default: current UTC).
            excluded_ids: Fact IDs to omit from results (novelty-aware retrieval).

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Empty list if no facts stored or none match.
        """
        return self._execute_recall(query, top_k=top_k, now=now, excluded_ids=excluded_ids)

    def recall_by_tag(self, tag: str) -> list[Fact]:
        """Return all facts that carry the given tag.

        Tags are assigned at add() time via the ``tags=`` parameter.

        Args:
            tag: The tag string to filter by.

        Returns:
            List of Facts whose tags include ``tag`` (may be empty).
        """
        return [f for f in self._storage.load(self._agent_id) if tag in f.tags]

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

    Trust starts at ``_PROVENANCE_DISCOUNT`` (0.8) for new agents and can
    be adjusted via :meth:`update_trust` based on feedback quality.
    Agents that consistently provide relevant facts earn higher trust;
    unreliable agents can be penalized.

    Usage::

        pool = SharedMemoryPool(storage=SQLiteStorage("mem.db"))
        pool.register("devops_agent")
        pool.register("coding_agent")

        # DevOps agent publishes a fact
        pool.publish("devops_agent", [fact_id], kb=devops_kb)

        # Coding agent queries the shared pool
        results = pool.recall("what database?", "coding_agent", top_k=5)

        # Boost devops_agent trust after positive feedback
        pool.update_trust("devops_agent", 0.05)

    Args:
        storage: Backend used to persist the shared namespace.
    """

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage: StorageBackend = storage or YAMLStorage()
        self._retriever = TFIDFRetriever()
        self._agents: set[str] = set()
        self._trust_scores: dict[str, float] = {}
        # MESI: per-agent high-water mark of versions pulled from shared pool.
        # Used by sync_dirty() to avoid re-fetching unchanged facts.
        self._known_version: dict[str, int] = {}

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

    def update_trust(self, agent_id: str, delta: float) -> float:
        """Adjust trust score for an agent (Marsh 1994 differential trust).

        Trust is clamped to [0.1, 1.0].  Positive ``delta`` rewards agents
        whose shared facts proved relevant; negative ``delta`` penalizes
        unreliable sources.

        Args:
            agent_id: The agent whose trust to adjust.
            delta: Amount to add (positive = reward, negative = penalize).

        Returns:
            The updated trust score.
        """
        current = self._trust_scores.get(agent_id, _PROVENANCE_DISCOUNT)
        updated = max(0.1, min(1.0, current + delta))
        self._trust_scores[agent_id] = updated
        return updated

    def get_trust(self, agent_id: str) -> float:
        """Return the current trust score for an agent.

        Args:
            agent_id: The agent to query.

        Returns:
            Trust score (0.1-1.0), defaulting to ``_PROVENANCE_DISCOUNT``.
        """
        return self._trust_scores.get(agent_id, _PROVENANCE_DISCOUNT)

    def publish(
        self,
        agent_id: str,
        fact_ids: list[str],
        *,
        kb: KnowledgeBase,
    ) -> list[Fact]:
        """Copy facts from an agent's private KB into the shared pool.

        Uses entity-addressed CAS: for each fact with entity+attribute, the
        existing active fact for that entity+attribute is closed (valid_until=now,
        mesi_state='I') and the new fact is inserted (mesi_state='M' if replacing,
        'S' if new). Generic facts (no entity) fall back to ID-based dedup.

        Args:
            agent_id: The agent publishing the facts.
            fact_ids: IDs of facts to publish from the agent's KB.
            kb: The agent's KnowledgeBase instance.

        Returns:
            List of facts that were published (copies, not mutations of private KB).

        Raises:
            ValueError: If agent_id is not registered.
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id!r} is not registered. Call register() first.")

        private_facts = kb.list_facts()
        id_set = set(fact_ids)
        to_publish = [f for f in private_facts if f.id in id_set]

        if not to_publish:
            return []

        now = datetime.now(UTC)
        shared = self._storage.load(_SHARED_NAMESPACE)

        # Index active shared facts by (entity, attribute) for O(1) lookup.
        active_by_entity: dict[tuple[str, str], Fact] = {}
        existing_ids: set[str] = set()
        for f in shared:
            existing_ids.add(f.id)
            if f.entity and f.attribute and f.is_active(now):
                active_by_entity[(f.entity.lower().strip(), f.attribute.lower().strip())] = f

        published: list[Fact] = []
        for fact in to_publish:
            new_fact = copy.deepcopy(fact)
            new_fact.origin_agent_id = agent_id
            new_fact.visibility = "pool"
            new_fact.valid_from = now
            new_fact.valid_until = None

            key = (fact.entity.lower().strip(), fact.attribute.lower().strip())
            if fact.entity and fact.attribute and key in active_by_entity:
                old = active_by_entity[key]
                if old.id == fact.id:
                    # Same fact already published as the active version — no-op.
                    continue
                # Entity-addressed CAS: close old version, insert new.
                old.valid_until = now
                old.mesi_state = MESIState.INVALID
                old.version += 1
                new_fact.mesi_state = MESIState.MODIFIED
                new_fact.version = old.version
            elif new_fact.id not in existing_ids:
                new_fact.mesi_state = MESIState.SHARED
                new_fact.version = 1
            else:
                # ID already in pool and no entity key — skip duplicate.
                continue

            shared.append(new_fact)
            published.append(new_fact)

        if published:
            self._storage.save(_SHARED_NAMESPACE, shared)
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

        if not active:
            return []

        pairs = self._retriever.search(query, active, top_k=top_k)

        # Apply per-agent trust discount for foreign facts (Marsh 1994).
        discounted: list[tuple[Fact, float]] = []
        for fact, score in pairs:
            if fact.origin_agent_id and fact.origin_agent_id != requesting_agent_id:
                trust = self._trust_scores.get(fact.origin_agent_id, _PROVENANCE_DISCOUNT)
                score *= trust
            discounted.append((fact, score))

        discounted.sort(key=lambda x: x[1], reverse=True)
        return discounted[:top_k]

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

    def list_shared_facts(self) -> list[Fact]:
        """Return all facts in the shared pool.

        Returns:
            List of all shared Facts (including closed/invalidated).
        """
        return self._storage.load(_SHARED_NAMESPACE)
