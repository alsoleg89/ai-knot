"""KnowledgeBase — the main public API for ai_knot."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ai_knot.extractor import Extractor, resolve_against_existing
from ai_knot.forgetting import apply_decay
from ai_knot.providers import LLMProvider
from ai_knot.retriever import TFIDFRetriever
from ai_knot.storage.base import SnapshotCapable, StorageBackend
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType, SnapshotDiff

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
        **provider_kwargs: str,
    ) -> None:
        self._agent_id = agent_id
        self._storage: StorageBackend = storage or YAMLStorage()
        self._retriever = TFIDFRetriever()
        self._default_provider = provider
        self._default_api_key = api_key
        self._default_model = model
        self._default_provider_kwargs: dict[str, str] = dict(provider_kwargs)

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
            to_insert, _ = resolve_against_existing(
                new_facts, existing, threshold=conflict_threshold
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

    async def arecall(self, query: str, *, top_k: int = 5) -> str:
        """Async variant of :meth:`recall` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.recall(query, top_k=top_k))

    async def arecall_facts(self, query: str, *, top_k: int = 5) -> list[Fact]:
        """Async variant of :meth:`recall_facts` — non-blocking for asyncio applications.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.recall_facts(query, top_k=top_k))

    def recall(self, query: str, *, top_k: int = 5) -> str:
        """Retrieve relevant facts as a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.

        Returns:
            Formatted multi-line string, or "" if no facts found.
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return ""

        # Apply decay before searching.
        facts = apply_decay(facts)

        pairs = self._retriever.search(query, facts, top_k=top_k)
        if not pairs:
            return ""

        results = [f for f, _ in pairs]

        # Increment access_count on returned facts and persist.
        returned_ids = {r.id for r in results}
        now = datetime.now(UTC)
        for fact in facts:
            if fact.id in returned_ids:
                interval = (now - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = now
        self._storage.save(self._agent_id, facts)

        # Format for prompt injection.
        lines = [f"[{r.type.value}] {r.content}" for r in results]
        return "\n".join(lines)

    def list_facts(self) -> list[Fact]:
        """Return all stored facts for this agent.

        Returns:
            List of all Facts, in storage order.
        """
        return self._storage.load(self._agent_id)

    def recall_facts(self, query: str, *, top_k: int = 5) -> list[Fact]:
        """Structured alternative to recall() — returns Fact objects.

        Use when you need IDs, types, importance scores, or other metadata.
        Use recall() when you only need a formatted string for prompt injection.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.

        Returns:
            List of relevant Facts (may be empty), sorted by relevance.
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return []

        facts = apply_decay(facts)
        pairs = self._retriever.search(query, facts, top_k=top_k)
        if not pairs:
            return []

        results = [f for f, _ in pairs]
        returned_ids = {r.id for r in results}
        now = datetime.now(UTC)
        for fact in facts:
            if fact.id in returned_ids:
                interval = (now - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = now
        self._storage.save(self._agent_id, facts)
        return results

    def recall_facts_with_scores(self, query: str, *, top_k: int = 5) -> list[tuple[Fact, float]]:
        """Like recall_facts() but also returns the relevance score for each fact.

        The score is a hybrid value (TF-IDF + retention + importance). Use it
        to rank or filter results in integration adapters.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.

        Returns:
            List of (Fact, score) pairs sorted by relevance (most relevant first).
            Empty list if no facts stored or none match.
        """
        facts = self._storage.load(self._agent_id)
        if not facts:
            return []

        facts = apply_decay(facts)
        pairs = self._retriever.search(query, facts, top_k=top_k)
        if not pairs:
            return []

        returned_ids = {f.id for f, _ in pairs}
        now = datetime.now(UTC)
        for fact in facts:
            if fact.id in returned_ids:
                interval = (now - fact.last_accessed).total_seconds() / 3600.0
                fact.access_intervals.append(interval)
                if len(fact.access_intervals) > 20:
                    fact.access_intervals = fact.access_intervals[-20:]
                fact.access_count += 1
                fact.last_accessed = now
        self._storage.save(self._agent_id, facts)
        return pairs

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

    def decay(self) -> None:
        """Apply Ebbinghaus forgetting curve to all stored facts."""
        facts = self._storage.load(self._agent_id)
        if not facts:
            return
        apply_decay(facts)
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
        facts = self._storage.load(self._agent_id)
        if not facts:
            return {
                "total_facts": 0,
                "by_type": {"semantic": 0, "procedural": 0, "episodic": 0},
                "avg_importance": 0.0,
                "avg_retention": 0.0,
            }

        type_counts = Counter(f.type.value for f in facts)
        return {
            "total_facts": len(facts),
            "by_type": {
                "semantic": type_counts.get("semantic", 0),
                "procedural": type_counts.get("procedural", 0),
                "episodic": type_counts.get("episodic", 0),
            },
            "avg_importance": sum(f.importance for f in facts) / len(facts),
            "avg_retention": sum(f.retention_score for f in facts) / len(facts),
        }
