"""KnowledgeBase — the main public API for agentmemo."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from agentmemo.extractor import Extractor
from agentmemo.forgetting import apply_decay
from agentmemo.retriever import TFIDFRetriever
from agentmemo.storage.base import StorageBackend
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import ConversationTurn, Fact, MemoryType

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Agent knowledge store with extraction, retrieval, and forgetting.

    Usage::

        kb = KnowledgeBase(agent_id="my_agent")
        kb.add("User prefers Python", importance=0.9)
        context = kb.recall("what language?")
        # → "[procedural] User prefers Python"

    Args:
        agent_id: Unique identifier for this agent's memory namespace.
        storage: Storage backend (defaults to YAMLStorage in .agentmemo/).
    """

    def __init__(
        self,
        agent_id: str,
        storage: StorageBackend | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._storage: StorageBackend = storage or YAMLStorage()
        self._retriever = TFIDFRetriever()

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

    def learn(
        self,
        turns: list[ConversationTurn],
        *,
        api_key: str | None = None,
        provider: str = "openai",
    ) -> list[Fact]:
        """Extract and store facts from a conversation using an LLM.

        Args:
            turns: Conversation messages to extract knowledge from.
            api_key: LLM API key. Required for extraction.
            provider: "openai" or "anthropic".

        Returns:
            List of newly extracted and stored Facts.
        """
        if not turns:
            return []
        if not api_key:
            logger.warning("No API key provided — skipping LLM extraction")
            return []

        extractor = Extractor(api_key=api_key, provider=provider)
        new_facts = extractor.extract(turns)

        if new_facts:
            existing = self._storage.load(self._agent_id)
            existing.extend(new_facts)
            self._storage.save(self._agent_id, existing)
            logger.info(
                "Learned %d facts from conversation for agent '%s'",
                len(new_facts),
                self._agent_id,
            )
        return new_facts

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

        results = self._retriever.search(query, facts, top_k=top_k)
        if not results:
            return ""

        # Increment access_count on returned facts and persist.
        returned_ids = {r.id for r in results}
        now = datetime.now(timezone.utc)
        for fact in facts:
            if fact.id in returned_ids:
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
