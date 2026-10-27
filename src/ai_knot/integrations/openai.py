"""OpenAI integration — memory-enabled wrapper.

Wraps OpenAI-style message lists with automatic knowledge retrieval
and injection into the system prompt.
"""

from __future__ import annotations

import copy

from ai_knot.integrations._task_context import task_query, user_texts
from ai_knot.knowledge import KnowledgeBase


class MemoryEnabledOpenAI:
    """Wraps an OpenAI-compatible client to inject agent memory.

    Usage:
        kb = KnowledgeBase(agent_id="assistant")
        client = MemoryEnabledOpenAI(knowledge_base=kb)
        enriched = client.enrich_messages(messages)
        # Pass enriched messages to your OpenAI client.

    Args:
        knowledge_base: The KnowledgeBase to use for recall.
        auto_learn: If True, automatically extract facts from responses.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        auto_learn: bool = False,
    ) -> None:
        self._kb = knowledge_base
        self._auto_learn = auto_learn

    def enrich_messages(
        self,
        messages: list[dict[str, str]],
        *,
        task_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Inject relevant knowledge into the system prompt.

        If a system message exists, appends memory context to it.
        If no system message exists, creates one with memory context.

        Args:
            messages: OpenAI-format message list.
            task_context: Optional host task hint for an anaphoric follow-up.

        Returns:
            New message list with memory context injected.
        """
        query = task_query(user_texts(messages), task_context=task_context)
        if not query:
            return messages
        context = self._kb.recall(query)

        if not context:
            return messages

        memory_block = f"\n\n## Agent Memory\n{context}"

        enriched = copy.deepcopy(messages)

        # Check for existing system message.
        system_indices = [i for i, m in enumerate(enriched) if m.get("role") == "system"]

        if system_indices:
            # Append to existing system prompt.
            idx = system_indices[0]
            enriched[idx]["content"] += memory_block
        else:
            # Insert new system message at the beginning.
            enriched.insert(0, {"role": "system", "content": memory_block.strip()})

        return enriched
