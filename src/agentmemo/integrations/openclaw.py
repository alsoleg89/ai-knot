"""OpenClaw memory adapter.

Provides two integration paths:

1. OpenClawMemoryAdapter — Python class mirroring the Mem0Provider interface
   used by OpenClaw community plugins. For Python-native agents or when
   migrating from a Mem0-based setup.

2. generate_mcp_config() — returns the mcpServers JSON snippet to wire
   agentmemo-mcp into OpenClaw at the TypeScript level (the recommended
   runtime integration for OpenClaw users).

Runtime integration example (openclaw.json)::

    import json
    from agentmemo.integrations.openclaw import generate_mcp_config
    print(json.dumps(generate_mcp_config("my_agent"), indent=2))

Python-native example::

    from agentmemo import KnowledgeBase
    from agentmemo.integrations.openclaw import OpenClawMemoryAdapter

    kb = KnowledgeBase("my_agent")
    memory = OpenClawMemoryAdapter(kb)
    memory.add([{"role": "user", "content": "I love Python"}])
    results = memory.search("programming language", top_k=3)
"""

from __future__ import annotations

from typing import Any

from agentmemo.knowledge import KnowledgeBase


def _fact_to_item(fact: Any, *, score: float | None = None) -> dict[str, Any]:
    return {
        "id": fact.id,
        "memory": fact.content,
        "score": score,
        "metadata": {
            "type": str(fact.type),
            "importance": fact.importance,
            "created_at": fact.created_at.isoformat(),
        },
    }


class OpenClawMemoryAdapter:
    """Wraps KnowledgeBase to match the OpenClaw Mem0Provider interface.

    Mirrors the five-method interface used by community plugins such as
    openclaw-mem0 and openclaw-supermemory, so Python-native agents can
    swap providers without changing call sites.

    Args:
        knowledge_base: The KnowledgeBase instance to delegate to.
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str | None = None,  # noqa: ARG002 — accepted for interface compat
        **_: Any,
    ) -> dict[str, Any]:
        """Store information from a message list.

        Single-message lists store the message directly. Multi-turn lists
        store the last user message. For LLM-powered extraction from
        multi-turn conversations, call ``kb.learn()`` directly and wrap
        the result.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            user_id: Accepted for interface compatibility; ignored (isolation
                is handled by ``agent_id`` at KnowledgeBase construction).

        Returns:
            ``{"results": [{"id": str, "memory": str, "event": "ADD"}]}``
        """
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            fact = self._kb.add(user_msgs[-1]["content"])
            return {"results": [{"id": fact.id, "memory": fact.content, "event": "ADD"}]}
        return {"results": []}

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Search memories by text query.

        Args:
            query: Natural language search string.
            top_k: Maximum number of results.

        Returns:
            List of MemoryItem dicts (id, memory, score, metadata).
        """
        raw = self._kb.recall(query, top_k=top_k)
        if not raw:
            return []
        # recall() returns "[type] content\n[type] content\n..."
        # Reconstruct MemoryItems by matching content against stored facts.
        contents = {line.split("] ", 1)[1] for line in raw.splitlines() if "] " in line}
        facts = [f for f in self._kb.list_facts() if f.content in contents]
        return [_fact_to_item(f, score=f.retention_score) for f in facts]

    def get(self, memory_id: str) -> dict[str, Any]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: 8-char hex fact ID.

        Returns:
            MemoryItem dict.

        Raises:
            KeyError: If no fact with that ID exists.
        """
        for fact in self._kb.list_facts():
            if fact.id == memory_id:
                return _fact_to_item(fact)
        raise KeyError(memory_id)

    def get_all(
        self,
        *,
        user_id: str | None = None,  # noqa: ARG002 — accepted for interface compat
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Return all stored memories.

        Args:
            user_id: Accepted for interface compatibility; ignored.

        Returns:
            List of MemoryItem dicts for all stored facts.
        """
        return [_fact_to_item(f) for f in self._kb.list_facts()]

    def delete(self, memory_id: str, **_: Any) -> None:
        """Remove a memory by ID.

        Args:
            memory_id: 8-char hex fact ID.
        """
        self._kb.forget(memory_id)


def generate_mcp_config(
    agent_id: str = "default",
    data_dir: str = ".agentmemo",
    storage: str = "sqlite",
) -> dict[str, Any]:
    """Return the OpenClaw mcpServers config snippet for agentmemo-mcp.

    Paste the returned dict into your ``openclaw.json`` under
    ``"mcpServers"``. SQLite is the default backend because WAL mode
    (already enabled) handles concurrent agent reads without write locks.

    Args:
        agent_id: Agent namespace passed to agentmemo-mcp.
        data_dir: Directory where agentmemo stores its data.
            The SQLite file will be at ``{data_dir}/agentmemo.db``.
        storage: Backend type — ``"sqlite"`` (recommended) or ``"yaml"``.

    Returns:
        Dict ready for ``json.dumps()``.

    Example::

        import json
        from agentmemo.integrations.openclaw import generate_mcp_config

        print(json.dumps(generate_mcp_config("my_agent"), indent=2))
        # {
        #   "mcpServers": {
        #     "agentmemo": {
        #       "command": "agentmemo-mcp",
        #       "env": {
        #         "AGENTMEMO_AGENT_ID": "my_agent",
        #         "AGENTMEMO_DATA_DIR": ".agentmemo",
        #         "AGENTMEMO_STORAGE": "sqlite"
        #       }
        #     }
        #   }
        # }
    """
    return {
        "mcpServers": {
            "agentmemo": {
                "command": "agentmemo-mcp",
                "env": {
                    "AGENTMEMO_AGENT_ID": agent_id,
                    "AGENTMEMO_DATA_DIR": data_dir,
                    "AGENTMEMO_STORAGE": storage,
                },
            }
        }
    }
