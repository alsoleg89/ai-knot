"""OpenClaw memory adapter.

Provides two integration paths:

**Which path should I use?**

+----------------------+-------------------------------------------+
| Situation            | Solution                                  |
+======================+===========================================+
| OpenClaw TypeScript  | generate_mcp_config() → paste into        |
| app (recommended)    | ~/.openclaw/openclaw.json                 |
+----------------------+-------------------------------------------+
| Python agent         | OpenClawMemoryAdapter(kb)                 |
| (LangChain, custom)  |                                           |
+----------------------+-------------------------------------------+

For multi-turn fact extraction in Python: use ``kb.learn(turns, api_key=...)``
directly — ``add()`` stores only the last user message without an LLM.

Runtime integration example::

    import json
    from ai_knot.integrations.openclaw import generate_mcp_config

    print(json.dumps(generate_mcp_config("my_agent"), indent=2))
    # Paste output into ~/.openclaw/openclaw.json (macOS/Linux)
    # or %APPDATA%\\OpenClaw\\openclaw.json (Windows)

Python-native example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.openclaw import OpenClawMemoryAdapter

    kb = KnowledgeBase("my_agent")
    memory = OpenClawMemoryAdapter(kb)
    memory.add([{"role": "user", "content": "I love Python"}])
    results = memory.search("programming language", top_k=3)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact


def _fact_to_item(fact: Fact, *, score: float | None = None) -> dict[str, Any]:
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
        user_id: str | None = None,  # noqa: ARG002 — kept as user_id for API compat, not _user_id
        **_: Any,
    ) -> dict[str, Any]:
        """Store information from a message list.

        Stores the last user message. For LLM-powered extraction from
        multi-turn conversations (extracting multiple facts at once),
        call ``kb.learn()`` directly and wrap the result.

        Args:
            messages: List of ``{"role": str, "content": str}`` dicts.
            user_id: Accepted for interface compatibility; ignored (isolation
                is handled by ``agent_id`` at KnowledgeBase construction).

        Returns:
            ``{"results": [{"id": str, "memory": str, "event": "ADD"}]}``
        """
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if len(user_msgs) > 1:
            warnings.warn(
                "add() received multi-turn history but has no LLM — storing only the last "
                "user message. For full extraction call kb.learn(turns, api_key=...) directly.",
                stacklevel=2,
            )
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
            ``score`` is a hybrid relevance value (TF-IDF + retention + importance).
        """
        pairs = self._kb.recall_facts_with_scores(query, top_k=top_k)
        return [_fact_to_item(f, score=round(score, 4)) for f, score in pairs]

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
        user_id: str | None = None,  # noqa: ARG002 — kept as user_id for API compat, not _user_id
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Return all stored memories.

        Args:
            user_id: Accepted for interface compatibility; ignored.

        Returns:
            List of MemoryItem dicts for all stored facts.
        """
        return [_fact_to_item(f) for f in self._kb.list_facts()]

    def update(self, memory_id: str, data: str, **_: Any) -> dict[str, Any]:
        """Replace the content of an existing memory.

        The fact is deleted and re-created, so the returned item has a
        new ID. If ID stability matters, use delete() + add() explicitly
        and record the new ID.

        Args:
            memory_id: 8-char hex ID of the fact to replace.
            data: New content string.

        Returns:
            MemoryItem dict for the replacement fact.
        """
        self._kb.forget(memory_id)
        fact = self._kb.add(data)
        return _fact_to_item(fact)

    def delete(self, memory_id: str, **_: Any) -> None:
        """Remove a memory by ID.

        Args:
            memory_id: 8-char hex fact ID.
        """
        self._kb.forget(memory_id)


def generate_mcp_config(
    agent_id: str = "default",
    data_dir: str = ".ai_knot",
    storage: Literal["sqlite", "yaml"] = "sqlite",
) -> dict[str, Any]:
    """Return the OpenClaw mcpServers config snippet for ai-knot-mcp.

    Paste the returned dict into your OpenClaw config file:

    - macOS / Linux: ``~/.openclaw/openclaw.json``
    - Windows:       ``%APPDATA%\\OpenClaw\\openclaw.json``

    SQLite is the default backend because WAL mode (already enabled)
    handles concurrent agent reads without write locks.

    Args:
        agent_id: Agent namespace passed to ai-knot-mcp.
        data_dir: Directory where ai-knot stores its data.
            The SQLite file will be at ``{data_dir}/ai_knot.db``.
        storage: Backend type — ``"sqlite"`` (recommended) or ``"yaml"``.

    Returns:
        Dict ready for ``json.dumps()``.

    Raises:
        ValueError: If ``storage`` is not ``"sqlite"`` or ``"yaml"``.

    Example::

        import json
        from ai_knot.integrations.openclaw import generate_mcp_config

        print(json.dumps(generate_mcp_config("my_agent"), indent=2))
        # Paste into ~/.openclaw/openclaw.json under "mcpServers"
    """
    if storage not in ("sqlite", "yaml"):
        raise ValueError(f"storage must be 'sqlite' or 'yaml', got {storage!r}")
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mcp package is required to use ai-knot-mcp. Install with: pip install 'ai-knot[mcp]'"
        ) from exc
    return {
        "mcpServers": {
            "ai-knot": {
                "command": "ai-knot-mcp",
                "env": {
                    "AI_KNOT_AGENT_ID": agent_id,
                    "AI_KNOT_DATA_DIR": data_dir,
                    "AI_KNOT_STORAGE": storage,
                },
            }
        }
    }
