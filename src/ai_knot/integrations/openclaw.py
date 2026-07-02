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

    from ai_knot import Fact, KnowledgeBase
    from ai_knot.integrations.openclaw import OpenClawMemoryAdapter

    kb = KnowledgeBase("my_agent")
    memory = OpenClawMemoryAdapter(kb)
    created = memory.add([{"role": "user", "content": "I love Python"}])
    results = memory.search("programming language", top_k=3)
    memory.list()
    structured = kb.add_resolved([
        Fact(content="User works at Acme", entity="user", attribute="employer", value_text="Acme")
    ])[0]
    current = memory.update(structured.id, "User now works at Globex")
    memory.lineage(current["id"])
    memory.get_all(include_inactive=True)
    memory.delete(created["results"][0]["id"])
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import Any, Literal

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact, MemoryOp


def _parse_now(value: str | None) -> datetime | None:
    """Parse an ISO-8601 anchor for active/inactive inspection views."""
    if not value:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _fact_to_item(
    fact: Fact,
    *,
    score: float | None = None,
    active_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "id": fact.id,
        "memory": fact.content,
        "score": score,
        "metadata": {
            "type": str(fact.type),
            "importance": fact.importance,
            "tags": list(fact.tags),
            "created_at": fact.created_at.isoformat(),
            "event_time": fact.event_time.isoformat() if fact.event_time else None,
            "valid_from": fact.valid_from.isoformat(),
            "valid_until": fact.valid_until.isoformat() if fact.valid_until else None,
            "active": fact.is_active(active_at),
            "slot_key": fact.slot_key or None,
            "entity": fact.entity or None,
            "attribute": fact.attribute or None,
            "value_text": fact.value_text or None,
            "version": fact.version,
        },
    }


class OpenClawMemoryAdapter:
    """Wraps KnowledgeBase to match the OpenClaw Mem0Provider interface.

    Mirrors the five-method interface used by community plugins such as
    openclaw-mem0 and openclaw-supermemory, so Python-native agents can
    swap providers without changing call sites.

    The provider-compatible verbs stay available as ``add()``, ``search()``,
    ``get_all()``, ``update()``, and ``delete()``. For cross-surface ai-knot
    consistency, the adapter also exposes ``recall()``, ``list()``,
    ``lineage()``, and ``forget()`` aliases.

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

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Alias for search() using ai-knot's recall wording."""
        return self.search(query, top_k=top_k, **kwargs)

    def get(self, memory_id: str, *, now: str | None = None) -> dict[str, Any]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: 8-char hex fact ID.
            now: Optional ISO-8601 anchor for active/inactive status.

        Returns:
            MemoryItem dict.

        Raises:
            KeyError: If no fact with that ID exists.
        """
        anchor = _parse_now(now)
        for fact in self._kb.list_facts():
            if fact.id == memory_id:
                return _fact_to_item(fact, active_at=anchor)
        raise KeyError(memory_id)

    def get_all(
        self,
        *,
        user_id: str | None = None,  # noqa: ARG002 — kept as user_id for API compat, not _user_id
        include_inactive: bool = False,
        now: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Return stored memories.

        Args:
            user_id: Accepted for interface compatibility; ignored.
            include_inactive: When True, include superseded / inactive memories.
            now: Optional ISO-8601 anchor for active/inactive filtering.

        Returns:
            List of MemoryItem dicts for the current active facts by default.
        """
        anchor = _parse_now(now)
        facts = self._kb.list_facts()
        if not include_inactive:
            facts = [fact for fact in facts if fact.is_active(anchor)]
        return [_fact_to_item(f, active_at=anchor) for f in facts]

    def list(
        self,
        *,
        user_id: str | None = None,
        include_inactive: bool = False,
        now: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Alias for get_all() so the base ai-knot loop stays add/search/list/delete."""
        return self.get_all(
            user_id=user_id,
            include_inactive=include_inactive,
            now=now,
            **kwargs,
        )

    def lineage(self, memory_id: str, *, now: str | None = None) -> list[dict[str, Any]]:
        """Return the supersession chain for one memory, newest -> oldest.

        Args:
            memory_id: 8-char hex fact ID to trace.
            now: Optional ISO-8601 anchor for active/inactive status.

        Returns:
            MemoryItem dicts from the given fact back to the original it
            superseded. Returns ``[]`` when the ID is unknown.
        """
        anchor = _parse_now(now)
        return [_fact_to_item(fact, active_at=anchor) for fact in self._kb.lineage(memory_id)]

    def update(self, memory_id: str, data: str, **_: Any) -> dict[str, Any]:
        """Replace the content of an existing memory.

        For structured facts (slot-addressed or entity+attribute memories), the
        update flows through ai-knot's supersession pipeline so lineage is
        preserved and the old fact becomes inactive instead of disappearing.
        For unstructured facts, the adapter falls back to delete + add because
        there is no stable slot to supersede.

        Args:
            memory_id: 8-char hex ID of the fact to replace.
            data: New content string.

        Returns:
            MemoryItem dict for the replacement fact. The returned item has a
            new ID; provider callers should use it as the new current handle.

        Raises:
            KeyError: If no fact with that ID exists.
        """
        current = self._kb.get(memory_id)

        if current.slot_key or (current.entity and current.attribute):
            replacement = Fact(
                content=data,
                type=current.type,
                importance=current.importance,
                tags=list(current.tags),
                entity=current.entity,
                attribute=current.attribute,
                slot_key=current.slot_key,
                value_text=data,
                op=MemoryOp.UPDATE,
            )
            inserted = self._kb.add_resolved([replacement])
            if inserted:
                return _fact_to_item(inserted[0])

        self._kb.forget(memory_id)
        fact = self._kb.add(
            data,
            type=current.type,
            importance=current.importance,
            tags=list(current.tags),
        )
        return _fact_to_item(fact)

    def delete(self, memory_id: str, **_: Any) -> None:
        """Remove a memory by ID.

        Args:
            memory_id: 8-char hex fact ID.
        """
        self._kb.forget(memory_id)

    def forget(self, memory_id: str, **kwargs: Any) -> None:
        """Alias for delete() using ai-knot's memory wording."""
        self.delete(memory_id, **kwargs)


def generate_mcp_config(
    agent_id: str = "default",
    data_dir: str = ".ai_knot",
    storage: Literal["sqlite", "yaml"] = "sqlite",
) -> dict[str, Any]:
    """Return the MCP server config snippet for ai-knot-mcp.

    Paste the returned dict into your MCP client config file:

    - Claude Desktop: ``~/Library/Application Support/Claude/claude_desktop_config.json``
    - OpenClaw: ``~/.openclaw/openclaw.json``

    SQLite is the default backend because WAL mode (already enabled)
    handles concurrent agent reads without write locks.

    Args:
        agent_id: Agent namespace passed to ai-knot-mcp.
        data_dir: Directory where ai-knot stores its data (resolved to an
            absolute path so the config works from any working directory).
        storage: Backend type — ``"sqlite"`` (recommended) or ``"yaml"``.

    Returns:
        Dict ready for ``json.dumps()``.

    Raises:
        ValueError: If ``storage`` is not ``"sqlite"`` or ``"yaml"``.

    Example::

        import json
        from ai_knot.integrations.openclaw import generate_mcp_config

        print(json.dumps(generate_mcp_config("my_agent"), indent=2))
        # Paste into your MCP client config under "mcpServers"
    """
    from pathlib import Path

    if storage not in ("sqlite", "yaml"):
        raise ValueError(f"storage must be 'sqlite' or 'yaml', got {storage!r}")

    abs_data_dir = str(Path(data_dir).resolve())
    return {
        "mcpServers": {
            "ai-knot": {
                "command": "ai-knot-mcp",
                "env": {
                    "AI_KNOT_AGENT_ID": agent_id,
                    "AI_KNOT_DATA_DIR": abs_data_dir,
                    "AI_KNOT_STORAGE": storage,
                },
            }
        }
    }
