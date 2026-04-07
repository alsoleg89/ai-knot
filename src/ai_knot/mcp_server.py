"""MCP server for ai-knot — exposes KnowledgeBase as MCP tools.

Run with::

    ai-knot-mcp

or::

    AI_KNOT_AGENT_ID=my_agent python -m ai_knot.mcp_server

Configuration is via environment variables:

- ``AI_KNOT_AGENT_ID``   — agent namespace (default: "default")
- ``AI_KNOT_STORAGE``    — backend: "yaml" or "sqlite" (default: "sqlite")
- ``AI_KNOT_DATA_DIR``   — base directory for file backends (default: ".ai_knot")
- ``AI_KNOT_DB_PATH``    — full path to SQLite file (overrides DATA_DIR for sqlite)
- ``AI_KNOT_PROVIDER``   — LLM provider for learn() (e.g. "anthropic")
- ``AI_KNOT_API_KEY``    — API key for the LLM provider
- ``AI_KNOT_MODEL``      — model name override for learn()
"""

from __future__ import annotations

import json
import os
from typing import Any

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage
from ai_knot.types import ConversationTurn, MemoryType


def _build_kb() -> KnowledgeBase:
    """Construct a KnowledgeBase from environment variables.

    Returns:
        A configured KnowledgeBase instance.
    """
    agent_id = os.environ.get("AI_KNOT_AGENT_ID", "default")
    backend = os.environ.get("AI_KNOT_STORAGE", "sqlite")
    data_dir = os.environ.get("AI_KNOT_DATA_DIR", ".ai_knot")
    db_path = os.environ.get("AI_KNOT_DB_PATH")

    dsn = db_path or os.path.join(data_dir, "ai_knot.db") if backend == "sqlite" else None
    storage = create_storage(backend, base_dir=data_dir, dsn=dsn)
    return KnowledgeBase(agent_id=agent_id, storage=storage)


# ---------------------------------------------------------------------------
# Tool implementations (pure functions, testable without mcp installed)
# ---------------------------------------------------------------------------


def tool_add(
    kb: KnowledgeBase,
    content: str,
    *,
    type: str = "semantic",
    importance: float = 0.8,
    tags: list[str] | None = None,
) -> str:
    """Add a fact to the knowledge base.

    Args:
        kb: The knowledge base instance.
        content: The fact text.
        type: Memory type — "semantic", "procedural", or "episodic".
        importance: Importance score (0.0–1.0).
        tags: Optional labels.

    Returns:
        Confirmation string with the new fact's ID.
    """
    if not 0.0 <= importance <= 1.0:
        raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")
    try:
        memory_type = MemoryType(type)
    except ValueError:
        raise ValueError(
            f"Unknown memory type {type!r}. Use: semantic, procedural, episodic"
        ) from None
    fact = kb.add(content, type=memory_type, importance=importance, tags=tags or [])
    return f"Added fact [{fact.id}]: {fact.content}"


def tool_recall(kb: KnowledgeBase, query: str, *, top_k: int = 5) -> str:
    """Recall relevant facts for a query.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return.

    Returns:
        Formatted string of relevant facts, or a message if none found.
    """
    result = kb.recall(query, top_k=top_k)
    return result if result else "No relevant facts found."


def tool_forget(kb: KnowledgeBase, fact_id: str) -> str:
    """Remove a fact by its ID.

    Args:
        kb: The knowledge base instance.
        fact_id: The 8-char hex ID of the fact.

    Returns:
        Confirmation string.
    """
    kb.forget(fact_id)
    return f"Fact {fact_id!r} removed."


def tool_list_facts(kb: KnowledgeBase) -> str:
    """List all stored facts.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON-formatted list of facts, or a message if empty.
    """
    facts = kb.list_facts()
    if not facts:
        return "No facts stored."
    data = [
        {
            "id": f.id,
            "content": f.content,
            "type": f.type.value,
            "importance": f.importance,
            "retention": round(f.retention_score, 3),
        }
        for f in facts
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def tool_stats(kb: KnowledgeBase) -> str:
    """Return statistics about the knowledge base.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON-formatted statistics.
    """
    return json.dumps(kb.stats(), ensure_ascii=False, indent=2)


def tool_recall_json(kb: KnowledgeBase, query: str, *, top_k: int = 5) -> str:
    """Recall relevant facts as a JSON array of structured objects.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return.

    Returns:
        JSON array of MemoryItem objects, or "[]" if nothing found.
    """
    facts = kb.recall_facts(query, top_k=top_k)
    data = [
        {
            "id": f.id,
            "memory": f.content,
            "type": f.type.value,
            "importance": f.importance,
            "retention": round(f.retention_score, 3),
        }
        for f in facts
    ]
    return json.dumps(data, ensure_ascii=False)


def tool_learn(
    kb: KnowledgeBase,
    messages: list[dict[str, str]],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Extract and store knowledge from a multi-turn conversation.

    Uses ``kb.learn()`` with LLM extraction when provider credentials are
    available (``AI_KNOT_PROVIDER`` / ``AI_KNOT_API_KEY``); falls back to
    storing the last user message verbatim when no LLM is configured.

    Args:
        kb: The knowledge base instance.
        messages: List of ``{"role": "user"|"assistant", "content": "..."}`` dicts.
        provider: LLM provider (overrides ``AI_KNOT_PROVIDER`` env var).
        api_key: API key (overrides ``AI_KNOT_API_KEY`` env var).
        model: Model name override (overrides ``AI_KNOT_MODEL`` env var).

    Returns:
        JSON summary with ``{"stored": n, "ids": [...]}`` or an error message.
    """
    effective_provider = provider or os.environ.get("AI_KNOT_PROVIDER")
    effective_key = api_key or os.environ.get("AI_KNOT_API_KEY")
    effective_model = model or os.environ.get("AI_KNOT_MODEL")

    turns = [
        ConversationTurn(role=m.get("role", "user"), content=m.get("content", "")) for m in messages
    ]

    try:
        if effective_provider and effective_key:
            facts = kb.learn(
                turns,
                api_key=effective_key,
                provider=effective_provider,
                model=effective_model,
            )
        else:
            # Degraded mode: store last user message without LLM extraction.
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not last_user:
                return json.dumps({"stored": 0, "ids": [], "note": "no user message found"})
            fact = kb.add(last_user)
            facts = [fact]

        return json.dumps({"stored": len(facts), "ids": [f.id for f in facts]}, ensure_ascii=False)
    except (ValueError, RuntimeError, OSError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def tool_list_snapshots(kb: KnowledgeBase) -> str:
    """Return names of all saved snapshots.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON array of snapshot names, or a message if none exist.
    """
    try:
        names = kb.list_snapshots()
    except NotImplementedError as exc:
        return f"Snapshots not supported by this storage backend: {exc}"
    return json.dumps(names, ensure_ascii=False) if names else "[]"


def tool_snapshot(kb: KnowledgeBase, name: str) -> str:
    """Save the current knowledge base state as a named snapshot.

    Args:
        kb: The knowledge base instance.
        name: Snapshot identifier.

    Returns:
        Confirmation string, or an error if snapshots are not supported.
    """
    try:
        kb.snapshot(name)
        return f"Snapshot {name!r} saved."
    except NotImplementedError as exc:
        return f"Snapshots not supported by this storage backend: {exc}"


def tool_restore(kb: KnowledgeBase, name: str) -> str:
    """Restore the knowledge base from a named snapshot.

    Args:
        kb: The knowledge base instance.
        name: Snapshot identifier to restore.

    Returns:
        Confirmation string, or an error message.
    """
    try:
        kb.restore(name)
        return f"Restored from snapshot {name!r}."
    except NotImplementedError as exc:
        return f"Snapshots not supported by this storage backend: {exc}"
    except KeyError:
        return f"Snapshot {name!r} not found."


# ---------------------------------------------------------------------------
# MCP server entry point
# ---------------------------------------------------------------------------


def _make_server(kb: KnowledgeBase) -> Any:
    """Build and return the FastMCP application.

    Args:
        kb: Pre-configured KnowledgeBase to expose as tools.

    Returns:
        A ``FastMCP`` application instance.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "mcp package is required. Install with: pip install 'ai-knot[mcp]'"
        ) from exc

    app = FastMCP(
        "ai-knot",
        instructions=(
            "Use these tools to manage persistent agent memory. "
            "learn: extract and store knowledge from a conversation (preferred). "
            "add: store a single fact directly. recall: retrieve relevant context as text. "
            "recall_json: retrieve relevant context as structured JSON. "
            "forget: remove a fact by ID. list_facts: view all stored facts. "
            "stats: memory statistics. snapshot/restore: version the memory state. "
            "list_snapshots: see available snapshots."
        ),
    )

    @app.tool()
    def add(
        content: str,
        type: str = "semantic",
        importance: float = 0.8,
        tags: list[str] | None = None,
    ) -> str:
        """Add a fact to agent memory.

        Args:
            content: The knowledge string to remember.
            type: Classification — semantic, procedural, or episodic.
            importance: How important (0.0–1.0). Higher = remembered longer.
            tags: Optional labels for later retrieval via recall_by_tag.
        """
        return tool_add(kb, content, type=type, importance=importance, tags=tags)

    @app.tool()
    def recall(query: str, top_k: int = 5) -> str:
        """Recall relevant facts from memory for the current query.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
        """
        return tool_recall(kb, query, top_k=top_k)

    @app.tool()
    def forget(fact_id: str) -> str:
        """Remove a specific fact by its ID.

        Args:
            fact_id: The 8-char hex ID shown in list_facts output.
        """
        return tool_forget(kb, fact_id)

    @app.tool()
    def list_facts() -> str:
        """List all facts stored in memory (JSON format)."""
        return tool_list_facts(kb)

    @app.tool()
    def stats() -> str:
        """Return memory statistics: total facts, counts by type, averages."""
        return tool_stats(kb)

    @app.tool()
    def snapshot(name: str) -> str:
        """Save the current memory state as a named snapshot.

        Args:
            name: Snapshot identifier (e.g. "before_campaign_v2").
        """
        return tool_snapshot(kb, name)

    @app.tool()
    def restore(name: str) -> str:
        """Restore memory from a named snapshot.

        Args:
            name: The snapshot to restore.
        """
        return tool_restore(kb, name)

    @app.tool()
    def recall_json(query: str, top_k: int = 5) -> str:
        """Recall relevant facts as a JSON array (id, memory, type, importance, retention).

        Use instead of recall() when you need structured data rather than plain text.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
        """
        return tool_recall_json(kb, query, top_k=top_k)

    @app.tool()
    def learn(messages: list[dict[str, str]]) -> str:
        """Extract and store knowledge from a conversation.

        Preferred over add() for multi-turn conversations — the LLM identifies
        facts automatically. Requires AI_KNOT_PROVIDER and AI_KNOT_API_KEY env
        vars; falls back to storing the last user message if no LLM is configured.

        Args:
            messages: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        """
        return tool_learn(kb, messages)

    @app.tool()
    def list_snapshots() -> str:
        """List all saved memory snapshots by name."""
        return tool_list_snapshots(kb)

    return app


def main() -> None:
    """Entry point for the ai-knot MCP server."""
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
    except ImportError:
        import sys

        print(
            "Error: mcp package not installed.\nInstall with: pip install 'ai-knot[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)
    kb = _build_kb()
    app = _make_server(kb)
    app.run()


if __name__ == "__main__":
    main()
