"""MCP server for agentmemo — exposes KnowledgeBase as MCP tools.

Run with::

    agentmemo-mcp

or::

    AGENTMEMO_AGENT_ID=my_agent python -m agentmemo.mcp_server

Configuration is via environment variables:

- ``AGENTMEMO_AGENT_ID``   — agent namespace (default: "default")
- ``AGENTMEMO_STORAGE``    — backend: "yaml" or "sqlite" (default: "yaml")
- ``AGENTMEMO_DATA_DIR``   — base directory for file backends (default: ".agentmemo")
- ``AGENTMEMO_DB_PATH``    — full path to SQLite file (overrides DATA_DIR for sqlite)
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage import create_storage
from agentmemo.types import MemoryType


def _build_kb() -> KnowledgeBase:
    """Construct a KnowledgeBase from environment variables.

    Returns:
        A configured KnowledgeBase instance.
    """
    agent_id = os.environ.get("AGENTMEMO_AGENT_ID", "default")
    backend = os.environ.get("AGENTMEMO_STORAGE", "yaml")
    data_dir = os.environ.get("AGENTMEMO_DATA_DIR", ".agentmemo")
    db_path = os.environ.get("AGENTMEMO_DB_PATH")

    dsn = db_path or os.path.join(data_dir, "agentmemo.db") if backend == "sqlite" else None
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
            "mcp package is required. Install with: pip install 'agentmemo[mcp]'"
        ) from exc

    app = FastMCP(
        "agentmemo",
        instructions=(
            "Use these tools to manage persistent agent memory. "
            "add: store a new fact. recall: retrieve relevant context for a query. "
            "forget: remove a fact by ID. list_facts: view all stored facts. "
            "stats: memory statistics. snapshot/restore: version the memory state."
        ),
    )

    @app.tool()
    def add(content: str, type: str = "semantic", importance: float = 0.8) -> str:
        """Add a fact to agent memory.

        Args:
            content: The knowledge string to remember.
            type: Classification — semantic, procedural, or episodic.
            importance: How important (0.0–1.0). Higher = remembered longer.
        """
        return tool_add(kb, content, type=type, importance=importance)

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

    return app


def main() -> None:
    """Entry point for the agentmemo MCP server."""
    kb = _build_kb()
    app = _make_server(kb)
    app.run()


if __name__ == "__main__":
    main()
