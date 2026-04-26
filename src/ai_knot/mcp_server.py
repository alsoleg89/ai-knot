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

import os
from typing import Any

from ai_knot._mcp_tools import (
    tool_add,
    tool_capabilities,
    tool_forget,
    tool_health,
    tool_learn,
    tool_list_facts,
    tool_list_snapshots,
    tool_recall,
    tool_recall_json,
    tool_recall_with_trace,
    tool_restore,
    tool_snapshot,
    tool_stats,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated string of floats into a tuple."""
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _build_kb() -> KnowledgeBase:
    """Construct a KnowledgeBase from environment variables.

    Supported env vars (all optional):

    - ``AI_KNOT_AGENT_ID``          — agent namespace (default: "default")
    - ``AI_KNOT_STORAGE``           — "yaml" or "sqlite" (default: "sqlite")
    - ``AI_KNOT_DATA_DIR``          — base directory (default: ".ai_knot")
    - ``AI_KNOT_DB_PATH``           — full path to SQLite file
    - ``AI_KNOT_LLM_RECALL``        — "1" to enable LLM query expansion
    - ``AI_KNOT_PROVIDER``          — LLM provider for llm_recall / learn
    - ``AI_KNOT_API_KEY``           — API key for the LLM provider
    - ``AI_KNOT_MODEL``             — model name override
    - ``AI_KNOT_RRF_WEIGHTS``       — comma-separated RRF weights (6 floats)
    - ``AI_KNOT_EXPANSION_WEIGHT``  — float, LLM expansion blend weight
    - ``AI_KNOT_EPISODIC_TTL``      — float, episodic TTL in hours
    - ``AI_KNOT_EMBED_URL``         — Ollama base URL for embeddings
    - ``AI_KNOT_EMBED_MODEL``       — embedding model (default: nomic-embed-text)

    Returns:
        A configured KnowledgeBase instance.
    """
    agent_id = os.environ.get("AI_KNOT_AGENT_ID", "default")
    backend = os.environ.get("AI_KNOT_STORAGE", "sqlite")
    data_dir = os.environ.get("AI_KNOT_DATA_DIR", ".ai_knot")
    db_path = os.environ.get("AI_KNOT_DB_PATH")

    dsn = db_path or os.path.join(data_dir, "ai_knot.db") if backend == "sqlite" else None
    storage = create_storage(backend, base_dir=data_dir, dsn=dsn)

    llm_recall = os.environ.get("AI_KNOT_LLM_RECALL", "") in ("1", "true", "yes")
    provider = os.environ.get("AI_KNOT_PROVIDER")
    api_key = os.environ.get("AI_KNOT_API_KEY")
    model = os.environ.get("AI_KNOT_MODEL")

    raw_rrf = os.environ.get("AI_KNOT_RRF_WEIGHTS")
    rrf_weights = _parse_float_tuple(raw_rrf) if raw_rrf else None

    raw_exp = os.environ.get("AI_KNOT_EXPANSION_WEIGHT")
    expansion_weight = float(raw_exp) if raw_exp else None

    raw_ttl = os.environ.get("AI_KNOT_EPISODIC_TTL")
    episodic_ttl_hours = float(raw_ttl) if raw_ttl else 72.0

    embed_url = os.environ.get("AI_KNOT_EMBED_URL", "http://localhost:11434")
    embed_model = os.environ.get("AI_KNOT_EMBED_MODEL", "nomic-embed-text")
    embed_api_key = os.environ.get("AI_KNOT_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")

    return KnowledgeBase(
        agent_id=agent_id,
        storage=storage,
        provider=provider,
        api_key=api_key,
        model=model,
        llm_recall=llm_recall,
        rrf_weights=rrf_weights,
        expansion_weight=expansion_weight,
        episodic_ttl_hours=episodic_ttl_hours,
        embed_url=embed_url,
        embed_model=embed_model,
        embed_api_key=embed_api_key,
    )


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
            "forget: remove a fact by ID. list_facts: view all stored facts as JSON. "
            "stats: memory statistics. snapshot/restore: version the memory state. "
            "list_snapshots: see available snapshots. "
            "health: check server status. capabilities: list all available tools."
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
    def recall_with_trace(query: str, top_k: int = 5) -> str:
        """Diagnostic: recall with per-stage pipeline trace (JSON).

        Returns context string plus stage1_candidates, pack_fact_ids, and
        full trace dict. Intended for benchmark diagnostics — not for
        production agent use.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
        """
        return tool_recall_with_trace(kb, query, top_k=top_k)

    @app.tool()
    def list_snapshots() -> str:
        """List all saved memory snapshots by name."""
        return tool_list_snapshots(kb)

    @app.tool()
    def health() -> str:
        """Return server health status and version as JSON."""
        return tool_health()

    @app.tool()
    def capabilities() -> str:
        """Return the list of available tools as a JSON array."""
        return tool_capabilities()

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
