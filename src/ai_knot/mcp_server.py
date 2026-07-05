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

import builtins
import os
from typing import Any

from ai_knot._mcp_tools import (
    tool_add,
    tool_add_resolved,
    tool_capabilities,
    tool_delete,
    tool_forget,
    tool_get,
    tool_health,
    tool_learn,
    tool_list,
    tool_list_facts,
    tool_list_snapshots,
    tool_memory_lineage,
    tool_recall,
    tool_recall_json,
    tool_recall_with_trace,
    tool_restore,
    tool_search,
    tool_snapshot,
    tool_stats,
)
from ai_knot.config import AIKnotConfig
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage import create_storage

_MCP_TRANSPORTS = {"stdio", "sse", "streamable-http"}


def _build_kb() -> KnowledgeBase:
    """Construct a KnowledgeBase from the environment.

    All ``AI_KNOT_*`` configuration is parsed and validated by
    :meth:`ai_knot.config.AIKnotConfig.from_env`, so a misconfiguration raises a
    clear ``ValueError`` here at startup instead of surfacing later on a recall.

    Returns:
        A configured KnowledgeBase instance.
    """
    cfg = AIKnotConfig.from_env()
    storage = create_storage(
        cfg.storage.backend, base_dir=cfg.storage.data_dir, dsn=cfg.storage.dsn
    )
    return KnowledgeBase(
        agent_id=cfg.agent_id,
        storage=storage,
        provider=cfg.llm.provider,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        llm_recall=cfg.llm.llm_recall,
        rrf_weights=cfg.recall.rrf_weights,
        expansion_weight=cfg.recall.expansion_weight,
        episodic_ttl_hours=cfg.recall.episodic_ttl_hours,
        embed_url=cfg.embed.url,
        embed_model=cfg.embed.model,
        embed_api_key=cfg.embed.api_key,
    )


# ---------------------------------------------------------------------------
# MCP server entry point
# ---------------------------------------------------------------------------


def _make_server(
    kb: KnowledgeBase,
    *,
    host: str | None = None,
    port: int | None = None,
    mount_path: str | None = None,
    sse_path: str | None = None,
    message_path: str | None = None,
    streamable_http_path: str | None = None,
) -> Any:
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

    app_kwargs: dict[str, Any] = {
        "instructions": (
            "Use these tools to manage persistent agent memory. "
            "learn: extract and store knowledge from a conversation (preferred). "
            "add: store a single fact directly. "
            "search/recall: retrieve relevant context as text. "
            "recall_json: retrieve relevant context as structured JSON. "
            "get: inspect one stored fact by ID as JSON. "
            "delete/forget: remove a fact by ID. "
            "list/list_facts: view all stored facts as JSON. "
            "stats: memory statistics. snapshot/restore: version the memory state. "
            "list_snapshots: see available snapshots. "
            "health: check server status. capabilities: list all available tools."
        ),
        "website_url": "https://github.com/alsoleg89/ai-knot",
    }
    if host is not None:
        app_kwargs["host"] = host
    if port is not None:
        app_kwargs["port"] = port
    if mount_path is not None:
        app_kwargs["mount_path"] = mount_path
    if sse_path is not None:
        app_kwargs["sse_path"] = sse_path
    if message_path is not None:
        app_kwargs["message_path"] = message_path
    if streamable_http_path is not None:
        app_kwargs["streamable_http_path"] = streamable_http_path

    app = FastMCP("ai-knot", **app_kwargs)

    @app.tool()
    def add(
        content: str,
        type: str = "semantic",
        importance: float = 0.8,
        tags: list[str] | None = None,
        event_time: str | None = None,
    ) -> str:
        """Add a fact to agent memory.

        Args:
            content: The knowledge string to remember.
            type: Classification — semantic, procedural, or episodic.
            importance: How important (0.0–1.0). Higher = remembered longer.
            tags: Optional labels for later retrieval via recall_by_tag.
            event_time: ISO-8601 timestamp of when the memory was formed. Used to
                resolve relative-time expressions ("yesterday", "last week") in
                ``content`` into absolute dates. Defaults to now() when omitted.
        """
        return tool_add(
            kb,
            content,
            type=type,
            importance=importance,
            tags=tags,
            event_time=event_time,
        )

    @app.tool()
    def add_resolved(facts: list[dict[str, Any]]) -> str:
        """Insert pre-structured facts through supersession (no LLM extraction).

        Each fact is a dict with a required ``content`` and optional ``entity``,
        ``attribute``, ``value_text``, ``slot_key`` and ``event_time`` (ISO-8601).
        A fact addressing an existing slot with a different value supersedes it,
        exactly as inside ``learn()``. Use this to ingest already-resolved facts
        (e.g. structured knowledge shared by another agent) without an LLM call.
        """
        return tool_add_resolved(kb, facts)

    @app.tool()
    def recall(query: str, top_k: int = 5, now: str | None = None) -> str:
        """Recall relevant facts from memory for the current query.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Optional ISO-8601 point-in-time anchor. Excludes facts whose
                validity ended by ``now`` (superseded knowledge-updates) and
                computes decay as of ``now``. Defaults to the current time.
        """
        return tool_recall(kb, query, top_k=top_k, now=now)

    @app.tool()
    def search(query: str, top_k: int = 5, now: str | None = None) -> str:
        """Alias for recall() using the market-standard search verb."""
        return tool_search(kb, query, top_k=top_k, now=now)

    @app.tool()
    def get(fact_id: str) -> str:
        """Return one stored fact by ID as JSON.

        Args:
            fact_id: The 8-char hex ID to inspect.
        """
        return tool_get(kb, fact_id)

    @app.tool()
    def forget(fact_id: str) -> str:
        """Remove a specific fact by its ID.

        Args:
            fact_id: The 8-char hex ID shown in list_facts output.
        """
        return tool_forget(kb, fact_id)

    @app.tool()
    def delete(fact_id: str) -> str:
        """Alias for forget() using the CRUD-style delete verb."""
        return tool_delete(kb, fact_id)

    @app.tool()
    def list(include_inactive: bool = False, now: str | None = None) -> str:
        """Alias for list_facts() using the familiar list verb."""
        return tool_list(kb, include_inactive=include_inactive, now=now)

    @app.tool()
    def list_facts(include_inactive: bool = False, now: str | None = None) -> str:
        """List active facts by default; pass include_inactive for full history."""
        return tool_list_facts(kb, include_inactive=include_inactive, now=now)

    @app.tool()
    def memory_lineage(fact_id: str, now: str | None = None) -> str:
        """Trace a fact's supersession lineage (newest → oldest) as JSON.

        The audit trail of how a slot's value evolved: each entry's
        supersedes_id points at the fact it replaced.

        Args:
            fact_id: The 8-char hex ID to trace (from list_facts/recall_json).
            now: Optional ISO-8601 activity anchor for active/inactive status.
        """
        return tool_memory_lineage(kb, fact_id, now=now)

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
    def recall_json(query: str, top_k: int = 5, now: str | None = None) -> str:
        """Recall relevant facts as a JSON array (id, memory, type, importance, retention).

        Use instead of recall() when you need structured data rather than plain text.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Optional ISO-8601 point-in-time anchor (see recall()).
        """
        return tool_recall_json(kb, query, top_k=top_k, now=now)

    @app.tool()
    def learn(
        messages: builtins.list[dict[str, str]],
        provider: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        event_time: str | None = None,
    ) -> str:
        """Extract and store knowledge from a conversation.

        Preferred over add() for multi-turn conversations — the LLM identifies
        facts automatically. Requires AI_KNOT_PROVIDER and AI_KNOT_API_KEY env
        vars; falls back to storing the last user message if no LLM is configured.

        Args:
            messages: Conversation as [{"role": "user"|"assistant", "content": "..."}].
            provider: Optional provider override for extraction.
            api_key: Optional API key override for extraction.
            model: Optional model override for extraction.
            event_time: Optional ISO-8601 real-world anchor for the conversation.
        """
        return tool_learn(
            kb,
            messages,
            provider=provider,
            api_key=api_key,
            model=model,
            event_time=event_time,
        )

    @app.tool()
    def recall_with_trace(query: str, top_k: int = 5, now: str | None = None) -> str:
        """Diagnostic: recall with per-stage pipeline trace (JSON).

        Returns context string plus stage1_candidates, pack_fact_ids, and
        full trace dict. Intended for benchmark diagnostics — not for
        production agent use.

        Args:
            query: What the agent needs to know right now.
            top_k: Maximum number of facts to return.
            now: Optional ISO-8601 point-in-time anchor (see recall()).
        """
        return tool_recall_with_trace(kb, query, top_k=top_k, now=now)

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


def _resolve_transport(value: str | None) -> str:
    """Normalize and validate the MCP transport selector."""
    transport = (value or "stdio").strip().lower()
    if transport not in _MCP_TRANSPORTS:
        raise ValueError("AI_KNOT_MCP_TRANSPORT must be one of: stdio, sse, streamable-http")
    return transport


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
    try:
        transport = _resolve_transport(os.environ.get("AI_KNOT_MCP_TRANSPORT"))
    except ValueError as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    kb = _build_kb()
    app = _make_server(kb)
    app.run(transport)


if __name__ == "__main__":
    main()
