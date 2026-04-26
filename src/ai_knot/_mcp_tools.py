"""Pure tool-handler functions for the ai-knot MCP server.

These functions implement the logic for each MCP tool and are kept separate
from the server setup so they can be tested without the ``mcp`` package.
"""

from __future__ import annotations

import json
import os

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import ConversationTurn, MemoryType


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


def tool_health() -> str:
    """Return server health and version.

    Returns:
        JSON object with ``{"status": "ok", "version": "..."}``
    """
    from ai_knot import __version__

    return json.dumps({"status": "ok", "version": __version__})


def tool_capabilities() -> str:
    """Return the list of available tools with short descriptions.

    Returns:
        JSON array of ``{"name": "...", "description": "..."}`` objects.
    """
    tools = [
        {"name": "add", "description": "Store a single fact"},
        {"name": "learn", "description": "Extract and store facts from a conversation"},
        {"name": "recall", "description": "Retrieve relevant facts as text"},
        {"name": "recall_json", "description": "Retrieve relevant facts as JSON array"},
        {"name": "forget", "description": "Remove a fact by ID"},
        {"name": "list_facts", "description": "List all stored facts as JSON array"},
        {"name": "stats", "description": "Memory statistics"},
        {"name": "snapshot", "description": "Save current state as a named snapshot"},
        {"name": "restore", "description": "Restore state from a named snapshot"},
        {"name": "list_snapshots", "description": "List available snapshots"},
        {"name": "health", "description": "Server health and version"},
        {"name": "capabilities", "description": "List available tools"},
    ]
    return json.dumps(tools, ensure_ascii=False)


def tool_recall_with_trace(kb: KnowledgeBase, query: str, *, top_k: int = 5) -> str:
    """Diagnostic variant of recall — returns context string plus per-stage trace.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return.

    Returns:
        JSON object with ``{"context", "pack_fact_ids", "trace"}``; trace keys:
        stage1_candidates (from_bm25/from_rare_tokens/from_entity_hop), stage3_rrf,
        stage3b_dense_guarantee, stage4a_ddsa, stage4b_mmr.
        Intended for diagnostics only — not for production use.
    """
    pairs, trace = kb.recall_facts_with_trace(query, top_k=top_k)
    context = kb.recall(query, top_k=top_k)
    pack_fact_ids = [f.id for f, _ in pairs]
    return json.dumps(
        {"context": context, "pack_fact_ids": pack_fact_ids, "trace": trace},
        ensure_ascii=False,
    )


def tool_list_facts(kb: KnowledgeBase) -> str:
    """List all stored facts.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON array of facts (empty array ``[]`` when nothing is stored).
    """
    facts = kb.list_facts()
    if not facts:
        return "[]"
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
