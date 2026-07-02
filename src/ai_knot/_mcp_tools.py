"""Pure tool-handler functions for the ai-knot MCP server.

These functions implement the logic for each MCP tool and are kept separate
from the server setup so they can be tested without the ``mcp`` package.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import ConversationTurn, Fact, MemoryOp, MemoryType

# Upper bound on top_k accepted from MCP clients. An unbounded top_k would let a
# malformed or hostile call pull the entire store into one recall (latency and
# context blow-up); clamping keeps a single tool call bounded.
_MAX_TOP_K = 200


def _clamp_top_k(top_k: int) -> int:
    """Clamp a client-supplied ``top_k`` into ``[1, _MAX_TOP_K]``."""
    return max(1, min(int(top_k), _MAX_TOP_K))


def _parse_event_time(event_time: str | None) -> datetime | None:
    """Parse an ISO-8601 event_time string into a datetime (or None).

    Accepts a trailing 'Z' (UTC) for convenience. Returns None on empty/invalid
    input so a malformed timestamp degrades to "no temporal anchor" rather than
    failing the whole add().
    """
    if not event_time:
        return None
    try:
        dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive inputs (e.g. a date-only "2023-05-08" or "...T23:40:00" without a zone)
    # are treated as UTC, so the value is comparable with the store's timezone-aware
    # validity bounds — otherwise recall(now=…) → is_active raises "can't compare
    # offset-naive and offset-aware datetimes" and the whole recall fails.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _listed_facts(
    kb: KnowledgeBase,
    *,
    include_inactive: bool,
    now: str | None,
) -> tuple[list[Fact], datetime]:
    """Return facts for listing plus the anchor used for activity checks."""
    anchor = _parse_event_time(now) or datetime.now(UTC)
    facts = kb.list_facts()
    if not include_inactive:
        facts = [fact for fact in facts if fact.is_active(anchor)]
    return facts, anchor


def tool_add(
    kb: KnowledgeBase,
    content: str,
    *,
    type: str = "semantic",
    importance: float = 0.8,
    tags: list[str] | None = None,
    event_time: str | None = None,
) -> str:
    """Add a fact to the knowledge base.

    Args:
        kb: The knowledge base instance.
        content: The fact text.
        type: Memory type — "semantic", "procedural", or "episodic".
        importance: Importance score (0.0–1.0).
        tags: Optional labels.
        event_time: ISO-8601 timestamp of when the memory was formed (the
            real-world anchor for resolving relative-time in ``content``). In
            production this is now(); on historical import it is the original
            message timestamp.

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
    fact = kb.add(
        content,
        type=memory_type,
        importance=importance,
        tags=tags or [],
        event_time=_parse_event_time(event_time),
    )
    return f"Added fact [{fact.id}]: {fact.content}"


def tool_add_resolved(kb: KnowledgeBase, facts: list[dict[str, Any]]) -> str:
    """Insert pre-structured facts through the supersession pipeline (no LLM).

    Each entry in *facts* is a dict with a required ``content`` and optional
    ``entity``, ``attribute``, ``value_text``, ``slot_key``, ``op`` and
    ``event_time`` (ISO-8601). A fact addressing an existing active slot with a
    *different* value supersedes it (knowledge-update), exactly as inside
    ``learn()``; ``op="update"`` forces supersession, ``op="delete"`` closes the
    matched memory without inserting a replacement, and ``op="noop"`` skips
    insertion/mutation entirely. A fact with no slot and no ``entity`` +
    ``attribute`` is inserted as-is. This is the clean, dependency-free
    structured-knowledge seam — the multi-agent path for ingesting
    already-resolved facts (e.g. from another agent) without an LLM extraction
    call.

    Args:
        kb: The knowledge base instance.
        facts: List of structured fact specs (see above).

    Returns:
        JSON array of the inserted/updated facts: ``id``, ``content``,
        ``slot_key``, ``version``.
    """
    built: list[Fact] = []
    for spec in facts:
        content = str(spec.get("content", "")).strip()
        if not content:
            raise ValueError("each fact requires a non-empty 'content'")
        fact = Fact(
            content=content,
            entity=str(spec.get("entity", "")),
            attribute=str(spec.get("attribute", "")),
            value_text=str(spec.get("value_text", "")),
            slot_key=str(spec.get("slot_key", "")),
        )
        raw_op = str(spec.get("op", MemoryOp.ADD.value)).strip().lower() or MemoryOp.ADD.value
        try:
            fact.op = MemoryOp(raw_op)
        except ValueError:
            raise ValueError(
                f"invalid op {raw_op!r}; expected one of {[op.value for op in MemoryOp]}"
            ) from None
        event_time = _parse_event_time(spec.get("event_time"))
        if event_time is not None:
            fact.event_time = event_time
        built.append(fact)
    inserted = kb.add_resolved(built)
    return json.dumps(
        [
            {
                "id": f.id,
                "content": f.content,
                "slot_key": f.slot_key,
                "version": f.version,
            }
            for f in inserted
        ]
    )


def tool_recall(kb: KnowledgeBase, query: str, *, top_k: int = 5, now: str | None = None) -> str:
    """Recall relevant facts for a query.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return (clamped to ``[1, 200]``).
        now: Optional ISO-8601 point-in-time anchor. Facts whose validity has
            ended by ``now`` (superseded knowledge-updates) are excluded and
            decay is computed as of ``now``. Defaults to the current time.

    Returns:
        Formatted string of relevant facts, or a message if none found.
    """
    result = kb.recall(query, top_k=_clamp_top_k(top_k), now=_parse_event_time(now))
    return result if result else "No relevant facts found."


def tool_search(kb: KnowledgeBase, query: str, *, top_k: int = 5, now: str | None = None) -> str:
    """Alias for :func:`tool_recall` using the market-standard search verb."""
    return tool_recall(kb, query, top_k=top_k, now=now)


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


def tool_delete(kb: KnowledgeBase, fact_id: str) -> str:
    """Alias for :func:`tool_forget` using the CRUD-style delete verb."""
    return tool_forget(kb, fact_id)


def tool_get(kb: KnowledgeBase, fact_id: str) -> str:
    """Return one stored fact by ID as JSON.

    Args:
        kb: The knowledge base instance.
        fact_id: The 8-char hex ID of the fact.

    Returns:
        JSON object describing the stored fact.

    Raises:
        ValueError: If the fact ID is unknown.
    """
    try:
        fact = kb.get(fact_id)
    except KeyError as exc:
        raise ValueError(f"No fact found with id {fact_id}.") from exc

    return json.dumps(
        {
            "id": fact.id,
            "content": fact.content,
            "type": fact.type.value,
            "importance": fact.importance,
            "retention_score": fact.retention_score,
            "access_count": fact.access_count,
            "tags": list(fact.tags),
            "created_at": fact.created_at.isoformat(),
            "last_accessed": fact.last_accessed.isoformat(),
            "event_time": fact.event_time.isoformat() if fact.event_time else None,
            "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
            "valid_until": fact.valid_until.isoformat() if fact.valid_until else None,
            "active": fact.is_active(),
        },
        ensure_ascii=False,
    )


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
        {
            "name": "add_resolved",
            "description": "Insert pre-structured facts through supersession (no LLM)",
        },
        {"name": "learn", "description": "Extract and store facts from a conversation"},
        {"name": "recall", "description": "Retrieve relevant facts as text"},
        {"name": "search", "description": "Retrieve relevant facts as text (alias for recall)"},
        {"name": "recall_json", "description": "Retrieve relevant facts as JSON array"},
        {"name": "get", "description": "Return one stored fact as JSON by fact_id"},
        {"name": "forget", "description": "Remove a fact by ID"},
        {"name": "delete", "description": "Remove a fact by ID (alias for forget)"},
        {
            "name": "list",
            "description": "List all stored facts as JSON array (alias for list_facts)",
        },
        {"name": "list_facts", "description": "List all stored facts as JSON array"},
        {
            "name": "memory_lineage",
            "description": "Trace a fact's supersession lineage (audit trail)",
        },
        {"name": "stats", "description": "Memory statistics"},
        {"name": "snapshot", "description": "Save current state as a named snapshot"},
        {"name": "restore", "description": "Restore state from a named snapshot"},
        {"name": "list_snapshots", "description": "List available snapshots"},
        {"name": "health", "description": "Server health and version"},
        {"name": "capabilities", "description": "List available tools"},
    ]
    return json.dumps(tools, ensure_ascii=False)


def tool_recall_with_trace(
    kb: KnowledgeBase, query: str, *, top_k: int = 5, now: str | None = None
) -> str:
    """Diagnostic variant of recall — returns context string plus per-stage trace.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return (clamped to ``[1, 200]``).
        now: Optional ISO-8601 point-in-time anchor (see :func:`tool_recall`).

    Returns:
        JSON object with ``{"context", "pack_fact_ids", "trace"}``; trace keys:
        stage1_candidates (from_bm25/from_rare_tokens/from_entity_hop), stage3_rrf,
        stage3b_dense_guarantee, stage4a_ddsa, stage4b_mmr.
        Intended for diagnostics only — not for production use.
    """
    top_k = _clamp_top_k(top_k)
    parsed_now = _parse_event_time(now)
    pairs, trace = kb.recall_facts_with_trace(query, top_k=top_k, now=parsed_now)
    context = kb.recall(query, top_k=top_k, now=parsed_now)
    pack_fact_ids = [f.id for f, _ in pairs]
    return json.dumps(
        {"context": context, "pack_fact_ids": pack_fact_ids, "trace": trace},
        ensure_ascii=False,
    )


def tool_list_facts(
    kb: KnowledgeBase,
    *,
    include_inactive: bool = False,
    now: str | None = None,
) -> str:
    """List stored facts.

    Args:
        kb: The knowledge base instance.
        include_inactive: When True, include superseded / inactive facts too.
        now: Optional ISO-8601 activity anchor used for active filtering.

    Returns:
        JSON array of facts (empty array ``[]`` when nothing is stored).
    """
    facts, anchor = _listed_facts(kb, include_inactive=include_inactive, now=now)
    if not facts:
        return "[]"
    data = [
        {
            "id": f.id,
            "content": f.content,
            "type": f.type.value,
            "importance": f.importance,
            "retention_score": round(f.retention_score, 3),
            "retention": round(f.retention_score, 3),
            "access_count": f.access_count,
            "tags": list(f.tags),
            "created_at": f.created_at.isoformat(),
            "last_accessed": f.last_accessed.isoformat(),
            "event_time": f.event_time.isoformat() if f.event_time else None,
            "valid_from": f.valid_from.isoformat() if f.valid_from else None,
            "valid_until": f.valid_until.isoformat() if f.valid_until else None,
            "active": f.is_active(anchor),
        }
        for f in facts
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def tool_list(
    kb: KnowledgeBase,
    *,
    include_inactive: bool = False,
    now: str | None = None,
) -> str:
    """Alias for :func:`tool_list_facts` using the familiar list verb."""
    return tool_list_facts(kb, include_inactive=include_inactive, now=now)


def tool_memory_lineage(kb: KnowledgeBase, fact_id: str, *, now: str | None = None) -> str:
    """Return the supersession lineage of a fact as JSON (newest → oldest).

    Args:
        kb: The knowledge base instance.
        fact_id: The 8-char hex ID to trace.
        now: Optional ISO-8601 activity anchor used for active/inactive status.

    Returns:
        JSON array of lineage rows from the given fact back to the original it
        replaced; ``"[]"`` when the fact is unknown.
    """
    chain = kb.lineage(fact_id)
    anchor = _parse_event_time(now) or datetime.now(UTC)
    data = [
        {
            "id": f.id,
            "content": f.content,
            "type": f.type.value,
            "importance": f.importance,
            "retention_score": f.retention_score,
            "access_count": f.access_count,
            "tags": list(f.tags),
            "created_at": f.created_at.isoformat(),
            "last_accessed": f.last_accessed.isoformat(),
            "event_time": f.event_time.isoformat() if f.event_time else None,
            "valid_from": f.valid_from.isoformat() if f.valid_from else None,
            "valid_until": f.valid_until.isoformat() if f.valid_until else None,
            "slot_key": f.slot_key or None,
            "entity": f.entity or None,
            "attribute": f.attribute or None,
            "value_text": f.value_text or None,
            "version": f.version,
            "supersedes_id": f.provenance.supersedes_id or None,
            "published_by": f.provenance.published_by or None,
            "active": f.is_active(anchor),
        }
        for f in chain
    ]
    return json.dumps(data, ensure_ascii=False)


def tool_stats(kb: KnowledgeBase) -> str:
    """Return statistics about the knowledge base.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON-formatted statistics.
    """
    return json.dumps(kb.stats(), ensure_ascii=False, indent=2)


def tool_recall_json(
    kb: KnowledgeBase, query: str, *, top_k: int = 5, now: str | None = None
) -> str:
    """Recall relevant facts as a JSON array of structured objects.

    Args:
        kb: The knowledge base instance.
        query: What the agent needs to know.
        top_k: Maximum number of facts to return (clamped to ``[1, 200]``).
        now: Optional ISO-8601 point-in-time anchor (see :func:`tool_recall`).

    Returns:
        JSON array of MemoryItem objects, or "[]" if nothing found.
    """
    facts = kb.recall_facts(query, top_k=_clamp_top_k(top_k), now=_parse_event_time(now))
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
    event_time: str | None = None,
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
        event_time: ISO-8601 real-world anchor for the whole conversation.

    Returns:
        JSON summary with ``{"stored": n, "ids": [...]}`` or an error message.
    """
    effective_provider = provider or os.environ.get("AI_KNOT_PROVIDER")
    effective_key = api_key or os.environ.get("AI_KNOT_API_KEY")
    effective_model = model or os.environ.get("AI_KNOT_MODEL")
    parsed_event_time = _parse_event_time(event_time)

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
                event_time=parsed_event_time,
            )
        else:
            # Degraded mode: store last user message without LLM extraction.
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            )
            if not last_user:
                return json.dumps({"stored": 0, "ids": [], "note": "no user message found"})
            fact = kb.add(last_user, event_time=parsed_event_time)
            facts = [fact]
        payload = {"stored": len(facts), "ids": [f.id for f in facts]}
        if not (effective_provider and effective_key):
            payload["note"] = "provider not configured; stored the last user message verbatim"
        return json.dumps(payload, ensure_ascii=False)
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
