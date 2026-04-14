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
    effective_key = (
        api_key
        or os.environ.get("AI_KNOT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
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
# New-gen Track A tools
# ---------------------------------------------------------------------------


def tool_ingest_episode(
    kb: KnowledgeBase,
    *,
    session_id: str,
    turn_id: str,
    speaker: str = "user",
    observed_at: str | None = None,
    raw_text: str,
    session_date: str | None = None,
    source_meta: dict[str, object] | None = None,
    parent_episode_id: str | None = None,
    materialize: bool = True,
) -> str:
    """Ingest a single raw episode into the knowledge base.

    Stores the turn as a ``RawEpisode`` and deterministically materializes
    it into ``AtomicClaim``s. Does NOT call the LLM.

    Args:
        kb: The knowledge base instance.
        session_id: Session identifier (groups turns into conversations).
        turn_id: Unique turn identifier within the session.
        speaker: Speaker role, e.g. "user" or "assistant".
        observed_at: ISO datetime string when the turn was observed. Defaults to now.
        raw_text: The raw conversation text for this turn.
        session_date: Optional ISO date for session-level temporal anchor.
        source_meta: Optional metadata dict (e.g. dataset name, LLM trace).
        parent_episode_id: Optional parent episode for session enveloping.
        materialize: If True (default), materialize into AtomicClaims immediately.

    Returns:
        JSON string with the episode ID.
    """
    from datetime import UTC, datetime

    obs = datetime.fromisoformat(observed_at) if observed_at else datetime.now(UTC)
    sdate = datetime.fromisoformat(session_date) if session_date else None

    try:
        episode = kb.ingest_episode(
            session_id=session_id,
            turn_id=turn_id,
            speaker=speaker,
            observed_at=obs,
            raw_text=raw_text,
            session_date=sdate,
            source_meta=source_meta or {},
            parent_episode_id=parent_episode_id,
            materialize=materialize,
        )
        return json.dumps({"episode_id": episode.id, "session_id": session_id}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def tool_query(
    kb: KnowledgeBase,
    question: str,
    *,
    top_k: int = 60,
    render: str = "structured",
) -> str:
    """Contract-first query over memory. Returns the answer text.

    Uses the full contract-first pipeline: analyze_query → retrieve_bundles
    → expand_claims → build_evidence_profile → choose_strategy → operator.

    Args:
        kb: The knowledge base instance.
        question: Natural-language question.
        top_k: Maximum bundles to retrieve (default 60).
        render: "structured" (deterministic) or "narrative" (LLM fallback).

    Returns:
        Answer text string.
    """
    try:
        answer = kb.query(question, top_k=top_k, render=render)
        return answer.text
    except (RuntimeError, NotImplementedError):
        # Storage does not support v2 planes — fall back to legacy recall.
        return kb.recall(question, top_k=min(top_k, 10))


def tool_query_json(
    kb: KnowledgeBase,
    question: str,
    *,
    top_k: int = 60,
    render: str = "structured",
) -> str:
    """Contract-first query returning full QueryAnswer as JSON.

    Returns the complete structured payload: text, items (with confidence
    and provenance), and trace (strategy, evidence profile, latency).

    Args:
        kb: The knowledge base instance.
        question: Natural-language question.
        top_k: Maximum bundles to retrieve.
        render: "structured" or "narrative".

    Returns:
        JSON string of QueryAnswer.
    """
    try:
        return json.dumps(kb.query_json(question, top_k=top_k, render=render), ensure_ascii=False)
    except (RuntimeError, NotImplementedError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def tool_rebuild_materialized(
    kb: KnowledgeBase,
    *,
    scope: str = "all",
    force: bool = False,
) -> str:
    """Rebuild the atomic_claims and support_bundles planes from raw_episodes.

    This is a deterministic, idempotent operation. It does NOT call the LLM.
    Use after schema upgrades or when materialization_version is stale.

    Args:
        kb: The knowledge base instance.
        scope: "all" (default) or a specific materialization scope.
        force: Force rebuild even if materialization_version is current.

    Returns:
        Rebuild report as JSON string.
    """
    try:
        report = kb.rebuild_materialized(scope=scope, force=force)
        return json.dumps(report, ensure_ascii=False)
    except AttributeError:
        return json.dumps({"error": "rebuild_materialized not available on this storage backend"})
    except Exception as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def tool_stats_query(kb: KnowledgeBase) -> str:
    """Return statistics about the v2 query planes.

    Includes counts of raw_episodes, atomic_claims, support_bundles, and
    the current materialization_version.

    Args:
        kb: The knowledge base instance.

    Returns:
        JSON object with plane counts and materialization metadata.
    """
    storage = kb._storage
    agent_id = kb._agent_id

    result: dict[str, object] = {}

    # Counts from v2 planes (graceful degradation if not supported).
    if hasattr(storage, "load_episodes"):
        try:
            episodes = storage.load_episodes(agent_id)
            result["n_raw_episodes"] = len(episodes)
        except Exception as exc:
            result["n_raw_episodes"] = f"error: {exc}"

    if hasattr(storage, "load_claims"):
        try:
            claims = storage.load_claims(agent_id, active_only=False)
            result["n_atomic_claims"] = len(claims)
            active = sum(1 for c in claims if c.valid_until is None)
            result["n_active_claims"] = active
        except Exception as exc:
            result["n_atomic_claims"] = f"error: {exc}"

    if hasattr(storage, "load_materialization_meta"):
        try:
            meta = storage.load_materialization_meta(agent_id)
            result["materialization_version"] = meta.get("materialization_version", 0)
            result["schema_version"] = meta.get("schema_version", 1)
            result["last_rebuild_at"] = meta.get("last_rebuild_at")
            result["rebuild_status"] = meta.get("rebuild_status", "ready")
        except Exception as exc:
            result["materialization_meta"] = f"error: {exc}"

    # Legacy stats merged in.
    result["legacy"] = kb.stats()
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_explain_query(
    kb: KnowledgeBase,
    question: str,
    *,
    top_k: int = 60,
) -> str:
    """Return only the AnswerTrace for a question without executing the full render.

    Useful for debugging: shows which strategy was chosen, how many bundles
    and claims were found, and what the evidence profile looks like.

    Args:
        kb: The knowledge base instance.
        question: Natural-language question to explain.
        top_k: Maximum bundles to retrieve.

    Returns:
        JSON string of AnswerTrace.
    """
    try:
        trace = kb.explain_query(question, top_k=top_k)
        # Serialize the trace dataclass.
        import dataclasses

        def _to_json(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_json(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, tuple):
                return [_to_json(x) for x in obj]
            if isinstance(obj, (list,)):
                return [_to_json(x) for x in obj]
            return obj

        return json.dumps(_to_json(trace), ensure_ascii=False, default=str, indent=2)
    except RuntimeError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
