"""AutoGen memory adapter.

ai-knot complements AutoGen's short-term model context with a distilled,
self-hosted fact store. The adapter implements the async methods expected by
AutoGen's ``Memory`` protocol, recalling only the facts relevant to the current
user task and appending them to the agent's model context as a ``SystemMessage``.

There is **no hard dependency** on AutoGen. Importing this module is safe
without ``autogen-core`` / ``autogen-agentchat`` installed; those imports are
required only when the adapter is used inside a real AutoGen run.

Example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.autogen import AiKnotAutoGenMemory

    kb = KnowledgeBase("assistant")
    kb.add("User prefers Python over Java")
    kb.add("User deploys APIs with Docker and Kubernetes")

    memory = AiKnotAutoGenMemory(kb)
    # Pass `memory=[memory]` into an AutoGen AssistantAgent(...)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ai_knot.integrations._task_context import task_query, user_texts
from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact, MemoryType

_AUTOGEN_IMPORT_ERROR = (
    "AutoGen memory integration requires AutoGen packages. "
    'Install with: pip install "ai-knot[autogen]" '
    '(or pip install autogen-agentchat "autogen-ext[openai]")'
)


def _content_to_text(content: object) -> str:
    """Best-effort conversion of AutoGen-style content payloads to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [_content_to_text(item) for item in content]
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            value = content.get(key)
            text = _content_to_text(value) if value is not None else ""
            if text:
                return text
        return ""
    for attr in ("text", "content", "value"):
        value = getattr(content, attr, None)
        text = _content_to_text(value) if value is not None else ""
        if text:
            return text
    return ""


def _memory_type(value: object) -> MemoryType:
    """Coerce metadata into an ai-knot ``MemoryType``."""
    if isinstance(value, MemoryType):
        return value
    if isinstance(value, str):
        try:
            return MemoryType(value)
        except ValueError:
            return MemoryType.SEMANTIC
    return MemoryType.SEMANTIC


def _importance(value: object, *, default: float = 0.8) -> float:
    """Coerce metadata into a bounded importance score."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _tags(value: object) -> list[str]:
    """Coerce metadata tags to a clean string list."""
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _fact_metadata(fact: Fact, *, score: float | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "id": fact.id,
        "type": str(fact.type),
        "importance": fact.importance,
        "created_at": fact.created_at.isoformat(),
    }
    if fact.tags:
        metadata["tags"] = list(fact.tags)
    if fact.event_time is not None:
        metadata["event_time"] = fact.event_time.isoformat()
    if score is not None:
        metadata["score"] = round(score, 4)
    return metadata


def _autogen_memory_symbols() -> tuple[Any, Any, Any, Any]:
    """Import AutoGen memory symbols only at runtime."""
    try:
        from autogen_core.memory import (
            MemoryContent,
            MemoryMimeType,
            MemoryQueryResult,
            UpdateContextResult,
        )
    except ImportError as exc:  # pragma: no cover - exercised by explicit import tests
        raise ImportError(_AUTOGEN_IMPORT_ERROR) from exc
    return MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult


def _autogen_system_message() -> Any:
    """Import AutoGen's ``SystemMessage`` only at runtime."""
    try:
        from autogen_core.models import SystemMessage
    except ImportError as exc:  # pragma: no cover - exercised by explicit import tests
        raise ImportError(_AUTOGEN_IMPORT_ERROR) from exc
    return SystemMessage


class AiKnotAutoGenMemory:
    """AutoGen ``Memory``-compatible adapter backed by ``KnowledgeBase``.

    The adapter keeps AutoGen's short-term context management intact and adds a
    second layer: persistent facts recalled from ai-knot and injected only when
    they are relevant to the current turn.

    Args:
        knowledge_base: The KnowledgeBase to query for long-term memory.
        name: Optional identifier for this memory instance.
        top_k: Maximum number of facts to recall per turn.
        heading: Heading used in the injected memory block.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        name: str = "ai_knot_memory",
        top_k: int = 5,
        heading: str = "Relevant memory content",
    ) -> None:
        self._kb = knowledge_base
        self._name = name
        self._top_k = top_k
        self._heading = heading

    @property
    def name(self) -> str:
        """Return the memory instance identifier."""
        return self._name

    def extract_query(self, messages: Sequence[object], *, task_context: str | None = None) -> str:
        """Build a bounded user-task query from this context's messages only."""
        return task_query(user_texts(messages), task_context=task_context)

    def build_memory_context(self, memories: Sequence[object]) -> str:
        """Format recalled memories as a ``SystemMessage`` payload."""
        lines = []
        for index, memory in enumerate(memories, start=1):
            value = _content_to_text(getattr(memory, "content", memory))
            if value:
                lines.append(f"{index}. {value}")
        if not lines:
            return ""
        return f"\n{self._heading}:\n" + "\n".join(lines) + "\n"

    async def query(
        self,
        query: str | object = "",
        cancellation_token: object | None = None,  # noqa: ARG002 - protocol compat
        **kwargs: Any,
    ) -> Any:
        """Return the most relevant ai-knot facts as AutoGen ``MemoryContent`` objects."""
        MemoryContent, MemoryMimeType, MemoryQueryResult, _ = _autogen_memory_symbols()
        query_text = _content_to_text(getattr(query, "content", query))
        if not query_text:
            return MemoryQueryResult(results=[])

        raw_limit = kwargs.get("top_k", kwargs.get("limit", self._top_k))
        top_k = int(raw_limit) if raw_limit is not None else self._top_k
        text_mime = getattr(MemoryMimeType, "TEXT", "text/plain")

        pairs = self._kb.recall_facts_with_scores(query_text, top_k=top_k)
        results = [
            MemoryContent(
                content=fact.answer_surface,
                mime_type=text_mime,
                metadata=_fact_metadata(fact, score=score),
            )
            for fact, score in pairs
        ]
        return MemoryQueryResult(results=results)

    async def update_context(self, model_context: Any, *, task_context: str | None = None) -> Any:
        """Append relevant long-term memory to an AutoGen model context."""
        _, _, MemoryQueryResult, UpdateContextResult = _autogen_memory_symbols()
        SystemMessage = _autogen_system_message()

        messages = await model_context.get_messages()
        query_text = self.extract_query(messages, task_context=task_context)
        if not query_text:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        memories = await self.query(query_text)
        memory_context = self.build_memory_context(memories.results)
        if memory_context:
            await model_context.add_message(SystemMessage(content=memory_context))
        return UpdateContextResult(memories=memories)

    async def add(
        self,
        content: object,
        cancellation_token: object | None = None,  # noqa: ARG002 - protocol compat
    ) -> None:
        """Persist a new AutoGen ``MemoryContent`` item into ai-knot."""
        body = _content_to_text(getattr(content, "content", content))
        if not body:
            return

        metadata = getattr(content, "metadata", None) or {}
        self._kb.add(
            body,
            type=_memory_type(metadata.get("type")),
            importance=_importance(metadata.get("importance")),
            tags=_tags(metadata.get("tags")),
        )

    async def clear(self) -> None:
        """Forget every fact for this agent namespace."""
        self._kb.clear_all()

    async def close(self) -> None:
        """AutoGen ``Memory`` compatibility hook — nothing to close."""
        return None
