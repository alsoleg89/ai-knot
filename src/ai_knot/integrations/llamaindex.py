"""LlamaIndex memory adapter.

ai-knot complements LlamaIndex's chat history with deterministic, self-hosted
long-term memory. The adapter follows the same high-level seam as the official
Mem0 integration: pass a memory object into `memory=...`, keep short-term chat
history in LlamaIndex, and prepend only the relevant long-term facts on read.

There is **no hard dependency** on ``llama-index-core``. Importing this module
is safe without LlamaIndex installed; the class falls back to a lightweight
in-memory chat-history shim so the adapter remains testable and the zero-network
surface demo still works.

Example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory

    kb = KnowledgeBase("assistant")
    kb.add("User deploys APIs with Docker and Kubernetes")
    kb.add("User prefers Python over Java")

    memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=4)
    # Then pass ``memory`` into SimpleChatEngine.from_defaults(..., memory=memory)
    # or FunctionAgent(...).run(..., memory=memory)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import ConversationTurn, MemoryType

try:  # pragma: no cover - exercised via import-safe tests
    from llama_index.core.base.llms.types import ChatMessage as _LlamaChatMessage
    from llama_index.core.base.llms.types import MessageRole as _LlamaMessageRole
    from llama_index.core.bridge.pydantic import ConfigDict, Field, PrivateAttr, SerializeAsAny
    from llama_index.core.memory import BaseMemory as _LlamaBaseMemory
    from llama_index.core.memory import Memory as _LlamaPrimaryMemory
except ImportError:  # pragma: no cover - normal test path
    _LLAMAINDEX_AVAILABLE = False
    _LlamaBaseMemory = object
else:  # pragma: no cover - exercised when the extra is installed
    _LLAMAINDEX_AVAILABLE = True


@dataclass
class _InMemoryChatHistory:
    """Fallback primary memory when LlamaIndex is not installed."""

    messages: list[Any] = field(default_factory=list)

    def get(self, input: str | None = None, **kwargs: Any) -> list[Any]:  # noqa: ARG002
        return list(self.messages)

    def get_all(self) -> list[Any]:
        return list(self.messages)

    def put(self, message: Any) -> None:
        self.messages.append(message)

    def set(self, messages: list[Any]) -> None:
        self.messages = list(messages)

    def reset(self) -> None:
        self.messages = []


def _role_name(message: object) -> str:
    role: object | None = message.get("role") if isinstance(message, dict) else getattr(
        message, "role", None
    )
    if hasattr(role, "value"):
        role = role.value
    if role is None:
        return ""
    return str(role).strip().lower()


def _text_from_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_text_from_content(part) for part in content]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if isinstance(content, dict):
        for key in ("text", "value", "content"):
            if key in content:
                text = _text_from_content(content[key])
                if text:
                    return text
        return ""

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    nested = getattr(content, "content", None)
    if nested is not None and nested is not content:
        return _text_from_content(nested)

    return ""


def _message_text(message: object) -> str:
    if isinstance(message, dict):
        return _text_from_content(message.get("content"))
    return _text_from_content(getattr(message, "content", None))


def _make_message(role: str, content: str) -> Any:
    if _LLAMAINDEX_AVAILABLE:
        message_role = {
            "user": _LlamaMessageRole.USER,
            "assistant": _LlamaMessageRole.ASSISTANT,
            "system": _LlamaMessageRole.SYSTEM,
        }.get(role, _LlamaMessageRole.USER)
        return _LlamaChatMessage(role=message_role, content=content)
    return {"role": role, "content": content}


def _inject_memory_block(messages: list[Any], memory_block: str) -> list[Any]:
    updated = list(messages)
    if updated and _role_name(updated[0]) == "system":
        existing = _message_text(updated[0]).strip()
        combined = f"{existing}\n\n{memory_block}" if existing else memory_block
        updated[0] = _make_message("system", combined)
        return updated
    return [_make_message("system", memory_block), *updated]


def _build_query(messages: list[Any], explicit_input: str | None, *, limit: int) -> str:
    if explicit_input and explicit_input.strip():
        return explicit_input.strip()

    recent: list[str] = []
    for message in reversed(messages):
        role = _role_name(message)
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        recent.append(text)
        if len(recent) >= max(1, limit):
            break
    return "\n".join(reversed(recent)).strip()


def _messages_to_turns(messages: list[Any], *, include_assistant: bool) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for message in messages:
        role = _role_name(message)
        if role not in {"user", "assistant", "system"}:
            continue
        if role == "assistant" and not include_assistant:
            continue
        content = _message_text(message).strip()
        if not content:
            continue
        turns.append(ConversationTurn(role=role, content=content))
    return turns


def _messages_to_texts(messages: list[Any], *, include_assistant: bool) -> list[str]:
    texts: list[str] = []
    for message in messages:
        role = _role_name(message)
        if role == "user" or (include_assistant and role == "assistant"):
            text = _message_text(message).strip()
            if text:
                texts.append(text)
    return texts


def _memory_type(value: MemoryType | str) -> MemoryType:
    return value if isinstance(value, MemoryType) else MemoryType(value)


class _AiKnotLlamaIndexMemoryMixin:
    """Shared behavior for the real and fallback LlamaIndex adapters."""

    primary_memory: Any
    top_k: int
    heading: str
    query_message_limit: int
    store_assistant_messages: bool
    extract_on_write: bool
    fact_type: MemoryType | str
    default_importance: float
    dedup_exact: bool
    provider: str | None
    api_key: str | None
    model: str | None
    provider_kwargs: dict[str, str]
    tags: list[str]
    _kb: KnowledgeBase

    def get(self, input: str | None = None, **kwargs: Any) -> list[Any]:
        """Return chat history with relevant ai-knot facts injected up front."""
        messages = list(self.primary_memory.get(input=input, **kwargs))
        query = _build_query(messages, input, limit=self.query_message_limit)
        if not query:
            return messages

        context = self._kb.recall(query, top_k=self.top_k)
        if not context:
            return messages
        return _inject_memory_block(messages, f"{self.heading}\n{context}")

    async def aget(self, input: str | None = None, **kwargs: Any) -> list[Any]:
        return await asyncio.to_thread(self.get, input=input, **kwargs)

    def get_all(self) -> list[Any]:
        return list(self.primary_memory.get_all())

    async def aget_all(self) -> list[Any]:
        return await asyncio.to_thread(self.get_all)

    def put(self, message: Any) -> None:
        """Store the chat message in primary memory and learn from it."""
        self._store_messages([message])
        self.primary_memory.put(message)

    async def aput(self, message: Any) -> None:
        await asyncio.to_thread(self.put, message)

    def put_messages(self, messages: list[Any]) -> None:
        for message in messages:
            self.put(message)

    async def aput_messages(self, messages: list[Any]) -> None:
        await asyncio.to_thread(self.put_messages, messages)

    def set(self, messages: list[Any]) -> None:
        """Replace chat history while only learning from newly appended messages."""
        existing = self.primary_memory.get_all()
        if len(messages) >= len(existing):
            new_messages = list(messages[len(existing) :])
        else:
            new_messages = list(messages)
        self._store_messages(new_messages)
        self.primary_memory.set(list(messages))

    async def aset(self, messages: list[Any]) -> None:
        await asyncio.to_thread(self.set, messages)

    def reset(self) -> None:
        self.primary_memory.reset()

    async def areset(self) -> None:
        await asyncio.to_thread(self.reset)

    def _store_messages(self, messages: list[Any]) -> None:
        if not messages:
            return

        if self.extract_on_write:
            turns = _messages_to_turns(
                messages,
                include_assistant=self.store_assistant_messages,
            )
            if turns:
                self._kb.learn(
                    turns,
                    provider=self.provider,
                    api_key=self.api_key,
                    model=self.model,
                    **self.provider_kwargs,
                )
            return

        for text in _messages_to_texts(messages, include_assistant=self.store_assistant_messages):
            self._store_text(text)

    def _store_text(self, text: str) -> None:
        content = text.strip()
        if not content:
            return
        if self.dedup_exact and any(
            fact.is_active() and fact.content.strip() == content for fact in self._kb.list_facts()
        ):
            return
        self._kb.add(
            content,
            type=_memory_type(self.fact_type),
            importance=self.default_importance,
            tags=list(self.tags),
        )


if _LLAMAINDEX_AVAILABLE:  # pragma: no cover - exercised when the extra is installed
    class AiKnotLlamaIndexMemory(_AiKnotLlamaIndexMemoryMixin, _LlamaBaseMemory):
        """LlamaIndex-compatible long-term memory backed by ``KnowledgeBase``.

        Args:
            knowledge_base: The KnowledgeBase to use for long-term memory.
            primary_memory: Short-term LlamaIndex memory instance. Defaults to
                ``Memory.from_defaults()``.
            top_k: Maximum number of facts to recall on each read.
            heading: Heading for the injected memory block.
            query_message_limit: How many recent chat messages to fold into the
                recall query when LlamaIndex does not pass ``input=...``.
            store_assistant_messages: Persist assistant messages alongside user
                messages on writes. Disabled by default to keep memory tighter.
            extract_on_write: When True, route writes through ``kb.learn(...)``
                instead of storing raw user text.
            fact_type: Fact type used for direct raw-message persistence.
            default_importance: Importance score used for direct raw-message
                persistence.
            dedup_exact: Skip storing an exact active duplicate fact.
            provider: Optional ``kb.learn(...)`` provider override when
                ``extract_on_write=True``.
            api_key: Optional ``kb.learn(...)`` API key override.
            model: Optional ``kb.learn(...)`` model override.
            provider_kwargs: Extra provider-specific ``kb.learn(...)`` kwargs.
            tags: Tags attached to raw-message facts stored directly by the
                adapter. Defaults to ``["llamaindex"]``.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        primary_memory: SerializeAsAny[_LlamaBaseMemory] = Field(
            description="Primary short-term LlamaIndex chat memory."
        )
        top_k: int = Field(default=5, description="Maximum recalled facts per read.")
        heading: str = Field(default="## Agent Memory", description="Injected memory heading.")
        query_message_limit: int = Field(
            default=3,
            description=(
                "Recent chat messages to fold into recall when no explicit input is passed."
            ),
        )
        store_assistant_messages: bool = Field(
            default=False,
            description="Store assistant messages as facts alongside user messages.",
        )
        extract_on_write: bool = Field(
            default=False,
            description="Use kb.learn(...) on writes instead of storing raw messages directly.",
        )
        fact_type: str = Field(default="semantic", description="Fact type for raw-message writes.")
        default_importance: float = Field(
            default=0.7,
            description="Importance score for raw-message writes.",
        )
        dedup_exact: bool = Field(
            default=True,
            description="Skip exact active duplicate facts on raw-message writes.",
        )
        provider: str | None = Field(
            default=None,
            description="Optional kb.learn(...) provider override.",
        )
        api_key: str | None = Field(
            default=None,
            description="Optional kb.learn(...) API key override.",
        )
        model: str | None = Field(
            default=None,
            description="Optional kb.learn(...) model override.",
        )
        provider_kwargs: dict[str, str] = Field(
            default_factory=dict,
            description="Extra kb.learn(...) provider kwargs.",
        )
        tags: list[str] = Field(
            default_factory=lambda: ["llamaindex"],
            description="Tags attached to raw-message facts persisted directly by the adapter.",
        )

        _kb: KnowledgeBase = PrivateAttr()

        def __init__(self, knowledge_base: KnowledgeBase, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._kb = knowledge_base

        @classmethod
        def class_name(cls) -> str:
            return "AiKnotLlamaIndexMemory"

        @classmethod
        def from_defaults(
            cls,
            *,
            knowledge_base: KnowledgeBase,
            primary_memory: _LlamaBaseMemory | None = None,
            top_k: int = 5,
            heading: str = "## Agent Memory",
            query_message_limit: int = 3,
            store_assistant_messages: bool = False,
            extract_on_write: bool = False,
            fact_type: MemoryType | str = MemoryType.SEMANTIC,
            default_importance: float = 0.7,
            dedup_exact: bool = True,
            provider: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            provider_kwargs: dict[str, str] | None = None,
            tags: list[str] | None = None,
        ) -> AiKnotLlamaIndexMemory:
            return cls(
                knowledge_base,
                primary_memory=primary_memory or _LlamaPrimaryMemory.from_defaults(),
                top_k=max(1, top_k),
                heading=heading,
                query_message_limit=max(1, query_message_limit),
                store_assistant_messages=store_assistant_messages,
                extract_on_write=extract_on_write,
                fact_type=_memory_type(fact_type).value,
                default_importance=default_importance,
                dedup_exact=dedup_exact,
                provider=provider,
                api_key=api_key,
                model=model,
                provider_kwargs=dict(provider_kwargs or {}),
                tags=list(tags or ["llamaindex"]),
            )

else:
    class AiKnotLlamaIndexMemory(_AiKnotLlamaIndexMemoryMixin):
        """Import-safe fallback with the same public methods as the real adapter."""

        def __init__(
            self,
            knowledge_base: KnowledgeBase,
            *,
            primary_memory: Any | None = None,
            top_k: int = 5,
            heading: str = "## Agent Memory",
            query_message_limit: int = 3,
            store_assistant_messages: bool = False,
            extract_on_write: bool = False,
            fact_type: MemoryType | str = MemoryType.SEMANTIC,
            default_importance: float = 0.7,
            dedup_exact: bool = True,
            provider: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            provider_kwargs: dict[str, str] | None = None,
            tags: list[str] | None = None,
        ) -> None:
            self._kb = knowledge_base
            self.primary_memory = primary_memory or _InMemoryChatHistory()
            self.top_k = max(1, top_k)
            self.heading = heading
            self.query_message_limit = max(1, query_message_limit)
            self.store_assistant_messages = store_assistant_messages
            self.extract_on_write = extract_on_write
            self.fact_type = _memory_type(fact_type)
            self.default_importance = default_importance
            self.dedup_exact = dedup_exact
            self.provider = provider
            self.api_key = api_key
            self.model = model
            self.provider_kwargs = dict(provider_kwargs or {})
            self.tags = list(tags or ["llamaindex"])

        @classmethod
        def class_name(cls) -> str:
            return "AiKnotLlamaIndexMemory"

        @classmethod
        def from_defaults(
            cls,
            *,
            knowledge_base: KnowledgeBase,
            primary_memory: Any | None = None,
            top_k: int = 5,
            heading: str = "## Agent Memory",
            query_message_limit: int = 3,
            store_assistant_messages: bool = False,
            extract_on_write: bool = False,
            fact_type: MemoryType | str = MemoryType.SEMANTIC,
            default_importance: float = 0.7,
            dedup_exact: bool = True,
            provider: str | None = None,
            api_key: str | None = None,
            model: str | None = None,
            provider_kwargs: dict[str, str] | None = None,
            tags: list[str] | None = None,
        ) -> AiKnotLlamaIndexMemory:
            return cls(
                knowledge_base,
                primary_memory=primary_memory,
                top_k=top_k,
                heading=heading,
                query_message_limit=query_message_limit,
                store_assistant_messages=store_assistant_messages,
                extract_on_write=extract_on_write,
                fact_type=fact_type,
                default_importance=default_importance,
                dedup_exact=dedup_exact,
                provider=provider,
                api_key=api_key,
                model=model,
                provider_kwargs=provider_kwargs,
                tags=tags,
            )


__all__ = ["AiKnotLlamaIndexMemory"]
