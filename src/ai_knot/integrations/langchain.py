"""LangChain / LangGraph memory adapters.

Three dependency-light integration points let a LangChain or LangGraph agent
use ai-knot as long-term memory **without** ai-knot taking a hard dependency on
LangChain:

**Which path should I use?**

+-----------------------------+---------------------------------------------+
| Situation                   | Solution                                    |
+=============================+=============================================+
| Generic function-calling    | ``create_basic_memory_functions(kb)`` —     |
| agent / custom runtime      | plain Python callables for the literal      |
|                             | ``add/search/list/delete`` loop             |
+-----------------------------+---------------------------------------------+
| Agent tool flow /           | ``create_basic_memory_tools(kb)`` —         |
| generic tool-calling agent  | explicit ``add/search/list/delete`` tools   |
|                             | for the first-run memory loop               |
+-----------------------------+---------------------------------------------+
| Agent tool flow /           | ``create_manage_memory_tool(kb)`` +         |
| ``create_react_agent(...)`` | ``create_search_memory_tool(kb)`` —         |
|                             | tool-style helpers that mirror the          |
|                             | LangMem onboarding shape                    |
+-----------------------------+---------------------------------------------+
| RAG / retrieval-augmented   | ``AiKnotRetriever(kb)`` — duck-typed        |
| chain or LangGraph node     | retriever (``invoke`` / ``get_relevant_     |
|                             | documents``)                                |
+-----------------------------+---------------------------------------------+
| Conversational chain that   | ``AiKnotChatMemory(kb)`` — mirrors          |
| wants drop-in memory        | BaseChatMemory (``save_context`` /          |
|                             | ``load_memory_variables``)                  |
+-----------------------------+---------------------------------------------+

``langchain`` is **not** imported here. If ``langchain_core`` is installed, the
retriever returns real ``Document`` objects and the tool helpers return real
``StructuredTool`` objects. Otherwise ai-knot falls back to lightweight shims
with the same practical invocation surface, so the adapters remain fully usable
and testable with only ai-knot installed.

Tool-flow example::

    from langgraph.prebuilt import create_react_agent
    from ai_knot.integrations.langchain import (
        create_basic_memory_tools,
    )

    tools = create_basic_memory_tools(kb, top_k=3)
    agent = create_react_agent(model, tools=tools)

Generic-callable example::

    from ai_knot.integrations import create_basic_memory_functions

    functions = create_basic_memory_functions(kb, top_k=3, include_get=True)
    # Pass these into any runtime that accepts ordinary Python callables.

Retriever example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.langchain import AiKnotRetriever

    kb = KnowledgeBase("my_agent")
    kb.add("User ships in Go and avoids Java")
    retriever = AiKnotRetriever(kb, top_k=3)
    docs = retriever.invoke("what language does the user use?")
    print(docs[0].page_content)   # "User ships in Go and avoids Java"

Conversational-memory example::

    from ai_knot.integrations.langchain import AiKnotChatMemory

    memory = AiKnotChatMemory(kb)
    memory.save_context({"input": "I deploy everything in Docker"}, {"output": "Got it."})
    memory.load_memory_variables({"input": "how should I deploy?"})
    # {"history": "[1] I deploy everything in Docker"}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact, MemoryType


@dataclass
class _Document:
    """Fallback Document shim when ``langchain_core`` is not installed.

    Matches the attribute surface LangChain consumers rely on
    (``page_content`` + ``metadata``).
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Tool:
    """Fallback LangChain-tool shim when ``langchain_core`` is not installed."""

    name: str
    description: str
    _func: Callable[..., str]

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self._func(*args, **kwargs)

    def invoke(self, input: Any) -> str:
        """Match the common LangChain ``Tool.invoke(...)`` shape."""
        if isinstance(input, dict):
            return self._func(**input)
        if input is None:
            return self._func()
        return self._func(input)

    async def ainvoke(self, input: Any) -> str:
        """Async shim for agent runtimes that await tool invocation."""
        return self.invoke(input)


def _make_document(content: str, metadata: dict[str, Any]) -> Any:
    """Return a real LangChain ``Document`` if available, else the shim."""
    try:
        from langchain_core.documents import Document  # type: ignore[import-not-found]
    except ImportError:
        return _Document(page_content=content, metadata=metadata)
    return Document(page_content=content, metadata=metadata)


def _make_tool(*, func: Callable[..., str], name: str, description: str) -> Any:
    """Return a real LangChain ``StructuredTool`` if available, else a shim."""
    try:
        from langchain_core.tools import StructuredTool  # type: ignore[import-not-found]
    except ImportError:
        return _Tool(name=name, description=description, _func=func)
    return StructuredTool.from_function(func=func, name=name, description=description)


def _finalize_callable(
    func: Callable[..., str],
    *,
    name: str,
    description: str,
) -> Callable[..., str]:
    """Attach runtime metadata that tool / function-calling runtimes inspect."""
    func.__name__ = name
    func.__doc__ = description
    return func


def _fact_metadata(fact: Fact, *, score: float | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "id": fact.id,
        "type": str(fact.type),
        "importance": fact.importance,
        "created_at": fact.created_at.isoformat(),
    }
    if score is not None:
        meta["score"] = round(score, 4)
    return meta


def facts_to_documents(facts: list[Fact]) -> list[Any]:
    """Convert ai-knot facts into LangChain ``Document`` objects (or shims)."""
    return [_make_document(f.content, _fact_metadata(f)) for f in facts]


def _list_facts_text(knowledge_base: KnowledgeBase) -> str:
    facts = knowledge_base.list_facts()
    if not facts:
        return "No stored facts."
    return "\n".join(f"[{fact.id}] {fact.content}" for fact in facts)


def _fact_detail_text(fact: Fact) -> str:
    tags = ", ".join(fact.tags) if fact.tags else "none"
    active = "yes" if fact.is_active() else "no"
    retention_pct = f"{fact.retention_score * 100:.0f}%"
    lines = [
        f"[{fact.id}] {fact.content}",
        f"type={fact.type.value}",
        f"importance={fact.importance:.2f}",
        f"retention={retention_pct}",
        f"accessed={fact.access_count}x",
        f"active={active}",
        f"tags={tags}",
    ]
    if fact.event_time is not None:
        lines.append(f"event_time={fact.event_time.isoformat()}")
    lines.append(f"valid_from={fact.valid_from.isoformat() if fact.valid_from else 'n/a'}")
    lines.append(f"valid_until={fact.valid_until.isoformat() if fact.valid_until else '(active)'}")
    return "\n".join(lines)


def create_add_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "add_memory",
    description: str | None = None,
) -> Any:
    """Build a single-purpose add tool over ai-knot memory."""

    func = create_add_memory_function(
        knowledge_base,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_add_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "add_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python add callable over ai-knot memory."""

    resolved_description = description or (
        "Store a new long-term fact in ai-knot memory. Use this when the user "
        "states a durable preference, constraint, stack choice, or decision "
        "worth remembering across sessions."
    )

    def add_memory(
        content: str | None = None,
        importance: float = 0.8,
        fact_type: str = "semantic",
    ) -> str:
        if not content or not content.strip():
            return "Missing content."
        try:
            fact = knowledge_base.add(
                content.strip(),
                importance=importance,
                type=MemoryType(fact_type),
            )
        except (TypeError, ValueError) as exc:
            return str(exc)
        return f"Stored fact {fact.id}: {fact.content}"

    return _finalize_callable(add_memory, name=name, description=resolved_description)


def create_search_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int = 5,
    name: str = "search_memory",
    description: str | None = None,
) -> Any:
    """Build a LangGraph/LangChain-style search tool over ai-knot memory."""

    func = create_search_memory_function(
        knowledge_base,
        top_k=top_k,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_search_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int = 5,
    name: str = "search_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python search callable over ai-knot memory."""

    resolved_description = description or (
        "Search ai-knot long-term memory for facts relevant to the current user "
        "request. Use this before answering when past preferences, stack "
        "choices, or prior decisions may matter."
    )

    def search_memory(query: str, top_k: int = top_k) -> str:
        recalled = knowledge_base.recall(query, top_k=top_k)
        return recalled or "No relevant memory found."

    return _finalize_callable(search_memory, name=name, description=resolved_description)


def create_list_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "list_memory",
    description: str | None = None,
) -> Any:
    """Build a single-purpose list tool over ai-knot memory."""

    func = create_list_memory_function(
        knowledge_base,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_list_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "list_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python list callable over ai-knot memory."""

    resolved_description = description or (
        "List the active facts currently stored in ai-knot long-term memory. "
        "Use this to inspect or debug what is actually persisted."
    )

    def list_memory() -> str:
        return _list_facts_text(knowledge_base)

    return _finalize_callable(list_memory, name=name, description=resolved_description)


def create_get_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "get_memory",
    description: str | None = None,
) -> Any:
    """Build a single-purpose by-id inspection tool over ai-knot memory."""

    func = create_get_memory_function(
        knowledge_base,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_get_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "get_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python by-id inspection callable over ai-knot memory."""

    resolved_description = description or (
        "Inspect one stored ai-knot fact by fact_id. Use this when you already "
        "have an ID from list/search/debug output and want to verify the exact "
        "stored content or metadata before deleting or superseding it."
    )

    def get_memory(fact_id: str | None = None) -> str:
        if not fact_id:
            return "Missing fact_id."
        try:
            fact = knowledge_base.get(fact_id)
        except KeyError:
            return f"No fact found with id {fact_id}."
        return _fact_detail_text(fact)

    return _finalize_callable(get_memory, name=name, description=resolved_description)


def create_delete_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "delete_memory",
    description: str | None = None,
) -> Any:
    """Build a single-purpose delete tool over ai-knot memory."""

    func = create_delete_memory_function(
        knowledge_base,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_delete_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "delete_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python delete callable over ai-knot memory."""

    resolved_description = description or (
        "Delete one stored fact from ai-knot long-term memory by fact_id. "
        "Use this when memory is stale, incorrect, or no longer relevant."
    )

    def delete_memory(fact_id: str | None = None) -> str:
        if not fact_id:
            return "Missing fact_id."
        by_id = {fact.id: fact for fact in knowledge_base.list_facts()}
        if fact_id not in by_id:
            return f"No fact found with id {fact_id}."
        fact = by_id[fact_id]
        knowledge_base.delete(fact_id)
        return f"Deleted fact {fact_id}: {fact.content}"

    return _finalize_callable(delete_memory, name=name, description=resolved_description)


def create_basic_memory_functions(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int = 5,
    include_get: bool = False,
) -> list[Callable[..., str]]:
    """Return plain Python callables for the literal add/search/list/delete loop."""

    functions: list[Callable[..., str]] = [
        create_add_memory_function(knowledge_base),
        create_search_memory_function(knowledge_base, top_k=top_k),
        create_list_memory_function(knowledge_base),
        create_delete_memory_function(knowledge_base),
    ]
    if include_get:
        functions.append(create_get_memory_function(knowledge_base))
    return functions


def create_basic_memory_tools(
    knowledge_base: KnowledgeBase,
    *,
    top_k: int = 5,
    include_get: bool = False,
) -> list[Any]:
    """Return the literal add/search/list/delete tool loop for agents.

    This mirrors the first-run surface shown in the README and CLI and is the
    clearest path when you want an agent runtime to see explicit CRUD/search
    verbs instead of routing everything through one generic manage tool.

    Set ``include_get=True`` when you also want a targeted by-id inspection tool
    for correction/debug loops after the agent already knows a ``fact_id``.
    """
    return [
        _make_tool(func=func, name=func.__name__, description=func.__doc__ or "")
        for func in create_basic_memory_functions(
            knowledge_base,
            top_k=top_k,
            include_get=include_get,
        )
    ]


def create_manage_memory_tool(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "manage_memory",
    description: str | None = None,
) -> Any:
    """Build a LangGraph/LangChain-style memory-management tool over ai-knot."""

    func = create_manage_memory_function(
        knowledge_base,
        name=name,
        description=description,
    )
    return _make_tool(func=func, name=name, description=func.__doc__ or "")


def create_manage_memory_function(
    knowledge_base: KnowledgeBase,
    *,
    name: str = "manage_memory",
    description: str | None = None,
) -> Callable[..., str]:
    """Build a plain Python memory-management callable over ai-knot."""

    resolved_description = description or (
        "Manage ai-knot long-term memory facts. Use action='add' to store a "
        "fact, action='list' to inspect active facts, action='get' to inspect "
        "one fact by ID, and action='delete' to remove one by fact_id."
    )

    def manage_memory(
        action: str,
        content: str | None = None,
        fact_id: str | None = None,
    ) -> str:
        normalized = action.strip().lower()

        if normalized == "add":
            if not content or not content.strip():
                return "Missing content for action='add'."
            fact = knowledge_base.add(content.strip())
            return f"Stored fact {fact.id}: {fact.content}"

        if normalized == "list":
            return _list_facts_text(knowledge_base)

        if normalized == "get":
            if not fact_id:
                return "Missing fact_id for action='get'."
            try:
                fact = knowledge_base.get(fact_id)
            except KeyError:
                return f"No fact found with id {fact_id}."
            return _fact_detail_text(fact)

        if normalized == "delete":
            if not fact_id:
                return "Missing fact_id for action='delete'."
            by_id = {fact.id: fact for fact in knowledge_base.list_facts()}
            if fact_id not in by_id:
                return f"No fact found with id {fact_id}."
            fact = by_id[fact_id]
            knowledge_base.delete(fact_id)
            return f"Deleted fact {fact_id}: {fact.content}"

        return "Unknown action. Use 'add', 'list', 'get', or 'delete'."

    return _finalize_callable(manage_memory, name=name, description=resolved_description)


class AiKnotRetriever:
    """A LangChain-compatible retriever backed by an ai-knot KnowledgeBase.

    Implements the modern Runnable ``invoke`` entry point and the classic
    ``get_relevant_documents`` method, so it drops into RAG chains and
    LangGraph nodes that expect a retriever. Recall never calls an LLM.

    Args:
        knowledge_base: The KnowledgeBase to query.
        top_k: Default maximum number of documents to return.
    """

    def __init__(self, knowledge_base: KnowledgeBase, *, top_k: int = 5) -> None:
        self._kb = knowledge_base
        self._top_k = top_k

    def get_relevant_documents(self, query: str, *, top_k: int | None = None) -> list[Any]:
        """Return documents relevant to *query*, ranked by ai-knot's fusion score."""
        pairs = self._kb.recall_facts_with_scores(query, top_k=top_k or self._top_k)
        return [_make_document(f.content, _fact_metadata(f, score=s)) for f, s in pairs]

    # Modern LangChain Runnable surface — chains call ``.invoke(query)``.
    def invoke(self, query: str, config: dict[str, Any] | None = None, **_: Any) -> list[Any]:
        """Runnable-style alias for :meth:`get_relevant_documents`."""
        return self.get_relevant_documents(query)

    async def ainvoke(
        self, query: str, config: dict[str, Any] | None = None, **_: Any
    ) -> list[Any]:
        """Async alias — recall is in-process, so this just delegates."""
        return self.get_relevant_documents(query)


class AiKnotChatMemory:
    """Drop-in conversational memory mirroring LangChain's ``BaseChatMemory``.

    ``save_context`` distills the user turn into a stored fact; the next
    ``load_memory_variables`` recalls only the facts relevant to the current
    input — so the prompt carries durable knowledge, not the whole transcript.

    Args:
        knowledge_base: The KnowledgeBase to read and write.
        memory_key: Key under which recalled context is returned (default
            ``"history"``, matching LangChain convention).
        input_key: Key in the ``inputs`` dict holding the user's message
            (default ``"input"``).
        top_k: Maximum facts to recall per turn.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        memory_key: str = "history",
        input_key: str = "input",
        top_k: int = 5,
    ) -> None:
        self._kb = knowledge_base
        self._memory_key = memory_key
        self._input_key = input_key
        self._top_k = top_k

    @property
    def memory_variables(self) -> list[str]:
        """The keys this memory injects — LangChain reads this to wire the prompt."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Recall facts relevant to the current input for prompt injection."""
        query = self._extract(inputs)
        recalled = self._kb.recall(query, top_k=self._top_k) if query else ""
        return {self._memory_key: recalled}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Persist the user's turn as a fact. The AI output is not stored."""
        content = self._extract(inputs)
        if content:
            self._kb.add(content)

    def clear(self) -> None:
        """Forget every fact for this agent."""
        for fact in self._kb.list_facts():
            self._kb.forget(fact.id)

    def _extract(self, inputs: dict[str, Any]) -> str:
        if self._input_key in inputs:
            return str(inputs[self._input_key])
        # Fall back to the first string value, mirroring BaseChatMemory leniency.
        for value in inputs.values():
            if isinstance(value, str):
                return value
        return ""
