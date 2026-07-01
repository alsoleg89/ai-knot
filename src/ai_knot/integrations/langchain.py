"""LangChain / LangGraph memory adapters.

Two thin, dependency-light integration points so a LangChain or LangGraph agent
can use ai-knot as its long-term memory **without** ai-knot taking a hard
dependency on LangChain:

**Which path should I use?**

+-----------------------------+---------------------------------------------+
| Situation                   | Solution                                    |
+=============================+=============================================+
| RAG / retrieval-augmented   | ``AiKnotRetriever(kb)`` — duck-typed        |
| chain or LangGraph node     | retriever (``invoke`` / ``get_relevant_     |
|                             | documents``)                                |
+-----------------------------+---------------------------------------------+
| Conversational chain that   | ``AiKnotChatMemory(kb)`` — mirrors          |
| wants drop-in memory        | BaseChatMemory (``save_context`` /          |
|                             | ``load_memory_variables``)                  |
+-----------------------------+---------------------------------------------+

``langchain`` is **not** imported here. If ``langchain_core`` is installed, the
retriever returns real ``Document`` objects; otherwise it returns a lightweight
shim with the same ``page_content`` / ``metadata`` attributes, so the adapter is
fully usable (and testable) with only ai-knot installed.

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

from dataclasses import dataclass, field
from typing import Any

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact


@dataclass
class _Document:
    """Fallback Document shim when ``langchain_core`` is not installed.

    Matches the attribute surface LangChain consumers rely on
    (``page_content`` + ``metadata``).
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_document(content: str, metadata: dict[str, Any]) -> Any:
    """Return a real LangChain ``Document`` if available, else the shim."""
    try:
        from langchain_core.documents import Document  # type: ignore[import-not-found]
    except ImportError:
        return _Document(page_content=content, metadata=metadata)
    return Document(page_content=content, metadata=metadata)


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
