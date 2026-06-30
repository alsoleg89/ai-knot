"""Tests for the LangChain / LangGraph adapters — no langchain dependency required."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.integrations.langchain import (
    AiKnotChatMemory,
    AiKnotRetriever,
    facts_to_documents,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="lc_test", storage=YAMLStorage(base_dir=str(tmp_path)))


class TestAiKnotRetriever:
    def test_invoke_returns_relevant_documents(self, kb: KnowledgeBase) -> None:
        kb.add("User ships in Go and avoids Java")
        kb.add("Team standup is at 10am")
        retriever = AiKnotRetriever(kb, top_k=3)

        docs = retriever.invoke("what language does the user use?")

        assert docs, "expected at least one document"
        assert any("Go" in d.page_content for d in docs)
        # The irrelevant standup fact should not crowd out the answer.
        assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)

    def test_get_relevant_documents_alias(self, kb: KnowledgeBase) -> None:
        kb.add("User deploys with Docker")
        retriever = AiKnotRetriever(kb)
        docs = retriever.get_relevant_documents("docker deployment")
        assert any("Docker" in d.page_content for d in docs)

    def test_metadata_carries_score_and_type(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        retriever = AiKnotRetriever(kb)
        docs = retriever.get_relevant_documents("python")
        assert docs[0].metadata["type"] == "semantic"
        assert "score" in docs[0].metadata
        assert "id" in docs[0].metadata

    def test_top_k_override(self, kb: KnowledgeBase) -> None:
        for i in range(8):
            kb.add(f"Deployment fact number {i}")
        retriever = AiKnotRetriever(kb, top_k=5)
        assert len(retriever.get_relevant_documents("deployment", top_k=2)) <= 2

    def test_ainvoke(self, kb: KnowledgeBase) -> None:
        import asyncio

        kb.add("User ships in Go")
        retriever = AiKnotRetriever(kb)
        docs = asyncio.run(retriever.ainvoke("language"))
        assert any("Go" in d.page_content for d in docs)


class TestAiKnotChatMemory:
    def test_memory_variables(self, kb: KnowledgeBase) -> None:
        assert AiKnotChatMemory(kb).memory_variables == ["history"]
        assert AiKnotChatMemory(kb, memory_key="ctx").memory_variables == ["ctx"]

    def test_save_then_load(self, kb: KnowledgeBase) -> None:
        memory = AiKnotChatMemory(kb)
        memory.save_context({"input": "I deploy everything in Docker"}, {"output": "Noted."})

        loaded = memory.load_memory_variables({"input": "how should I deploy?"})

        assert "history" in loaded
        assert "Docker" in loaded["history"]

    def test_save_context_persists_fact(self, kb: KnowledgeBase) -> None:
        AiKnotChatMemory(kb).save_context({"input": "User likes tea"}, {"output": "ok"})
        assert any("tea" in f.content for f in kb.list_facts())

    def test_load_empty_when_no_match(self, kb: KnowledgeBase) -> None:
        memory = AiKnotChatMemory(kb)
        assert memory.load_memory_variables({"input": "anything"}) == {"history": ""}

    def test_clear_forgets_all(self, kb: KnowledgeBase) -> None:
        memory = AiKnotChatMemory(kb)
        memory.save_context({"input": "fact one"}, {"output": "ok"})
        memory.save_context({"input": "fact two"}, {"output": "ok"})
        memory.clear()
        assert kb.list_facts() == []

    def test_extract_falls_back_to_first_string(self, kb: KnowledgeBase) -> None:
        # No "input" key — should pick the first string value.
        memory = AiKnotChatMemory(kb)
        memory.save_context({"question": "User uses Kubernetes"}, {"output": "ok"})
        assert any("Kubernetes" in f.content for f in kb.list_facts())


class TestFactsToDocuments:
    def test_converts_with_attributes(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        docs = facts_to_documents(kb.list_facts())
        assert len(docs) == 1
        assert docs[0].page_content == "User prefers Python"
        assert docs[0].metadata["type"] == "semantic"
