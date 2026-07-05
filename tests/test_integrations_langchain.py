"""Tests for the LangChain / LangGraph adapters — no langchain dependency required."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.integrations.langchain import (
    AiKnotChatMemory,
    AiKnotRetriever,
    create_add_memory_function,
    create_add_memory_tool,
    create_basic_memory_functions,
    create_basic_memory_tools,
    create_delete_memory_function,
    create_delete_memory_tool,
    create_get_memory_function,
    create_get_memory_tool,
    create_list_memory_function,
    create_list_memory_tool,
    create_manage_memory_function,
    create_manage_memory_tool,
    create_search_memory_function,
    create_search_memory_tool,
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


class TestLangGraphToolHelpers:
    def test_basic_memory_functions_expose_add_search_list_delete_loop(
        self, kb: KnowledgeBase
    ) -> None:
        functions = create_basic_memory_functions(kb, top_k=3)
        by_name = {func.__name__: func for func in functions}

        assert set(by_name) == {"add_memory", "search_memory", "list_memory", "delete_memory"}

        stored = by_name["add_memory"]("User deploys with Docker Compose")
        assert "Stored fact" in stored

        searched = by_name["search_memory"]("how does the user deploy?")
        assert "Docker Compose" in searched

        listed = by_name["list_memory"]()
        assert "Docker Compose" in listed

        fact_id = kb.list_facts()[0].id
        deleted = by_name["delete_memory"](fact_id)
        assert f"Deleted fact {fact_id}" in deleted
        assert kb.list_facts() == []

    def test_memory_functions_can_be_built_individually(self, kb: KnowledgeBase) -> None:
        add_memory = create_add_memory_function(kb)
        search_memory = create_search_memory_function(kb, top_k=3)
        list_memory = create_list_memory_function(kb)
        get_memory = create_get_memory_function(kb)
        delete_memory = create_delete_memory_function(kb)

        assert "Store a new long-term fact" in (add_memory.__doc__ or "")
        assert add_memory("User prefers quiet standups").startswith("Stored fact")
        fact_id = kb.list_facts()[0].id
        assert "quiet standups" in search_memory("what does the user prefer?")
        assert "quiet standups" in list_memory()
        assert fact_id in get_memory(fact_id)
        assert f"Deleted fact {fact_id}" in delete_memory(fact_id)

    def test_manage_memory_function_add_list_get_delete(self, kb: KnowledgeBase) -> None:
        manage_memory = create_manage_memory_function(kb)

        stored = manage_memory("add", "User deploys with Docker Compose")
        assert "Stored fact" in stored

        listed = manage_memory("list")
        assert "Docker Compose" in listed

        fact_id = kb.list_facts()[0].id
        fetched = manage_memory("get", fact_id=fact_id)
        assert "Docker Compose" in fetched
        assert f"[{fact_id}]" in fetched

        deleted = manage_memory("delete", fact_id=fact_id)
        assert f"Deleted fact {fact_id}" in deleted
        assert kb.list_facts() == []

    def test_basic_memory_functions_can_optionally_include_get(self, kb: KnowledgeBase) -> None:
        functions = create_basic_memory_functions(kb, top_k=3, include_get=True)

        assert {func.__name__ for func in functions} == {
            "add_memory",
            "search_memory",
            "list_memory",
            "delete_memory",
            "get_memory",
        }

    def test_basic_memory_tools_expose_add_search_list_delete_loop(self, kb: KnowledgeBase) -> None:
        tools = create_basic_memory_tools(kb, top_k=3)
        by_name = {tool.name: tool for tool in tools}

        assert set(by_name) == {"add_memory", "search_memory", "list_memory", "delete_memory"}

        stored = by_name["add_memory"].invoke({"content": "User deploys with Docker Compose"})
        assert "Stored fact" in stored

        searched = by_name["search_memory"].invoke({"query": "how does the user deploy?"})
        assert "Docker Compose" in searched

        listed = by_name["list_memory"].invoke({})
        assert "Docker Compose" in listed

        fact_id = kb.list_facts()[0].id
        deleted = by_name["delete_memory"].invoke({"fact_id": fact_id})
        assert f"Deleted fact {fact_id}" in deleted
        assert kb.list_facts() == []

    def test_search_memory_tool_returns_recalled_text(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        tool = create_search_memory_tool(kb, top_k=3)

        result = tool.invoke({"query": "what language does the user prefer?"})

        assert tool.name == "search_memory"
        assert "Python" in result

    def test_search_memory_tool_returns_empty_message(self, kb: KnowledgeBase) -> None:
        tool = create_search_memory_tool(kb)

        result = tool.invoke({"query": "nothing is stored yet"})

        assert result == "No relevant memory found."

    def test_add_list_delete_tools_can_be_used_individually(self, kb: KnowledgeBase) -> None:
        add_tool = create_add_memory_tool(kb)
        list_tool = create_list_memory_tool(kb)
        delete_tool = create_delete_memory_tool(kb)

        stored = add_tool.invoke({"content": "User prefers quiet standups"})
        assert "Stored fact" in stored

        listed = list_tool.invoke({})
        assert "quiet standups" in listed

        fact_id = kb.list_facts()[0].id
        deleted = delete_tool.invoke({"fact_id": fact_id})
        assert f"Deleted fact {fact_id}" in deleted
        assert kb.list_facts() == []

    def test_add_list_delete_tools_validate_inputs(self, kb: KnowledgeBase) -> None:
        add_tool = create_add_memory_tool(kb)
        list_tool = create_list_memory_tool(kb)
        delete_tool = create_delete_memory_tool(kb)

        assert add_tool.invoke({}) == "Missing content."
        assert "must be between 0.0 and 1.0" in add_tool.invoke(
            {"content": "Bad importance", "importance": 1.5}
        )
        assert list_tool.invoke({}) == "No stored facts."
        assert delete_tool.invoke({}) == "Missing fact_id."
        assert delete_tool.invoke({"fact_id": "deadbeef"}) == "No fact found with id deadbeef."

    def test_get_memory_tool_returns_targeted_fact_details(self, kb: KnowledgeBase) -> None:
        fact = kb.add("User deploys with Docker Compose")
        tool = create_get_memory_tool(kb)

        result = tool.invoke({"fact_id": fact.id})

        assert tool.name == "get_memory"
        assert fact.id in result
        assert "Docker Compose" in result
        assert "active=" in result

    def test_get_memory_tool_validation_messages(self, kb: KnowledgeBase) -> None:
        tool = create_get_memory_tool(kb)

        assert tool.invoke({}) == "Missing fact_id."
        assert tool.invoke({"fact_id": "deadbeef"}) == "No fact found with id deadbeef."

    def test_basic_memory_tools_can_optionally_include_get(self, kb: KnowledgeBase) -> None:
        tools = create_basic_memory_tools(kb, top_k=3, include_get=True)
        by_name = {tool.name: tool for tool in tools}

        assert set(by_name) == {
            "add_memory",
            "search_memory",
            "list_memory",
            "delete_memory",
            "get_memory",
        }

    def test_manage_memory_tool_add_list_get_delete(self, kb: KnowledgeBase) -> None:
        tool = create_manage_memory_tool(kb)

        stored = tool.invoke({"action": "add", "content": "User deploys with Docker Compose"})
        assert "Stored fact" in stored

        listed = tool.invoke({"action": "list"})
        assert "Docker Compose" in listed

        fact_id = kb.list_facts()[0].id
        fetched = tool.invoke({"action": "get", "fact_id": fact_id})
        assert "Docker Compose" in fetched
        assert f"[{fact_id}]" in fetched

        deleted = tool.invoke({"action": "delete", "fact_id": fact_id})
        assert f"Deleted fact {fact_id}" in deleted
        assert kb.list_facts() == []

    def test_manage_memory_tool_validation_messages(self, kb: KnowledgeBase) -> None:
        tool = create_manage_memory_tool(kb)

        assert tool.invoke({"action": "add"}) == "Missing content for action='add'."
        assert tool.invoke({"action": "get"}) == "Missing fact_id for action='get'."
        assert (
            tool.invoke({"action": "get", "fact_id": "deadbeef"})
            == "No fact found with id deadbeef."
        )
        assert tool.invoke({"action": "delete"}) == "Missing fact_id for action='delete'."
        assert (
            tool.invoke({"action": "delete", "fact_id": "deadbeef"})
            == "No fact found with id deadbeef."
        )
        assert (
            tool.invoke({"action": "oops"})
            == "Unknown action. Use 'add', 'list', 'get', or 'delete'."
        )


class TestFactsToDocuments:
    def test_converts_with_attributes(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        docs = facts_to_documents(kb.list_facts())
        assert len(docs) == 1
        assert docs[0].page_content == "User prefers Python"
        assert docs[0].metadata["type"] == "semantic"
