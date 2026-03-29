"""Tests for OpenClaw integration — no real API calls, no subprocess."""

from __future__ import annotations

import pathlib

import pytest

from agentmemo.integrations.openclaw import OpenClawMemoryAdapter, generate_mcp_config
from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="openclaw_test", storage=YAMLStorage(base_dir=str(tmp_path)))


@pytest.fixture
def adapter(kb: KnowledgeBase) -> OpenClawMemoryAdapter:
    return OpenClawMemoryAdapter(kb)


class TestOpenClawMemoryAdapter:
    def test_init(self, kb: KnowledgeBase, adapter: OpenClawMemoryAdapter) -> None:
        assert adapter._kb is kb

    def test_add_single_message(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        result = adapter.add([{"role": "user", "content": "Deploy at 2 AM UTC"}])

        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "Deploy at 2 AM UTC"
        assert len(kb.list_facts()) == 1

    def test_add_extracts_last_user_message(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I prefer dark mode"},
        ]
        result = adapter.add(messages)

        assert len(result["results"]) == 1
        assert "dark mode" in result["results"][0]["memory"]

    def test_add_no_user_messages_returns_empty(self, adapter: OpenClawMemoryAdapter) -> None:
        result = adapter.add([{"role": "assistant", "content": "Sure!"}])
        assert result == {"results": []}

    def test_search_returns_memory_items(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("User deploys on Fridays")

        items = adapter.search("deployment day")

        assert len(items) >= 1
        item = items[0]
        assert "id" in item
        assert "memory" in item
        assert "score" in item
        assert "metadata" in item
        assert "type" in item["metadata"]
        assert "importance" in item["metadata"]
        assert "created_at" in item["metadata"]

    def test_search_empty_kb(self, adapter: OpenClawMemoryAdapter) -> None:
        assert adapter.search("anything") == []

    def test_get_known_id(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        fact = kb.add("User lives in Berlin")
        item = adapter.get(fact.id)

        assert item["id"] == fact.id
        assert item["memory"] == "User lives in Berlin"

    def test_get_unknown_id_raises(self, adapter: OpenClawMemoryAdapter) -> None:
        with pytest.raises(KeyError):
            adapter.get("nonexist")

    def test_get_all_returns_all(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        kb.add("Fact one")
        kb.add("Fact two")
        kb.add("Fact three")

        items = adapter.get_all()

        assert len(items) == 3
        memories = {i["memory"] for i in items}
        assert memories == {"Fact one", "Fact two", "Fact three"}

    def test_get_all_accepts_user_id(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("Some fact")
        # user_id is silently ignored — should not raise
        items = adapter.get_all(user_id="any_user")
        assert len(items) == 1

    def test_delete_removes_fact(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        fact = kb.add("Temporary fact")
        assert len(adapter.get_all()) == 1

        adapter.delete(fact.id)

        assert adapter.get_all() == []

    def test_memory_item_score_in_search(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("User prefers pytest")
        items = adapter.search("testing framework")
        if items:
            assert items[0]["score"] is not None
            assert 0.0 <= items[0]["score"] <= 1.0


class TestGenerateMcpConfig:
    def test_defaults(self) -> None:
        cfg = generate_mcp_config()

        assert cfg["mcpServers"]["agentmemo"]["command"] == "agentmemo-mcp"
        env = cfg["mcpServers"]["agentmemo"]["env"]
        assert env["AGENTMEMO_AGENT_ID"] == "default"
        assert env["AGENTMEMO_DATA_DIR"] == ".agentmemo"
        assert env["AGENTMEMO_STORAGE"] == "sqlite"

    def test_custom_params(self) -> None:
        cfg = generate_mcp_config(agent_id="bot_a", data_dir="/data/mem", storage="yaml")

        env = cfg["mcpServers"]["agentmemo"]["env"]
        assert env["AGENTMEMO_AGENT_ID"] == "bot_a"
        assert env["AGENTMEMO_DATA_DIR"] == "/data/mem"
        assert env["AGENTMEMO_STORAGE"] == "yaml"

    def test_is_json_serialisable(self) -> None:
        import json

        cfg = generate_mcp_config("agent_x")
        dumped = json.dumps(cfg)
        assert "agentmemo-mcp" in dumped
        assert "agent_x" in dumped
