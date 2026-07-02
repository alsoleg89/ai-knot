"""Tests for OpenClaw integration — no real API calls, no subprocess."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from ai_knot.integrations.openclaw import OpenClawMemoryAdapter, generate_mcp_config
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


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

    def test_recall_alias_matches_search(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("User deploys on Fridays")

        assert adapter.recall("deployment schedule") == adapter.search("deployment schedule")

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

    def test_get_all_can_include_inactive(
        self,
        adapter: OpenClawMemoryAdapter,
        kb: KnowledgeBase,
    ) -> None:
        old = kb.add_resolved(
            [
                Fact(
                    content="User works at Acme",
                    entity="user",
                    attribute="employer",
                    value_text="Acme",
                )
            ]
        )[0]
        kb.add_resolved(
            [
                Fact(
                    content="User now works at Globex",
                    entity="user",
                    attribute="employer",
                    value_text="Globex",
                )
            ]
        )

        active_items = adapter.get_all()
        history_items = adapter.get_all(include_inactive=True)

        assert len(active_items) == 1
        assert active_items[0]["memory"] == "User now works at Globex"
        assert len(history_items) == 2
        old_item = next(item for item in history_items if item["id"] == old.id)
        assert old_item["metadata"]["active"] is False

    def test_get_all_accepts_user_id(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("Some fact")
        # user_id is silently ignored — should not raise
        items = adapter.get_all(user_id="any_user")
        assert len(items) == 1

    def test_list_alias_matches_get_all(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        kb.add("Some fact")

        assert adapter.list() == adapter.get_all()

    def test_get_accepts_now_anchor_for_active_status(
        self,
        adapter: OpenClawMemoryAdapter,
        kb: KnowledgeBase,
    ) -> None:
        fact = kb.add("Future migration is scheduled", event_time=datetime(2099, 1, 1, tzinfo=UTC))

        item = adapter.get(fact.id, now="2000-01-01T00:00:00+00:00")

        assert item["metadata"]["active"] is False

    def test_delete_removes_fact(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        fact = kb.add("Temporary fact")
        assert len(adapter.get_all()) == 1

        adapter.delete(fact.id)

        assert adapter.get_all() == []

    def test_forget_alias_removes_fact(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add("Temporary fact")

        adapter.forget(fact.id)

        assert adapter.get_all() == []

    def test_search_score_is_float(self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase) -> None:
        kb.add("User prefers pytest")
        items = adapter.search("testing framework")
        assert len(items) >= 1, "Expected at least one result for 'testing framework'"
        assert all(isinstance(item["score"], float) for item in items)
        assert all(item["score"] >= 0.0 for item in items)

    def test_add_multi_turn_warns(self, adapter: OpenClawMemoryAdapter) -> None:
        messages = [
            {"role": "user", "content": "I work at Sber"},
            {"role": "assistant", "content": "Got it"},
            {"role": "user", "content": "I prefer dark mode"},
        ]
        with pytest.warns(UserWarning, match="multi-turn"):
            adapter.add(messages)

    def test_update_replaces_content(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add("Old content")
        adapter.update(fact.id, "New content")
        contents = [f.content for f in kb.list_facts()]
        assert "New content" in contents
        assert "Old content" not in contents

    def test_update_returns_memory_item(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add("Will be replaced")
        item = adapter.update(fact.id, "Replacement text")
        assert item["memory"] == "Replacement text"
        assert "id" in item
        assert "metadata" in item
        assert item["metadata"]["active"] is True

    def test_update_returned_id_differs_from_input(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add("Old content")
        result = adapter.update(fact.id, "New content")
        assert result["id"] != fact.id

    def test_update_old_id_no_longer_accessible_for_unstructured_facts(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add("Old content")
        new_item = adapter.update(fact.id, "New content")
        with pytest.raises(KeyError):
            adapter.get(fact.id)
        assert adapter.get(new_item["id"])["memory"] == "New content"

    def test_update_structured_fact_preserves_lineage(
        self, adapter: OpenClawMemoryAdapter, kb: KnowledgeBase
    ) -> None:
        fact = kb.add_resolved(
            [
                Fact(
                    content="User works at Acme",
                    entity="user",
                    attribute="employer",
                    value_text="Acme",
                )
            ]
        )[0]

        new_item = adapter.update(fact.id, "User now works at Globex")

        assert new_item["id"] != fact.id
        assert new_item["memory"] == "User now works at Globex"
        assert new_item["metadata"]["slot_key"] == "user::employer"
        assert new_item["metadata"]["version"] == 1
        assert adapter.get(fact.id)["metadata"]["active"] is False
        assert adapter.list() == [adapter.get(new_item["id"])]
        history = adapter.get_all(include_inactive=True)
        assert len(history) == 2

    def test_lineage_returns_newest_to_oldest_chain(
        self,
        adapter: OpenClawMemoryAdapter,
        kb: KnowledgeBase,
    ) -> None:
        original = kb.add_resolved(
            [
                Fact(
                    content="User works at Acme",
                    entity="user",
                    attribute="employer",
                    value_text="Acme",
                )
            ]
        )[0]
        current = adapter.update(original.id, "User now works at Globex")

        chain = adapter.lineage(current["id"])

        assert [item["id"] for item in chain] == [current["id"], original.id]
        assert chain[0]["metadata"]["active"] is True
        assert chain[1]["metadata"]["active"] is False
        assert chain[0]["metadata"]["slot_key"] == "user::employer"
        assert chain[0]["metadata"]["version"] == 1

    def test_lineage_returns_empty_for_unknown_id(self, adapter: OpenClawMemoryAdapter) -> None:
        assert adapter.lineage("deadbeef") == []

    def test_lineage_accepts_now_anchor_for_active_flags(
        self,
        adapter: OpenClawMemoryAdapter,
        kb: KnowledgeBase,
    ) -> None:
        current = kb.add_resolved(
            [
                Fact(
                    content="User works at Acme",
                    entity="user",
                    attribute="employer",
                    value_text="Acme",
                )
            ]
        )[0]

        chain = adapter.lineage(current.id, now="2000-01-01T00:00:00+00:00")

        assert [item["id"] for item in chain] == [current.id]
        assert chain[0]["metadata"]["active"] is False

    def test_update_missing_id_raises(
        self, adapter: OpenClawMemoryAdapter
    ) -> None:
        with pytest.raises(KeyError):
            adapter.update("deadbeef", "Replacement text")


class TestGenerateMcpConfig:
    def test_defaults(self) -> None:
        cfg = generate_mcp_config()

        assert cfg["mcpServers"]["ai-knot"]["command"] == "ai-knot-mcp"
        env = cfg["mcpServers"]["ai-knot"]["env"]
        assert env["AI_KNOT_AGENT_ID"] == "default"
        # data_dir is resolved to an absolute path.
        assert env["AI_KNOT_DATA_DIR"].endswith(".ai_knot")
        assert env["AI_KNOT_DATA_DIR"].startswith("/")
        assert env["AI_KNOT_STORAGE"] == "sqlite"

    def test_custom_params(self) -> None:
        cfg = generate_mcp_config(agent_id="bot_a", data_dir="/data/mem", storage="yaml")

        env = cfg["mcpServers"]["ai-knot"]["env"]
        assert env["AI_KNOT_AGENT_ID"] == "bot_a"
        assert env["AI_KNOT_DATA_DIR"] == "/data/mem"
        assert env["AI_KNOT_STORAGE"] == "yaml"

    def test_is_json_serialisable(self) -> None:
        import json

        cfg = generate_mcp_config("agent_x")
        dumped = json.dumps(cfg)
        assert "ai-knot-mcp" in dumped
        assert "agent_x" in dumped

    def test_invalid_storage_raises(self) -> None:
        with pytest.raises(ValueError, match="sqlite.*yaml"):
            generate_mcp_config(storage="postgres")  # type: ignore[arg-type]
