"""Unit tests for ai_knot._mcp_tools — pure tool-handler functions."""

from __future__ import annotations

import json
import pathlib

import pytest

from ai_knot._mcp_tools import (
    tool_add,
    tool_capabilities,
    tool_forget,
    tool_health,
    tool_learn,
    tool_list_facts,
    tool_list_snapshots,
    tool_recall,
    tool_recall_json,
    tool_recall_with_trace,
    tool_restore,
    tool_snapshot,
    tool_stats,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="mcp-test", storage=YAMLStorage(base_dir=str(tmp_path)))


# ---- tool_add ---------------------------------------------------------------


class TestToolAdd:
    def test_returns_confirmation_with_id(self, kb: KnowledgeBase) -> None:
        msg = tool_add(kb, "User prefers Python")
        assert msg.startswith("Added fact [")
        assert "User prefers Python" in msg

    def test_invalid_importance_rejected(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="importance must be between 0.0 and 1.0"):
            tool_add(kb, "x", importance=1.5)

    def test_unknown_memory_type_rejected(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="Unknown memory type"):
            tool_add(kb, "x", type="invalid")

    def test_tags_are_persisted(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "User uses Docker", tags=["devops", "tools"])
        facts = kb.list_facts()
        assert any("devops" in f.tags for f in facts)


# ---- tool_recall ------------------------------------------------------------


class TestToolRecall:
    def test_returns_facts(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Caroline runs marathons every weekend")
        result = tool_recall(kb, "Caroline running")
        assert "Caroline" in result

    def test_empty_kb_returns_no_facts_message(self, kb: KnowledgeBase) -> None:
        result = tool_recall(kb, "anything")
        assert result == "No relevant facts found."


# ---- tool_forget ------------------------------------------------------------


class TestToolForget:
    def test_removes_existing_fact(self, kb: KnowledgeBase) -> None:
        msg_added = tool_add(kb, "Temporary fact")
        # extract id from "Added fact [abcd1234]: ..."
        fact_id = msg_added.split("[")[1].split("]")[0]

        msg = tool_forget(kb, fact_id)
        assert fact_id in msg
        assert "removed" in msg.lower()
        assert kb.list_facts() == []


# ---- tool_health ------------------------------------------------------------


class TestToolHealth:
    def test_returns_ok_with_version(self) -> None:
        data = json.loads(tool_health())
        assert data["status"] == "ok"
        assert isinstance(data["version"], str) and data["version"]


# ---- tool_capabilities ------------------------------------------------------


class TestToolCapabilities:
    def test_lists_all_documented_tools(self) -> None:
        data = json.loads(tool_capabilities())
        names = {item["name"] for item in data}
        expected = {
            "add",
            "learn",
            "recall",
            "recall_json",
            "forget",
            "list_facts",
            "stats",
            "snapshot",
            "restore",
            "list_snapshots",
            "health",
            "capabilities",
        }
        assert names == expected


# ---- tool_list_facts --------------------------------------------------------


class TestToolListFacts:
    def test_empty_kb_returns_empty_array(self, kb: KnowledgeBase) -> None:
        assert tool_list_facts(kb) == "[]"

    def test_returns_facts_with_metadata(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "First fact")
        tool_add(kb, "Second very different fact")
        data = json.loads(tool_list_facts(kb))
        assert len(data) == 2
        assert all(set(d) >= {"id", "content", "type", "importance", "retention"} for d in data)


# ---- tool_stats -------------------------------------------------------------


class TestToolStats:
    def test_returns_valid_json(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Some fact")
        data = json.loads(tool_stats(kb))
        assert isinstance(data, dict)
        assert data.get("total_facts", 0) >= 1


# ---- tool_recall_json -------------------------------------------------------


class TestToolRecallJson:
    def test_empty_kb_returns_empty_array(self, kb: KnowledgeBase) -> None:
        assert tool_recall_json(kb, "anything") == "[]"

    def test_returns_structured_objects(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Tokyo has good ramen")
        data = json.loads(tool_recall_json(kb, "Tokyo ramen"))
        assert isinstance(data, list)
        if data:
            assert {"id", "memory", "type", "importance", "retention"} <= set(data[0])


# ---- tool_learn -------------------------------------------------------------


class TestToolLearnDegraded:
    """Without provider+key, learn falls back to storing the last user message."""

    def test_stores_last_user_message_without_llm(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AI_KNOT_PROVIDER", raising=False)
        monkeypatch.delenv("AI_KNOT_API_KEY", raising=False)

        result = tool_learn(
            kb,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "I prefer dark mode"},
            ],
        )
        data = json.loads(result)
        assert data["stored"] == 1
        assert len(data["ids"]) == 1

        facts = kb.list_facts()
        assert any("dark mode" in f.content for f in facts)

    def test_no_user_message_returns_empty(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AI_KNOT_PROVIDER", raising=False)
        monkeypatch.delenv("AI_KNOT_API_KEY", raising=False)

        result = tool_learn(kb, messages=[{"role": "assistant", "content": "Hi"}])
        data = json.loads(result)
        assert data["stored"] == 0
        assert data["ids"] == []


# ---- snapshot / restore / list_snapshots ------------------------------------


class TestSnapshotLifecycle:
    def test_snapshot_save_list_and_restore_roundtrip(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Fact before snapshot")

        msg_save = tool_snapshot(kb, "v1")
        assert "v1" in msg_save and "saved" in msg_save.lower()

        names_json = tool_list_snapshots(kb)
        assert "v1" in names_json

        # Mutate state then restore.
        tool_add(kb, "A second very different fact added after snapshot")
        assert len(kb.list_facts()) == 2

        msg_restore = tool_restore(kb, "v1")
        assert "v1" in msg_restore and "restored" in msg_restore.lower()
        assert len(kb.list_facts()) == 1

    def test_restore_unknown_snapshot_returns_not_found(self, kb: KnowledgeBase) -> None:
        msg = tool_restore(kb, "does-not-exist")
        assert "not found" in msg.lower()

    def test_list_snapshots_empty_returns_empty_array(self, kb: KnowledgeBase) -> None:
        assert tool_list_snapshots(kb) == "[]"


# ---- tool_recall_with_trace -------------------------------------------------


class TestToolRecallWithTrace:
    def test_empty_kb_returns_empty_context_with_trace_keys(self, kb: KnowledgeBase) -> None:
        result = json.loads(tool_recall_with_trace(kb, "anything"))
        assert "context" in result
        assert "pack_fact_ids" in result
        assert "trace" in result
        assert isinstance(result["pack_fact_ids"], list)
        assert isinstance(result["trace"], dict)

    def test_returns_context_and_pack_ids_after_add(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "The Eiffel Tower is in Paris")
        result = json.loads(tool_recall_with_trace(kb, "Eiffel Tower"))
        assert isinstance(result["context"], str)
        assert isinstance(result["pack_fact_ids"], list)
        # trace should contain stage1_candidates key
        trace = result["trace"]
        assert "stage1_candidates" in trace

    def test_trace_has_all_required_stage_keys(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Rome has the Colosseum")
        tool_add(kb, "Paris has the Eiffel Tower")
        result = json.loads(tool_recall_with_trace(kb, "famous landmarks"))
        trace = result["trace"]
        # stage1_candidates is always present (may be empty if only dense path fires)
        assert "stage1_candidates" in trace
        stage1 = trace["stage1_candidates"]
        assert {"from_bm25", "from_rare_tokens", "from_entity_hop"} <= set(stage1.keys())
        # pack_fact_ids are valid hex strings
        for fid in result["pack_fact_ids"]:
            assert len(fid) == 8
