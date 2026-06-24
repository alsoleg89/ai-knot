"""Unit tests for ai_knot._mcp_tools — pure tool-handler functions."""

from __future__ import annotations

import json
import pathlib

import pytest

from ai_knot._mcp_tools import (
    _MAX_TOP_K,
    _clamp_top_k,
    tool_add,
    tool_add_resolved,
    tool_capabilities,
    tool_forget,
    tool_health,
    tool_learn,
    tool_list_facts,
    tool_list_snapshots,
    tool_memory_lineage,
    tool_recall,
    tool_recall_json,
    tool_recall_with_trace,
    tool_restore,
    tool_snapshot,
    tool_stats,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


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

    def test_event_time_anchor_is_persisted(self, kb: KnowledgeBase) -> None:
        from datetime import UTC, datetime

        tool_add(kb, "User joined Globex", event_time="2023-05-08T00:00:00+00:00")
        facts = kb.list_facts()
        assert facts[0].event_time == datetime(2023, 5, 8, tzinfo=UTC)
        # The anchor must NOT leak into the indexed content (no date text-prefix).
        assert "2023" not in facts[0].content

    def test_event_time_omitted_when_absent(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "User likes hiking")
        assert kb.list_facts()[0].event_time is None

    def test_event_time_bad_input_ignored(self, kb: KnowledgeBase) -> None:
        # An unparseable anchor must not block the add; it is silently dropped.
        msg = tool_add(kb, "User likes biking", event_time="not-a-date")
        assert msg.startswith("Added fact [")
        assert kb.list_facts()[0].event_time is None


# ---- tool_recall ------------------------------------------------------------


class TestToolRecall:
    def test_returns_facts(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Caroline runs marathons every weekend")
        result = tool_recall(kb, "Caroline running")
        assert "Caroline" in result

    def test_empty_kb_returns_no_facts_message(self, kb: KnowledgeBase) -> None:
        result = tool_recall(kb, "anything")
        assert result == "No relevant facts found."

    def test_now_anchor_excludes_facts_not_yet_valid(self, kb: KnowledgeBase) -> None:
        # A fact is valid from its creation time onward; recalling "as of" a point
        # before that anchor must exclude it (the time-aware / knowledge-update seam).
        tool_add(kb, "Caroline took up running")
        assert (
            tool_recall(kb, "Caroline running", now="2000-01-01T00:00:00+00:00")
            == "No relevant facts found."
        )
        assert "Caroline" in tool_recall(kb, "Caroline running")

    def test_bad_now_degrades_to_no_anchor(self, kb: KnowledgeBase) -> None:
        # A malformed timestamp must degrade to "no anchor" (current time), never crash.
        tool_add(kb, "Caroline took up running")
        assert "Caroline" in tool_recall(kb, "Caroline running", now="not-a-date")

    def test_naive_now_does_not_crash(self, kb: KnowledgeBase) -> None:
        # A timezone-naive ``now`` (date-only, or a datetime with no offset — exactly
        # what LongMemEval's question_date supplies) must be treated as UTC, not raise
        # "can't compare offset-naive and offset-aware datetimes" inside is_active.
        tool_add(kb, "Caroline took up running")
        # Future naive anchor → fact is active → recalled (no crash).
        assert "Caroline" in tool_recall(kb, "Caroline running", now="2099-01-01T23:40:00")
        # Past naive date-only anchor → fact not yet valid → excluded (no crash).
        assert tool_recall(kb, "Caroline running", now="2000-01-01") == "No relevant facts found."

    def test_now_anchor_applies_to_json_and_trace_variants(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Caroline took up running")
        past = "2000-01-01T00:00:00+00:00"
        assert tool_recall_json(kb, "Caroline running", now=past) == "[]"
        trace = json.loads(tool_recall_with_trace(kb, "Caroline running", now=past))
        assert trace["pack_fact_ids"] == []

    def test_top_k_is_clamped(self, kb: KnowledgeBase) -> None:
        assert _clamp_top_k(99999) == _MAX_TOP_K
        assert _clamp_top_k(0) == 1
        assert _clamp_top_k(-5) == 1
        assert _clamp_top_k(5) == 5
        # An out-of-range top_k must not raise through the tool.
        tool_add(kb, "Caroline took up running")
        assert "Caroline" in tool_recall(kb, "Caroline running", top_k=99999)


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
            "add_resolved",
            "learn",
            "recall",
            "recall_json",
            "forget",
            "list_facts",
            "memory_lineage",
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


# ---- tool_memory_lineage ----------------------------------------------------


class TestToolMemoryLineage:
    @staticmethod
    def _slot_fact(value: str) -> Fact:
        return Fact(
            content=f"Alex earns {value}",
            entity="Alex",
            attribute="salary",
            value_text=value,
            slot_key="Alex::salary",
        )

    def test_traces_supersession_chain(self, kb: KnowledgeBase) -> None:
        old = kb.add_resolved([self._slot_fact("80k")])[0]
        mid = kb.add_resolved([self._slot_fact("95k")])[0]
        new = kb.add_resolved([self._slot_fact("120k")])[0]
        data = json.loads(tool_memory_lineage(kb, new.id))
        assert [row["id"] for row in data] == [new.id, mid.id, old.id]
        assert data[0]["value_text"] == "120k"
        assert data[0]["supersedes_id"] == mid.id
        assert data[0]["active"] is True
        assert data[-1]["active"] is False  # the oldest was superseded

    def test_unknown_fact_returns_empty(self, kb: KnowledgeBase) -> None:
        assert tool_memory_lineage(kb, "ffffffff") == "[]"

    def test_single_fact_has_no_predecessor(self, kb: KnowledgeBase) -> None:
        only = kb.add_resolved([self._slot_fact("80k")])[0]
        data = json.loads(tool_memory_lineage(kb, only.id))
        assert [row["id"] for row in data] == [only.id]
        assert data[0]["supersedes_id"] == ""


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


class TestToolAddResolved:
    def test_inserts_and_returns_json(self, kb: KnowledgeBase) -> None:
        out = tool_add_resolved(
            kb,
            [{"content": "User works at Globex", "entity": "user", "attribute": "employer"}],
        )
        rows = json.loads(out)
        assert len(rows) == 1
        assert rows[0]["content"] == "User works at Globex"
        assert rows[0]["slot_key"] == "user::employer"

    def test_same_slot_new_value_supersedes(self, kb: KnowledgeBase) -> None:
        tool_add_resolved(
            kb,
            [
                {
                    "content": "User works at Acme",
                    "entity": "user",
                    "attribute": "employer",
                    "value_text": "Acme",
                }
            ],
        )
        tool_add_resolved(
            kb,
            [
                {
                    "content": "User works at Globex",
                    "entity": "user",
                    "attribute": "employer",
                    "value_text": "Globex",
                }
            ],
        )
        active = [f for f in kb.list_facts() if f.is_active() and f.attribute == "employer"]
        assert len(active) == 1
        assert active[0].value_text == "Globex"

    def test_empty_content_rejected(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="non-empty 'content'"):
            tool_add_resolved(kb, [{"entity": "user", "attribute": "employer"}])

    def test_event_time_preserved(self, kb: KnowledgeBase) -> None:
        out = tool_add_resolved(
            kb,
            [
                {
                    "content": "User joined the company",
                    "entity": "user",
                    "attribute": "join",
                    "value_text": "joined",
                    "event_time": "2023-05-08T00:00:00+00:00",
                }
            ],
        )
        rows = json.loads(out)
        fact = next(f for f in kb.list_facts() if f.id == rows[0]["id"])
        assert fact.event_time is not None

    def test_capabilities_lists_add_resolved(self) -> None:
        caps = json.loads(tool_capabilities())
        assert any(c["name"] == "add_resolved" for c in caps)
