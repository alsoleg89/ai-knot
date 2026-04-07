"""Tests for the MCP server tool implementations.

All tests operate on tool_* functions directly — no mcp package required.
"""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.mcp_server import (
    _build_kb,
    tool_add,
    tool_forget,
    tool_learn,
    tool_list_facts,
    tool_list_snapshots,
    tool_recall,
    tool_recall_json,
    tool_restore,
    tool_snapshot,
    tool_stats,
)
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    """KnowledgeBase backed by a temp YAML directory."""
    return KnowledgeBase(agent_id="mcp_test", storage=YAMLStorage(base_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# _build_kb — environment variable configuration
# ---------------------------------------------------------------------------


class TestBuildKb:
    """Verify that _build_kb() reads env vars correctly."""

    def test_defaults(self, tmp_path: pathlib.Path) -> None:
        with patch.dict(os.environ, {"AI_KNOT_DATA_DIR": str(tmp_path)}, clear=False):
            kb = _build_kb()
        assert kb is not None

    def test_custom_agent_id(self, tmp_path: pathlib.Path) -> None:
        env = {"AI_KNOT_AGENT_ID": "my_agent", "AI_KNOT_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=False):
            kb = _build_kb()
        assert kb._agent_id == "my_agent"

    def test_sqlite_backend(self, tmp_path: pathlib.Path) -> None:
        db = str(tmp_path / "test.db")
        env = {
            "AI_KNOT_STORAGE": "sqlite",
            "AI_KNOT_DB_PATH": db,
            "AI_KNOT_DATA_DIR": str(tmp_path),
        }
        with patch.dict(os.environ, env, clear=False):
            kb = _build_kb()
        assert kb is not None


# ---------------------------------------------------------------------------
# tool_add
# ---------------------------------------------------------------------------


class TestToolAdd:
    """Tests for tool_add()."""

    def test_adds_fact_returns_confirmation(self, kb: KnowledgeBase) -> None:
        result = tool_add(kb, "User works at Sber")
        assert "Added fact" in result
        assert len(kb.list_facts()) == 1

    def test_default_type_is_semantic(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Some fact")
        assert kb.list_facts()[0].type == MemoryType.SEMANTIC

    def test_custom_type(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Always use pytest", type="procedural")
        assert kb.list_facts()[0].type == MemoryType.PROCEDURAL

    def test_custom_importance(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "Critical fact", importance=0.95)
        assert kb.list_facts()[0].importance == pytest.approx(0.95)

    def test_invalid_importance_raises(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="importance"):
            tool_add(kb, "Bad fact", importance=1.5)

    def test_invalid_type_raises(self, kb: KnowledgeBase) -> None:
        with pytest.raises(ValueError, match="Unknown memory type"):
            tool_add(kb, "Bad fact", type="unknown")

    def test_result_contains_fact_id(self, kb: KnowledgeBase) -> None:
        result = tool_add(kb, "User prefers Python")
        fact_id = kb.list_facts()[0].id
        assert fact_id in result

    def test_tags_stored(self, kb: KnowledgeBase) -> None:
        tool_add(kb, "User works at Sber", tags=["profile"])
        assert "profile" in kb.list_facts()[0].tags


# ---------------------------------------------------------------------------
# tool_recall
# ---------------------------------------------------------------------------


class TestToolRecall:
    """Tests for tool_recall()."""

    def test_returns_relevant_fact(self, kb: KnowledgeBase) -> None:
        kb.add("User deploys in Docker", importance=0.9)
        result = tool_recall(kb, "deployment")
        assert "Docker" in result

    def test_empty_kb_returns_no_facts_message(self, kb: KnowledgeBase) -> None:
        result = tool_recall(kb, "anything")
        assert "No relevant facts" in result

    def test_top_k_limits_results(self, kb: KnowledgeBase) -> None:
        for i in range(10):
            kb.add(f"Fact number {i} about deployment Docker")
        result = tool_recall(kb, "deployment Docker", top_k=2)
        assert result.count("[") <= 2


# ---------------------------------------------------------------------------
# tool_forget
# ---------------------------------------------------------------------------


class TestToolForget:
    """Tests for tool_forget()."""

    def test_removes_fact(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Fact to remove")
        tool_forget(kb, fact.id)
        assert kb.list_facts() == []

    def test_returns_confirmation(self, kb: KnowledgeBase) -> None:
        fact = kb.add("Fact to remove")
        result = tool_forget(kb, fact.id)
        assert fact.id in result

    def test_nonexistent_id_no_error(self, kb: KnowledgeBase) -> None:
        result = tool_forget(kb, "00000000")
        assert "00000000" in result


# ---------------------------------------------------------------------------
# tool_list_facts
# ---------------------------------------------------------------------------


class TestToolListFacts:
    """Tests for tool_list_facts()."""

    def test_empty_kb(self, kb: KnowledgeBase) -> None:
        result = tool_list_facts(kb)
        assert "No facts" in result

    def test_returns_json(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber")
        result = tool_list_facts(kb)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "User works at Sber"

    def test_json_fields_present(self, kb: KnowledgeBase) -> None:
        kb.add("Test fact", importance=0.7)
        data = json.loads(tool_list_facts(kb))
        assert set(data[0].keys()) >= {"id", "content", "type", "importance", "retention"}


# ---------------------------------------------------------------------------
# tool_stats
# ---------------------------------------------------------------------------


class TestToolStats:
    """Tests for tool_stats()."""

    def test_empty_kb(self, kb: KnowledgeBase) -> None:
        result = tool_stats(kb)
        data = json.loads(result)
        assert data["total_facts"] == 0

    def test_counts_by_type(self, kb: KnowledgeBase) -> None:
        kb.add("Semantic", type=MemoryType.SEMANTIC)
        kb.add("Procedural", type=MemoryType.PROCEDURAL)
        result = tool_stats(kb)
        data = json.loads(result)
        assert data["total_facts"] == 2
        assert data["by_type"]["semantic"] == 1
        assert data["by_type"]["procedural"] == 1


# ---------------------------------------------------------------------------
# tool_snapshot / tool_restore
# ---------------------------------------------------------------------------


class TestToolSnapshotRestore:
    """Tests for snapshot/restore tools (YAML backend supports snapshots)."""

    def test_snapshot_confirmation(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        with patch.object(kb, "snapshot", create=True):
            result = tool_snapshot(kb, "v1")
        assert "v1" in result
        assert "Snapshot" in result

    def test_restore_confirmation(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        with patch.object(kb, "restore", create=True):
            result = tool_restore(kb, "v1")
        assert "v1" in result

    def test_restore_missing_snapshot_returns_message(self, kb: KnowledgeBase) -> None:
        with patch.object(kb, "restore", create=True, side_effect=KeyError("nonexistent")):
            result = tool_restore(kb, "nonexistent")
        assert "not found" in result

    def test_snapshot_unsupported_backend_returns_message(self, kb: KnowledgeBase) -> None:
        with patch.object(
            kb, "snapshot", create=True, side_effect=NotImplementedError("no snapshots")
        ):
            result = tool_snapshot(kb, "v1")
        assert "not supported" in result

    def test_restore_unsupported_backend_returns_message(self, kb: KnowledgeBase) -> None:
        with patch.object(
            kb, "restore", create=True, side_effect=NotImplementedError("no snapshots")
        ):
            result = tool_restore(kb, "v1")
        assert "not supported" in result


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# tool_recall_json
# ---------------------------------------------------------------------------


class TestToolRecallJson:
    def test_recall_json_returns_valid_json(self, kb: KnowledgeBase) -> None:
        kb.add("User deploys on Fridays")
        result = tool_recall_json(kb, "deployment day")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_recall_json_has_expected_keys(self, kb: KnowledgeBase) -> None:
        kb.add("User prefers Python")
        result = tool_recall_json(kb, "language preference")
        items = json.loads(result)
        assert len(items) >= 1
        item = items[0]
        assert "id" in item
        assert "memory" in item
        assert "type" in item
        assert "importance" in item
        assert "retention" in item

    def test_recall_json_empty_kb(self, kb: KnowledgeBase) -> None:
        result = tool_recall_json(kb, "anything")
        assert json.loads(result) == []

    def test_recall_json_top_k(self, kb: KnowledgeBase) -> None:
        for i in range(10):
            kb.add(f"Deployment fact {i}")
        result = tool_recall_json(kb, "deployment", top_k=2)
        items = json.loads(result)
        assert len(items) <= 2


# ---------------------------------------------------------------------------
# tool_list_snapshots
# ---------------------------------------------------------------------------


class TestToolListSnapshots:
    def test_list_snapshots_empty(self, kb: KnowledgeBase) -> None:
        result = tool_list_snapshots(kb)
        assert json.loads(result) == []

    def test_list_snapshots_returns_names(self, tmp_path: pathlib.Path) -> None:
        from ai_knot.storage.yaml_storage import YAMLStorage

        snap_kb = KnowledgeBase(agent_id="snap_test", storage=YAMLStorage(base_dir=str(tmp_path)))
        snap_kb.add("Some fact")
        snap_kb.snapshot("release_1")
        snap_kb.snapshot("release_2")
        result = tool_list_snapshots(snap_kb)
        names = json.loads(result)
        assert "release_1" in names
        assert "release_2" in names

    def test_list_snapshots_unsupported_backend(self, kb: KnowledgeBase) -> None:
        with patch.object(kb, "list_snapshots", side_effect=NotImplementedError("no snapshots")):
            result = tool_list_snapshots(kb)
        assert "not supported" in result


# ---------------------------------------------------------------------------
# tool_recall / tool_recall_json empty-state contract
# ---------------------------------------------------------------------------


class TestRecallEmptyContract:
    """recall returns a human string; recall_json always returns valid JSON."""

    def test_recall_empty_returns_message(self, kb: KnowledgeBase) -> None:
        result = tool_recall(kb, "anything")
        assert "No relevant facts" in result

    def test_recall_json_empty_returns_valid_json_array(self, kb: KnowledgeBase) -> None:
        result = tool_recall_json(kb, "anything")
        assert json.loads(result) == []

    def test_recall_json_never_raises_on_empty(self, kb: KnowledgeBase) -> None:
        result = tool_recall_json(kb, "anything")
        parsed = json.loads(result)  # must not raise JSONDecodeError
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# _make_server — MCP wiring (mocked)
# ---------------------------------------------------------------------------


class TestMakeServer:
    """Verify _make_server() raises ImportError when mcp is not installed."""

    def test_import_error_without_mcp(self, kb: KnowledgeBase) -> None:
        from ai_knot.mcp_server import _make_server

        with (
            patch.dict(
                "sys.modules", {"mcp": None, "mcp.server": None, "mcp.server.fastmcp": None}
            ),
            pytest.raises((ImportError, TypeError)),
        ):
            _make_server(kb)

    def test_make_server_with_mock_mcp(self, kb: KnowledgeBase) -> None:
        from ai_knot.mcp_server import _make_server

        mock_app = MagicMock()
        mock_fastmcp_cls = MagicMock(return_value=mock_app)
        mock_fastmcp_module = MagicMock()
        mock_fastmcp_module.FastMCP = mock_fastmcp_cls

        with patch.dict(
            "sys.modules",
            {
                "mcp": MagicMock(),
                "mcp.server": MagicMock(),
                "mcp.server.fastmcp": mock_fastmcp_module,
            },
        ):
            app = _make_server(kb)

        assert app is mock_app
        call_args = mock_fastmcp_cls.call_args
        assert call_args[0][0] == "ai-knot"
        assert "instructions" in call_args[1]


# ---------------------------------------------------------------------------
# tool_learn — multi-turn ingestion with degraded-mode fallback
# ---------------------------------------------------------------------------


class TestToolLearn:
    def test_learn_degraded_stores_last_user_message(self, kb: KnowledgeBase) -> None:
        """Without LLM credentials, stores the last user message verbatim."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "I use PostgreSQL 16 as my main database."},
        ]
        result = tool_learn(kb, messages)
        data = json.loads(result)
        assert data["stored"] == 1
        assert len(data["ids"]) == 1
        facts = kb.list_facts()
        assert any("PostgreSQL" in f.content for f in facts)

    def test_learn_degraded_returns_json(self, kb: KnowledgeBase) -> None:
        """tool_learn always returns valid JSON."""
        result = tool_learn(kb, [{"role": "user", "content": "test fact"}])
        data = json.loads(result)
        assert "stored" in data
        assert "ids" in data

    def test_learn_degraded_no_user_message(self, kb: KnowledgeBase) -> None:
        """Returns zero stored when there is no user message in conversation."""
        result = tool_learn(kb, [{"role": "assistant", "content": "Hello"}])
        data = json.loads(result)
        assert data["stored"] == 0
        assert data["ids"] == []

    def test_learn_degraded_empty_messages(self, kb: KnowledgeBase) -> None:
        """Returns zero stored for empty conversation."""
        result = tool_learn(kb, [])
        data = json.loads(result)
        assert data["stored"] == 0

    def test_learn_provider_env_vars(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tool_learn reads provider credentials from environment variables."""
        monkeypatch.setenv("AI_KNOT_PROVIDER", "anthropic")
        monkeypatch.setenv("AI_KNOT_API_KEY", "test-key")

        # With a bad API key the LLM call will fail; we expect an error JSON,
        # not an exception propagating out of tool_learn.
        result = tool_learn(kb, [{"role": "user", "content": "test"}])
        data = json.loads(result)
        # Either stored successfully (unlikely with fake key) or got an error field.
        assert "stored" in data or "error" in data
