"""Tests for the MCP server tool implementations.

All tests operate on tool_* functions directly — no mcp package required.
"""

from __future__ import annotations

import json
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from agentmemo.knowledge import KnowledgeBase
from agentmemo.mcp_server import (
    _build_kb,
    tool_add,
    tool_forget,
    tool_list_facts,
    tool_recall,
    tool_restore,
    tool_snapshot,
    tool_stats,
)
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import MemoryType

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
        with patch.dict(os.environ, {"AGENTMEMO_DATA_DIR": str(tmp_path)}, clear=False):
            kb = _build_kb()
        assert kb is not None

    def test_custom_agent_id(self, tmp_path: pathlib.Path) -> None:
        env = {"AGENTMEMO_AGENT_ID": "my_agent", "AGENTMEMO_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=False):
            kb = _build_kb()
        assert kb._agent_id == "my_agent"

    def test_sqlite_backend(self, tmp_path: pathlib.Path) -> None:
        db = str(tmp_path / "test.db")
        env = {
            "AGENTMEMO_STORAGE": "sqlite",
            "AGENTMEMO_DB_PATH": db,
            "AGENTMEMO_DATA_DIR": str(tmp_path),
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
# _make_server — MCP wiring (mocked)
# ---------------------------------------------------------------------------


class TestMakeServer:
    """Verify _make_server() raises ImportError when mcp is not installed."""

    def test_import_error_without_mcp(self, kb: KnowledgeBase) -> None:
        from agentmemo.mcp_server import _make_server

        with (
            patch.dict(
                "sys.modules", {"mcp": None, "mcp.server": None, "mcp.server.fastmcp": None}
            ),
            pytest.raises((ImportError, TypeError)),
        ):
            _make_server(kb)

    def test_make_server_with_mock_mcp(self, kb: KnowledgeBase) -> None:
        from agentmemo.mcp_server import _make_server

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
        assert call_args[0][0] == "agentmemo"
        assert "instructions" in call_args[1]
