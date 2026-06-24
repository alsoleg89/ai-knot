"""Tests for snapshot / restore / diff functionality."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, SnapshotDiff

# ---------------------------------------------------------------------------
# Parametrized storage fixtures
# ---------------------------------------------------------------------------


def _yaml_kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="snap_test", storage=YAMLStorage(base_dir=str(tmp_path)))


def _sqlite_kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id="snap_test",
        storage=SQLiteStorage(db_path=str(tmp_path / "test.db")),
    )


@pytest.fixture(params=["yaml", "sqlite"])
def kb(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> KnowledgeBase:
    """KnowledgeBase parametrized over both storage backends."""
    if request.param == "yaml":
        return _yaml_kb(tmp_path)
    return _sqlite_kb(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_facts(kb: KnowledgeBase, contents: list[str]) -> list[Fact]:
    return [kb.add(c) for c in contents]


# ---------------------------------------------------------------------------
# SnapshotCapable — basic operations
# ---------------------------------------------------------------------------


class TestSnapshotBasics:
    """Core snapshot save / restore / list behaviour."""

    def test_list_snapshots_empty(self, kb: KnowledgeBase) -> None:
        assert kb.list_snapshots() == []

    def test_snapshot_saves_current_state(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber")
        kb.snapshot("v1")
        assert "v1" in kb.list_snapshots()

    def test_snapshot_and_restore_round_trip(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber")
        kb.snapshot("v1")

        # Mutate current state.
        kb.clear_all()
        kb.add("User switched to Yandex")
        assert len(kb.list_facts()) == 1

        # Restore should bring back the original fact.
        kb.restore("v1")
        facts = kb.list_facts()
        assert len(facts) == 1
        assert facts[0].content == "User works at Sber"

    def test_restore_does_not_affect_snapshot(self, kb: KnowledgeBase) -> None:
        """Restoring should not delete the snapshot itself."""
        kb.add("Fact A")
        kb.snapshot("v1")
        kb.restore("v1")
        assert "v1" in kb.list_snapshots()

    def test_snapshot_overwrites_same_name(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.snapshot("v1")
        kb.clear_all()
        kb.add("Fact B")
        kb.snapshot("v1")  # overwrite

        kb.clear_all()
        kb.restore("v1")
        facts = kb.list_facts()
        assert len(facts) == 1
        assert facts[0].content == "Fact B"

    def test_multiple_snapshots_listed(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.snapshot("v1")
        kb.add("Fact B")
        kb.snapshot("v2")
        names = kb.list_snapshots()
        assert "v1" in names
        assert "v2" in names

    def test_restore_missing_snapshot_raises_key_error(self, kb: KnowledgeBase) -> None:
        with pytest.raises(KeyError):
            kb.restore("nonexistent")

    def test_snapshot_empty_kb(self, kb: KnowledgeBase) -> None:
        """Snapshot of an empty KB should restore to empty."""
        kb.snapshot("empty")
        kb.add("Fact A")
        kb.restore("empty")
        assert kb.list_facts() == []


# ---------------------------------------------------------------------------
# SnapshotDiff
# ---------------------------------------------------------------------------


class TestSnapshotDiff:
    """Tests for kb.diff()."""

    def test_diff_same_snapshot_no_changes(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.snapshot("v1")
        diff = kb.diff("v1", "v1")
        assert isinstance(diff, SnapshotDiff)
        assert diff.added == []
        assert diff.removed == []

    def test_diff_added_facts(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.snapshot("v1")
        kb.add("Fact B")
        kb.snapshot("v2")
        diff = kb.diff("v1", "v2")
        assert len(diff.added) == 1
        assert diff.added[0].content == "Fact B"
        assert diff.removed == []

    def test_diff_removed_facts(self, kb: KnowledgeBase) -> None:
        fact_a = kb.add("Fact A")
        kb.add("Fact B")
        kb.snapshot("v1")
        kb.forget(fact_a.id)
        kb.snapshot("v2")
        diff = kb.diff("v1", "v2")
        assert len(diff.removed) == 1
        assert diff.removed[0].content == "Fact A"
        assert diff.added == []

    def test_diff_with_current(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        kb.snapshot("v1")
        kb.add("Fact B")
        diff = kb.diff("v1", "current")
        assert len(diff.added) == 1
        assert diff.added[0].content == "Fact B"

    def test_diff_current_current_no_changes(self, kb: KnowledgeBase) -> None:
        kb.add("Fact A")
        diff = kb.diff("current", "current")
        assert diff.added == []
        assert diff.removed == []

    def test_diff_snapshot_a_b_metadata(self, kb: KnowledgeBase) -> None:
        kb.snapshot("v1")
        kb.snapshot("v2")
        diff = kb.diff("v1", "v2")
        assert diff.snapshot_a == "v1"
        assert diff.snapshot_b == "v2"

    def test_diff_missing_snapshot_raises(self, kb: KnowledgeBase) -> None:
        kb.snapshot("v1")
        with pytest.raises(KeyError):
            kb.diff("v1", "missing")


# ---------------------------------------------------------------------------
# Unsupported backend guard
# ---------------------------------------------------------------------------


class TestUnsupportedBackend:
    """KnowledgeBase raises NotImplementedError on backends without snapshots."""

    def test_snapshot_raises_on_plain_backend(self, tmp_path: pathlib.Path) -> None:
        class _MinimalStorage:
            def save(self, agent_id: str, facts: list[Fact]) -> None:
                pass

            def load(self, agent_id: str) -> list[Fact]:
                return []

            def delete(self, agent_id: str, fact_id: str) -> None:
                pass

            def list_agents(self) -> list[str]:
                return []

        kb = KnowledgeBase(agent_id="test", storage=_MinimalStorage())  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError):
            kb.snapshot("v1")

    def test_restore_raises_on_plain_backend(self, tmp_path: pathlib.Path) -> None:
        class _MinimalStorage:
            def save(self, agent_id: str, facts: list[Fact]) -> None:
                pass

            def load(self, agent_id: str) -> list[Fact]:
                return []

            def delete(self, agent_id: str, fact_id: str) -> None:
                pass

            def list_agents(self) -> list[str]:
                return []

        kb = KnowledgeBase(agent_id="test", storage=_MinimalStorage())  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError):
            kb.restore("v1")
