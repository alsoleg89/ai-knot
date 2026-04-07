"""Tests for conflict resolution in learn() via resolve_against_existing()."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from ai_knot.extractor import resolve_against_existing
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    """KnowledgeBase backed by a temp YAML directory."""
    return KnowledgeBase(agent_id="test", storage=YAMLStorage(base_dir=str(tmp_path)))


def _fact(content: str, importance: float = 0.8) -> Fact:
    return Fact(content=content, type=MemoryType.SEMANTIC, importance=importance)


# ---------------------------------------------------------------------------
# Unit tests for resolve_against_existing()
# ---------------------------------------------------------------------------


class TestResolveAgainstExisting:
    """Direct tests for the conflict-resolution helper (temporal close semantics).

    When a new fact matches an existing one:
    - The old fact is **temporally closed** (``valid_until`` set to now).
    - The new fact (with importance carried forward +0.05) is added to ``to_insert``.
    Both old (closed) and new (versioned) facts are kept in storage — history is preserved.
    """

    def test_duplicate_closes_old_and_inserts_new(self) -> None:
        """Duplicate fact should close the old version and insert a new one."""
        old = _fact("User works at Sber as Operations Director")
        new = [_fact("User works at Sber as Operations Director")]
        to_insert, closed = resolve_against_existing(new, [old])
        # New version inserted; old fact temporally closed.
        assert len(to_insert) == 1
        assert len(closed) == 1
        assert closed[0] is old
        assert old.valid_until is not None

    def test_similar_fact_closes_old_and_inserts_new(self) -> None:
        """Similar fact should trigger temporal close of the old version."""
        old = _fact("User works at Sber as Operations Director")
        new = [_fact("User works at Sber as a Director")]
        to_insert, closed = resolve_against_existing(new, [old], threshold=0.5)
        assert len(to_insert) == 1
        assert len(closed) == 1
        assert old.valid_until is not None

    def test_truly_new_fact_is_inserted(self) -> None:
        existing = [_fact("User works at Sber")]
        new = [_fact("User prefers Python over Java")]
        to_insert, closed = resolve_against_existing(new, existing)
        assert len(to_insert) == 1
        assert closed == []

    def test_similar_fact_bumps_importance_on_new_version(self) -> None:
        """Importance is carried forward (+0.05) to the new version, old is unchanged."""
        old = _fact("User deploys in Docker", importance=0.7)
        new = [_fact("User deploys everything in Docker", importance=0.8)]
        to_insert, _ = resolve_against_existing(new, [old], threshold=0.5)
        # Old fact retains its original importance (not mutated).
        assert old.importance == pytest.approx(0.7)
        # New version gets bumped importance from old.
        assert to_insert[0].importance == pytest.approx(0.75)

    def test_importance_capped_at_1_0_on_new_version(self) -> None:
        old = _fact("User deploys in Docker", importance=0.98)
        new = [_fact("User deploys everything in Docker")]
        to_insert, _ = resolve_against_existing(new, [old], threshold=0.5)
        assert old.importance == pytest.approx(0.98)
        assert to_insert[0].importance == pytest.approx(1.0)

    def test_similar_fact_closes_old_valid_until_set(self) -> None:
        """The old fact must have valid_until set; last_accessed is not mutated."""
        before = datetime(2025, 1, 1, tzinfo=UTC)
        old = _fact("User works at Sber")
        old.last_accessed = before
        new = [_fact("User works at Sber")]
        resolve_against_existing(new, [old])
        # Old fact: valid_until set (closed), last_accessed NOT mutated.
        assert old.valid_until is not None
        assert old.last_accessed == before

    def test_threshold_parameter_high_no_merge(self) -> None:
        existing = [_fact("User works at Sber in Moscow")]
        new = [_fact("User prefers Python for backend development")]
        # With very high threshold these clearly different facts won't merge.
        to_insert, closed = resolve_against_existing(new, existing, threshold=0.99)
        assert len(to_insert) == 1
        assert closed == []

    def test_threshold_parameter_low_merges(self) -> None:
        old = _fact("User works at Sber")
        new = [_fact("User works at Sber as a Director")]
        to_insert, closed = resolve_against_existing(new, [old], threshold=0.3)
        assert len(to_insert) == 1  # new version inserted
        assert len(closed) == 1    # old version closed
        assert old.valid_until is not None

    def test_empty_existing_all_inserted(self) -> None:
        new = [_fact("Fact A"), _fact("Fact B")]
        to_insert, closed = resolve_against_existing(new, [])
        assert len(to_insert) == 2
        assert closed == []

    def test_empty_new_returns_empty(self) -> None:
        existing = [_fact("Fact A")]
        to_insert, closed = resolve_against_existing([], existing)
        assert to_insert == []
        assert closed == []

    def test_mixed_new_and_duplicate(self) -> None:
        old = _fact("User works at Sber")
        new = [
            _fact("User works at Sber"),  # duplicate → temporal close + new version
            _fact("User prefers Python"),  # genuinely new
        ]
        to_insert, closed = resolve_against_existing(new, [old])
        # Both the new version of the duplicate AND the genuinely new fact are inserted.
        assert len(to_insert) == 2
        contents = {f.content for f in to_insert}
        assert "User works at Sber" in contents
        assert "User prefers Python" in contents
        # Old fact is closed.
        assert len(closed) == 1
        assert old.valid_until is not None


# ---------------------------------------------------------------------------
# Integration tests via KnowledgeBase.learn()
# ---------------------------------------------------------------------------


class TestLearnConflictResolution:
    """Integration tests for conflict_threshold in KnowledgeBase.learn()."""

    def _mock_extract(self, facts: list[Fact]):  # type: ignore[return]
        """Patch Extractor.extract to return a fixed list of facts."""
        return patch("ai_knot.knowledge.Extractor.extract", return_value=facts)

    def test_learn_closes_old_and_inserts_new_version(self, kb: KnowledgeBase) -> None:
        """A near-duplicate closes the old fact and inserts a new version."""
        kb.add("User works at Sber as Operations Director", importance=0.9)
        new = [_fact("User works at Sber as Operations Director")]
        with self._mock_extract(new):
            inserted = kb.learn(
                [ConversationTurn(role="user", content="x")],
                api_key="fake",
            )
        # New version is returned as inserted.
        assert len(inserted) == 1
        # Storage holds both old (closed) and new version.
        all_facts = kb.list_facts()
        assert len(all_facts) == 2
        # Only one is active.
        active = [f for f in all_facts if f.is_active()]
        assert len(active) == 1

    def test_learn_inserts_new_facts(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber", importance=0.9)
        new = [_fact("User prefers Python")]
        with self._mock_extract(new):
            inserted = kb.learn(
                [ConversationTurn(role="user", content="x")],
                api_key="fake",
            )
        assert len(inserted) == 1
        assert len(kb.list_facts()) == 2

    def test_learn_conflict_threshold_kwarg(self, kb: KnowledgeBase) -> None:
        """Under low threshold, old fact is closed; new version has bumped importance."""
        kb.add("User deploys in Docker", importance=0.7)
        new = [_fact("User deploys everything in Docker")]
        with self._mock_extract(new):
            inserted = kb.learn(
                [ConversationTurn(role="user", content="x")],
                api_key="fake",
                conflict_threshold=0.3,
            )
        # New version is inserted (importance carried forward from old).
        assert len(inserted) == 1
        assert inserted[0].importance == pytest.approx(0.75)
        # Storage has old (closed) + new version = 2 facts; 1 active.
        all_facts = kb.list_facts()
        assert len(all_facts) == 2
        active = [f for f in all_facts if f.is_active()]
        assert len(active) == 1
        assert active[0].importance == pytest.approx(0.75)

    def test_learn_empty_turns_returns_empty(self, kb: KnowledgeBase) -> None:
        result = kb.learn([], api_key="fake")
        assert result == []
