"""Tests for conflict resolution in learn() via resolve_against_existing()."""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from agentmemo.extractor import resolve_against_existing
from agentmemo.knowledge import KnowledgeBase
from agentmemo.storage.yaml_storage import YAMLStorage
from agentmemo.types import ConversationTurn, Fact, MemoryType

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
    """Direct tests for the conflict-resolution helper."""

    def test_exact_duplicate_not_inserted(self) -> None:
        existing = [_fact("User works at Sber as Operations Director")]
        new = [_fact("User works at Sber as Operations Director")]
        to_insert, updated = resolve_against_existing(new, existing)
        assert to_insert == []
        assert len(updated) == 1

    def test_similar_fact_not_inserted(self) -> None:
        existing = [_fact("User works at Sber as Operations Director")]
        new = [_fact("User works at Sber as a Director")]
        to_insert, _ = resolve_against_existing(new, existing, threshold=0.5)
        assert to_insert == []

    def test_truly_new_fact_is_inserted(self) -> None:
        existing = [_fact("User works at Sber")]
        new = [_fact("User prefers Python over Java")]
        to_insert, updated = resolve_against_existing(new, existing)
        assert len(to_insert) == 1
        assert updated == []

    def test_similar_fact_bumps_importance(self) -> None:
        existing = [_fact("User deploys in Docker", importance=0.7)]
        new = [_fact("User deploys everything in Docker", importance=0.8)]
        resolve_against_existing(new, existing, threshold=0.5)
        assert existing[0].importance == pytest.approx(0.75)

    def test_importance_capped_at_1_0(self) -> None:
        existing = [_fact("User deploys in Docker", importance=0.98)]
        new = [_fact("User deploys everything in Docker")]
        resolve_against_existing(new, existing, threshold=0.5)
        assert existing[0].importance == pytest.approx(1.0)

    def test_similar_fact_updates_last_accessed(self) -> None:
        before = datetime(2025, 1, 1, tzinfo=UTC)
        existing_fact = _fact("User works at Sber")
        existing_fact.last_accessed = before
        new = [_fact("User works at Sber")]
        resolve_against_existing(new, [existing_fact])
        assert existing_fact.last_accessed > before

    def test_threshold_parameter_high_no_merge(self) -> None:
        existing = [_fact("User works at Sber")]
        new = [_fact("User works at Sber as a Director")]
        # With very high threshold these won't merge.
        to_insert, updated = resolve_against_existing(new, existing, threshold=0.99)
        assert len(to_insert) == 1
        assert updated == []

    def test_threshold_parameter_low_merges(self) -> None:
        existing = [_fact("User works at Sber")]
        new = [_fact("User works at Sber as a Director")]
        to_insert, updated = resolve_against_existing(new, existing, threshold=0.3)
        assert to_insert == []
        assert len(updated) == 1

    def test_empty_existing_all_inserted(self) -> None:
        new = [_fact("Fact A"), _fact("Fact B")]
        to_insert, updated = resolve_against_existing(new, [])
        assert len(to_insert) == 2
        assert updated == []

    def test_empty_new_returns_empty(self) -> None:
        existing = [_fact("Fact A")]
        to_insert, updated = resolve_against_existing([], existing)
        assert to_insert == []
        assert updated == []

    def test_mixed_new_and_duplicate(self) -> None:
        existing = [_fact("User works at Sber")]
        new = [
            _fact("User works at Sber"),  # duplicate
            _fact("User prefers Python"),  # new
        ]
        to_insert, updated = resolve_against_existing(new, existing)
        assert len(to_insert) == 1
        assert to_insert[0].content == "User prefers Python"
        assert len(updated) == 1


# ---------------------------------------------------------------------------
# Integration tests via KnowledgeBase.learn()
# ---------------------------------------------------------------------------


class TestLearnConflictResolution:
    """Integration tests for conflict_threshold in KnowledgeBase.learn()."""

    def _mock_extract(self, facts: list[Fact]):  # type: ignore[return]
        """Patch Extractor.extract to return a fixed list of facts."""
        return patch("agentmemo.knowledge.Extractor.extract", return_value=facts)

    def test_learn_does_not_duplicate_existing_fact(self, kb: KnowledgeBase) -> None:
        kb.add("User works at Sber as Operations Director", importance=0.9)
        new = [_fact("User works at Sber as Operations Director")]
        with self._mock_extract(new):
            inserted = kb.learn(
                [ConversationTurn(role="user", content="x")],
                api_key="fake",
            )
        assert inserted == []
        assert len(kb.list_facts()) == 1

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
        kb.add("User deploys in Docker", importance=0.7)
        new = [_fact("User deploys everything in Docker")]
        with self._mock_extract(new):
            inserted = kb.learn(
                [ConversationTurn(role="user", content="x")],
                api_key="fake",
                conflict_threshold=0.3,
            )
        assert inserted == []
        facts = kb.list_facts()
        assert len(facts) == 1
        assert facts[0].importance == pytest.approx(0.75)

    def test_learn_empty_turns_returns_empty(self, kb: KnowledgeBase) -> None:
        result = kb.learn([], api_key="fake")
        assert result == []
