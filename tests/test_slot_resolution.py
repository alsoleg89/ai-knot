"""Tests for slot-based resolution: resolve_by_slot() and the learn() slot pipeline."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from ai_knot.extractor import resolve_by_slot
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(
    content: str,
    *,
    slot_key: str = "",
    value_text: str = "",
    importance: float = 0.8,
    state_confidence: float = 1.0,
    entity: str = "",
    attribute: str = "",
) -> Fact:
    return Fact(
        content=content,
        type=MemoryType.SEMANTIC,
        importance=importance,
        slot_key=slot_key,
        value_text=value_text,
        state_confidence=state_confidence,
        entity=entity,
        attribute=attribute,
    )


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="test", storage=YAMLStorage(base_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# Unit tests for resolve_by_slot()
# ---------------------------------------------------------------------------


class TestResolveBySlot:
    """Direct tests for the slot resolver."""

    def test_no_slot_key_returns_branch(self) -> None:
        existing = [_fact("User works at Sber", slot_key="User::employer")]
        new = _fact("User has been at Sber 3 years")  # no slot_key
        op, matched = resolve_by_slot(new, existing)
        assert op == "branch"
        assert matched is None

    def test_no_match_returns_branch(self) -> None:
        existing = [_fact("...", slot_key="Alex::salary", value_text="80000")]
        new = _fact("...", slot_key="Alex::role", value_text="PM")
        op, matched = resolve_by_slot(new, existing)
        assert op == "branch"
        assert matched is None

    def test_empty_existing_returns_branch(self) -> None:
        new = _fact("...", slot_key="Alex::salary", value_text="95000")
        op, matched = resolve_by_slot(new, [])
        assert op == "branch"
        assert matched is None

    def test_same_value_returns_reinforce(self) -> None:
        old = _fact("Alex earns 95k", slot_key="Alex::salary", value_text="95000")
        new = _fact("Alex's salary is 95000", slot_key="Alex::salary", value_text="95000")
        op, matched = resolve_by_slot(new, [old])
        assert op == "reinforce"
        assert matched is old

    def test_same_value_case_insensitive(self) -> None:
        old = _fact("...", slot_key="U::title", value_text="Senior PM")
        new = _fact("...", slot_key="U::title", value_text="senior pm")
        op, matched = resolve_by_slot(new, [old])
        assert op == "reinforce"
        assert matched is old

    def test_different_value_returns_supersede(self) -> None:
        old = _fact("Alex earns 80k", slot_key="Alex::salary", value_text="80000")
        new = _fact("Alex got a raise to 95k", slot_key="Alex::salary", value_text="95000")
        op, matched = resolve_by_slot(new, [old])
        assert op == "supersede"
        assert matched is old

    def test_missing_old_value_text_returns_supersede(self) -> None:
        """Old fact has no value_text → can't confirm same value → supersede."""
        old = _fact("Alex earns a lot", slot_key="Alex::salary", value_text="")
        new = _fact("Alex earns 95k", slot_key="Alex::salary", value_text="95000")
        op, matched = resolve_by_slot(new, [old])
        assert op == "supersede"
        assert matched is old

    def test_missing_new_value_text_returns_supersede(self) -> None:
        """New fact has no value_text → supersede (safer than reinforce)."""
        old = _fact("Alex earns 80k", slot_key="Alex::salary", value_text="80000")
        new = _fact("Alex's pay changed", slot_key="Alex::salary", value_text="")
        op, matched = resolve_by_slot(new, [old])
        assert op == "supersede"
        assert matched is old

    def test_first_slot_match_wins(self) -> None:
        """Only the first fact with a matching slot_key is considered."""
        old1 = _fact("first", slot_key="A::x", value_text="v1")
        old2 = _fact("second", slot_key="A::x", value_text="v1")
        new = _fact("new", slot_key="A::x", value_text="v1")
        op, matched = resolve_by_slot(new, [old1, old2])
        assert op == "reinforce"
        assert matched is old1  # first match wins


# ---------------------------------------------------------------------------
# Integration tests via KnowledgeBase.learn()
# ---------------------------------------------------------------------------


class TestLearnSlotTransitions:
    """Integration tests for slot-based resolution in learn()."""

    def _mock_extract(self, facts: list[Fact]):  # type: ignore[return]
        return patch("ai_knot.knowledge.Extractor.extract", return_value=facts)

    # --- reinforce ---

    def test_reinforce_does_not_insert_new_fact(self, kb: KnowledgeBase) -> None:
        """Reinforce: same slot + same value → bump confidence, no new fact inserted."""
        old = kb.add("Alex earns 95k", importance=0.8)
        old.slot_key = "Alex::salary"
        old.value_text = "95000"
        kb._storage.save(kb._agent_id, [old])

        new = [_fact("Alex's salary is 95k", slot_key="Alex::salary", value_text="95000")]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        assert inserted == []
        all_facts = kb.list_facts()
        assert len(all_facts) == 1
        # Importance and confidence bumped on the original.
        assert all_facts[0].importance == pytest.approx(0.82)
        assert all_facts[0].state_confidence == pytest.approx(1.0)  # already at cap

    def test_reinforce_bumps_state_confidence(self, kb: KnowledgeBase) -> None:
        old = kb.add("Alex is a PM", importance=0.7)
        old.slot_key = "Alex::role"
        old.value_text = "PM"
        old.state_confidence = 0.8
        kb._storage.save(kb._agent_id, [old])

        new = [_fact("Alex is Product Manager", slot_key="Alex::role", value_text="PM")]
        with self._mock_extract(new):
            kb.learn([ConversationTurn("user", "x")], api_key="fake")

        stored = kb.list_facts()[0]
        assert stored.state_confidence == pytest.approx(0.85)

    # --- supersede ---

    def test_supersede_closes_old_inserts_new(self, kb: KnowledgeBase) -> None:
        """Supersede: same slot, new value → close old, insert new version."""
        old = kb.add("Alex earns 80k", importance=0.7)
        old.slot_key = "Alex::salary"
        old.value_text = "80000"
        kb._storage.save(kb._agent_id, [old])

        new = [_fact("Alex got a raise to 95k", slot_key="Alex::salary", value_text="95000")]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        # New version inserted.
        assert len(inserted) == 1
        assert inserted[0].value_text == "95000"
        # Importance carried forward (+0.05).
        assert inserted[0].importance == pytest.approx(0.75)
        # Version incremented.
        assert inserted[0].version == 1

        # Storage: old closed, new active.
        all_facts = kb.list_facts()
        assert len(all_facts) == 2
        active = [f for f in all_facts if f.is_active()]
        closed = [f for f in all_facts if not f.is_active()]
        assert len(active) == 1
        assert len(closed) == 1
        assert active[0].value_text == "95000"
        assert closed[0].value_text == "80000"
        assert closed[0].valid_until is not None

    def test_supersede_version_increments(self, kb: KnowledgeBase) -> None:
        """Each supersede increments the version counter."""
        f = kb.add("Alex role: PM")
        f.slot_key = "Alex::role"
        f.value_text = "PM"
        f.version = 3
        kb._storage.save(kb._agent_id, [f])

        new = [_fact("Alex is now Director", slot_key="Alex::role", value_text="Director")]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        assert inserted[0].version == 4

    # --- branch ---

    def test_branch_inserts_new_slot(self, kb: KnowledgeBase) -> None:
        """Branch: new slot not in storage → inserted without modification."""
        kb.add("Some unrelated fact")

        new = [_fact("Alex joined in 2022", slot_key="Alex::start_date", value_text="2022")]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        assert len(inserted) == 1
        assert inserted[0].slot_key == "Alex::start_date"

    # --- mixed ---

    def test_mixed_slot_and_unslotted_in_same_learn(self, kb: KnowledgeBase) -> None:
        """Slotted facts use slot resolution; unslotted use lexical dedup."""
        old_slot = kb.add("Alex earns 80k")
        old_slot.slot_key = "Alex::salary"
        old_slot.value_text = "80000"
        old_plain = kb.add("User prefers dark mode")
        kb._storage.save(kb._agent_id, [old_slot, old_plain])

        new = [
            _fact("Alex raised to 95k", slot_key="Alex::salary", value_text="95000"),
            _fact("User prefers dark mode"),  # unslotted, lexical duplicate
            _fact("Python is preferred language"),  # unslotted, genuinely new
        ]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        # supersede (1) + branch for "Python..." (1) + new version of unslotted dup (1)
        # The lexical dup "User prefers dark mode" triggers temporal close + new version
        assert len(inserted) >= 2
        contents = {f.content for f in inserted}
        assert "Alex raised to 95k" in contents
        assert "Python is preferred language" in contents

    def test_multiple_supersedes_in_one_learn(self, kb: KnowledgeBase) -> None:
        """Multiple slot facts can be superseded in a single learn() call."""
        f1 = kb.add("Alex role: PM")
        f1.slot_key = "Alex::role"
        f1.value_text = "PM"
        f2 = kb.add("Alex salary: 80k")
        f2.slot_key = "Alex::salary"
        f2.value_text = "80000"
        kb._storage.save(kb._agent_id, [f1, f2])

        new = [
            _fact("Alex is now Director", slot_key="Alex::role", value_text="Director"),
            _fact("Alex earns 120k now", slot_key="Alex::salary", value_text="120000"),
        ]
        with self._mock_extract(new):
            inserted = kb.learn([ConversationTurn("user", "x")], api_key="fake")

        assert len(inserted) == 2
        active = [f for f in kb.list_facts() if f.is_active()]
        assert len(active) == 2
        values = {f.value_text for f in active}
        assert values == {"Director", "120000"}
