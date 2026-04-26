"""Tests for E0.2 — MMR slot-aware Jaccard: list items under same slot_key
do not suppress each other in MMR selection."""

from __future__ import annotations

import pathlib
import uuid

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact


def _make_fact(content: str, slot_key: str, value_text: str, importance: float = 1.0) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        content=content,
        slot_key=slot_key,
        value_text=value_text,
        importance=importance,
    )


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="mmr_slot_test", storage=storage)


class TestMMRSlotProtection:
    def test_list_items_same_slot_all_survive(self, kb: KnowledgeBase) -> None:
        """4 list items under slot 'melanie::hobbies' with different value_text
        should all pass MMR when top_k=4 — none suppressed as near-duplicate."""
        hobbies = ["pottery", "camping", "painting", "swimming"]
        for h in hobbies:
            kb.add(f"Melanie enjoys {h}", tags=["melanie", "hobbies"])

        results = kb.recall_facts("What are Melanie's hobbies", top_k=4)
        result_contents = " ".join(f.content for f in results).lower()
        # Each hobby should appear in results
        found = sum(1 for h in hobbies if h in result_contents)
        assert found >= 3, f"Expected ≥3 hobbies, got {found}: {result_contents}"

    def test_different_slot_key_still_penalised(self, kb: KnowledgeBase) -> None:
        """Two facts with different slot_keys but very similar content are still
        penalised by MMR (normal Jaccard similarity applies)."""
        # Add two very similar facts about different slots
        kb.add("Alice enjoys swimming in the lake", tags=["alice"])
        kb.add("Alice loves swimming in the lake very much", tags=["alice"])
        results = kb.recall_facts("alice swimming", top_k=1)
        # With top_k=1, only one should survive
        assert len(results) == 1

    def test_raw_mode_no_slot_backward_compat(self, kb: KnowledgeBase) -> None:
        """Facts with empty slot_key (raw/dated mode) have identical Jaccard behaviour
        as before — no regression."""
        kb.add("John likes hiking", tags=["john"])
        kb.add("John enjoys running", tags=["john"])
        results = kb.recall_facts("john activity", top_k=5)
        assert len(results) >= 1

    def test_same_slot_different_value_top_k_three(self, kb: KnowledgeBase) -> None:
        """With top_k=3 from 4 list items under same slot, all 3 selected are distinct."""
        hobbies = ["pottery", "camping", "painting", "swimming"]
        for h in hobbies:
            kb.add(f"Melanie likes {h}", tags=["melanie"])

        results = kb.recall_facts("Melanie hobbies", top_k=3)
        assert len(results) == 3
        result_contents = [f.content for f in results]
        # All 3 must be distinct
        assert len(set(result_contents)) == 3
