"""Tests for _consolidate_phase() — entity-level aggregate creation."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MemoryType


def _make_fact(
    entity: str,
    attribute: str,
    value: str,
    importance: float = 0.8,
) -> Fact:
    return Fact(
        content=f"{entity} {attribute} {value}",
        type=MemoryType.SEMANTIC,
        importance=importance,
        entity=entity,
        attribute=attribute,
        slot_key=f"{entity}::{attribute}",
        value_text=value,
    )


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="test_agent", storage=storage)


class TestConsolidatePhase:
    """_consolidate_phase() produces correct entity-level aggregates."""

    def test_aggregate_created_for_two_facts(self, kb: KnowledgeBase) -> None:
        facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        assert len(aggregates) == 1
        agg = aggregates[0]
        assert agg.entity == "melanie"
        assert agg.attribute == "_aggregate"
        assert agg.slot_key == "melanie::_agg"
        assert "pottery" in agg.content
        assert "swimming" in agg.content
        assert "aggregate" in agg.tags
        assert agg.verification_source == "aggregate"

    def test_no_aggregate_for_single_fact(self, kb: KnowledgeBase) -> None:
        facts = [_make_fact("melanie", "hobby", "pottery")]
        aggregates = kb._consolidate_phase(facts, [])
        assert aggregates == []

    def test_aggregate_content_is_compact(self, kb: KnowledgeBase) -> None:
        """Content should be 'entity: val1, val2, ...' without attr= prefixes."""
        facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
            _make_fact("melanie", "instrument", "clarinet"),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        assert len(aggregates) == 1
        content = aggregates[0].content
        assert content.startswith("melanie:")
        assert "=" not in content  # no attr= prefixes
        assert "pottery" in content
        assert "swimming" in content
        assert "clarinet" in content

    def test_canonical_surface_is_keywords(self, kb: KnowledgeBase) -> None:
        facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        assert "melanie" in aggregates[0].canonical_surface
        assert "pottery" in aggregates[0].canonical_surface
        assert "swimming" in aggregates[0].canonical_surface

    def test_no_aggregate_on_aggregate(self, kb: KnowledgeBase) -> None:
        """Facts with _aggregate attribute are skipped — prevents chaining."""
        facts = [
            _make_fact("melanie", "_aggregate", "pottery, swimming"),
            _make_fact("melanie", "sport", "swimming"),
        ]
        # Only 1 non-aggregate fact → no aggregate created
        aggregates = kb._consolidate_phase(facts, [])
        assert aggregates == []

    def test_skips_facts_without_entity(self, kb: KnowledgeBase) -> None:
        facts = [
            Fact(content="raw fact one", entity="", attribute=""),
            Fact(content="raw fact two", entity="", attribute=""),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        assert aggregates == []

    def test_separate_aggregates_per_entity(self, kb: KnowledgeBase) -> None:
        facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
            _make_fact("oliver", "pet", "dog"),
            _make_fact("oliver", "sport", "tennis"),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        entities = {a.entity for a in aggregates}
        assert entities == {"melanie", "oliver"}

    def test_importance_is_max_of_group(self, kb: KnowledgeBase) -> None:
        facts = [
            _make_fact("melanie", "hobby", "pottery", importance=0.6),
            _make_fact("melanie", "sport", "swimming", importance=0.95),
        ]
        aggregates = kb._consolidate_phase(facts, [])
        assert aggregates[0].importance == pytest.approx(0.95)

    def test_cas_skip_when_content_unchanged(self, kb: KnowledgeBase) -> None:
        """If an identical aggregate already exists, no new aggregate is returned."""
        facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
        ]
        # Pre-create the aggregate in existing
        first_run = kb._consolidate_phase(facts, [])
        assert len(first_run) == 1

        # Second run with same facts + first aggregate in existing
        second_run = kb._consolidate_phase(facts, first_run)
        assert second_run == []

    def test_cas_supersedes_stale_aggregate(self, kb: KnowledgeBase) -> None:
        """When content changes, old aggregate gets valid_until set."""
        old_facts = [
            _make_fact("melanie", "hobby", "pottery"),
            _make_fact("melanie", "sport", "swimming"),
        ]
        old_aggs = kb._consolidate_phase(old_facts, [])
        assert len(old_aggs) == 1
        old_agg = old_aggs[0]

        # Add a new fact for melanie — aggregate content changes
        new_facts = old_facts + [_make_fact("melanie", "instrument", "clarinet")]
        new_aggs = kb._consolidate_phase(new_facts, old_aggs)

        # Old aggregate should be superseded
        assert old_agg.valid_until is not None
        # New aggregate returned
        assert len(new_aggs) == 1
        assert "clarinet" in new_aggs[0].content

    def test_cap_at_30_pairs(self, kb: KnowledgeBase) -> None:
        """Aggregates include at most 30 attribute-value pairs."""
        facts = [_make_fact("entity", f"attr{i}", f"val{i}") for i in range(40)]
        aggregates = kb._consolidate_phase(facts, [])
        assert len(aggregates) == 1
        # 30 values → 30 comma-separated items in content
        values_part = aggregates[0].content.split(": ", 1)[1]
        items = [v.strip() for v in values_part.split(",") if v.strip()]
        assert len(items) <= 30
