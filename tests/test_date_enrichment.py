"""Tests for C6c: enrich_date_tags and its integration via kb.add()."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot._date_enrichment import enrich_date_tags
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact

# ---------------------------------------------------------------------------
# Unit tests for enrich_date_tags
# ---------------------------------------------------------------------------


def test_bracket_prefix_date_enriched() -> None:
    """[27 June, 2023] prefix → iso, month-year, month, year tags."""
    f = Fact(content="[27 June, 2023] Melanie went to the beach")
    enrich_date_tags(f)
    assert "2023-06-27" in f.tags
    assert "june 2023" in f.tags
    assert "june" in f.tags
    assert "2023" in f.tags


def test_iso_date_enriched() -> None:
    f = Fact(content="Project started on 2023-08-15")
    enrich_date_tags(f)
    assert "2023-08-15" in f.tags
    assert "august 2023" in f.tags
    assert "august" in f.tags
    assert "2023" in f.tags


def test_mdy_date_enriched() -> None:
    """June 27, 2023 style."""
    f = Fact(content="They met on June 27, 2023 at noon")
    enrich_date_tags(f)
    assert "2023-06-27" in f.tags
    assert "june 2023" in f.tags


def test_month_year_only_enriched() -> None:
    f = Fact(content="He graduated in May 2022")
    enrich_date_tags(f)
    assert "may 2022" in f.tags
    assert "may" in f.tags
    assert "2022" in f.tags
    # No ISO tag (no day available)
    assert not any(t.count("-") == 2 for t in f.tags)


def test_no_date_leaves_tags_untouched() -> None:
    f = Fact(content="Melanie enjoys pottery", tags=["hobby"])
    enrich_date_tags(f)
    assert f.tags == ["hobby"]


def test_no_date_empty_content() -> None:
    f = Fact(content="")
    enrich_date_tags(f)
    assert f.tags == []


def test_multiple_dates_all_captured() -> None:
    f = Fact(content="Started in May 2023, finished by August 2023")
    enrich_date_tags(f)
    assert "may 2023" in f.tags
    assert "august 2023" in f.tags


def test_tags_capped_at_10() -> None:
    # Fill 9 existing tags, then a date with 4 canonical forms → only 1 more fits.
    existing = [f"tag{i}" for i in range(9)]
    f = Fact(content="[27 June, 2023] test", tags=existing)
    enrich_date_tags(f)
    assert len(f.tags) == 10


def test_existing_tags_preserved_first() -> None:
    f = Fact(content="[27 June, 2023] event", tags=["hobby", "sport"])
    enrich_date_tags(f)
    assert f.tags[0] == "hobby"
    assert f.tags[1] == "sport"
    assert "june 2023" in f.tags


def test_false_positive_room_number_not_tagged() -> None:
    """Bare "2023" in "Room 2023" must not produce a year-only tag.

    Our patterns all require multi-component dates (day+month+year,
    month+year, ISO) — a standalone number won't match.
    """
    f = Fact(content="Meeting in Room 2023 about the project")
    enrich_date_tags(f)
    assert "2023" not in f.tags


def test_invalid_month_name_ignored() -> None:
    f = Fact(content="Blabla 27 2023 something")
    enrich_date_tags(f)
    assert f.tags == []


def test_witness_surface_also_parsed() -> None:
    f = Fact(content="She went camping", witness_surface="[10 October, 2023] went camping")
    enrich_date_tags(f)
    assert "october 2023" in f.tags
    assert "2023-10-10" in f.tags


def test_duplicate_dates_not_doubled() -> None:
    """Same date appearing twice in content → tags added only once."""
    f = Fact(content="[27 June, 2023] also June 27, 2023 mentioned again")
    enrich_date_tags(f)
    assert f.tags.count("june 2023") == 1
    assert f.tags.count("june") == 1


# ---------------------------------------------------------------------------
# Integration: KnowledgeBase.add() applies date enrichment
# ---------------------------------------------------------------------------


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="test_dates", storage=storage)


def test_raw_mode_kb_add_enriches(kb: KnowledgeBase) -> None:
    kb.add("[27 June, 2023] Melanie went swimming")
    facts = kb.list_facts()
    assert len(facts) == 1
    assert "2023-06-27" in facts[0].tags
    assert "june 2023" in facts[0].tags


def test_kb_add_no_date_no_tags(kb: KnowledgeBase) -> None:
    kb.add("Melanie enjoys pottery")
    facts = kb.list_facts()
    assert facts[0].tags == []


def test_recall_prefers_date_tagged_fact(kb: KnowledgeBase) -> None:
    """Fact with matching date tag ranks higher than unrelated fact."""
    kb.add("[27 June, 2023] Melanie went to the beach")
    kb.add("[10 October, 2023] Rajesh wrote Rust code")
    # Query mentions "June 2023" — should recall Melanie's fact first.
    pairs = kb.recall_facts("what happened in June 2023?", top_k=2)
    assert len(pairs) >= 1
    assert "Melanie" in pairs[0].content or "June" in pairs[0].content
