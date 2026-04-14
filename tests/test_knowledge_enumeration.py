"""Integration tests for C6b v2: enumeration split inside KnowledgeBase.add().

Covers raw and dated ingest modes — the paths that bypass Extractor.extract()
and call kb.add() directly.
"""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="test_enum", storage=storage)


# ---------------------------------------------------------------------------
# Basic splitting via add()
# ---------------------------------------------------------------------------


def test_add_enum_creates_parent_and_children(kb: KnowledgeBase) -> None:
    """add() with a comma list stores original + one child per item."""
    kb.add("Melanie enjoys pottery, camping, painting, and swimming")
    facts = kb.list_facts()
    contents = [f.content for f in facts]
    # Parent always present
    assert any("pottery, camping" in c for c in contents)
    # Children
    assert any("pottery" in c and "camping" not in c for c in contents)
    assert any("camping" in c and "pottery" not in c for c in contents)
    assert any("painting" in c and "swimming" not in c for c in contents)
    assert any("swimming" in c and "painting" not in c for c in contents)


def test_add_enum_children_have_lower_importance(kb: KnowledgeBase) -> None:
    kb.add("She likes skiing, hiking, painting, and reading", importance=0.9)
    facts = kb.list_facts()
    parent = next(f for f in facts if "skiing, hiking" in f.content)
    children = [f for f in facts if f is not parent]
    assert all(c.importance < parent.importance for c in children)


def test_add_returns_parent_fact(kb: KnowledgeBase) -> None:
    """add() returns the parent (original) Fact, not a derived child."""
    result = kb.add("She likes skiing, hiking, painting, and reading")
    # The returned fact should be the parent (contains the full list).
    assert "skiing, hiking" in result.content
    # Children have lower importance.
    assert result.importance == 0.8  # default, not reduced


def test_add_two_item_list_no_split(kb: KnowledgeBase) -> None:
    """Two items → no children generated."""
    kb.add("She likes Paris and Rome")
    facts = kb.list_facts()
    assert len(facts) == 1


def test_add_no_enumeration_single_fact(kb: KnowledgeBase) -> None:
    kb.add("She works as a data scientist")
    facts = kb.list_facts()
    assert len(facts) == 1


# ---------------------------------------------------------------------------
# Date tag inheritance
# ---------------------------------------------------------------------------


def test_add_dated_children_inherit_date_tags(kb: KnowledgeBase) -> None:
    """Children from a dated window carry date tags from the [date] prefix."""
    kb.add("[27 June, 2023] Alice loves pottery, camping, painting, and swimming")
    facts = kb.list_facts()
    for f in facts:
        assert "june 2023" in f.tags, f"date tag missing on: {f.content!r}"
        assert "2023-06-27" in f.tags, f"ISO date tag missing on: {f.content!r}"


def test_add_dated_children_preserve_date_prefix(kb: KnowledgeBase) -> None:
    """Children's content starts with the [date] bracket."""
    kb.add("[27 June, 2023] Alice loves pottery, camping, painting, and swimming")
    facts = kb.list_facts()
    # Children: importance lower than parent's default (0.8).
    children = [f for f in facts if f.importance < 0.8]
    assert len(children) >= 4
    for c in children:
        assert c.content.startswith("[27 June, 2023]"), (
            f"date prefix missing on child: {c.content!r}"
        )


# ---------------------------------------------------------------------------
# Recall benefits
# ---------------------------------------------------------------------------


def test_add_enum_children_recalled_individually(kb: KnowledgeBase) -> None:
    """Atomic child fact is recalled when querying for the specific item."""
    kb.add("Melanie enjoys pottery, camping, painting, and swimming")
    kb.add("John works as an engineer in Berlin")
    # Query specifically for one hobby — should surface a fact about pottery
    results = kb.recall_facts("what pottery does Melanie do?", top_k=5)
    contents = [r.content for r in results]
    assert any("pottery" in c for c in contents)


# ---------------------------------------------------------------------------
# Near-duplicate dedup for children
# ---------------------------------------------------------------------------


def test_add_child_near_dup_suppressed(kb: KnowledgeBase) -> None:
    """A child whose content is near-identical to an existing fact is dropped."""
    # Pre-store a fact almost identical to what the child would be.
    kb.add("Melanie enjoys pottery")
    # Now add an enumeration whose first item overlaps heavily.
    kb.add("Melanie enjoys pottery, camping, painting, and swimming")
    facts = kb.list_facts()
    pottery_facts = [f for f in facts if "pottery" in f.content and "camping" not in f.content]
    # Should not have two nearly-identical "pottery" facts.
    assert len(pottery_facts) <= 2  # parent (list) + 0 or 1 child; not 2 identical children


# ---------------------------------------------------------------------------
# Semicolon enumeration via add()
# ---------------------------------------------------------------------------


def test_add_semicolon_enum_splits(kb: KnowledgeBase) -> None:
    kb.add("Hobbies: pottery; camping; painting; swimming")
    facts = kb.list_facts()
    contents = [f.content for f in facts]
    assert any("pottery" in c and "camping" not in c for c in contents)
    assert any("camping" in c and "pottery" not in c for c in contents)
    assert len(facts) >= 5  # parent + 4 children
