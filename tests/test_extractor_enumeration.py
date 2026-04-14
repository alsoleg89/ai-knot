"""Tests for C6b: split_enumerations post-process in extractor.py."""

from __future__ import annotations

from ai_knot.extractor import (
    _ITEM_SPLIT_COMMA,
    _ITEM_SPLIT_SEMI,
    _extract_enum_items,
    split_enumerations,
)
from ai_knot.types import Fact, MemoryType

# Backward-compat alias used in existing tests below.
_split_enumerations = split_enumerations

# ---------------------------------------------------------------------------
# _extract_enum_items
# ---------------------------------------------------------------------------


def test_extract_items_basic_comma_list() -> None:
    items = _extract_enum_items("pottery, camping, painting, swimming", _ITEM_SPLIT_COMMA)
    assert items == ["pottery", "camping", "painting", "swimming"]


def test_extract_items_with_and() -> None:
    items = _extract_enum_items("pottery, camping, painting, and swimming", _ITEM_SPLIT_COMMA)
    assert items == ["pottery", "camping", "painting", "swimming"]


def test_extract_items_long_item_filtered() -> None:
    # Any item > 20 chars is dropped.
    items = _extract_enum_items(
        "pottery, long-form creative pottery sessions, swimming", _ITEM_SPLIT_COMMA
    )
    assert "long-form creative pottery sessions" not in items
    assert "pottery" in items
    assert "swimming" in items


def test_extract_items_strips_punctuation() -> None:
    items = _extract_enum_items("running, swimming, dancing.", _ITEM_SPLIT_COMMA)
    assert "dancing" in items  # trailing dot stripped


def test_extract_items_semicolon_list() -> None:
    items = _extract_enum_items("pottery; camping; painting; swimming", _ITEM_SPLIT_SEMI)
    assert items == ["pottery", "camping", "painting", "swimming"]


# ---------------------------------------------------------------------------
# _split_enumerations
# ---------------------------------------------------------------------------


def test_split_basic_four_items() -> None:
    f = Fact(
        content="Melanie enjoys pottery, camping, painting, and swimming",
        entity="melanie",
        importance=0.9,
        tags=["hobby"],
    )
    out = _split_enumerations([f])
    contents = [x.content for x in out]
    assert "Melanie enjoys pottery" in contents
    assert "Melanie enjoys camping" in contents
    assert "Melanie enjoys painting" in contents
    assert "Melanie enjoys swimming" in contents
    assert f in out  # original preserved


def test_split_preserves_entity_and_tags() -> None:
    f = Fact(
        content="She visited beach, mountains, and forest",
        entity="caroline",
        tags=["travel"],
        importance=0.8,
    )
    out = _split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert len(derived) == 3
    assert all(d.entity == "caroline" for d in derived)
    assert all("travel" in d.tags for d in derived)


def test_split_derived_importance_reduced() -> None:
    f = Fact(content="Hobbies: skiing, hiking, painting, and reading", importance=0.9)
    out = _split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert all(d.importance < f.importance for d in derived)
    assert all(d.importance >= 0.0 for d in derived)


def test_split_derived_gets_new_ids() -> None:
    f = Fact(content="Melanie enjoys pottery, camping, painting, and swimming")
    out = _split_enumerations([f])
    ids = [x.id for x in out]
    assert len(ids) == len(set(ids)), "all IDs must be unique"


def test_split_derived_tags_are_independent_copy() -> None:
    """Mutating a derived fact's tags must not affect the original."""
    f = Fact(content="Activities: running, swimming, cycling, climbing", tags=["sport"])
    out = _split_enumerations([f])
    derived = [x for x in out if x is not f][0]
    derived.tags.append("mutated")
    assert "mutated" not in f.tags


def test_no_split_two_items() -> None:
    # Fewer than 3 items → no split.
    f = Fact(content="They visited Paris and Rome")
    assert _split_enumerations([f]) == [f]


def test_no_split_address_two_items() -> None:
    f = Fact(content="Lives in New York, NY")
    assert _split_enumerations([f]) == [f]


def test_no_split_all_long_items() -> None:
    # All items are > 20 chars → _extract_enum_items returns < 3 valid items.
    f = Fact(
        content="Activities include long-form creative pottery sessions, "
        "extensive multi-day camping trips, and deep-dive painting workshops",
    )
    out = _split_enumerations([f])
    # No extras generated (items too long), just the original.
    assert len(out) == 1


def test_split_empty_content_skipped() -> None:
    f = Fact(content="")
    assert _split_enumerations([f]) == [f]


def test_split_preserves_type() -> None:
    f = Fact(
        content="Bob camped at beach, mountains, and forest",
        type=MemoryType.EPISODIC,
    )
    out = _split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert all(d.type == MemoryType.EPISODIC for d in derived)


def test_split_source_snippets_reset() -> None:
    """Derived facts start with empty source_snippets (re-populated later by ATC)."""
    f = Fact(
        content="Melanie swims, paints, and camps",
        source_snippets=["original snippet"],
    )
    out = _split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert all(d.source_snippets == [] for d in derived)


def test_split_multiple_facts() -> None:
    f1 = Fact(content="Alice plays piano, guitar, and violin")
    f2 = Fact(content="No enumeration here")
    out = _split_enumerations([f1, f2])
    assert f2 in out  # unchanged
    assert f1 in out  # original preserved
    derived = [x for x in out if x is not f1 and x is not f2]
    assert len(derived) == 3  # piano, guitar, violin


# ---------------------------------------------------------------------------
# Semicolon enumeration
# ---------------------------------------------------------------------------


def test_split_semicolon_list() -> None:
    f = Fact(
        content="Hobbies: pottery; camping; painting; swimming",
        entity="melanie",
        importance=0.9,
        tags=["hobby"],
    )
    out = split_enumerations([f])
    contents = [x.content for x in out]
    assert "Hobbies: pottery" in contents
    assert "Hobbies: camping" in contents
    assert "Hobbies: painting" in contents
    assert "Hobbies: swimming" in contents
    assert f in out


def test_split_semicolon_fewer_than_three_no_split() -> None:
    f = Fact(content="Options: A; B")
    assert split_enumerations([f]) == [f]


# ---------------------------------------------------------------------------
# Dated-window prefix handling (C6b v2)
# ---------------------------------------------------------------------------


def test_dated_prefix_preserved_on_children() -> None:
    """Children should carry the [date] bracket for date-tag enrichment."""
    f = Fact(content="[27 June, 2023] Alice loves pottery, camping, painting, and swimming")
    out = split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert len(derived) == 4
    for d in derived:
        assert d.content.startswith("[27 June, 2023]"), f"date prefix missing: {d.content!r}"


def test_prior_turn_trimmed_from_verb_prefix() -> None:
    """Turn separator '/' should stop the verb prefix from absorbing prior turns."""
    content = "[2023-06-27] Bob: hi / Alice: I love pottery, camping, painting, and swimming"
    f = Fact(content=content)
    out = split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert len(derived) == 4
    for d in derived:
        # "Bob: hi" must NOT appear in children
        assert "Bob: hi" not in d.content, f"prior turn leaked: {d.content!r}"
        # Date prefix must still be there
        assert "[2023-06-27]" in d.content


def test_no_date_prefix_falls_back_to_normal_prefix() -> None:
    """Without [date] bracket the prefix is the plain verb phrase as before."""
    f = Fact(content="Melanie enjoys pottery, camping, painting, and swimming")
    out = split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert "Melanie enjoys pottery" in [d.content for d in derived]
    assert "Melanie enjoys swimming" in [d.content for d in derived]


def test_sentence_boundary_trimmed_from_verb_prefix() -> None:
    """Period-space boundary should stop the prefix at the last sentence."""
    content = "She visited Paris. She loves pottery, camping, painting, and swimming."
    f = Fact(content=content)
    out = split_enumerations([f])
    derived = [x for x in out if x is not f]
    assert len(derived) == 4
    for d in derived:
        assert "visited Paris" not in d.content, f"prior sentence leaked: {d.content!r}"
