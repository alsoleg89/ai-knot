"""Tests that InvertedIndex indexes all Fact fields and exposes them via properties."""

from __future__ import annotations

from ai_knot._inverted_index import InvertedIndex
from ai_knot.types import Fact


def _make_fact(
    id: str,
    content: str,
    *,
    tags: list[str] | None = None,
    canonical_surface: str = "",
    source_snippets: list[str] | None = None,
    slot_key: str = "",
    entity: str = "",
    attribute: str = "",
    value_text: str = "",
) -> Fact:
    return Fact(
        id=id,
        content=content,
        tags=tags or [],
        canonical_surface=canonical_surface,
        source_snippets=source_snippets or [],
        slot_key=slot_key,
        entity=entity,
        attribute=attribute,
        value_text=value_text,
    )


class TestInvertedIndexProperties:
    def test_content_postings_property_exists(self) -> None:
        idx = InvertedIndex([_make_fact("1", "hello world")])
        postings = idx.content_postings
        assert isinstance(postings, dict)

    def test_tags_postings_property_exists(self) -> None:
        idx = InvertedIndex([_make_fact("1", "x", tags=["hobby"])])
        assert isinstance(idx.tags_postings, dict)

    def test_content_postings_indexes_tokens(self) -> None:
        f = _make_fact("1", "Alex works at FinServe")
        idx = InvertedIndex([f])
        # Tokenizer stems tokens: "Alex" → "alex", "works" → "work",
        # "FinServe" → ["fin", "serv"] (split + stemmed).
        assert "alex" in idx.content_postings
        assert "1" in idx.content_postings["alex"]
        # At least one token from "FinServe" is indexed.
        finserve_tokens = {"fin", "serv", "finserv", "finserve"}
        assert any(t in idx.content_postings for t in finserve_tokens)

    def test_tags_postings_indexes_tag_tokens(self) -> None:
        f1 = _make_fact("1", "x", tags=["employer", "role"])
        f2 = _make_fact("2", "y", tags=["python", "preferences"])
        idx = InvertedIndex([f1, f2])
        # Tokenizer stems: "employer" → "employ", "role" stays "role",
        # "python" stays "python", "preferences" → "preferenc".
        assert "employ" in idx.tags_postings
        assert "1" in idx.tags_postings["employ"]
        assert "python" in idx.tags_postings
        assert "2" in idx.tags_postings["python"]

    def test_slot_tokens_non_empty_for_slot_key(self) -> None:
        f = _make_fact("1", "x", slot_key="alex chen::employer")
        idx = InvertedIndex([f])
        assert idx.slot_tokens["1"]  # non-empty frozenset

    def test_slot_tokens_empty_for_no_slot_key(self) -> None:
        f = _make_fact("2", "y", slot_key="")
        idx = InvertedIndex([f])
        assert idx.slot_tokens["2"] == frozenset()

    def test_facts_property_accessible(self) -> None:
        f = _make_fact("1", "content")
        idx = InvertedIndex([f])
        assert "1" in idx.facts
        assert idx.facts["1"] is f

    def test_canonical_surface_indexed(self) -> None:
        f = _make_fact("1", "Alex works there", canonical_surface="person works at company")
        idx = InvertedIndex([f])
        # Access internal canonical postings via score lookup.
        scores = idx.score("person company")
        assert scores.get("1", 0.0) > 0.0

    def test_source_snippets_indexed_as_evidence(self) -> None:
        f = _make_fact("1", "short", source_snippets=["Alex works at FinServe Capital"])
        idx = InvertedIndex([f])
        scores = idx.score("FinServe Capital")
        assert scores.get("1", 0.0) > 0.0

    def test_two_facts_in_same_posting(self) -> None:
        f1 = _make_fact("1", "melanie likes pottery", tags=["hobby"])
        f2 = _make_fact("2", "melanie enjoys camping", tags=["hobby"])
        idx = InvertedIndex([f1, f2])
        # "hobby" is stemmed to "hobbi" by the tokenizer.
        hobby_tok = "hobbi" if "hobbi" in idx.tags_postings else "hobby"
        assert len(idx.tags_postings[hobby_tok]) == 2

    def test_properties_do_not_mutate_index(self) -> None:
        f = _make_fact("1", "foo", tags=["bar"])
        idx = InvertedIndex([f])
        _ = idx.content_postings
        _ = idx.tags_postings
        _ = idx.slot_tokens
        # After property access, score still works correctly.
        scores = idx.score("foo")
        assert scores.get("1", 0.0) > 0.0
