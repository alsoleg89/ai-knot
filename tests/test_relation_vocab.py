"""Tests for shared relation vocabulary module."""

from __future__ import annotations

from ai_knot.relation_vocab import (
    alias_relations,
    canonical_relation_for_phrase,
    canonical_relation_for_token,
    matches_relation,
)


def test_canonical_relation_inflected_forms():
    assert canonical_relation_for_token("attended") is not None
    assert canonical_relation_for_token("bought") is not None
    assert canonical_relation_for_token("signed") is not None


def test_canonical_relation_for_phrase_compound():
    assert canonical_relation_for_phrase("sign up for pottery") == "signed_up_for"
    assert canonical_relation_for_phrase("moved to Berlin") == "moved_to"
    assert canonical_relation_for_phrase("passed away") == "passed_away"
    assert canonical_relation_for_phrase("find it satisfying") == "finds_satisfying"


def test_alias_relations():
    assert "signed_up_for" in alias_relations("sign")
    assert "met_with" in alias_relations("meet")
    assert "liked" not in alias_relations("like")  # aliases are compound forms
    assert "likes" in alias_relations("like")


def test_matches_relation_exact():
    assert matches_relation("attended", "attended") is True
    assert matches_relation("liked", "liked") is True


def test_matches_relation_via_alias():
    assert matches_relation("signed_up_for", "sign") is True
    assert matches_relation("met_with", "meet") is True
    assert matches_relation("likes", "like") is True


def test_matches_relation_no_match():
    assert matches_relation("attended", "bought") is False
    assert matches_relation(None, "attend") is False
    assert matches_relation("attend", None) is False
