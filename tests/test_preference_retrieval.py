"""Tests for Phase 4 preference-augmented retrieval."""

from ai_knot.preference_retrieval import AFFECT_LEXICON, retrieve_preference_episodes


def test_affect_lexicon_contains_core_words():
    for word in ("like", "love", "hate", "want", "prefer", "enjoy", "feel", "wish"):
        assert word in AFFECT_LEXICON


def test_affect_lexicon_is_frozenset():
    assert isinstance(AFFECT_LEXICON, frozenset)


def test_retrieve_no_entities_returns_empty():
    result = retrieve_preference_episodes(object(), "agent", entities=())
    assert result == []


def test_retrieve_no_search_fn_returns_empty():
    class NoSearch:
        pass

    result = retrieve_preference_episodes(NoSearch(), "agent", entities=("Alice",))
    assert result == []


def test_retrieve_calls_search_with_affect_query():
    calls = []

    class FakeStorage:
        def search_episodes_by_entities(
            self,
            agent_id: str,
            entities: tuple[str, ...],
            query: str,
            top_k: int,
            diversity: bool,
        ) -> list[object]:
            calls.append({"query": query, "entities": entities, "top_k": top_k})
            return []

    retrieve_preference_episodes(FakeStorage(), "agent", entities=("Alice",), top_k=15)
    assert len(calls) == 1
    assert "like" in calls[0]["query"]
    assert calls[0]["top_k"] == 15
    assert calls[0]["entities"] == ("Alice",)
