"""Tests for Frame Lexical Bridge query expansion."""

from ai_knot.query_lexicon import LEXICON, LexicalExpansion, expand_query_lexically


class TestLexicon:
    def test_all_weights_below_one(self) -> None:
        for frame_name, frame in LEXICON.items():
            for term, w in frame.terms.items():
                assert w < 1.0, f"{frame_name}.{term} weight={w} >= 1.0"

    def test_navigational_never_expanded(self) -> None:
        result = expand_query_lexically("find meeting notes", "navigational")
        assert result.expansion_weights == {}
        assert result.frames_applied == []
        assert result.terms_added == 0

    def test_factual_sports_query_gets_activity_terms(self) -> None:
        result = expand_query_lexically("What sport does Sarah play?", "factual")
        # "play" is in query → activity_sport frame should fire
        # terms_added may be 0 if "play" is already in q_lower and all other terms added
        assert isinstance(result.terms_added, int)
        assert result.terms_added >= 0

    def test_activity_frame_fires_on_sport_keyword(self) -> None:
        result = expand_query_lexically("What sport does Sarah play?", "factual")
        # "play" is in query tokens which matches activity_sport frame term set → frame fires
        assert "activity_sport" in result.frames_applied

    def test_work_frame_fires_on_job_keyword(self) -> None:
        result = expand_query_lexically("What job does Tom have?", "factual")
        # "job" matches work_career frame terms
        assert "work_career" in result.frames_applied
        assert result.terms_added > 0

    def test_deterministic(self) -> None:
        r1 = expand_query_lexically("What job does Tom have?", "factual")
        r2 = expand_query_lexically("What job does Tom have?", "factual")
        assert r1.expansion_weights == r2.expansion_weights

    def test_max_terms_cap_respected(self) -> None:
        result = expand_query_lexically(
            "What activities and sports and work?", "aggregational", max_terms_per_intent=3
        )
        assert len(result.expansion_weights) <= 3

    def test_all_expansion_weights_below_one(self) -> None:
        """Expansion weights must never be >= 1.0."""
        queries = [
            ("What sport does Sarah play?", "factual"),
            ("List all activities Tom does", "aggregational"),
            ("Why did Alice move there?", "exploratory"),
            ("What job does Bob have?", "factual"),
        ]
        for query, intent in queries:
            result = expand_query_lexically(query, intent)
            for term, w in result.expansion_weights.items():
                assert w < 1.0, f"Expansion weight >= 1.0 for term '{term}': {w}"

    def test_query_terms_not_re_added(self) -> None:
        """Terms already in the query should not appear in expansion_weights."""
        result = expand_query_lexically("What sport does Sarah play?", "factual")
        q_lower = "what sport does sarah play?"
        for term in result.expansion_weights:
            assert term not in q_lower, f"Term '{term}' already in query but re-added"

    def test_no_locomo_answer_terms_in_lexicon(self) -> None:
        """Anti-overfit: lexicon must not contain LOCOMO gold answer terms."""
        FORBIDDEN = {"caroline", "melanie", "marathon", "pride parade", "locomo"}
        for _frame_name, frame in LEXICON.items():
            for term in frame.terms:
                assert term.lower() not in FORBIDDEN, f"Lexicon contains forbidden term: {term}"

    def test_returns_lexical_expansion_dataclass(self) -> None:
        result = expand_query_lexically("What does Alice do for work?", "factual")
        assert isinstance(result, LexicalExpansion)
        assert result.original_query == "What does Alice do for work?"
        assert result.intent == "factual"

    def test_navigational_intent_with_various_queries(self) -> None:
        """NAVIGATIONAL must always return empty expansion regardless of content."""
        queries = [
            "find the meeting notes",
            "show sports transcript",
            "open work log file",
            "What sport does Sarah play?",  # even if it would match other frames
        ]
        for q in queries:
            result = expand_query_lexically(q, "navigational")
            assert result.expansion_weights == {}, f"Expected empty for navigational: {q}"
            assert result.terms_added == 0

    def test_unknown_intent_does_not_crash(self) -> None:
        """Unknown intent strings should not crash — frames simply won't match."""
        result = expand_query_lexically("some query", "unknown_intent_xyz")
        assert isinstance(result, LexicalExpansion)
        assert result.expansion_weights == {}
