"""Tests for the ATC (Asymmetric Token Containment) verification guardrail."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_knot.extractor import Extractor, _atc_score, _verify_facts_atc
from ai_knot.types import ConversationTurn, Fact, MemoryType

# ---------------------------------------------------------------------------
# _atc_score unit tests
# ---------------------------------------------------------------------------


class TestAtcScore:
    """Unit tests for _atc_score."""

    def test_fully_contained(self) -> None:
        """All snippet tokens appear in source — score should be 1.0."""
        snippet = "user works at Sber"
        source = "The user works at Sber as Operations Director"
        assert _atc_score(snippet, source) == pytest.approx(1.0)

    def test_partial_containment(self) -> None:
        """Half the snippet tokens appear in source."""
        # snippet tokens (after tokenize): ["user", "like", "python"] — 3 tokens
        # source has "user" and "python" but not "like" → 2/3
        snippet = "user likes python"
        source = "user prefers python over java"
        score = _atc_score(snippet, source)
        assert 0.0 < score < 1.0

    def test_zero_containment(self) -> None:
        """No snippet tokens appear in source — score should be 0.0."""
        snippet = "quantum entanglement physics"
        source = "user deploys docker containers"
        assert _atc_score(snippet, source) == pytest.approx(0.0)

    def test_empty_snippet_returns_one(self) -> None:
        """Empty snippet is vacuously supported — should return 1.0."""
        assert _atc_score("", "any source text here") == pytest.approx(1.0)

    def test_empty_source_returns_zero(self) -> None:
        """Non-empty snippet against empty source — no tokens can match."""
        assert _atc_score("user works at Sber", "") == pytest.approx(0.0)

    def test_both_empty_returns_one(self) -> None:
        """Both empty — vacuously supported."""
        assert _atc_score("", "") == pytest.approx(1.0)

    def test_score_in_valid_range(self) -> None:
        """Score is always in [0.0, 1.0]."""
        score = _atc_score("the quick brown fox", "lazy dog jumps high")
        assert 0.0 <= score <= 1.0

    def test_asymmetry(self) -> None:
        """ATC is asymmetric: score(A, B) != score(B, A) in general."""
        a = "python"
        b = "python java ruby go"
        score_ab = _atc_score(a, b)
        score_ba = _atc_score(b, a)
        # score(a, b) should be 1.0 (single token "python" is in b)
        # score(b, a) should be 0.25 (only "python" of 4 tokens is in a)
        assert score_ab > score_ba


# ---------------------------------------------------------------------------
# _verify_facts_atc unit tests
# ---------------------------------------------------------------------------


class TestVerifyFactsAtc:
    """Unit tests for _verify_facts_atc."""

    def _make_fact(self, content: str) -> Fact:
        return Fact(content=content, type=MemoryType.SEMANTIC, importance=0.8)

    def test_sets_verification_source(self) -> None:
        """All facts should have verification_source set to 'atc'."""
        facts = [self._make_fact("user works at Sber")]
        source = "user works at Sber as Operations Director"
        _verify_facts_atc(facts, source)
        assert facts[0].verification_source == "atc"

    def test_supported_above_threshold(self) -> None:
        """Fact with high ATC score should be marked as supported."""
        facts = [self._make_fact("user works at Sber")]
        source = "user works at Sber as Operations Director"
        _verify_facts_atc(facts, source, threshold=0.6)
        assert facts[0].supported is True

    def test_unsupported_below_threshold(self) -> None:
        """Fact with low ATC score should be marked as not supported."""
        facts = [self._make_fact("quantum entanglement physics experiments")]
        source = "user deploys docker containers on kubernetes"
        _verify_facts_atc(facts, source, threshold=0.6)
        assert facts[0].supported is False

    def test_support_confidence_set(self) -> None:
        """support_confidence should be set to the ATC score."""
        facts = [self._make_fact("user deploys docker")]
        source = "user deploys docker containers"
        _verify_facts_atc(facts, source)
        score = _atc_score("user deploys docker", source)
        assert facts[0].support_confidence == pytest.approx(score)

    def test_modifies_in_place(self) -> None:
        """The function should return the same list object."""
        facts = [self._make_fact("user works at Sber")]
        source = "user works at Sber"
        result = _verify_facts_atc(facts, source)
        assert result is facts

    def test_empty_facts_list(self) -> None:
        """Empty facts list should return empty list without error."""
        result = _verify_facts_atc([], "some source text")
        assert result == []

    def test_multiple_facts_annotated(self) -> None:
        """All facts in the list should be annotated."""
        facts = [
            self._make_fact("user works at Sber"),
            self._make_fact("quantum entanglement mystery"),
        ]
        source = "user works at Sber as Operations Director"
        _verify_facts_atc(facts, source, threshold=0.6)
        assert all(f.verification_source == "atc" for f in facts)
        # First fact should be supported, second should not
        assert facts[0].supported is True
        assert facts[1].supported is False

    def test_threshold_boundary(self) -> None:
        """Fact with score exactly at threshold should be supported."""
        # Craft a snippet where we know the exact score
        snippet = "alpha beta"
        source = "alpha beta gamma delta"
        # tokenize("alpha beta") = ["alpha", "beta"] — 2 tokens, both in source
        # score = 1.0 >= any reasonable threshold
        facts = [self._make_fact(snippet)]
        _verify_facts_atc(facts, source, threshold=1.0)
        assert facts[0].supported is True


# ---------------------------------------------------------------------------
# Integration tests: ATC in the full extraction pipeline
# ---------------------------------------------------------------------------


class TestExtractorAtcIntegration:
    """Integration tests: extracted facts are ATC-verified."""

    MOCK_LLM_RESPONSE = [
        {"content": "User deploys in Docker", "type": "semantic", "importance": 0.8},
        {"content": "User dislikes async code", "type": "procedural", "importance": 0.85},
        {"content": "quantum entanglement vortex", "type": "semantic", "importance": 0.5},
    ]

    def _turns(self) -> list[ConversationTurn]:
        return [
            ConversationTurn(role="user", content="I deploy everything in Docker"),
            ConversationTurn(role="assistant", content="Got it"),
            ConversationTurn(role="user", content="I really dislike async code"),
        ]

    def test_facts_have_atc_verification_source(self) -> None:
        """All extracted facts should have verification_source == 'atc'."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        with patch.object(extractor, "_call_llm", return_value=self.MOCK_LLM_RESPONSE):
            facts = extractor.extract(self._turns())
        assert all(f.verification_source == "atc" for f in facts)

    def test_facts_have_support_confidence_set(self) -> None:
        """All extracted facts should have support_confidence set."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        with patch.object(extractor, "_call_llm", return_value=self.MOCK_LLM_RESPONSE):
            facts = extractor.extract(self._turns())
        # support_confidence should be a float in [0.0, 1.0]
        for fact in facts:
            assert 0.0 <= fact.support_confidence <= 1.0

    def test_supported_field_is_bool(self) -> None:
        """supported field should be a bool on all facts."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        with patch.object(extractor, "_call_llm", return_value=self.MOCK_LLM_RESPONSE):
            facts = extractor.extract(self._turns())
        assert all(isinstance(f.supported, bool) for f in facts)

    def test_supported_fact_detected(self) -> None:
        """A fact whose content appears in conversation text should be supported."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        # Use a response where the first fact is clearly supported by source
        llm_response = [
            {"content": "User deploys Docker", "type": "semantic", "importance": 0.9},
        ]
        with patch.object(extractor, "_call_llm", return_value=llm_response):
            facts = extractor.extract(self._turns())
        assert len(facts) == 1
        assert facts[0].supported is True

    def test_unsupported_fact_detected(self) -> None:
        """A hallucinated fact not in source should be marked unsupported."""
        extractor = Extractor(api_key="fake-key", provider="openai")
        llm_response = [
            {
                "content": "quantum entanglement vortex singularity",
                "type": "semantic",
                "importance": 0.5,
            },
        ]
        with patch.object(extractor, "_call_llm", return_value=llm_response):
            facts = extractor.extract(self._turns())
        assert len(facts) == 1
        assert facts[0].supported is False
