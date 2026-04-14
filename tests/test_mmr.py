"""Tests for KnowledgeBase._mmr_select — MMR diversity reranking."""

from __future__ import annotations

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact


def _pairs(specs: list[tuple[str, float]]) -> list[tuple[Fact, float]]:
    """Build (Fact, score) pairs from (content, score) specs."""
    return [(Fact(content=c), s) for c, s in specs]


class TestMMRSelect:
    """Unit tests for _mmr_select static method."""

    def test_exact_duplicates_collapsed(self) -> None:
        """Three identical facts → only one survives in top_k=2.

        Scores must be close (realistic BM25 spread) so the diversity
        penalty for a duplicate (max_sim=1.0) outweighs the relevance gap.
        With λ=0.5: MMR(dup) = 0.5*0.9 - 0.5*1.0 = -0.05
                    MMR(other) = 0.5*0.85 - 0.5*0 = 0.425 → other wins.
        """
        pairs = _pairs(
            [
                ("alpha bravo charlie delta", 1.0),
                ("alpha bravo charlie delta", 0.9),  # exact duplicate
                ("alpha bravo charlie delta", 0.85),  # exact duplicate
                ("echo foxtrot golf hotel", 0.85),  # different topic, same score range
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        assert len(result) == 2
        contents = [f.content for f, _ in result]
        # First = highest scorer
        assert contents[0] == "alpha bravo charlie delta"
        # Second = diverse fact, not another duplicate
        assert contents[1] == "echo foxtrot golf hotel"

    def test_near_duplicate_windows_collapsed(self) -> None:
        """Overlapping 3-turn windows (84% overlap) → representative + diverse fact."""
        pairs = _pairs(
            [
                # Window A: turns 0-2
                ("Melanie: I love pottery. Caroline: That's great. Melanie: I made bowls", 1.0),
                # Window B: turns 1-3 (84% overlap with A)
                ("Caroline: That's great. Melanie: I made bowls. Caroline: Wow", 0.9),
                # Window C: turns 2-4 (84% overlap with B)
                ("Melanie: I made bowls. Caroline: Wow. Melanie: went swimming today", 0.8),
                # Completely different topic
                ("Caroline visited the art museum last Tuesday", 0.6),
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        assert len(result) == 2
        # First selection is the highest-scored window
        assert "pottery" in result[0][0].content or "bowls" in result[0][0].content
        # Second selection should NOT be another pottery/bowls window
        second_content = result[1][0].content
        assert "museum" in second_content or "swimming" in second_content

    def test_point_query_unaffected(self) -> None:
        """Single highly-relevant fact must stay #1 regardless of diversity pressure."""
        pairs = _pairs(
            [
                ("Caroline's email is caroline@example.com", 1.0),
                ("Caroline joined the support group", 0.3),
                ("Caroline likes reading books", 0.2),
                ("Caroline lives in New York", 0.15),
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=3)
        assert result[0][0].content == "Caroline's email is caroline@example.com"

    def test_extracted_fact_not_suppressed_by_raw_turn(self) -> None:
        """Extracted fact (short) should NOT be penalized against raw turn (long).

        Jaccard("Melanie: swimming", raw_turn) ≈ 0.2 — low similarity.
        With Jaccard-only MMR: extracted survives as #1, raw turn is selected #2
        (higher BM25 score than 'other'), Caroline is left out.

        Regression for the containment-bias bug where
        containment(extracted ⊆ raw) = 1.0 would kill the extracted fact.
        """
        # Extracted: short, precise (2 meaningful tokens after stemming)
        extracted = ("Melanie: swimming", 1.0)
        # Raw turn: long, extracted tokens are a small subset (Jaccard ≈ 0.2)
        raw = (
            "[8 May, 2023] Melanie: I mentioned that I love swimming and pottery with friends",
            0.95,
        )
        # Different topic, lower score
        other = ("Caroline attended pride parade last June", 0.6)

        pairs = _pairs([extracted, raw, other])
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        assert len(result) == 2
        contents = [f.content for f, _ in result]
        # Extracted wins (highest score)
        assert contents[0] == extracted[0]
        # Raw is NOT treated as a duplicate (Jaccard ≈ 0.2 < 0.5):
        # it beats "other" on BM25 → raw is selected second, not Caroline
        assert "Melanie" in contents[1] or "swimming" in contents[1]

    def test_jaccard_only_not_containment(self) -> None:
        """Verify Jaccard is used, not containment.

        containment(ft ⊆ st) = inter/len(ft).
        For ft={"a","b"} and st={"a","b","c","d","e","f","g","h","i","j"}:
          containment = 2/2 = 1.0  (old buggy code would suppress ft)
          jaccard     = 2/10 = 0.2  (new code: ft is NOT a near-duplicate)

        With Jaccard-only, the short fact should survive alongside the long one.
        """
        short_fact = ("alpha beta", 1.0)  # tokens: {alpha, beta} — 2 tokens
        long_fact = (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            0.9,
        )  # 10 tokens, short_fact ⊆ long_fact (containment=1.0 but jaccard=0.2)
        unrelated = ("completely different topic here", 0.5)

        pairs = _pairs([short_fact, long_fact, unrelated])
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        contents = [f.content for f, _ in result]
        # short_fact is first (highest score)
        assert contents[0] == short_fact[0]
        # long_fact is second (NOT suppressed — Jaccard(short, long) = 0.2 is low)
        assert "gamma" in contents[1]

    def test_temporal_dates_best_stays_first(self) -> None:
        """Multiple date facts for different events → correct date must stay #1."""
        pairs = _pairs(
            [
                # Directly relevant: answers "When did Caroline go to book club?"
                ("Caroline attended the book club meeting on 7 May 2023", 1.0),
                # Other dated facts (different events, different dates)
                ("Melanie signed up for pottery class on 2 July 2023", 0.4),
                ("Caroline started therapy on 15 March 2023", 0.3),
                ("Melanie went to the beach on 20 August 2023", 0.25),
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=3)
        # The most relevant temporal fact must still be #1
        assert "7 May" in result[0][0].content

    def test_fewer_candidates_than_top_k_returns_all(self) -> None:
        """When candidates ≤ top_k, return all unchanged."""
        pairs = _pairs([("fact A", 1.0), ("fact B", 0.5)])
        result = KnowledgeBase._mmr_select(pairs, top_k=5)
        assert result == pairs

    def test_single_pair(self) -> None:
        """Single candidate returns as-is."""
        pairs = _pairs([("only fact", 1.0)])
        result = KnowledgeBase._mmr_select(pairs, top_k=1)
        assert len(result) == 1
        assert result[0][0].content == "only fact"

    def test_all_identical_scores(self) -> None:
        """All same score → diversity drives selection, no crash."""
        pairs = _pairs(
            [
                ("alpha beta gamma", 1.0),
                ("alpha beta delta", 1.0),  # overlaps with first
                ("epsilon zeta eta", 1.0),  # different
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        assert len(result) == 2
        # Second selection must be the non-overlapping fact
        assert result[1][0].content == "epsilon zeta eta"

    def test_date_prefix_stripped_before_comparison(self) -> None:
        """[8 May] prefix must not prevent dedup of otherwise-identical facts."""
        pairs = _pairs(
            [
                ("[8 May, 2023] Melanie went swimming with the kids", 1.0),
                ("[9 May, 2023] Melanie went swimming with the kids", 0.9),
                ("Caroline painted abstract art today", 0.5),
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        contents = [f.content for f, _ in result]
        # The two [date]-prefixed facts are identical after stripping → only one survives
        swimming_count = sum(1 for c in contents if "swimming" in c)
        assert swimming_count == 1
        assert any("Caroline" in c for c in contents)

    def test_original_scores_preserved(self) -> None:
        """MMR must return original RRF scores, not normalized ones."""
        pairs = _pairs(
            [
                ("fact alpha", 0.01666),  # typical RRF score
                ("fact beta", 0.00833),
                ("fact gamma gamma gamma", 0.004),
            ]
        )
        result = KnowledgeBase._mmr_select(pairs, top_k=2)
        assert result[0][1] == pytest.approx(0.01666)
        assert result[1][1] == pytest.approx(0.00833)
