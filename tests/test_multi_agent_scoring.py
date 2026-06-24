"""Tests for multi_agent.scoring — specificity and near-miss detection."""

from __future__ import annotations

from ai_knot.multi_agent.scoring import DiversityPolicy, NearMissDetector, SpecificityScorer
from ai_knot.types import Fact


def _fact(content: str, *, slot_key: str = "", snippets: list[str] | None = None) -> Fact:
    f = Fact(content=content)
    f.slot_key = slot_key
    if snippets:
        f.source_snippets = snippets
    return f


class TestSpecificityScorer:
    def setup_method(self) -> None:
        self.scorer = SpecificityScorer()

    def test_technical_fact_high_score(self) -> None:
        fact = _fact(
            "The cryptographic-protocols module implements homomorphic "
            "lattice reduction over ring-LWE enabling encrypted computation."
        )
        score = self.scorer.score(fact)
        assert score > 0.5

    def test_technical_beats_generic(self) -> None:
        technical = _fact(
            "The cryptographic-protocols module implements homomorphic "
            "lattice reduction over ring-LWE enabling encrypted computation.",
            slot_key="crypto::lattice_reduction",
        )
        generic = _fact(
            "Standard operations require regular performance monitoring "
            "and quarterly security audits."
        )
        # Slot-addressed technical fact should score higher.
        assert self.scorer.score(technical) > self.scorer.score(generic)

    def test_slot_bonus(self) -> None:
        fact_no_slot = _fact("User prefers Python for backend development.")
        fact_with_slot = _fact(
            "User prefers Python for backend development.",
            slot_key="user::language_preference",
        )
        assert self.scorer.score(fact_with_slot) > self.scorer.score(fact_no_slot)

    def test_empty_content(self) -> None:
        assert self.scorer.score(_fact("")) == 0.0


class TestNearMissDetector:
    def setup_method(self) -> None:
        self.detector = NearMissDetector()

    def test_overview_fact_penalised(self) -> None:
        fact = _fact(
            "A cryptographic overview covers encryption and decryption "
            "at a conceptual level without implementation specifics."
        )
        penalty = self.detector.penalty(fact)
        assert penalty >= 0.3

    def test_specific_fact_low_penalty(self) -> None:
        fact = _fact(
            "The cryptographic-protocols module Zeph0001 implements "
            "homomorphic lattice reduction over ring-LWE enabling "
            "encrypted computation with adaptive convergence tuning."
        )
        penalty = self.detector.penalty(fact)
        assert penalty < 0.3

    def test_penalty_capped_at_07(self) -> None:
        fact = _fact("A general introduction overview at a conceptual level.")
        penalty = self.detector.penalty(fact)
        assert penalty <= 0.7

    def test_no_penalty_for_normal_fact(self) -> None:
        fact = _fact("PostgreSQL supports JSONB indexing for semi-structured data.")
        penalty = self.detector.penalty(fact)
        assert penalty < 0.2


class TestDiversityPolicy:
    def setup_method(self) -> None:
        self.policy = DiversityPolicy()

    def test_per_agent_cap_three_publishers(self) -> None:
        cap = self.policy.per_agent_cap(top_k=10, n_publishers=3)
        assert 2 <= cap <= 5

    def test_per_agent_cap_single_publisher(self) -> None:
        cap = self.policy.per_agent_cap(top_k=10, n_publishers=1)
        assert cap == 10

    def test_per_domain_cap(self) -> None:
        cap = self.policy.per_domain_cap(top_k=10, n_facets=3)
        assert 2 <= cap <= 5
