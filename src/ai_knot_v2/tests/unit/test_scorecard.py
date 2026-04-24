"""Sprint 6 — Scorecard unit tests.

Verifies all 8 metrics compute correctly on synthetic data.
No LOCOMO data.
"""

from __future__ import annotations

import dataclasses

from ai_knot_v2.bench.scorecard import (
    Scorecard,
    compute_context_dilution_rate,
    compute_dependency_closure_recall,
    compute_evidence_utility_density,
    compute_gold_evidence_coverage,
    compute_noise_atom_ratio,
    compute_required_atom_recall,
    compute_scorecard,
    compute_temporal_validity_accuracy,
    compute_unsafe_omission_rate,
)
from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.types import ReaderBudget


def _atom(
    risk_class: str = "preference",
    risk_severity: float = 0.2,
    valid_from: int | None = None,
    valid_until: int | None = None,
    depends_on: tuple[str, ...] = (),
) -> MemoryAtom:
    uid = new_ulid()
    ep_id = new_ulid()
    return MemoryAtom(
        atom_id=uid,
        agent_id="agent-1",
        user_id="user-1",
        variables=("user_1",),
        causal_graph=(),
        kernel_kind="point",
        kernel_payload={},
        intervention_domain=("user_1",),
        predicate="is",
        subject="alice",
        object_value="doctor",
        polarity="pos",
        valid_from=valid_from,
        valid_until=valid_until,
        observation_time=1_700_000_000,
        belief_time=1_700_000_000,
        granularity="instant",  # type: ignore[arg-type]
        entity_orbit_id="entity:alice",
        transport_provenance=("session-1",),
        depends_on=depends_on,
        depended_by=(),
        risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=risk_severity,
        regret_charge=risk_severity,
        irreducibility_score=1.0,
        protection_energy=risk_severity * 2,
        action_affect_mask=0,
        credence=0.9,
        evidence_episodes=(ep_id,),
        synthesis_method="regex",
        validation_tests=(),
        contradiction_events=(),
    )


_BUDGET = ReaderBudget(max_atoms=10, max_tokens=500, require_dependency_closure=True)


class TestRequiredAtomRecall:
    def test_all_gold_in_result_is_1(self) -> None:
        a = _atom()
        assert compute_required_atom_recall([a], {a.atom_id}) == 1.0

    def test_none_gold_in_result_is_0(self) -> None:
        a = _atom()
        b = _atom()
        assert compute_required_atom_recall([a], {b.atom_id}) == 0.0

    def test_partial_gold(self) -> None:
        a, b, c = _atom(), _atom(), _atom()
        score = compute_required_atom_recall([a, b], {a.atom_id, b.atom_id, c.atom_id})
        assert abs(score - 2 / 3) < 1e-9

    def test_empty_gold_is_1(self) -> None:
        assert compute_required_atom_recall([_atom()], set()) == 1.0

    def test_empty_result_with_gold_is_0(self) -> None:
        a = _atom()
        assert compute_required_atom_recall([], {a.atom_id}) == 0.0


class TestGoldEvidenceCoverage:
    def test_all_episodes_covered(self) -> None:
        a = _atom()
        ep_id = a.evidence_episodes[0]
        assert compute_gold_evidence_coverage([a], {ep_id}) == 1.0

    def test_no_episodes_covered(self) -> None:
        a = _atom()
        assert compute_gold_evidence_coverage([a], {"fake-episode-id"}) == 0.0

    def test_empty_gold_is_1(self) -> None:
        assert compute_gold_evidence_coverage([_atom()], set()) == 1.0


class TestDependencyClosureRecall:
    def test_no_deps_is_1(self) -> None:
        a = _atom()
        assert compute_dependency_closure_recall([a], [a]) == 1.0

    def test_dep_present_is_1(self) -> None:
        dep = _atom()
        main = dataclasses.replace(_atom(), depends_on=(dep.atom_id,))
        assert compute_dependency_closure_recall([main, dep], [main, dep]) == 1.0

    def test_dep_missing_is_0(self) -> None:
        dep = _atom()
        main = dataclasses.replace(_atom(), depends_on=(dep.atom_id,))
        # dep is in all_atoms but NOT in result_atoms
        score = compute_dependency_closure_recall([main], [main, dep])
        assert score == 0.0


class TestTemporalValidityAccuracy:
    def test_no_temporal_is_1(self) -> None:
        a = _atom(valid_from=None, valid_until=None)
        assert compute_temporal_validity_accuracy([a]) == 1.0

    def test_valid_interval_is_1(self) -> None:
        a = _atom(valid_from=1000, valid_until=2000)
        assert compute_temporal_validity_accuracy([a], query_time=1500) == 1.0

    def test_inverted_interval_is_0(self) -> None:
        a = _atom(valid_from=2000, valid_until=1000)
        assert compute_temporal_validity_accuracy([a]) == 0.0

    def test_future_atom_with_query_time_is_0(self) -> None:
        a = _atom(valid_from=5000, valid_until=6000)
        assert compute_temporal_validity_accuracy([a], query_time=1000) == 0.0

    def test_empty_result_is_1(self) -> None:
        assert compute_temporal_validity_accuracy([]) == 1.0


class TestContextDilutionRate:
    def test_all_gold_no_dilution(self) -> None:
        a = _atom()
        assert compute_context_dilution_rate([a], {a.atom_id}) == 0.0

    def test_half_noise(self) -> None:
        a, b = _atom(), _atom()
        rate = compute_context_dilution_rate([a, b], {a.atom_id})
        assert abs(rate - 0.5) < 1e-9

    def test_all_noise(self) -> None:
        a = _atom()
        b = _atom()
        assert compute_context_dilution_rate([a], {b.atom_id}) == 1.0

    def test_empty_result_no_dilution(self) -> None:
        assert compute_context_dilution_rate([], {new_ulid()}) == 0.0

    def test_ambient_fallback_when_no_gold(self) -> None:
        a = _atom(risk_class="ambient")
        b = _atom(risk_class="medical")
        rate = compute_context_dilution_rate([a, b], set())
        assert abs(rate - 0.5) < 1e-9


class TestUnsafeOmissionRate:
    def test_no_high_risk_is_0(self) -> None:
        low = _atom(risk_severity=0.2)
        assert compute_unsafe_omission_rate([low], [low]) == 0.0

    def test_high_risk_present_is_0(self) -> None:
        high = _atom(risk_severity=0.9)
        assert compute_unsafe_omission_rate([high], [high]) == 0.0

    def test_high_risk_missing_is_1(self) -> None:
        high = _atom(risk_severity=0.9)
        low = _atom(risk_severity=0.1)
        rate = compute_unsafe_omission_rate([low], [high, low])
        assert rate == 1.0

    def test_partial_omission(self) -> None:
        h1, h2 = _atom(risk_severity=0.9), _atom(risk_severity=0.8)
        rate = compute_unsafe_omission_rate([h1], [h1, h2])
        assert abs(rate - 0.5) < 1e-9


class TestEvidenceUtilityDensity:
    def test_empty_result_is_0(self) -> None:
        assert compute_evidence_utility_density([], "query", _BUDGET) == 0.0

    def test_nonzero_for_atoms(self) -> None:
        a = _atom(risk_severity=0.5)
        density = compute_evidence_utility_density([a], "query", _BUDGET)
        assert density > 0.0

    def test_returns_float(self) -> None:
        assert isinstance(compute_evidence_utility_density([_atom()], "query", _BUDGET), float)


class TestNoiseAtomRatio:
    def test_all_gold_no_noise(self) -> None:
        a = _atom()
        assert compute_noise_atom_ratio([a], {a.atom_id}) == 0.0

    def test_all_noise(self) -> None:
        a, b = _atom(), _atom()
        assert compute_noise_atom_ratio([a], {b.atom_id}) == 1.0

    def test_empty_library_is_0(self) -> None:
        assert compute_noise_atom_ratio([], {new_ulid()}) == 0.0


class TestComputeScorecard:
    def test_returns_scorecard(self) -> None:
        a = _atom(risk_severity=0.5)
        sc = compute_scorecard([a], [a], "query", _BUDGET)
        assert isinstance(sc, Scorecard)

    def test_all_metrics_present(self) -> None:
        a = _atom()
        sc = compute_scorecard([a], [a], "query", _BUDGET)
        summary = sc.summary()
        assert len(summary) == 8

    def test_perfect_recall_passes_gate(self) -> None:
        sc = Scorecard(
            required_atom_recall=1.0,
            gold_evidence_coverage=1.0,
            dependency_closure_recall=1.0,
            temporal_validity_accuracy=1.0,
            context_dilution_rate=0.0,
            unsafe_omission_rate=0.0,
            evidence_utility_density=0.5,
            noise_atom_ratio=0.0,
        )
        assert sc.passes_gate() is True

    def test_zero_recall_fails_gate(self) -> None:
        sc = Scorecard(
            required_atom_recall=0.0,
            gold_evidence_coverage=0.0,
            dependency_closure_recall=0.0,
            temporal_validity_accuracy=1.0,
            context_dilution_rate=1.0,
            unsafe_omission_rate=1.0,
            evidence_utility_density=0.0,
            noise_atom_ratio=1.0,
        )
        assert sc.passes_gate() is False

    def test_regression_vs_baseline_fails_gate(self) -> None:
        baseline = Scorecard(
            required_atom_recall=0.8,
            gold_evidence_coverage=0.8,
            dependency_closure_recall=0.9,
            temporal_validity_accuracy=1.0,
            context_dilution_rate=0.1,
            unsafe_omission_rate=0.1,
            evidence_utility_density=0.5,
            noise_atom_ratio=0.1,
        )
        regressed = Scorecard(
            required_atom_recall=0.7,  # −0.1 > threshold
            gold_evidence_coverage=0.8,
            dependency_closure_recall=0.9,
            temporal_validity_accuracy=1.0,
            context_dilution_rate=0.1,
            unsafe_omission_rate=0.1,
            evidence_utility_density=0.5,
            noise_atom_ratio=0.1,
        )
        assert regressed.passes_gate(baseline) is False
