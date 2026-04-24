"""Multi-metric scorecard for ai-knot v2.

8 internal metrics computed from a single recall result against ground truth.
All metrics are deterministic; no LLM judge in this module.
"""

from __future__ import annotations

import dataclasses

from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.types import ReaderBudget
from ai_knot_v2.ops.planner import reader_cost, utility


@dataclasses.dataclass(frozen=True, slots=True)
class Scorecard:
    """8-metric scorecard for a single recall result.

    All values ∈ [0.0, 1.0] unless noted.
    Higher is better for all metrics except ContextDilutionRate and NoiseAtomRatio.
    """

    # 1. Required atom recall (are gold atoms in the pack?)
    required_atom_recall: float  # ∈ [0, 1] ↑

    # 2. Gold evidence coverage (fraction of gold spans covered)
    gold_evidence_coverage: float  # ∈ [0, 1] ↑

    # 3. Dependency closure recall (are all deps of selected atoms retrieved?)
    dependency_closure_recall: float  # ∈ [0, 1] ↑

    # 4. Temporal validity accuracy (correct valid_from/valid_until)
    temporal_validity_accuracy: float  # ∈ [0, 1] ↑

    # 5. Context dilution rate (fraction of noise atoms in pack)
    context_dilution_rate: float  # ∈ [0, 1] ↓ lower is better

    # 6. Unsafe omission rate (fraction of high-risk atoms missed)
    unsafe_omission_rate: float  # ∈ [0, 1] ↓ lower is better

    # 7. Evidence utility density (utility / total_tokens)
    evidence_utility_density: float  # ≥ 0 ↑

    # 8. Noise atom ratio (noise / signal in full library)
    noise_atom_ratio: float  # ∈ [0, 1] ↓ lower is better

    def passes_gate(
        self,
        baseline: Scorecard | None = None,
    ) -> bool:
        """Multi-metric gate: accept change only if all conditions hold.

        If baseline is None, checks absolute thresholds only.
        If baseline is provided, also checks monotonic conditions vs. baseline.
        """
        # Absolute thresholds (minimum acceptable)
        if self.required_atom_recall < 0.3:
            return False
        if self.unsafe_omission_rate > 0.5:
            return False

        if baseline is None:
            return True

        # Monotonic conditions vs baseline
        if self.required_atom_recall < baseline.required_atom_recall - 0.02:
            return False
        if self.gold_evidence_coverage < baseline.gold_evidence_coverage - 0.02:
            return False
        if self.context_dilution_rate > baseline.context_dilution_rate + 0.05:
            return False
        if self.unsafe_omission_rate > baseline.unsafe_omission_rate + 0.02:
            return False
        return self.dependency_closure_recall >= baseline.dependency_closure_recall - 0.02

    def summary(self) -> dict[str, float]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_required_atom_recall(
    result_atoms: list[MemoryAtom],
    gold_atom_ids: set[str],
) -> float:
    """Fraction of gold atoms present in result_atoms."""
    if not gold_atom_ids:
        return 1.0
    found = sum(1 for a in result_atoms if a.atom_id in gold_atom_ids)
    return found / len(gold_atom_ids)


def compute_gold_evidence_coverage(
    result_atoms: list[MemoryAtom],
    gold_episode_ids: set[str],
) -> float:
    """Fraction of gold episode IDs covered by any result atom's evidence_episodes."""
    if not gold_episode_ids:
        return 1.0
    covered: set[str] = set()
    for atom in result_atoms:
        covered.update(atom.evidence_episodes)
    return len(covered & gold_episode_ids) / len(gold_episode_ids)


def compute_dependency_closure_recall(
    result_atoms: list[MemoryAtom],
    all_atoms: list[MemoryAtom],
) -> float:
    """Fraction of required dependencies present in result."""
    result_ids = {a.atom_id for a in result_atoms}
    all_atom_map = {a.atom_id: a for a in all_atoms}

    required_deps: set[str] = set()
    for atom in result_atoms:
        for dep_id in atom.depends_on:
            if dep_id in all_atom_map:
                required_deps.add(dep_id)

    if not required_deps:
        return 1.0

    found = sum(1 for dep_id in required_deps if dep_id in result_ids)
    return found / len(required_deps)


def compute_temporal_validity_accuracy(
    result_atoms: list[MemoryAtom],
    query_time: int | None = None,
) -> float:
    """Fraction of atoms with temporally valid (non-contradictory) intervals.

    An atom is temporally valid if:
    - valid_from is None OR valid_from <= query_time (if provided)
    - valid_until is None OR valid_until >= query_time (if provided)
    - If both are set: valid_from <= valid_until

    If query_time is None, only checks interval consistency.
    """
    if not result_atoms:
        return 1.0

    valid_count = 0
    for atom in result_atoms:
        vf, vu = atom.valid_from, atom.valid_until
        # Interval consistency
        if vf is not None and vu is not None and vf > vu:
            continue
        # Query time containment
        if query_time is not None:
            if vf is not None and query_time < vf:
                continue
            if vu is not None and query_time > vu:
                continue
        valid_count += 1

    return valid_count / len(result_atoms)


def compute_context_dilution_rate(
    result_atoms: list[MemoryAtom],
    gold_atom_ids: set[str],
    relevant_risk_classes: set[str] | None = None,
) -> float:
    """Fraction of result atoms that are noise (not gold and low-relevance).

    If gold_atom_ids is empty, uses risk-class heuristic: ambient/low-risk = noise.
    """
    if not result_atoms:
        return 0.0

    if gold_atom_ids:
        noise = sum(1 for a in result_atoms if a.atom_id not in gold_atom_ids)
        return noise / len(result_atoms)

    # Fallback: ambient atoms are noise
    noise = sum(1 for a in result_atoms if a.risk_class == "ambient")
    return noise / len(result_atoms)


def compute_unsafe_omission_rate(
    result_atoms: list[MemoryAtom],
    all_atoms: list[MemoryAtom],
    high_risk_threshold: float = 0.7,
) -> float:
    """Fraction of high-risk library atoms missing from result.

    High-risk: risk_severity >= high_risk_threshold.
    """
    high_risk = [a for a in all_atoms if a.risk_severity >= high_risk_threshold]
    if not high_risk:
        return 0.0

    result_ids = {a.atom_id for a in result_atoms}
    missed = sum(1 for a in high_risk if a.atom_id not in result_ids)
    return missed / len(high_risk)


def compute_evidence_utility_density(
    result_atoms: list[MemoryAtom],
    query: str,
    budget: ReaderBudget,
) -> float:
    """Sum of utilities / total tokens used.

    Returns 0.0 if pack is empty or uses 0 tokens.
    """
    if not result_atoms:
        return 0.0

    total_utility = sum(utility(a, query, result_atoms) for a in result_atoms)
    total_tokens = sum(reader_cost(a) for a in result_atoms)
    if total_tokens == 0:
        return 0.0

    return total_utility / total_tokens


def compute_noise_atom_ratio(
    all_atoms: list[MemoryAtom],
    gold_atom_ids: set[str],
) -> float:
    """Fraction of library atoms that are noise (not in gold set).

    If gold is empty, uses ambient risk class as proxy for noise.
    """
    if not all_atoms:
        return 0.0

    if gold_atom_ids:
        noise = sum(1 for a in all_atoms if a.atom_id not in gold_atom_ids)
        return noise / len(all_atoms)

    noise = sum(1 for a in all_atoms if a.risk_class == "ambient")
    return noise / len(all_atoms)


# ---------------------------------------------------------------------------
# Full scorecard computation
# ---------------------------------------------------------------------------


def compute_scorecard(
    result_atoms: list[MemoryAtom],
    all_atoms: list[MemoryAtom],
    query: str,
    budget: ReaderBudget,
    gold_atom_ids: set[str] | None = None,
    gold_episode_ids: set[str] | None = None,
    query_time: int | None = None,
) -> Scorecard:
    """Compute all 8 metrics from a recall result and ground truth."""
    gold_atoms = gold_atom_ids or set()
    gold_episodes = gold_episode_ids or set()

    return Scorecard(
        required_atom_recall=compute_required_atom_recall(result_atoms, gold_atoms),
        gold_evidence_coverage=compute_gold_evidence_coverage(result_atoms, gold_episodes),
        dependency_closure_recall=compute_dependency_closure_recall(result_atoms, all_atoms),
        temporal_validity_accuracy=compute_temporal_validity_accuracy(result_atoms, query_time),
        context_dilution_rate=compute_context_dilution_rate(result_atoms, gold_atoms),
        unsafe_omission_rate=compute_unsafe_omission_rate(result_atoms, all_atoms),
        evidence_utility_density=compute_evidence_utility_density(result_atoms, query, budget),
        noise_atom_ratio=compute_noise_atom_ratio(all_atoms, gold_atoms),
    )
