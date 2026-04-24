"""Evidence Planner — greedy-utility selection within a bounded reader budget.

Core principle: maximize density of correct evidence per token spent.
No broad-context widening. No LLM calls.

Key functions:
- reader_cost(atom)            — token cost estimate for rendering this atom
- reduction_score(atom, ...)   — expected regret reduction from adding atom
- utility(atom, ...)           — reduction_score / reader_cost
- plan_evidence_pack(...)      — greedy selection with dependency-closure
- handle_contradictions(pack)  — sheaf-curvature detection → split or abstain
- temporal_allen_bonus(atom, query_vf, query_vu) — Allen-relation temporal match bonus
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.action_calculus import compute_action_affect_mask
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.temporal import AllenRelation, allen_relation, resolve_temporal
from ai_knot_v2.core.types import EvidencePack, ReaderBudget

# ---------------------------------------------------------------------------
# Token cost estimation
# ---------------------------------------------------------------------------

_BASE_TOKENS_PER_ATOM = 12  # predicate + subject + object in short form
_TOKENS_PER_CHAR = 0.25  # rough approximation


def reader_cost(atom: MemoryAtom) -> int:
    """Estimate token cost to render this atom in a context window.

    Formula: base + len(subject) * factor + len(object) * factor.
    Returns at least 1 token.
    """
    subj_chars = len(atom.subject or "")
    obj_chars = len(atom.object_value or "")
    return max(1, _BASE_TOKENS_PER_ATOM + int((subj_chars + obj_chars) * _TOKENS_PER_CHAR))


# ---------------------------------------------------------------------------
# Tri-temporal Allen-relation scoring (Sprint 11)
# ---------------------------------------------------------------------------

# Bonus for each Allen relation between atom valid-interval and query time window.
# Overlapping relations → high bonus; adjacent → small; disjoint → neutral (not penalized).
_ALLEN_BONUS: dict[AllenRelation, float] = {
    AllenRelation.EQUALS: 0.6,
    AllenRelation.DURING: 0.6,
    AllenRelation.CONTAINS: 0.5,
    AllenRelation.STARTS: 0.5,
    AllenRelation.FINISHES: 0.5,
    AllenRelation.STARTED_BY: 0.4,
    AllenRelation.FINISHED_BY: 0.4,
    AllenRelation.OVERLAPS: 0.4,
    AllenRelation.OVERLAPPED_BY: 0.4,
    AllenRelation.MEETS: 0.2,
    AllenRelation.MET_BY: 0.2,
    AllenRelation.PRECEDES: 0.0,
    AllenRelation.PRECEDED_BY: 0.0,
}

# Recency decay half-life in seconds (for observation_time axis).
# Atoms observed within 30 days get full bonus; score halves per half-life.
_RECENCY_HALFLIFE_SEC = 30 * 86400
_RECENCY_MAX_BONUS = 0.3


def temporal_allen_bonus(
    atom: MemoryAtom,
    query_vf: int | None,
    query_vu: int | None,
) -> float:
    """Compute tri-temporal bonus for an atom relative to a query time window.

    Three axes:
    1. Valid-time Allen relation: atom [valid_from, valid_until] vs query [vf, vu].
       Applied only when both atom and query have explicit intervals.
    2. Observation-time recency: exponential decay from now — prefers recently observed facts.
    3. Belief-time (future: currently equal to observation_time axis, reserved for Sprint 12).

    Returns additive bonus ∈ [0.0, 0.6].
    """
    bonus = 0.0

    # Axis 1: valid-time Allen relation
    if (
        query_vf is not None
        and query_vu is not None
        and atom.valid_from is not None
        and atom.valid_until is not None
    ):
        rel = allen_relation(atom.valid_from, atom.valid_until, query_vf, query_vu)
        bonus += _ALLEN_BONUS.get(rel, 0.0)

    # Axis 2: observation-time recency (exponential decay toward now)
    now = int(time.time())
    age_sec = max(0, now - atom.observation_time)
    import math

    decay = math.exp(-age_sec / _RECENCY_HALFLIFE_SEC)
    bonus += decay * _RECENCY_MAX_BONUS

    return bonus


def _query_temporal_window(query: str) -> tuple[int | None, int | None]:
    """Extract a temporal window from a query string.

    Uses resolve_temporal with today's date as reference.
    Returns (valid_from_epoch, valid_until_epoch), or (None, None) if no expression found.
    """
    vf, vu, _ = resolve_temporal(query, date.today())
    return vf, vu


# ---------------------------------------------------------------------------
# Reduction score (regret reduction heuristic)
# ---------------------------------------------------------------------------

_POLARITY_WEIGHT = {"pos": 1.0, "neg": 0.9}


def reduction_score(
    atom: MemoryAtom,
    query: str,
    current_pack: list[MemoryAtom],
    query_vf: int | None = None,
    query_vu: int | None = None,
) -> float:
    """Estimate how much adding this atom reduces expected answer regret.

    Heuristic components:
    1. Risk severity (high-risk facts reduce more regret when recalled)
    2. Text relevance (overlap of atom object/subject with query tokens)
    3. Action diversity (prefer atoms with new action coverage)
    4. Polarity correction (neg polarity slightly less confident)
    5. Credence weight
    6. Regret charge contribution
    7. Tri-temporal Allen-relation bonus (valid-time + recency)

    Returns score ∈ [0.0, ∞) (not normalized — higher is better).
    """
    score = 0.0

    # 1. Risk severity contribution
    score += atom.risk_severity * 2.0

    # 2. Text relevance (with possessive/punctuation stripping)
    import re as _re

    _strip = _re.compile(r"[?.!,;:\"']+$")
    _poss = _re.compile(r"'s?$|s'$")

    def _nw(w: str) -> str:
        w = _strip.sub("", w)
        w = _poss.sub("", w)
        return w.lower()

    q_words = {_nw(w) for w in query.split() if len(_nw(w)) > 3}
    obj_words = {w.lower() for w in (atom.object_value or "").split() if len(w) > 3}
    subj_words = {w.lower() for w in (atom.subject or "").split() if len(w) > 3}
    overlap = len(q_words & (obj_words | subj_words))
    score += overlap * 0.5

    # 3. Action diversity — reward covering new action bits
    if current_pack:
        current_coverage = 0
        for a in current_pack:
            current_coverage |= compute_action_affect_mask(a)
        atom_mask = compute_action_affect_mask(atom)
        new_bits = atom_mask & ~current_coverage
        score += bin(new_bits).count("1") * 0.3
    else:
        # First atom — full action coverage value
        atom_mask = compute_action_affect_mask(atom)
        score += bin(atom_mask).count("1") * 0.3

    # 4. Polarity weight
    score *= _POLARITY_WEIGHT.get(atom.polarity, 1.0)

    # 5. Credence weight
    score *= atom.credence

    # 6. Regret charge contribution (atoms with high regret_charge = more costly to omit)
    score += atom.regret_charge * 0.5

    # 7. Tri-temporal Allen-relation bonus
    score += temporal_allen_bonus(atom, query_vf, query_vu)

    return score


def utility(
    atom: MemoryAtom,
    query: str,
    current_pack: list[MemoryAtom],
    query_vf: int | None = None,
    query_vu: int | None = None,
) -> float:
    """Utility = reduction_score / reader_cost (information per token)."""
    cost = reader_cost(atom)
    return reduction_score(atom, query, current_pack, query_vf, query_vu) / cost


# ---------------------------------------------------------------------------
# Contradiction detection (sheaf curvature)
# ---------------------------------------------------------------------------

ContradictionPair = tuple[MemoryAtom, MemoryAtom]


def _atoms_contradict(a: MemoryAtom, b: MemoryAtom) -> bool:
    """Return True if two atoms assert contradictory claims about the same entity.

    Contradiction conditions (all must hold):
    1. Same entity orbit
    2. Same predicate
    3. Same subject (normalized)
    4. Different polarity OR different object_value for binary predicates
    """
    if a.entity_orbit_id != b.entity_orbit_id:
        return False
    if a.predicate != b.predicate:
        return False
    if (a.subject or "").lower() != (b.subject or "").lower():
        return False
    # Different polarity = direct contradiction
    if a.polarity != b.polarity:
        return True
    # Same polarity but different object on identity-type predicate = contradiction
    return a.predicate in ("is", "lives_in", "works_at") and a.object_value != b.object_value


def detect_contradictions(pack: list[MemoryAtom]) -> list[ContradictionPair]:
    """Return all pairs of contradicting atoms in pack."""
    pairs: list[ContradictionPair] = []
    for i, a in enumerate(pack):
        for b in pack[i + 1 :]:
            if _atoms_contradict(a, b):
                pairs.append((a, b))
    return pairs


def handle_contradictions(
    pack: list[MemoryAtom],
) -> tuple[list[MemoryAtom], list[str]]:
    """Apply sheaf-curvature contradiction resolution.

    Strategy:
    - For each contradicting pair: keep the higher-credence atom (split).
    - If credences are equal: remove both (safe-abstain).
    - Returns (resolved_pack, abstain_atom_ids).

    Never averages — always split or abstain.
    """
    contradictions = detect_contradictions(pack)
    if not contradictions:
        return pack, []

    remove_ids: set[str] = set()
    abstain_ids: list[str] = []

    for a, b in contradictions:
        if a.atom_id in remove_ids or b.atom_id in remove_ids:
            continue
        if a.credence > b.credence:
            remove_ids.add(b.atom_id)
        elif b.credence > a.credence:
            remove_ids.add(a.atom_id)
        else:
            # Equal credence — use observation_time as tiebreaker.
            # In temporal updates (job change, move, etc.) the later observation wins.
            # Only true simultaneous contradictions become abstains.
            if a.observation_time != b.observation_time:
                older = a if a.observation_time < b.observation_time else b
                remove_ids.add(older.atom_id)
            else:
                # Truly simultaneous contradiction → safe-abstain: remove both
                remove_ids.add(a.atom_id)
                remove_ids.add(b.atom_id)
                abstain_ids.extend([a.atom_id, b.atom_id])

    resolved = [a for a in pack if a.atom_id not in remove_ids]
    return resolved, abstain_ids


# ---------------------------------------------------------------------------
# Dependency closure within planner
# ---------------------------------------------------------------------------


def _close_dependencies(
    atoms: list[MemoryAtom],
    library: AtomLibrary,
    budget: ReaderBudget,
    token_budget: int,
) -> tuple[list[MemoryAtom], int]:
    """Add dependency atoms up to token budget. Returns (extended_list, tokens_used)."""
    atom_ids = {a.atom_id for a in atoms}
    result = list(atoms)
    tokens = token_budget

    for atom in list(atoms):
        for dep_id in atom.depends_on:
            if dep_id in atom_ids:
                continue
            dep = library.get(dep_id)
            if dep is None:
                continue
            cost = reader_cost(dep)
            if tokens - cost >= 0 and len(result) < budget.max_atoms:
                result.append(dep)
                atom_ids.add(dep_id)
                tokens -= cost

    return result, tokens


# ---------------------------------------------------------------------------
# Main: plan_evidence_pack
# ---------------------------------------------------------------------------


def plan_evidence_pack(
    atoms: list[MemoryAtom],
    query: str,
    budget: ReaderBudget,
    library: AtomLibrary | None = None,
) -> EvidencePack:
    """Greedy-utility selection within reader budget.

    1. Extract query temporal window (Allen-relation anchor).
    2. Score all atoms by utility(atom, query, current_pack, query_vf, query_vu).
    3. Greedily select highest-utility atom within token budget.
    4. Repeat until budget exhausted or no atoms left.
    5. Apply dependency closure (if library provided).
    6. Handle contradictions (split or abstain).
    7. Return EvidencePack with utility_scores metadata.
    """
    if not atoms:
        return EvidencePack(pack_id=new_ulid(), atoms=(), spans=())

    # Extract temporal anchor from query once (reused for all atoms)
    query_vf, query_vu = _query_temporal_window(query)

    token_budget = budget.max_tokens
    selected: list[MemoryAtom] = []
    remaining = list(atoms)
    utility_scores: dict[str, Any] = {}

    while remaining and len(selected) < budget.max_atoms and token_budget > 0:
        # Score all remaining atoms
        scored = [(utility(a, query, selected, query_vf, query_vu), a) for a in remaining]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_util, best_atom = scored[0]
        cost = reader_cost(best_atom)

        if cost > token_budget:
            # Skip if too expensive — try next
            skip_idx = next(
                (i for i, (_, a) in enumerate(scored) if reader_cost(a) <= token_budget),
                None,
            )
            if skip_idx is None:
                break
            best_util, best_atom = scored[skip_idx]
            cost = reader_cost(best_atom)

        selected.append(best_atom)
        utility_scores[best_atom.atom_id] = round(best_util, 4)
        token_budget -= cost
        remaining.remove(best_atom)

    # Dependency closure
    if library is not None and budget.require_dependency_closure:
        selected, _ = _close_dependencies(selected, library, budget, token_budget)

    # Contradiction resolution
    resolved, abstain_ids = handle_contradictions(selected)

    return EvidencePack(
        pack_id=new_ulid(),
        atoms=tuple(a.atom_id for a in resolved),
        spans=(),
        utility_scores={
            "atom_utilities": utility_scores,
            "abstain_atom_ids": abstain_ids,
            "tokens_used": budget.max_tokens - token_budget,
            "contradiction_count": len(abstain_ids) // 2,
        },
    )
