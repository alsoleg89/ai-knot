"""READ operation: query → intervention → candidates → planner → RecallResult.

Pipeline: extract_intervention → select_candidates (submodular-greedy) →
          plan_evidence_pack (planner) → build RecallResult.

No LLM. All selection is deterministic rule-based.
"""

from __future__ import annotations

import re

from ai_knot_v2.core.action_calculus import (
    action_distance,
    canonical_action_signature,
    compute_action_affect_mask,
    predict_action,
)
from ai_knot_v2.core.action_taxonomy import ActionClass
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.library import AtomLibrary
from ai_knot_v2.core.types import (
    Intervention,
    ReaderBudget,
    RecallResult,
)

# ---------------------------------------------------------------------------
# Default budget
# ---------------------------------------------------------------------------

DEFAULT_BUDGET = ReaderBudget(
    max_atoms=20,
    max_tokens=2000,
    require_dependency_closure=True,
)

# ---------------------------------------------------------------------------
# Intervention extraction
# ---------------------------------------------------------------------------

# Keyword → variable (causal do-calculus variable name)
_INTERVENTION_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmedical|doctor|diagnosis|symptom|medication|treatment\b", re.I), "health"),
    (re.compile(r"\bschedule|appointment|meeting|event|calendar\b", re.I), "schedule"),
    (re.compile(r"\bprefer|like|enjoy|want|favorite|interest\b", re.I), "preference"),
    (re.compile(r"\bname|age|address|identity|lives|occupation\b", re.I), "identity"),
    (re.compile(r"\bsalary|income|budget|earn|money|finance\b", re.I), "finance"),
    (re.compile(r"\bpromise|commit|guarantee|agreement|contract\b", re.I), "commitment"),
    (re.compile(r"\bdanger|hazard|emergency|unsafe|accident\b", re.I), "safety"),
]


def extract_intervention(query: str) -> tuple[Intervention, ActionClass]:
    """Extract a do-calculus intervention variable and predicted action from a query.

    Returns (Intervention(variable, value), predicted_action_class).
    Intervention.value is the normalized query text.
    """
    q_lower = query.lower().strip()
    variable = "general"
    for pattern, var in _INTERVENTION_KEYWORDS:
        if pattern.search(q_lower):
            variable = var
            break

    # Build signature for action prediction (no atoms available here)

    sig = canonical_action_signature([], query)
    predicted = predict_action(sig)

    return Intervention(variable=variable, value=q_lower), predicted


# ---------------------------------------------------------------------------
# Candidate selection (submodular-greedy with action_distance diversity)
# ---------------------------------------------------------------------------

_MIN_DIVERSITY_EPSILON = 0.1  # Minimum action_distance to add a diverse atom


def select_candidates(
    library: AtomLibrary,
    intervention: Intervention,
    query: str,
    budget: ReaderBudget,
) -> list[MemoryAtom]:
    """Select atoms from library via submodular-greedy diversity selection.

    1. Score atoms by relevance to intervention variable (risk_class match).
    2. Greedily add atoms that maximize marginal utility (relevance + diversity).
    3. Stop when max_atoms reached.
    """
    # Map intervention variable → relevant risk classes
    _VAR_TO_RISK: dict[str, list[str]] = {
        "health": ["medical", "safety"],
        "schedule": ["scheduling", "commitment"],
        "preference": ["preference"],
        "identity": ["identity"],
        "finance": ["finance"],
        "commitment": ["commitment", "legal"],
        "safety": ["safety", "medical"],
        "general": [
            "medical",
            "scheduling",
            "preference",
            "identity",
            "finance",
            "commitment",
            "safety",
            "legal",
            "ambient",
        ],
    }

    relevant_risk = _VAR_TO_RISK.get(intervention.variable, ["ambient"])

    # Gather all atoms from library, score by risk class match
    all_atoms = library.all_atoms()
    if not all_atoms:
        return []

    q_lower = query.lower()
    scored: list[tuple[float, MemoryAtom]] = []
    for atom in all_atoms:
        base_score = 0.0
        if atom.risk_class in relevant_risk:
            base_score += 1.0
        # Text overlap bonus: subject, object, AND predicate
        obj = (atom.object_value or "").lower()
        subj = (atom.subject or "").lower()
        pred = atom.predicate.replace("_", " ")
        words = {w for w in q_lower.split() if len(w) > 3}
        obj_words = {w for w in obj.split() if len(w) > 3}
        subj_words = {w for w in subj.split() if len(w) > 3}
        pred_words = {w for w in pred.split() if len(w) > 3}
        overlap = len(words & (obj_words | subj_words | pred_words))
        base_score += overlap * 0.3
        # High-risk atoms get priority
        base_score += atom.risk_severity * 0.5
        scored.append((base_score, atom))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Submodular-greedy: add atoms with diversity constraint
    selected: list[MemoryAtom] = []
    selected_masks: list[int] = []

    for _, atom in scored:
        if len(selected) >= budget.max_atoms:
            break

        atom_mask = compute_action_affect_mask(atom)

        if not selected:
            selected.append(atom)
            selected_masks.append(atom_mask)
            continue

        # Compute minimum distance to already-selected atoms
        min_dist = min(action_distance(atom_mask, m) for m in selected_masks)

        # Accept if diverse enough OR if mask is zero (ambient facts always pass)
        if min_dist >= _MIN_DIVERSITY_EPSILON or atom_mask == 0:
            selected.append(atom)
            selected_masks.append(atom_mask)

    return selected


# ---------------------------------------------------------------------------
# Main recall function
# ---------------------------------------------------------------------------


def recall(
    query: str,
    library: AtomLibrary,
    budget: ReaderBudget | None = None,
) -> RecallResult:
    """Full read path: query → intervention → select → planner → RecallResult."""
    from ai_knot_v2.ops.planner import plan_evidence_pack

    if budget is None:
        budget = DEFAULT_BUDGET

    intervention, _action = extract_intervention(query)
    candidates = select_candidates(library, intervention, query, budget)

    # Evidence planner: greedy-utility selection with contradiction resolution
    pack = plan_evidence_pack(candidates, query, budget, library)

    # Resolve atom objects from pack (after planner may have filtered some)
    pack_atom_ids = set(pack.atoms)
    result_atoms = [a for a in candidates if a.atom_id in pack_atom_ids]

    return RecallResult(
        atoms=result_atoms,
        evidence_pack_id=pack.pack_id,
        intervention=intervention,
    )
