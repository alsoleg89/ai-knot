# ESWP Level-3 Implementation Spec

Date: 2026-04-25
Status: executable specification for Stage 4 implementation
Related:
- `research/extraction_sufficient_witness_program.md` — theoretical justification
- `research/iwt_v2_implementation_audit_20260424.md` — IWT construct ↔ placeholder mapping
- `src/ai_knot_v2/CLAUDE.md` — architectural invariants

---

## Overview

Six discrete changes to bring ai-knot v2 from Sprint-1-placeholder to Level-3 ESWP.
Each step is a separate commit with its own test gate.

**Execution order** — by expected metric impact (highest first):

| Step | What changes | Gate metric | Risk |
|---|---|---|---|
| S1 | Reader-probe + render reordering (bench/) | cat1 ≥ 36% | Medium |
| S2 | ΔF-write regret_charge | CCB RWCA ≥ 0.70 on 20-history prototype | Medium |
| S3 | Sheaf-section gluing (planner) | ContextDilutionRate ↓ ≥ 15% | Medium |
| S4 | Landauer-ODE forget | Memory sublinear at 200+ sessions | Low |
| S5 | Bitemporal staleness bonus | Temporal questions +5pp | Low |
| S6 | RG-flow consolidation | Neutral on 2-conv, improves 10-conv | Low |

Stop rule: if any step drops 2-conv aggregate by >2pp → revert, do not stack.

---

## S1: Reader-Probe Validator + Render Reordering

**Goal:** Close ≥ 5 of 9 LLM-fail cat1 questions via pack reordering and extraction probe.
**Files:** 3 new + 1 modified. No core/ changes.

### S1a — Render-format reordering in bench/

**File:** `src/ai_knot_v2/bench/v2_locomo_runner.py`

Find where atoms are rendered to context string. Replace with regret_charge-sorted render:

```python
# BEFORE (approximate location — find the context assembly loop):
context_lines = [f"{a.subject} {a.predicate} {a.object_value}" for a in atoms]

# AFTER:
from ai_knot_v2.bench.ccb.render import render_pack_eswp
context_str = render_pack_eswp(atoms, query)
```

**New file:** `src/ai_knot_v2/bench/ccb/__init__.py` (empty)

**New file:** `src/ai_knot_v2/bench/ccb/render.py`

```python
"""ESWP render operator — sorts atoms by regret_charge × credence for extraction."""
from __future__ import annotations
from ai_knot_v2.core.atom import MemoryAtom


def render_pack_eswp(atoms: list[MemoryAtom], query: str) -> str:
    """Render atoms sorted by regret_charge × credence descending.

    Factual predicates rendered as assertions; event predicates with temporal tag.
    """
    FACTUAL_PREDS = frozenset(
        {"is", "has", "works_at", "lives_in", "prefers", "dislikes", "moved_to"}
    )
    ordered = sorted(atoms, key=lambda a: a.regret_charge * a.credence, reverse=True)
    lines: list[str] = []
    for atom in ordered:
        obj = atom.object_value or ""
        pred_display = atom.predicate.replace("_", " ")
        if atom.predicate in FACTUAL_PREDS:
            # Factual assertion: clear subject-predicate-object
            line = f"[fact] {atom.subject} {pred_display} {obj}"
        else:
            # Event with optional temporal window
            temp = ""
            if atom.valid_from is not None and atom.valid_until is not None:
                temp = f" (valid {atom.valid_from}–{atom.valid_until})"
            line = f"[event] {atom.subject} {pred_display} {obj}{temp}"
        if atom.polarity == "neg":
            line = line.replace("[fact]", "[neg-fact]", 1).replace("[event]", "[neg-event]", 1)
        lines.append(line)
    return "\n".join(lines)
```

### S1b — Extraction-sufficiency probe

**New file:** `src/ai_knot_v2/bench/ccb/probe.py`

```python
"""Extraction-sufficiency probe — bench/ only, never imported in core/ops/store/api/."""
from __future__ import annotations
from typing import Any
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.bench.ccb.render import render_pack_eswp


def token_f1(pred: str, gold: str) -> float:
    pred_toks = set(pred.lower().split())
    gold_toks = set(gold.lower().split())
    if not gold_toks:
        return 0.0
    prec = len(pred_toks & gold_toks) / max(1, len(pred_toks))
    rec = len(pred_toks & gold_toks) / len(gold_toks)
    denom = prec + rec
    return 2 * prec * rec / denom if denom > 0 else 0.0


def validate_extraction_sufficiency(
    atoms: list[MemoryAtom],
    query: str,
    reader: Any,
    expected_answer: str | None = None,
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Call reader on rendered pack; return (is_sufficient, f1_score).

    reader must have a .complete(system, user) -> str interface.
    This function must remain in bench/ — never call from core/.
    """
    rendered = render_pack_eswp(atoms, query)
    response = reader.complete(
        system="Answer the question from the context below. Be concise and direct.",
        user=f"Context:\n{rendered}\n\nQuestion: {query}",
    )

    if expected_answer is not None:
        score = token_f1(response, expected_answer)
    else:
        # Proxy: coverage of query content tokens in response
        q_words = {w.lower() for w in query.split() if len(w) > 3}
        r_words = {w.lower() for w in response.split()}
        score = len(q_words & r_words) / max(1, len(q_words))

    return score >= threshold, score
```

### S1 Gate

```bash
# Architecture invariant still holds
grep -r "openai\|anthropic\|gpt\|claude" src/ai_knot_v2/{core,ops,store,api}/ && echo FAIL || echo ok

# Unit tests
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_render.py -p no:cov -v

# 2-conv LOCOMO run
cd aiknotbench && npx tsx src/index.ts run -r s1-probe --top-k 60 --limit 2
# Accept: cat1 ≥ 36% (baseline 30.2%), cat1-4 aggregate not down >2pp
```

**New test:** `src/ai_knot_v2/tests/unit/test_render.py`

```python
from ai_knot_v2.bench.ccb.render import render_pack_eswp

def test_factual_pred_format(make_atom):
    atom = make_atom(predicate="is", object_value="doctor", risk_class="identity",
                     regret_charge=0.9, credence=1.0)
    rendered = render_pack_eswp([atom], "what does Alice do")
    assert "[fact]" in rendered
    assert "doctor" in rendered

def test_high_charge_first(make_atom):
    low = make_atom(regret_charge=0.1, credence=1.0, subject="Low", object_value="l")
    high = make_atom(regret_charge=0.9, credence=1.0, subject="High", object_value="h")
    rendered = render_pack_eswp([low, high], "query")
    assert rendered.index("High") < rendered.index("Low")

def test_no_llm_import():
    import ast, pathlib
    src = pathlib.Path("src/ai_knot_v2/bench/ccb/render.py").read_text()
    tree = ast.parse(src)
    imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
    assert not any("openai" in i or "anthropic" in i for i in imports)
```

---

## S2: ΔF-Write Regret Charge

**Goal:** Replace `regret_charge = risk_severity * 1.0` with action-diversity-weighted formula.
**Files:** 2 modified, 1 new.

### S2a — New helper module

**New file:** `src/ai_knot_v2/core/information.py`

```python
"""Information-theoretic helpers — no LLM, no external dependencies."""
from __future__ import annotations
from ai_knot_v2.core.atom import MemoryAtom


def compute_regret_charge_v2(atom: MemoryAtom, curvature: float = 0.0) -> float:
    """Compute regret charge as proxy for expected marginal free energy.

    Formula:
      delta_q_proxy = risk_severity × (1 + 0.3 × action_bits)
      curvature_term = 1 + 0.5 × curvature
      danger_term = 1 + 0.2 × risk_severity
      regret_charge = min(1.0, delta_q_proxy × curvature_term × danger_term)

    curvature = 1.0 if atom has contradiction events, else 0.0.
    """
    action_bits = bin(atom.action_affect_mask).count("1")
    delta_q = atom.risk_severity * (1.0 + 0.3 * action_bits)
    curvature_term = 1.0 + 0.5 * curvature
    danger_term = 1.0 + 0.2 * atom.risk_severity
    return min(1.0, delta_q * curvature_term * danger_term)


def compute_irreducibility(
    atom: MemoryAtom,
    peer_atoms: list[MemoryAtom],
) -> float:
    """Score how irreplaceable this atom is within its orbit.

    1.0 = no peers with same predicate → fully irreducible.
    Decreases with number of overlapping peers.
    """
    if not peer_atoms:
        return 1.0

    def _overlaps(a: MemoryAtom, b: MemoryAtom) -> bool:
        if a.valid_from is None or b.valid_from is None:
            return True  # untimed atoms are assumed overlapping
        if a.valid_until is None or b.valid_until is None:
            return True
        return a.valid_from <= b.valid_until and b.valid_from <= a.valid_until

    overlapping = sum(
        1 for p in peer_atoms
        if p.predicate == atom.predicate and _overlaps(atom, p)
    )
    redundancy = overlapping / (len(peer_atoms) + 1)
    return max(0.1, 1.0 - redundancy)
```

### S2b — Wire into atomizer

**File:** `src/ai_knot_v2/ops/atomizer.py`

Change lines 456–459 (protection_energy and regret_charge assignment):

```python
# BEFORE:
protection_energy = min(1.0, risk_severity * 2.0)
regret_charge = risk_severity * 1.0

# AFTER:
from ai_knot_v2.core.information import compute_regret_charge_v2
curvature = 0.0  # no contradiction events at write time (detected later)
regret_charge = compute_regret_charge_v2(
    # Build a minimal atom proxy for the formula (action_affect_mask computed below)
    # Use risk_severity and action_affect_mask=0 initially; updated post-construction
    MemoryAtom.__new__(MemoryAtom),  # placeholder — see note below
    curvature=curvature,
)
```

**Implementation note:** `compute_regret_charge_v2` needs `action_affect_mask` which is computed AFTER atom construction. Solution: compute `regret_charge` post-construction using a two-step approach:

```python
# In Atomizer.atomize(), after building the atom:
from ai_knot_v2.core.action_calculus import compute_action_affect_mask
from ai_knot_v2.core.information import compute_regret_charge_v2

# Compute action mask first
action_mask = compute_action_affect_mask_from_fields(
    risk_class, canon_pred, clause.object_raw or ""
)
regret_charge = _compute_regret_charge_for_fields(risk_severity, action_mask)
protection_energy = min(1.0, risk_severity * 2.0 + 0.1 * bin(action_mask).count("1") * 0.05)

# Then build atom with computed values
atoms.append(MemoryAtom(
    ...
    action_affect_mask=action_mask,
    regret_charge=regret_charge,
    ...
))
```

Add helper function inside `atomizer.py`:

```python
def _compute_regret_charge_for_fields(
    risk_severity: float,
    action_affect_mask: int,
    curvature: float = 0.0,
) -> float:
    action_bits = bin(action_affect_mask).count("1")
    delta_q = risk_severity * (1.0 + 0.3 * action_bits)
    curvature_term = 1.0 + 0.5 * curvature
    danger_term = 1.0 + 0.2 * risk_severity
    return min(1.0, delta_q * curvature_term * danger_term)
```

Also compute `action_affect_mask` before atom construction (currently it's hardcoded 0 at line 489):

```python
# BEFORE:
action_affect_mask=0,

# AFTER:
action_affect_mask=compute_action_affect_mask_for_fields(risk_class, canon_pred, clause.object_raw or ""),
```

Add standalone function in `atomizer.py` (wraps `action_calculus.compute_action_affect_mask` for pre-construction use):

```python
def compute_action_affect_mask_for_fields(
    risk_class: str,
    predicate: str,
    object_value: str,
) -> int:
    """Compute action_affect_mask before MemoryAtom is constructed."""
    # Create a minimal proxy — only fields used by compute_action_affect_mask
    import dataclasses
    from ai_knot_v2.core.atom import MemoryAtom
    proxy = MemoryAtom(
        atom_id="", agent_id="", user_id=None,
        variables=(), causal_graph=(), kernel_kind="point", kernel_payload={},
        intervention_domain=(), predicate=predicate, subject="", object_value=object_value,
        polarity="pos", valid_from=None, valid_until=None, observation_time=0, belief_time=0,
        granularity="instant", entity_orbit_id="", transport_provenance=(), depends_on=(),
        depended_by=(), risk_class=risk_class,  # type: ignore[arg-type]
        risk_severity=0.0, regret_charge=0.0, irreducibility_score=1.0, protection_energy=0.0,
        action_affect_mask=0, credence=1.0, evidence_episodes=(), synthesis_method="regex",
        validation_tests=(), contradiction_events=(),
    )
    return compute_action_affect_mask(proxy)
```

### S2 Gate

```bash
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_information.py -p no:cov -v
.venv/bin/pytest src/ai_knot_v2/tests/unit/ -p no:cov -q
cd aiknotbench && npx tsx src/index.ts run -r s2-write --top-k 60 --limit 2
# Accept: 2-conv aggregate not down >2pp; RSB v1 = 100%
```

**New test:** `src/ai_knot_v2/tests/unit/test_information.py`

```python
from ai_knot_v2.core.information import compute_regret_charge_v2, compute_irreducibility

def test_high_action_bits_increases_charge(make_atom):
    low = make_atom(risk_severity=0.5, action_affect_mask=0b00001)
    high = make_atom(risk_severity=0.5, action_affect_mask=0b11110)
    assert compute_regret_charge_v2(high) > compute_regret_charge_v2(low)

def test_curvature_increases_charge(make_atom):
    a = make_atom(risk_severity=0.5, action_affect_mask=0)
    assert compute_regret_charge_v2(a, curvature=1.0) > compute_regret_charge_v2(a, curvature=0.0)

def test_charge_bounded(make_atom):
    a = make_atom(risk_severity=1.0, action_affect_mask=0xFF)
    assert compute_regret_charge_v2(a, curvature=1.0) <= 1.0

def test_irreducibility_no_peers(make_atom):
    assert compute_irreducibility(make_atom(), []) == 1.0

def test_irreducibility_with_peers(make_atom):
    atom = make_atom(predicate="is", valid_from=0, valid_until=1000)
    peers = [make_atom(predicate="is", valid_from=0, valid_until=1000)]
    assert compute_irreducibility(atom, peers) < 0.9
```

---

## S3: Sheaf-Section Gluing in Planner

**Goal:** Replace greedy utility-only selection with orbit/action-class partition-aware selection.
**Files:** 1 modified.

### Changes to `src/ai_knot_v2/ops/planner.py`

Add `sheaf_section_gluing()` function and make it the default in `plan_evidence_pack()`:

```python
# Add after existing imports
from collections import defaultdict as _defaultdict


def sheaf_section_gluing(
    atoms: list[MemoryAtom],
    query: str,
    budget: ReaderBudget,
    library: AtomLibrary | None = None,
) -> EvidencePack:
    """Partition-first evidence selection.

    1. Partition atoms by (entity_orbit_id, coarse_action_class).
    2. Select best representative from each partition.
    3. Fill remaining budget with greedy-utility selection from leftover atoms.
    4. Apply dependency closure + contradiction resolution.
    """
    if not atoms:
        return EvidencePack(pack_id=new_ulid(), atoms=(), spans=())

    query_vf, query_vu = _query_temporal_window(query)

    # Step 1: partition by (orbit, coarse action class = top 8 bits of mask)
    sections: dict[tuple[str, int], list[MemoryAtom]] = _defaultdict(list)
    for atom in atoms:
        coarse = atom.action_affect_mask & 0xFF00
        sections[(atom.entity_orbit_id, coarse)].append(atom)

    # Step 2: pick best representative per section (highest regret_charge × credence)
    representatives: list[MemoryAtom] = []
    for sec_atoms in sections.values():
        best = max(sec_atoms, key=lambda a: a.regret_charge * a.credence)
        representatives.append(best)

    # Step 3: greedy fill from representatives first, sorted by utility
    token_budget = budget.max_tokens
    selected: list[MemoryAtom] = []
    utility_scores: dict[str, float] = {}

    representatives.sort(
        key=lambda a: utility(a, query, [], query_vf, query_vu), reverse=True
    )
    for atom in representatives:
        cost = reader_cost(atom)
        if len(selected) >= budget.max_atoms or token_budget - cost < 0:
            continue
        selected.append(atom)
        utility_scores[atom.atom_id] = round(
            utility(atom, query, selected[:-1], query_vf, query_vu), 4
        )
        token_budget -= cost

    # Step 4: fill with non-representative atoms
    covered = {a.atom_id for a in selected}
    remaining = [a for a in atoms if a.atom_id not in covered]
    remaining.sort(key=lambda a: utility(a, query, selected, query_vf, query_vu), reverse=True)

    for atom in remaining:
        cost = reader_cost(atom)
        if len(selected) >= budget.max_atoms or token_budget - cost < 0:
            break
        selected.append(atom)
        utility_scores[atom.atom_id] = round(
            utility(atom, query, selected[:-1], query_vf, query_vu), 4
        )
        token_budget -= cost

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
```

Update `plan_evidence_pack()` to delegate to `sheaf_section_gluing()` by default:

```python
# In plan_evidence_pack(), replace the selection loop body with:
return sheaf_section_gluing(atoms, query, budget, library)
```

The existing greedy loop becomes the Level-2 fallback callable directly as `_greedy_plan_evidence_pack()` if needed for ablation.

### S3 Gate

```bash
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_planner.py -p no:cov -v
cd aiknotbench && npx tsx src/index.ts run -r s3-sheaf --top-k 60 --limit 2
# Accept: cat1 not down >2pp; ContextDilutionRate logged in scorecard ↓ ≥ 15%
# Multi-metric gate in scorecard.py
```

**New test additions to** `src/ai_knot_v2/tests/unit/test_planner.py`:

```python
def test_sheaf_selects_from_different_orbits(make_atom):
    """With tight budget, sheaf should pick one from each orbit, not two from one."""
    orbit1_a = make_atom(entity_orbit_id="o1", action_affect_mask=0x0100, regret_charge=0.9)
    orbit1_b = make_atom(entity_orbit_id="o1", action_affect_mask=0x0100, regret_charge=0.8)
    orbit2_a = make_atom(entity_orbit_id="o2", action_affect_mask=0x0200, regret_charge=0.7)

    # Budget: max 2 atoms
    budget = ReaderBudget(max_atoms=2, max_tokens=1000, require_dependency_closure=False)
    pack = sheaf_section_gluing([orbit1_a, orbit1_b, orbit2_a], "query", budget)

    selected_ids = set(pack.atoms)
    # Should include orbit1_a (best in o1) and orbit2_a (only in o2)
    assert orbit1_b.atom_id not in selected_ids
    assert orbit2_a.atom_id in selected_ids
```

---

## S4: Landauer-ODE Forget

**Goal:** Add Landauer floor and access-rate boost to protection energy decay.
**Files:** 1 modified.

### Changes to `src/ai_knot_v2/ops/forget.py`

```python
# Add constants
LANDAUER_FLOOR_SCALE: float = 0.02  # normalized: high-risk atoms floor at 0.02 × risk_severity
ACCESS_RATE_HALFLIFE_DAYS: float = 7.0

def decay_protection_energy(
    atom: MemoryAtom,
    elapsed_days: float,
    access_count_recent: int = 0,
    contradiction_count: int = 0,
) -> MemoryAtom:
    """ODE decay with Landauer floor, access-rate boost, and curvature term."""
    # Base exponential decay (unchanged formula)
    k = BASE_DECAY_RATE / (1.0 + atom.risk_severity * 5.0)
    decayed = atom.protection_energy * math.exp(-k * elapsed_days)

    # Access-rate boost: each recent access adds 0.05 with halflife decay
    access_boost = (
        0.05 * access_count_recent * math.exp(-elapsed_days / ACCESS_RATE_HALFLIFE_DAYS)
    )

    # Curvature term: atoms in active contradictions decay slower
    curvature_boost = 0.03 * min(contradiction_count, 3)

    new_energy = decayed + access_boost + curvature_boost

    # Landauer floor: high-risk atoms cannot decay below thermodynamic floor
    landauer_floor = LANDAUER_FLOOR_SCALE * atom.risk_severity
    new_energy = max(landauer_floor, min(1.0, new_energy))

    return dataclasses.replace(atom, protection_energy=new_energy)
```

Existing call sites `decay_protection_energy(atom, elapsed_days)` continue to work (new params default to 0). Update `run_forget_pass()` to pass `contradiction_count`:

```python
def run_forget_pass(library, store, elapsed_days):
    for atom in atoms:
        contradiction_count = len(atom.contradiction_events)
        decayed = decay_protection_energy(atom, elapsed_days,
                                          contradiction_count=contradiction_count)
        ...
```

### S4 Gate

```bash
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_forget.py -p no:cov -v
# Existing tests still pass (backward compatible: new params default to 0)
# New test: Landauer floor holds for high-risk atom after large elapsed_days
```

---

## S5: Bitemporal Staleness Bonus

**Goal:** Add staleness penalty for open-ended atoms not recently confirmed.
**Files:** 1 modified.

### Changes to `src/ai_knot_v2/ops/planner.py`

Replace `temporal_allen_bonus()` call in `reduction_score()` with `bitemporal_allen_bonus()`:

```python
def bitemporal_allen_bonus(
    atom: MemoryAtom,
    query_vf: int | None,
    query_vu: int | None,
) -> float:
    """Extend temporal_allen_bonus with staleness penalty and belief-time bonus."""
    import math as _math
    bonus = temporal_allen_bonus(atom, query_vf, query_vu)

    now = int(time.time())

    # Staleness penalty: open-ended atom not updated in 6+ months
    if atom.valid_until is None:
        age_days = max(0, (now - atom.observation_time) / 86400)
        if age_days > 180:
            penalty = min(0.3, (age_days - 180) / 365 * 0.3)
            bonus -= penalty

    # Belief-time freshness: agent reconfirmed the fact after first observation
    if atom.belief_time > atom.observation_time:
        bonus += 0.1

    return max(0.0, bonus)
```

In `reduction_score()`, line 195:

```python
# BEFORE:
score += temporal_allen_bonus(atom, query_vf, query_vu)

# AFTER:
score += bitemporal_allen_bonus(atom, query_vf, query_vu)
```

### S5 Gate

```bash
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_planner.py -p no:cov -v
cd aiknotbench && npx tsx src/index.ts run -r s5-bitemp --top-k 60 --limit 2
# Accept: temporal questions not regressed; cat1-4 aggregate stable
```

---

## S6: RG-Flow Consolidation

**Goal:** Stratify interval merge by risk class (high-risk: 1-day adjacency; low-risk: 7-day).
**Files:** 1 modified.

### Changes to `src/ai_knot_v2/ops/consolidate.py`

```python
_HIGH_RISK_CLASSES: frozenset[str] = frozenset(
    {"safety", "medical", "legal", "identity"}
)
_LOW_RISK_ADJACENT_SECONDS: int = 7 * 86400  # 7 days for low-risk merge


def merge_intervals_rg(
    atoms: list[MemoryAtom],
) -> tuple[list[MemoryAtom], list[MemoryAtom]]:
    """RG-flow consolidation: risk-stratified interval merge.

    High-risk atoms: merge only with ADJACENT_SECONDS=86400 (1 day) — conservative.
    Low-risk atoms: merge with 7-day adjacency — more aggressive.
    """
    high_risk = [a for a in atoms if a.risk_class in _HIGH_RISK_CLASSES]
    low_risk = [a for a in atoms if a.risk_class not in _HIGH_RISK_CLASSES]

    high_result, high_removed = merge_intervals(high_risk)

    # Temporarily widen adjacency for low-risk group
    global ADJACENT_SECONDS
    original_adj = ADJACENT_SECONDS
    ADJACENT_SECONDS = _LOW_RISK_ADJACENT_SECONDS
    try:
        low_result, low_removed = merge_intervals(low_risk)
    finally:
        ADJACENT_SECONDS = original_adj

    return high_result + low_result, high_removed + low_removed


def consolidate_library(library: AtomLibrary, store: SqliteStore) -> ConsolidateResult:
    """Run RG-flow consolidation pass on the library."""
    atoms = library.all_atoms()
    original_ids = {a.atom_id for a in atoms}
    result_atoms, removed_originals = merge_intervals_rg(atoms)  # changed from merge_intervals
    ...  # rest unchanged
```

### S6 Gate

```bash
.venv/bin/pytest src/ai_knot_v2/tests/unit/test_consolidate.py -p no:cov -v
# Existing tests still pass; new test: high-risk atoms not merged at 7-day gap
```

---

## Full Test Suite Gate (after all steps)

```bash
# 1. Format
.venv/bin/ruff format src/ai_knot_v2/

# 2. Lint
.venv/bin/ruff check src/ai_knot_v2/

# 3. Types
.venv/bin/mypy --strict src/ai_knot_v2/core src/ai_knot_v2/ops src/ai_knot_v2/store src/ai_knot_v2/api

# 4. Architecture invariant
.venv/bin/pytest src/ai_knot_v2/tests/architecture/ -p no:cov -v

# 5. Unit tests
.venv/bin/pytest src/ai_knot_v2/tests/unit/ -p no:cov -q

# 6. RSB v1 = 100%
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --scenarios rsb

# 7. LOCOMO 2-conv
cd aiknotbench && npx tsx src/index.ts run -r level3-final --top-k 60 --limit 2

# 8. Author check
git log --format="%an" | sort -u  # must be: alsoleg89
```

---

## Anti-Regression Rules

1. `test_no_llm_in_core` must pass after every step.
2. `probe.py` and `render.py` are in `bench/` — grep confirms no import from `core/ops/store/api/`.
3. `MemoryAtom` schema unchanged — no fields added/removed.
4. If 2-conv cat1-4 aggregate drops >2pp from best-so-far → revert the step, do not stack.
5. No LOCOMO-specific patterns introduced — grep `locomo\|locom` in newly added code.
