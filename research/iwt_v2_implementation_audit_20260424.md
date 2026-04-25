# IWT → v2 Implementation Audit

Date: 2026-04-25
Status: Sprint 29-30 codebase (HEAD: 1e47590)
Purpose: Map every IWT theoretical construct to its current v2 placeholder, identify the delta, and specify the Level-3 replacement that brings the implementation from Sprint-1 approximation to ESWP Level-3.

Related:
- `research/extraction_sufficient_witness_program.md` — theoretical justification for each replacement
- `research/ccb_benchmark_design.md` — validation benchmark
- `src/ai_knot_v2/CLAUDE.md` — architectural invariants (no LLM in core)

---

## Legend

| Column | Meaning |
|---|---|
| IWT Construct | Formal object or operation defined in IWT Phase 2–4 |
| IWT Phase | Section in `research/invariant_witness_theory_program.md` |
| v2 Placeholder | Current implementation in `src/ai_knot_v2/` with `file:line` |
| What it does now | 1–2 sentences |
| What IWT requires | 1–2 sentences |
| Level-3 Approximation | Concrete formula + pseudocode ≤ 15 lines |
| Test pattern | How to verify the replacement |
| Expected metric delta | Empirical prediction tied to `cat1_55_investigation_20260423.md` buckets |

---

## Table 1: Write-Time Constructs

### Row 1: Regret Charge q(ω)

**IWT Construct:** Cohomological regret charge q(ω) = E_{Q ~ P_t}[Δ_Q(ω)] · (1 + λ ‖δω‖) · (1 + κ D_ω)

**IWT Phase:** §3.2

**v2 Placeholder:** `src/ai_knot_v2/ops/atomizer.py:459`
```python
regret_charge = risk_severity * 1.0   # Sprint 1 placeholder: irreducibility=1
```

**What it does now:** Sets regret_charge = risk_severity, treating every atom as equally irreducible (irreducibility_score hardcoded 1.0 at line 487). Ignores the cohomology curvature term (1 + λ ‖δω‖) and danger severity D_ω beyond what is already captured in risk_severity.

**What IWT requires:** risk_severity captures D_ω only. The full formula requires: (a) expected action regret Δ_Q(ω) over future task distribution P_t; (b) curvature penalty (1 + λ ‖δω‖) for atoms involved in temporal or identity contradictions; (c) danger severity D_ω from domain risk table. All three terms multiply.

**Level-3 Approximation:**
```python
def compute_regret_charge_v2(atom: MemoryAtom, curvature: float = 0.0) -> float:
    # Δ_Q proxy: risk_severity × action_affect_bits (higher diversity = higher expected regret)
    action_bits = bin(atom.action_affect_mask).count("1")
    delta_q_proxy = atom.risk_severity * (1.0 + 0.3 * action_bits)

    # Curvature term: atoms in detected contradictions get +λ bonus
    curvature_term = 1.0 + 0.5 * curvature  # curvature = |contradiction_events| > 0

    # Danger term: same as risk_severity (already in delta_q_proxy)
    danger_term = 1.0 + 0.2 * atom.risk_severity

    return min(1.0, delta_q_proxy * curvature_term * danger_term)
```

Where `curvature` = len(atom.contradiction_events) > 0 → 1.0 else 0.0.

**File to change:** `src/ai_knot_v2/ops/atomizer.py` line 459; add helper `compute_regret_charge_v2()` in `src/ai_knot_v2/core/information.py` (new file).

**Test pattern:**
```python
# tests/unit/test_regret_charge.py
atom_high_action = make_atom(risk_severity=0.9, action_affect_mask=0b11110, contradiction_events=())
atom_low_action = make_atom(risk_severity=0.9, action_affect_mask=0b00001, contradiction_events=())
assert compute_regret_charge_v2(atom_high_action) > compute_regret_charge_v2(atom_low_action)

atom_contrad = make_atom(risk_severity=0.5, contradiction_events=("evt1",))
atom_clean = make_atom(risk_severity=0.5, contradiction_events=())
assert compute_regret_charge_v2(atom_contrad) > compute_regret_charge_v2(atom_clean)
```

**Expected metric delta:** Improves rare-critical atom survival in 700-session CCB. LoCoMo direct impact: small (<2pp cat1), because 9/30 LLM-fail is not a retrieval problem. Improves H1' (rare-critical survival rate ≥ 0.95 at 10% budget). Addresses 6/30 cat1 hard-retrieval-miss indirectly by boosting regret_charge of correctly identified rare atoms.

---

### Row 2: Irreducibility Score

**IWT Construct:** Irreducibility: a witness ω is irreducible relative to M if removing it changes the minimal sufficient support set for some query Q.

**IWT Phase:** §3.2, §4.4

**v2 Placeholder:** `src/ai_knot_v2/ops/atomizer.py:487`
```python
irreducibility_score=1.0,  # Sprint 1 placeholder: always 1.0
```

Also `src/ai_knot_v2/bench/ablation.py:40-41`:
```python
def rcmt_only(atom, ...):
    return atom.irreducibility_score * atom.credence / reader_cost(atom)
```

**What it does now:** Every atom is treated as fully irreducible. The RCMT ablation mode uses this score but it is constant, making rcmt_only indistinguishable from credence/cost scoring.

**What IWT requires:** irreducibility_score should measure the fraction of plausible future queries for which this atom is the sole witness covering its constraint. High irreducibility = no other atom in M can substitute for this one; removing it changes the answer.

**Level-3 Approximation:**
```python
def compute_irreducibility(atom: MemoryAtom, library: AtomLibrary) -> float:
    # Find atoms in same orbit with same predicate
    peers = [a for a in library.query_by_entity(atom.entity_orbit_id)
             if a.predicate == atom.predicate and a.atom_id != atom.atom_id]
    if not peers:
        return 1.0  # no substitutes → fully irreducible

    # Temporal overlap: if peers cover the same interval, atom is redundant
    overlapping = [p for p in peers
                   if _intervals_overlap(atom.valid_from, atom.valid_until,
                                        p.valid_from, p.valid_until)]
    redundancy = len(overlapping) / (len(peers) + 1)
    return max(0.1, 1.0 - redundancy)
```

**File to change:** `src/ai_knot_v2/ops/atomizer.py` line 487 → call `compute_irreducibility(candidate, library)`; `library` must be passed to `Atomizer.atomize()`. Add helper in `src/ai_knot_v2/core/information.py`.

**Test pattern:**
```python
# atom with 2 peers covering same interval → low irreducibility
assert compute_irreducibility(atom_with_peers, library) < 0.5
# atom with no peers → irreducibility = 1.0
assert compute_irreducibility(atom_unique, empty_library) == 1.0
```

**Expected metric delta:** Improves ablation discrimination (rcmt-only mode becomes meaningful). Small direct LOCOMO impact; improves CCB RWCA for rare-critical atoms by ensuring they are not forgotten when peers exist.

---

### Row 3: Protection Energy Initial Value

**IWT Construct:** Protection energy E_ω — initial value at write time should encode D_ω (danger severity) and VoI estimate.

**IWT Phase:** §3.2, §3.3 FORGETTING

**v2 Placeholder:** `src/ai_knot_v2/ops/atomizer.py:456`
```python
protection_energy = min(1.0, risk_severity * 2.0)  # heuristic
```

**What it does now:** Linear scale of risk_severity capped at 1.0. No credence weighting. No dependency-boundary bonus.

**What IWT requires:** E_ω at write time should encode D_ω (danger) × credence × dependency-count bonus. Atoms that are dependency boundaries for other high-charge witnesses deserve higher initial protection.

**Level-3 Approximation:**
```python
def initial_protection_energy(atom: MemoryAtom) -> float:
    base = min(1.0, atom.risk_severity * 2.0)
    credence_factor = 0.5 + 0.5 * atom.credence  # credence ∈ [0.5, 1.0] scale
    dep_bonus = min(0.2, len(atom.depends_on) * 0.05)  # up to +0.2 for dep-boundary atoms
    return min(1.0, base * credence_factor + dep_bonus)
```

**File to change:** `src/ai_knot_v2/ops/atomizer.py` line 456.

**Test pattern:**
```python
low_cred = make_atom(risk_severity=0.8, credence=0.3, depends_on=())
high_cred = make_atom(risk_severity=0.8, credence=0.95, depends_on=())
assert initial_protection_energy(high_cred) > initial_protection_energy(low_cred)

dep_atom = make_atom(risk_severity=0.5, credence=0.9, depends_on=("a1", "a2"))
nodep_atom = make_atom(risk_severity=0.5, credence=0.9, depends_on=())
assert initial_protection_energy(dep_atom) > initial_protection_energy(nodep_atom)
```

**Expected metric delta:** Moderate improvement on CCB rare-critical survival by protecting boundary atoms. LoCoMo: <1pp direct (not a write-path failure for 2-conv runs).

---

## Table 2: Read-Time Constructs

### Row 4: Sheaf-Section Gluing (Read)

**IWT Construct:** READ solves min-sufficient support: S*(Q, M_T) subject to counterfactual separability. Implemented as partition into (orbit, interval, action_class) sections, one representative per section.

**IWT Phase:** §3.3 READ, §2.2 (Mechanism 2 — Sheaf Cohomology)

**v2 Placeholder:** `src/ai_knot_v2/ops/planner.py:330-398`
```python
def plan_evidence_pack(atoms, query, budget, library=None) -> EvidencePack:
    # Greedy utility selection: pick highest utility(atom, query, selected) each step
    while remaining and len(selected) < budget.max_atoms and token_budget > 0:
        scored = [(utility(a, ...), a) for a in remaining]
        scored.sort(...)
        best_atom = scored[0][1]
        selected.append(best_atom)
        ...
```

**What it does now:** Greedy selection maximizing utility = reduction_score / reader_cost. No partition structure; may select multiple atoms from the same (orbit, predicate) pair if they are high-utility, leaving other action-class partitions unrepresented.

**What IWT requires:** Sheaf-section gluing: partition candidate atoms into sections by (entity_orbit_id, action_class_bits); select one representative from each section (minimum cost, maximum credence); then greedily fill remaining budget with inter-section atoms. This ensures diverse coverage before depth within any single partition.

**Level-3 Approximation:**
```python
def sheaf_section_gluing(
    atoms: list[MemoryAtom],
    query: str,
    budget: ReaderBudget,
    library: AtomLibrary | None = None,
) -> EvidencePack:
    # Step 1: partition by (orbit, action_class_coarse)
    def section_key(a: MemoryAtom) -> tuple:
        coarse_action = a.action_affect_mask & 0xFF00  # top 8 bits = coarse class
        return (a.entity_orbit_id, coarse_action)

    sections: dict[tuple, list[MemoryAtom]] = defaultdict(list)
    for atom in atoms:
        sections[section_key(atom)].append(atom)

    # Step 2: pick best representative per section (highest regret_charge × credence)
    representatives = []
    for sec_atoms in sections.values():
        best = max(sec_atoms, key=lambda a: a.regret_charge * a.credence)
        representatives.append(best)

    # Step 3: greedy fill from representatives first, then remaining
    representatives.sort(key=lambda a: utility(a, query, [], None, None), reverse=True)
    selected, token_budget = [], budget.max_tokens
    for atom in representatives:
        cost = reader_cost(atom)
        if len(selected) < budget.max_atoms and token_budget - cost >= 0:
            selected.append(atom)
            token_budget -= cost

    # Step 4: fill remaining budget with non-representative high-utility atoms
    covered = {a.atom_id for a in selected}
    remaining = [a for a in atoms if a.atom_id not in covered]
    remaining.sort(key=lambda a: utility(a, query, selected, None, None), reverse=True)
    for atom in remaining:
        cost = reader_cost(atom)
        if len(selected) >= budget.max_atoms or token_budget - cost < 0:
            break
        selected.append(atom)
        token_budget -= cost

    # Steps 5-6: dependency closure + contradiction resolution (unchanged)
    ...
```

**File to change:** `src/ai_knot_v2/ops/planner.py` — add `sheaf_section_gluing()` alongside existing `plan_evidence_pack()`; update callers to use `sheaf_section_gluing` by default.

**Test pattern:**
```python
# Two atoms from same orbit/action should not both be selected when budget is tight
atoms = [atom_orbit1_action1_a, atom_orbit1_action1_b, atom_orbit2_action2]
pack = sheaf_section_gluing(atoms, query, budget_tight, ...)
# Should include one from orbit1 and one from orbit2, not two from orbit1
assert sum(1 for a in pack.atoms if "orbit1" in a) == 1
assert sum(1 for a in pack.atoms if "orbit2" in a) == 1
```

**Expected metric delta:** Addresses the 10/30 partial-retrieval cat1 cases by improving section diversity in the pack. Expected ContextDilutionRate ↓ ≥ 20%. LOCOMO 2-conv cat1 improvement: +2–4pp (primarily partial-retrieval bucket).

---

### Row 5: Render-Format Ordering (ESWP Extension)

**IWT Construct:** None (new in ESWP) — render operator ρ optimized for reader extractability.

**ESWP Phase:** §3.2, §3.3 READ

**v2 Placeholder:** No render operator exists. `bench/v2_locomo_runner.py` assembles context by iterating atoms in pack order (ULID order = insertion order).

**What it does now:** Atoms rendered in ULID insertion order; high-regret-charge atoms may appear anywhere in the context string.

**What ESWP requires:** render operator ρ sorts atoms by (regret_charge × credence) descending before context assembly; assigns render-format tags (factual_assertion > has_obj/copula predicates; transcript_fragment > event predicates); renders factual_assertion atoms in "Subject Verb Object" form.

**Level-3 Approximation:**
```python
def render_pack_eswp(atoms: list[MemoryAtom], query: str) -> str:
    # Sort by regret_charge × credence descending
    ordered = sorted(atoms, key=lambda a: a.regret_charge * a.credence, reverse=True)

    lines = []
    for atom in ordered:
        # Choose render format based on predicate type
        if atom.predicate in ("is", "has", "works_at", "lives_in", "prefers", "dislikes"):
            # Factual assertion format: S is/has P
            fmt = f"[fact] {atom.subject} {atom.predicate.replace('_', ' ')} {atom.object_value or ''}"
        else:
            # Event format: S verb O (temporal if available)
            temp = ""
            if atom.valid_from and atom.valid_until:
                temp = f" (valid {atom.valid_from}–{atom.valid_until})"
            fmt = f"[event] {atom.subject} {atom.predicate} {atom.object_value or ''}{temp}"
        lines.append(fmt)

    return "\n".join(lines)
```

**File to change:** `src/ai_knot_v2/bench/v2_locomo_runner.py` — replace current join with `render_pack_eswp(atoms, query)`.

**Test pattern:**
```python
atoms = [low_charge_atom, high_charge_atom]
rendered = render_pack_eswp(atoms, "query")
# High charge atom should appear first
assert rendered.index("high_charge_subject") < rendered.index("low_charge_subject")
```

**Expected metric delta:** Closes ~3–5 of the 9 LLM-fail cat1 cases (context dilution + reordering). Direct LOCOMO impact: H5' prediction +5 questions → cat1 ~41.9%.

---

## Table 3: Forget-Time Constructs

### Row 6: Protection Energy ODE (Landauer Floor)

**IWT Construct:** Protection energy ODE: dE_μ/dt = -α E_μ + β VoI_μ + χ D_μ + ζ C_μ - ψ R_μ - ξ c_μ

**IWT Phase:** §3.3 FORGETTING

**v2 Placeholder:** `src/ai_knot_v2/ops/forget.py:26-30`
```python
def decay_protection_energy(atom: MemoryAtom, elapsed_days: float) -> MemoryAtom:
    k = BASE_DECAY_RATE / (1.0 + atom.risk_severity * 5.0)
    new_energy = max(0.0, atom.protection_energy * math.exp(-k * elapsed_days))
    return dataclasses.replace(atom, protection_energy=new_energy)
```

**What it does now:** First-order linear ODE with k = BASE_DECAY_RATE / (1 + risk_severity × 5). Decay rate is slower for high-risk atoms. No Landauer floor (minimum energy), no access-rate boost, no curvature term, no redundancy decay.

**What IWT requires:** Multi-term ODE with Landauer floor E_barrier = kB T ln 2 (normalized to ~0.02 in code units), plus access-rate refresh term ζ C_μ, and curvature term.

**Level-3 Approximation:**
```python
LANDAUER_FLOOR = 0.02   # thermodynamic floor (normalized)
ACCESS_RATE_HALFLIFE = 7  # days

def decay_protection_energy_v2(
    atom: MemoryAtom,
    elapsed_days: float,
    contradiction_count: int = 0,
    access_count_recent: int = 0,
) -> MemoryAtom:
    # Decay term: same as before
    k = BASE_DECAY_RATE / (1.0 + atom.risk_severity * 5.0)
    decayed = atom.protection_energy * math.exp(-k * elapsed_days)

    # Access-rate refresh: each recent access adds a small boost
    access_boost = 0.05 * access_count_recent * math.exp(-elapsed_days / ACCESS_RATE_HALFLIFE)

    # Curvature term: atoms in active contradictions decay slower (need resolution)
    curvature_boost = 0.03 * min(contradiction_count, 3)

    new_energy = decayed + access_boost + curvature_boost

    # Landauer floor: cannot erase below thermodynamic floor for high-risk atoms
    landauer_floor = LANDAUER_FLOOR * atom.risk_severity
    new_energy = max(landauer_floor, min(1.0, new_energy))

    return dataclasses.replace(atom, protection_energy=new_energy)
```

**File to change:** `src/ai_knot_v2/ops/forget.py:26-30`. Callers must pass `access_count_recent` from store audit log; `contradiction_count` from atom.contradiction_events.

**Test pattern:**
```python
# High-risk atom must not decay below Landauer floor
high_risk = make_atom(risk_severity=1.0, protection_energy=0.1)
decayed = decay_protection_energy_v2(high_risk, elapsed_days=1000)
assert decayed.protection_energy >= LANDAUER_FLOOR * 1.0

# Frequently accessed atom decays slower
active = make_atom(risk_severity=0.3, protection_energy=0.5)
inactive = make_atom(risk_severity=0.3, protection_energy=0.5)
assert (decay_protection_energy_v2(active, 30, access_count_recent=5).protection_energy >
        decay_protection_energy_v2(inactive, 30, access_count_recent=0).protection_energy)
```

**Expected metric delta:** Memory sublinear growth in 700-session runs (H3'). LoCoMo 2-conv: <1pp direct (not a forget-path problem at 2-conv scale). Critical for 10-conv and CCB 700-session.

---

## Table 4: Consolidation-Time Constructs

### Row 7: RG-Flow Consolidation

**IWT Construct:** Consolidation = Renormalization Group flow: block-decimation of low-charge atoms per risk-class, preserving high-charge witnesses and extraction-sufficiency invariant.

**IWT Phase:** §3.3 CONSOLIDATION

**v2 Placeholder:** `src/ai_knot_v2/ops/consolidate.py:68-85`
```python
def _merge_group(atoms: list[MemoryAtom]) -> list[MemoryAtom]:
    # Greedy interval merging: merge overlapping or adjacent intervals
    timed.sort(key=lambda a: a.valid_from)
    for atom in timed[1:]:
        if _overlaps_or_adjacent(merged[-1], atom):
            merged[-1] = _merge_two(merged[-1], atom)
        else:
            merged.append(atom)
```

**What it does now:** Greedy interval merge by (entity_orbit_id, predicate, subject, object_value, polarity) key. All risk classes treated identically. No regret-charge preservation check. No extraction-sufficiency validation post-merge.

**What IWT requires:** Block-decimation by risk-class: high-risk atoms are preserved with full resolution; low-risk atoms from the same entity orbit can be merged more aggressively; the merged result must be extraction-sufficient (render-format tag must not degrade).

**Level-3 Approximation:**
```python
def merge_intervals_rg(atoms: list[MemoryAtom]) -> tuple[list[MemoryAtom], list[MemoryAtom]]:
    """RG-flow consolidation: risk-class-stratified interval merge."""
    # Separate into risk strata
    HIGH_RISK = {"safety", "medical", "legal", "identity"}

    high_risk_atoms = [a for a in atoms if a.risk_class in HIGH_RISK]
    low_risk_atoms = [a for a in atoms if a.risk_class not in HIGH_RISK]

    # High-risk: merge only exact-same-triple atoms (preserve distinct facts)
    high_result, high_removed = merge_intervals(high_risk_atoms)

    # Low-risk: merge more aggressively (adjacent within 7 days, not 1 day)
    # Temporarily widen ADJACENT_SECONDS for low-risk group
    original_adj = consolidate.ADJACENT_SECONDS
    consolidate.ADJACENT_SECONDS = 7 * 86400
    low_result, low_removed = merge_intervals(low_risk_atoms)
    consolidate.ADJACENT_SECONDS = original_adj

    # Preserve render-format: merged atom inherits max-credence constituent's synthesis_method
    for merged in high_result + low_result:
        pass  # synthesis_method already set to "fusion" by _merge_two

    return high_result + low_result, high_removed + low_removed
```

**File to change:** `src/ai_knot_v2/ops/consolidate.py` — add `merge_intervals_rg()` alongside `merge_intervals()`; update `consolidate_library()` to call `merge_intervals_rg`.

**Test pattern:**
```python
medical_atom1 = make_atom(risk_class="medical", valid_from=T1, valid_until=T2)
medical_atom2 = make_atom(risk_class="medical", valid_from=T2+86399, valid_until=T3)  # adjacent
preference_atom1 = make_atom(risk_class="preference", valid_from=T1, valid_until=T2)
preference_atom2 = make_atom(risk_class="preference", valid_from=T2+4*86400, valid_until=T3)  # within 7d

result, removed = merge_intervals_rg([medical_atom1, medical_atom2, preference_atom1, preference_atom2])
# Medical: adjacent=1day → should merge (T2+86399 - T2 = 86399 < ADJACENT_SECONDS=86400)
# Preference: adjacent=7days → should merge (4 days within new ADJACENT_SECONDS=7days)
assert len([a for a in result if a.risk_class == "medical"]) == 1
assert len([a for a in result if a.risk_class == "preference"]) == 1
```

**Expected metric delta:** Neutral on 2-conv LOCOMO (not triggered at 2-conv scale). Improves |M_T|/T sublinearity at 700 sessions (H3'). Prevents over-merging of high-risk medical/safety atoms that should stay distinct.

---

## Table 5: Identity/Groupoid Constructs

### Row 8: Groupoid Transport (Action-Invariant)

**IWT Construct:** Entity transport maps g_{ij} ∈ G such that action-relevant predicates remain invariant under transport: P(A | m_i, ω) = P(A | g_{ij} m_j, ω) ± ε.

**IWT Phase:** §3.5

**v2 Placeholder:** `src/ai_knot_v2/core/groupoid.py:33-41`
```python
def resolve(self, surface: str) -> str:
    norm = _normalize_str(surface)
    if norm not in self._orbits:
        self._orbits[norm] = f"entity:{norm}"
    return self._orbits[norm]
```

**What it does now:** String normalization (lowercase, strip articles, collapse whitespace) → canonical orbit ID. Holonomy detection via Floyd's cycle detection (`has_holonomy()`, `holonomy_orbits()`). No action-invariance check: two entity strings that resolve to different orbit IDs may refer to the same person if nicknames differ.

**What IWT requires:** Full transport maps g_{ij} that verify action-invariance: two mentions are the same entity if and only if all action-affecting predicates (risk_class in {safety, medical, identity, legal, scheduling}) are invariant under the rename. Today, "Alice" and "Alice Chen" may map to different orbits, fragmenting evidence.

**Level-3 Approximation:**
```python
def merge_by_predicate_invariance(
    self,
    surface_a: str,
    surface_b: str,
    atoms_a: list[MemoryAtom],
    atoms_b: list[MemoryAtom],
    risk_classes: frozenset[str] = frozenset({"safety", "medical", "identity"}),
) -> str | None:
    """Merge two orbits if their high-risk predicates are compatible.

    Returns merged orbit_id if merge is safe; None if predicates conflict.
    """
    orbit_a = self.resolve(surface_a)
    orbit_b = self.resolve(surface_b)
    if orbit_a == orbit_b:
        return orbit_a

    # Get high-risk predicates for each orbit
    preds_a = {(a.predicate, a.object_value) for a in atoms_a
               if a.risk_class in risk_classes and a.polarity == "pos"}
    preds_b = {(a.predicate, a.object_value) for a in atoms_b
               if a.risk_class in risk_classes and a.polarity == "pos"}

    # Check for predicate conflicts (same predicate, different object)
    for pred_a, obj_a in preds_a:
        for pred_b, obj_b in preds_b:
            if pred_a == pred_b and obj_a != obj_b:
                return None  # Conflict: cannot merge

    # No conflict → safe to merge
    return self.merge(surface_a, surface_b)
```

**File to change:** `src/ai_knot_v2/core/groupoid.py` — add `merge_by_predicate_invariance()`. Callers in `atomizer.py` can use this to merge "Alice" and "Alice Chen" when they share consistent predicates.

**Test pattern:**
```python
g = EntityGroupoid()
# Alice (doctor) and Alice Chen (doctor) → safe merge
result = g.merge_by_predicate_invariance("Alice", "Alice Chen",
    [make_atom(predicate="is", object_value="doctor", risk_class="identity")],
    [make_atom(predicate="is", object_value="doctor", risk_class="identity")])
assert result is not None

# Alice (doctor) and Alice Chen (nurse) → conflict, no merge
result = g.merge_by_predicate_invariance("Alice", "Alice Chen",
    [make_atom(predicate="is", object_value="doctor", risk_class="identity")],
    [make_atom(predicate="is", object_value="nurse", risk_class="identity")])
assert result is None
```

**Expected metric delta:** Reduces entity fragmentation. LoCoMo cat1: +1–3pp from better entity resolution. CCB-Identity: significant improvement.

---

## Table 6: Temporal Constructs

### Row 9: Bitemporal Representation

**IWT Construct:** Tri-temporal: valid_time τ^v, transaction_time τ^x, belief_time τ^b. READ must use all three axes to answer "what was X's state at time T" vs. "what did the agent believe at time T".

**IWT Phase:** §3.4

**v2 Placeholder:** `src/ai_knot_v2/core/atom.py:34-38`:
```python
valid_from: int | None
valid_until: int | None
observation_time: int     # = transaction_time τ^x
belief_time: int          # stored but = observation_time (not independently tracked)
```

`src/ai_knot_v2/ops/planner.py:94-110`: Allen-relation scoring uses only valid_from/valid_until vs. query temporal window. observation_time used only for recency decay. belief_time not used.

**What it does now:** Allen-relation scoring on valid-time. Recency decay on transaction-time (observation_time). belief_time stored but unused and equal to observation_time.

**What IWT requires:** Bitemporal planning: when answering "what is X now?", the planner should prefer atoms where valid_until > now AND observation_time > last_update_session (agent actually knows this fact is current). Atoms with valid_until = None (open-ended) but old observation_time should get a staleness hazard penalty.

**Level-3 Approximation:**
```python
def bitemporal_allen_bonus(
    atom: MemoryAtom,
    query_vf: int | None,
    query_vu: int | None,
    now: int,
) -> float:
    # Existing: valid-time Allen relation bonus (unchanged)
    bonus = temporal_allen_bonus(atom, query_vf, query_vu)

    # New: transaction-time staleness penalty
    if atom.valid_until is None:  # open-ended claim
        age_days = (now - atom.observation_time) / 86400
        if age_days > 180:  # 6 months without update → staleness hazard
            staleness_penalty = min(0.3, (age_days - 180) / 365 * 0.3)
            bonus -= staleness_penalty

    # New: belief-time freshness (if belief_time > observation_time, agent reconfirmed)
    if atom.belief_time > atom.observation_time:
        reconfirm_bonus = 0.1  # agent has reconfirmed this fact since first observation
        bonus += reconfirm_bonus

    return max(0.0, bonus)
```

**File to change:** `src/ai_knot_v2/ops/planner.py:75-110` — replace `temporal_allen_bonus()` call with `bitemporal_allen_bonus()`.

**Test pattern:**
```python
# Stale open-ended atom (learned 2 years ago, no valid_until) → lower bonus
old_open = make_atom(valid_from=T_old, valid_until=None, observation_time=T_old_obs)
new_open = make_atom(valid_from=T_new, valid_until=None, observation_time=T_now)
assert bitemporal_allen_bonus(old_open, None, None, T_now) < bitemporal_allen_bonus(new_open, None, None, T_now)

# Reconfirmed atom → higher bonus
reconfirmed = make_atom(belief_time=T_now, observation_time=T_old)
assert bitemporal_allen_bonus(reconfirmed, None, None, T_now) > bitemporal_allen_bonus(old_open, None, None, T_now)
```

**Expected metric delta:** H4' prediction: ≥ 15pp improvement on LongMemEval temporal update questions. LoCoMo 2-conv: +2–4pp on temporal-sensitive cat2/cat3 questions.

---

## Table 7: Extraction-Sufficiency (ESWP-Only)

### Row 10: Reader-Probe Validator

**IWT Construct:** None (new in ESWP). Extraction-sufficiency probe: I(y*_Q; π_R(Q, render(W_Q))) ≥ (1−ε) · H(y*_Q).

**ESWP Phase:** §3.2, §3.3 READ

**v2 Placeholder:** None. No reader-probe exists anywhere in the codebase.

**What it does now:** n/a

**What ESWP requires:** A function in bench/ (NOT in core/ops/store/api/) that takes an assembled pack, a query, and a reader model, calls the reader once, and returns an extraction-sufficiency score. If score < threshold, the function recommends pack expansion.

**Level-3 Implementation (bench/ only):**
```python
# src/ai_knot_v2/bench/ccb/probe.py

def validate_extraction_sufficiency(
    pack: EvidencePack,
    query: str,
    reader: Any,  # OpenAI/Anthropic client
    expected_answer: str | None = None,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """Call reader on rendered pack; return (is_sufficient, score).

    Score = token-level F1 between reader output and expected_answer if known,
    else proxy score = 1.0 / (1 + num_irrelevant_tokens_in_rendered_pack).
    """
    rendered = render_pack_eswp(pack.atoms_resolved, query)
    response = reader.complete(
        system="Answer the question from the context. Be concise.",
        user=f"Context:\n{rendered}\n\nQuestion: {query}"
    )
    output = response.content

    if expected_answer is not None:
        score = token_f1(output, expected_answer)
    else:
        # Proxy: coverage of query tokens in output
        q_tokens = set(query.lower().split())
        out_tokens = set(output.lower().split())
        score = len(q_tokens & out_tokens) / max(1, len(q_tokens))

    return score >= threshold, score


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 0.0
    precision = len(pred_tokens & gold_tokens) / max(1, len(pred_tokens))
    recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

**File to create:** `src/ai_knot_v2/bench/ccb/probe.py` (new).
**Architecture guard:** `tests/architecture/test_no_llm_in_core.py` must still pass — probe.py is in bench/, not core/.

**Test pattern:**
```python
# Sufficient pack: expected answer tokens in reader output → score ≥ threshold
pack_sufficient = make_pack_with_gold_atom()
is_suff, score = validate_extraction_sufficiency(pack_sufficient, query="...", reader=mock_reader, expected_answer="pottery")
assert is_suff

# Insufficient pack: diluted context → score < threshold
pack_diluted = make_pack_without_gold_atom()
is_suff, score = validate_extraction_sufficiency(pack_diluted, query="...", reader=mock_reader, expected_answer="pottery")
assert not is_suff
```

**Expected metric delta:** H5' prediction: closes ≥ 5 of 9 LLM-fail cat1 questions → cat1 ~41.9%. This is the highest-impact single change.

---

## Summary: Implementation Priority and Sequence

| Step | Row | File | IWT Component | Expected Impact | Risk |
|---|---|---|---|---|---|
| 1 | 10 | `bench/ccb/probe.py` (new) | Reader-probe validator | H5': +5 cat1 Q, cat1 ~41.9% | Medium — probe may not generalize |
| 2 | 5 | `bench/v2_locomo_runner.py` | Render-format ordering | +2–3 cat1 Q from reordering | Low |
| 3 | 1 | `ops/atomizer.py:459` + `core/information.py` | ΔF-write regret_charge | H1': rare-critical survival +20pp | Medium — action_affect_mask proxy |
| 4 | 4 | `ops/planner.py` | Sheaf-section gluing | H2': ContextDilutionRate ↓ 20% | Medium — partition quality |
| 5 | 6 | `ops/forget.py:26-30` | Landauer-ODE forget | H3': sublinear memory | Low (for 700-session runs) |
| 6 | 9 | `ops/planner.py:94-110` | Bitemporal bonus | H4': +15pp temporal questions | Low |
| 7 | 7 | `ops/consolidate.py` | RG-flow consolidation | H3' supporting | Low risk |
| 8 | 8 | `core/groupoid.py` | Predicate-invariant merge | Cat1 +1–3pp entity resolution | Medium — may cause false merges |

**Gate after each step:** run multi-metric scorecard: cat1 monotonic up (or target metric), ContextDilutionRate not up, DependencyClosureRecall not down, test_no_llm_in_core passes, no LOCOMO-specific code added.

**Stop-and-revert rule:** if 2-conv scorecard drops >2pp aggregate from best baseline → revert (memory rule `feedback_regression_stop_rule.md`).

---

## Confirmed Invariants (DO NOT CHANGE)

1. `MemoryAtom` schema: frozen from Sprint 1 (`core/atom.py`). No fields added, removed, or renamed.
2. No LLM imports in `core/` `ops/` `store/` `api/`. CI gate enforces this.
3. `bench/rsb/` — RSB v1 (100% pass rate). Not touched in any step above.
4. `aiknotbench/` — LOCOMO TypeScript harness. Not modified; only run for gate checks.
