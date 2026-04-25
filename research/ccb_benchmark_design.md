# Counterfactual Continuity Benchmark (CCB) — Design Specification

Date: 2026-04-25
Status: specification document; prototype implementation in `src/ai_knot_v2/bench/ccb/` (to be built in Stage 4)
Related:
- IWT Phase 6.2: conceptual definition of CCB
- `research/extraction_sufficient_witness_program.md`: H1'–H5' hypotheses tested by CCB
- `research/iwt_v2_implementation_audit_20260424.md`: Row 10 (reader-probe validator)

---

## 1. Motivation

Existing benchmarks (LoCoMo, LongMemEval, BEAM, MemoryArena, MemGround) share a structural limitation: **ground truth is established by averaging over many sessions with diverse queries**. A system can score well by retrieving the most common facts across sessions, even if it systematically fails on the rare, high-stakes constraints that matter most in practice.

CCB closes this gap by grounding evaluation in **counterfactual interventions**: we change one specific early witness and verify that the system's answers at much later sessions change correctly. This design has three properties:

1. **Unambiguous ground truth.** The correct answer is determined by the changed witness, not by crowd-sourced annotation or model judgment.
2. **Unfakable by full-context models.** Two near-identical histories that differ in one early witness are impossible to distinguish without witness-level tracking.
3. **Directly tests ESWP claims.** It measures both witness retention (IWT — does the changed witness survive?) and extraction sufficiency (ESWP — can the reader extract the updated fact from the pack?).

---

## 2. Benchmark Anatomy

### 2.1 Domains

CCB covers four domains that have natural rare-critical constraints and delayed causal dependencies:

| Domain | Code | Rare-Critical Pattern | Example |
|---|---|---|---|
| Medical-Logistics | MED | Drug allergy, contraindication, specialist exclusion | "allergic to penicillin" recorded in session 5; tested in session 312 when prescribing antibiotics |
| Scheduling-Commitment | SCH | Recurring obligation, deadline, conditional cancellation | "cannot miss Thursday slot — only window" recorded in session 12; tested in session 200 when rescheduling |
| Preference-Identity | PID | Core preference update, identity drift | "changed last name to Chen" recorded in session 8; tested in session 150 when addressing user |
| Financial-Legal | FIN | Insurance exclusion, contract clause, payment obligation | "insurer excludes Clinic A since session 30" tested in session 250 when booking appointment |

### 2.2 History Structure

Each CCB history consists of:

```
History H = (sessions_1..N, witness_w*, intervention)

where:
  sessions_i   : synthetic multi-turn conversation (5–15 turns)
  witness_w*   : the "planted" rare-critical witness (introduced in session k << N)
  intervention : a counterfactual change to witness_w* (e.g., change "penicillin" → "aspirin")
  probe_queries: questions at sessions k+50, k+100, k+200 that depend on w*
```

**Two parallel histories per seed:**
- `H_original`: history with witness_w* as originally planted
- `H_counterfactual`: history with intervention(w*) substituted

Ground truth for probe queries is determined by which history the system has been fed.

### 2.3 History Parameters

| Parameter | Default | Range |
|---|---|---|
| Sessions per history | 200 | 100–700 |
| Turns per session | 8 | 5–15 |
| Rare-critical witnesses per history | 3 | 1–5 |
| Probe query delay (sessions after plant) | 50, 100, 200 | 30–300 |
| Seeds per scenario | 5 | 3–10 |
| Intervention types | name-change, allergy-swap, location-update, exclusion-add | varies |

### 2.4 Scale Targets

| Phase | Histories | Domains | Seeds | Total probe queries |
|---|---|---|---|---|
| Prototype (MPU) | 20 per domain × 4 = 80 | 4 | 5 | ~1200 |
| Full (paper-ready) | 50 per domain × 4 = 200 | 4 | 5 | ~3000 |
| Extended | 250 per domain × 4 = 1000 | 4 | 5 | ~15000 |

---

## 3. Counterfactual Intervention Types

Each domain has 5 intervention templates. Interventions are applied deterministically to H_original to produce H_counterfactual.

### 3.1 Medical-Logistics (MED)

| Template | Original witness | Counterfactual | Probe question |
|---|---|---|---|
| MED-1 | "allergic to [drug_class_A]" | "allergic to [drug_class_B]" | "Is [subject] allergic to [drug_class_A]?" |
| MED-2 | "treated by [doctor_A] at [clinic_A]" | "treated by [doctor_B] at [clinic_B]" | "Which clinic does [subject] use?" |
| MED-3 | "[insurer] excludes [specialist_type] visits" | "[insurer] covers [specialist_type] visits" | "Does [insurer] cover [specialist_type]?" |
| MED-4 | "[diagnosis] first noted at session k" | "[diagnosis] resolved at session k+20" | "Does [subject] still have [diagnosis]?" |
| MED-5 | "contraindication: [drug_A] + [drug_B]" | "no known interaction: [drug_A] + [drug_B]" | "Can [subject] take [drug_A] with [drug_B]?" |

### 3.2 Scheduling-Commitment (SCH)

| Template | Original witness | Counterfactual | Probe question |
|---|---|---|---|
| SCH-1 | "weekly meeting on [day] at [time]" | "weekly meeting on [day+1] at [time]" | "When is [subject]'s weekly meeting?" |
| SCH-2 | "[deadline] is [date_A]" | "[deadline] is [date_B]" | "When is [subject]'s [deadline]?" |
| SCH-3 | "committed to [event] on [date]" | "[event] cancelled" | "Is [subject] attending [event]?" |
| SCH-4 | "[contact] available after [time]" | "[contact] available before [time]" | "When can [subject] reach [contact]?" |
| SCH-5 | "reminder for [task] set for [date]" | "[task] completed early, no reminder needed" | "Does [subject] still need [task] reminder?" |

### 3.3 Preference-Identity (PID)

| Template | Original witness | Counterfactual | Probe question |
|---|---|---|---|
| PID-1 | "name: [name_A]" | "name changed to [name_B]" | "What is [subject]'s name?" |
| PID-2 | "prefers [food_A]" | "no longer eats [food_A], prefers [food_B]" | "What does [subject] prefer to eat?" |
| PID-3 | "lives in [city_A]" | "moved to [city_B]" | "Where does [subject] live?" |
| PID-4 | "works at [company_A]" | "left [company_A], works at [company_B]" | "Where does [subject] work?" |
| PID-5 | "relationship status: [status_A]" | "relationship status changed to [status_B]" | "What is [subject]'s relationship status?" |

### 3.4 Financial-Legal (FIN)

| Template | Original witness | Counterfactual | Probe question |
|---|---|---|---|
| FIN-1 | "policy excludes [provider_A]" | "policy now covers [provider_A]" | "Does [subject]'s policy cover [provider_A]?" |
| FIN-2 | "subscription to [service_A] active" | "subscription to [service_A] cancelled" | "Is [subject]'s [service_A] subscription active?" |
| FIN-3 | "contract with [vendor_A] until [date_A]" | "contract extended until [date_B]" | "When does [subject]'s [vendor_A] contract end?" |
| FIN-4 | "tax filing status: [status_A]" | "tax filing status changed to [status_B]" | "What is [subject]'s tax filing status?" |
| FIN-5 | "budget limit: [amount_A] per month" | "budget limit changed to [amount_B] per month" | "What is [subject]'s monthly budget?" |

---

## 4. History Generator

### 4.1 Contract

```python
# src/ai_knot_v2/bench/ccb/generator.py

@dataclass(frozen=True)
class CCBScenario:
    domain: Literal["MED", "SCH", "PID", "FIN"]
    template_id: str  # e.g., "MED-1"
    seed: int
    n_sessions: int
    plant_session: int           # session where rare-critical witness is planted
    probe_delays: tuple[int, ...]  # probe at plant_session + delay
    intervention: dict[str, str]  # {field: original_value} → {field: counterfactual_value}

@dataclass(frozen=True)
class CCBHistory:
    scenario: CCBScenario
    sessions: tuple[Session, ...]    # full history
    planted_witness: MemoryAtomSpec  # description of planted witness
    probe_queries: tuple[ProbeQuery, ...]

@dataclass(frozen=True)
class ProbeQuery:
    query_text: str
    expected_original: str     # correct answer for H_original
    expected_counterfactual: str  # correct answer for H_counterfactual
    gold_witness_id: str       # atom that must be in pack to answer correctly

def generate_history(scenario: CCBScenario, rng: random.Random) -> tuple[CCBHistory, CCBHistory]:
    """Returns (H_original, H_counterfactual) for the given scenario."""
    ...
```

### 4.2 Generation Algorithm

1. **Background sessions (0 to plant_session - 1):** Generate neutral conversation sessions using domain-appropriate templates (schedule, preference, logistics). These sessions do not contain the planted witness. Use template-based text generation, not LLM.

2. **Plant session (session plant_session):** Insert one synthetic dialogue turn containing the planted witness in natural language form. E.g., for MED-1: "By the way, I just found out I'm severely allergic to penicillin. The doctor confirmed it."

3. **Filler sessions (plant_session + 1 to N - max_probe_delay - 1):** Continue with neutral background sessions. Optionally inject 2–3 "interference" facts that share surface tokens with the planted witness but are irrelevant (tests extraction disambiguation).

4. **Probe sessions (plant_session + probe_delay for each delay):** Each probe session contains a natural question about the planted witness in a new phrasing. E.g., "I have a chest infection — the doctor mentioned penicillin-based antibiotics. Should I be worried?"

5. **Counterfactual history:** Same as original, except plant session uses the counterfactual witness text (e.g., "I just found out I'm severely allergic to aspirin.") and all probe questions have updated expected answers.

### 4.3 Determinism

All generation must be deterministic given (scenario, seed). Use `random.Random(seed)` for all stochastic choices (name selection, filler topic selection, paraphrase selection). No LLM calls in the generator.

---

## 5. Scoring

### 5.1 Metrics

#### Risk-Weighted Counterfactual Accuracy (RWCA)

```
RWCA = Σ_{probe queries q} [w(q) × 1[answer_q matches expected_q]] / Σ w(q)

where w(q) = risk_severity of the planted witness for probe q
```

RWCA is the primary metric. It rewards correct answers proportionally to the danger severity of the underlying witness. A wrong answer on a penicillin allergy (risk_severity = 1.0) is penalized 10× more than a wrong answer on a preference (risk_severity = 0.1).

#### Extraction-Sufficiency Rate (ESR)

```
ESR = (# probe queries where gold witness is in pack AND reader extracts correctly) /
      (# probe queries where gold witness is in pack)

Interpretation: of the cases where the pack contains the right atom, what fraction does the reader extract correctly?
```

ESR is the ESWP-specific metric. It isolates A9 Reader-Extraction Blindness from retrieval failures.

#### Pack Coverage Rate (PCR)

```
PCR = (# probe queries where gold witness is in assembled pack) / (# total probe queries)

Interpretation: basic retrieval quality — does the pack contain the right atom?
```

PCR is the IWT-side metric. It measures whether the regret-charge mechanism protects the planted witness.

#### Counterfactual Discrimination Score (CDS)

```
CDS = |RWCA(H_original) - RWCA(H_counterfactual)| / max_possible_RWCA

Interpretation: can the system distinguish the two parallel histories?
A score near 0 = system ignores the planted witness entirely.
A score near 1 = system correctly propagates the intervention.
```

CDS is the benchmark-level metric. A full-context LLM with near-infinite memory would score CDS ≈ 1.0 on short histories but drop as history length increases.

### 5.2 Scorecard

```python
@dataclass(frozen=True)
class CCBScorecard:
    domain: str
    template_id: str
    n_sessions: int
    seed: int
    rwca_original: float
    rwca_counterfactual: float
    cds: float
    esr: float
    pcr: float
    probe_results: tuple[ProbeResult, ...]

@dataclass(frozen=True)
class ProbeResult:
    query_id: str
    probe_delay: int
    is_correct: bool
    gold_in_pack: bool
    reader_extracted: bool
    risk_weight: float
    answer_given: str
    expected_answer: str
```

### 5.3 Passing Criteria

| Metric | Minimum for "passes" | Target for paper |
|---|---|---|
| RWCA (Level-3 ESWP) | ≥ 0.80 | ≥ 0.90 |
| RWCA (best baseline) | < 0.65 | < 0.60 |
| ESR (Level-3 ESWP) | ≥ 0.75 | ≥ 0.85 |
| CDS (Level-3 ESWP) | ≥ 0.60 | ≥ 0.75 |
| PCR (Level-3 ESWP) | ≥ 0.85 | ≥ 0.90 |

Baselines that must be beaten for paper-level result: Mem0, A-MEM, LightMem, Zep/Graphiti, MemGPT, full-context-128K (truncated), v2-Sprint1-placeholder, recency-only.

---

## 6. Runner

### 6.1 Contract

```python
# src/ai_knot_v2/bench/ccb/runner.py

def run_ccb(
    adapter: MemoryAdapter,  # ai-knot v2 or baseline system
    scenarios: list[CCBScenario],
    reader: Any,  # LLM client (bench/ only)
    answerer: Any | None = None,  # optional separate answerer model
    budget: ReaderBudget = DEFAULT_BUDGET,
    seed: int = 42,
    probe_validation: bool = True,  # whether to call extraction-sufficiency probe
) -> list[CCBScorecard]:
    """Run CCB evaluation and return per-scenario scorecards."""
    scorecards = []
    for scenario in scenarios:
        h_orig, h_cf = generate_history(scenario, random.Random(seed))
        sc_orig = _evaluate_history(adapter, h_orig, reader, answerer, budget, probe_validation)
        sc_cf = _evaluate_history(adapter, h_cf, reader, answerer, budget, probe_validation)
        scorecard = _merge_scorecards(sc_orig, sc_cf, scenario)
        scorecards.append(scorecard)
    return scorecards


def _evaluate_history(
    adapter: MemoryAdapter,
    history: CCBHistory,
    reader: Any,
    answerer: Any | None,
    budget: ReaderBudget,
    probe_validation: bool,
) -> list[ProbeResult]:
    """Ingest history into adapter; run probe queries; return results."""
    # Ingest sessions 0..N-1
    for session in history.sessions:
        adapter.write_session(session)

    results = []
    for probe in history.probe_queries:
        pack = adapter.read(probe.query_text, budget)
        gold_in_pack = any(
            atom_contains_witness(a, history.planted_witness)
            for a in pack.atoms_resolved
        )

        # Get answer from reader/answerer
        ans_model = answerer or reader
        answer = ans_model.complete(
            system="Answer the question from the context.",
            user=f"Context:\n{render_pack_eswp(pack.atoms_resolved, probe.query_text)}\nQuestion: {probe.query_text}"
        )

        is_correct = token_f1(answer, probe.expected_original) > 0.5  # expected varies by history

        reader_extracted = False
        if probe_validation and gold_in_pack:
            is_suff, score = validate_extraction_sufficiency(
                pack, probe.query_text, reader, probe.expected_original
            )
            reader_extracted = is_suff

        results.append(ProbeResult(
            query_id=probe.query_text[:40],
            probe_delay=probe.scenario.probe_delays[0],  # simplified
            is_correct=is_correct,
            gold_in_pack=gold_in_pack,
            reader_extracted=reader_extracted,
            risk_weight=history.planted_witness.risk_severity,
            answer_given=answer,
            expected_answer=probe.expected_original,
        ))

    return results
```

### 6.2 MemoryAdapter Interface

```python
# src/ai_knot_v2/bench/ccb/adapter.py

class MemoryAdapter(Protocol):
    def write_session(self, session: Session) -> None: ...
    def read(self, query: str, budget: ReaderBudget) -> EvidencePack: ...
    def reset(self) -> None: ...
```

Concrete adapters:
- `AiKnotV2Adapter` — wraps ai-knot v2 write_episodes + plan_evidence_pack
- `Mem0Adapter`, `LightMemAdapter`, `ZepAdapter` — wrappers for baseline systems
- `FullContextAdapter` — keeps all sessions in context up to 128K tokens; truncates oldest

---

## 7. Validation Against Anti-Gaming Criteria

### 7.1 Anti-Gaming Properties

| Anti-gaming criterion | How CCB enforces it |
|---|---|
| Cannot be gamed by domain rules | Held-out intervention values (drug names, entity names) are unseen at design time; different per seed |
| Cannot be gamed by full-context LLM | Two near-identical histories differ only in one early session turn; LLM without witness tracking cannot distinguish them |
| Cannot be gamed by zero-shot | Probe questions use new phrasings at each delay step; no template reuse |
| Cannot be gamed by ignoring baseline | RWCA is risk-weighted; getting low-risk questions right while missing high-risk ones fails |

### 7.2 Difficulty Gradient

CCB is designed to have a natural difficulty gradient:
- **Easy:** probe at delay=50 (50 sessions after plant). Most systems retain the witness.
- **Medium:** probe at delay=100. Systems without explicit protection start failing.
- **Hard:** probe at delay=200 with 3 interference facts. Only systems with high regret-charge protection pass.

### 7.3 Compute Budget

| Phase | LLM calls per history | Estimated cost (gpt-4o-mini) |
|---|---|---|
| Ingestion (200 sessions) | 0 (deterministic) | $0 |
| Probe answers (6 queries × 2 histories) | 12 | ~$0.01 |
| Extraction probe (6 × 2) | 12 | ~$0.01 |
| **Total per history** | **24** | **~$0.02** |
| Full 200-history prototype | 4800 | **~$4.80** |
| Full 1000-history extended | 24000 | **~$24** |

These costs are consistent with bench cost discipline (memory: `feedback_bench_cost.md`): prototype runs are well within the $5 per run ceiling.

---

## 8. File Structure (to be created in Stage 4)

```
src/ai_knot_v2/bench/ccb/
├── __init__.py
├── adapter.py          # MemoryAdapter protocol + concrete adapters
├── generator.py        # CCBScenario, CCBHistory, generate_history()
├── probe.py            # validate_extraction_sufficiency(), token_f1()
├── runner.py           # run_ccb(), _evaluate_history()
├── scorer.py           # CCBScorecard, RWCA, ESR, CDS, PCR computation
├── render.py           # render_pack_eswp() (moved from bench/)
└── scenarios/
    ├── MED.yaml         # MED-1 through MED-5 templates
    ├── SCH.yaml         # SCH-1 through SCH-5 templates
    ├── PID.yaml         # PID-1 through PID-5 templates
    └── FIN.yaml         # FIN-1 through FIN-5 templates
```

Total estimated LoC for prototype: ~600 lines Python + ~200 lines YAML.

---

## 9. Integration with Existing Bench Infrastructure

- CCB runner uses `src/ai_knot_v2/bench/scorecard.py` for multi-metric gate validation.
- CCB adapter implements the same `MemoryAdapter` protocol as `v2_locomo_runner.py` adapters.
- CCB probe.py imports `plan_evidence_pack` from `ops/planner.py` (read-only, no core modifications).
- Architecture guard: CCB probe.py is in `bench/`; `tests/architecture/test_no_llm_in_core.py` still passes.

---

## 10. Relation to Existing Benchmarks

| Benchmark | What CCB adds |
|---|---|
| LoCoMo | CCB: ground truth by counterfactual intervention, not human annotation; risk-weighted scoring; extraction-sufficiency metric |
| LongMemEval | CCB: tests intervention propagation specifically; includes financial-legal domain |
| BEAM | CCB: shorter histories but deeper causal structure; rare-critical constraints, not arbitrary QA |
| MemoryArena | CCB: deterministic evaluation (no environment randomness); extraction-sufficiency metric |
| MemGround | CCB: explicit counterfactual pairs (not just state tracking); risk weighting |

CCB does not replace these benchmarks; it adds a complementary evaluation axis that none of them cover: can the system correctly propagate a known intervention through a long history?
