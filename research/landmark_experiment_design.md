# Landmark Experiment Design — ESWP Level-3

Date: 2026-04-25
Related:
- `research/extraction_sufficient_witness_program.md` §6.4
- `research/ccb_benchmark_design.md` — CCB runner and scoring
- `research/eswp_level3_implementation_spec.md` — what Level-3 is

---

## The Landmark Claim

[speculative] A single figure, reproducible by anyone who runs `bench/ccb/runner.py`:

> **At 10% memory budget over 200-session histories, ESWP Level-3 maintains RWCA ≥ 0.90 on CCB-RareCritical (4 domains × 5 seeds) while all 8 baselines fall below RWCA = 0.70, and the gap widens rather than narrows as history length increases from 100 to 200 sessions.**

The figure: **survival curve** = RWCA vs. memory budget fraction B/|H|, for each system, at fixed history length N=200.

---

## Why This Is a Landmark

1. **Non-monotone baseline behavior.** IWT predicts baselines show overload inversion at critical B/|H|. The landmark verifies this prediction empirically: a single threshold budget below which all baselines collapse.

2. **Unfakeable.** The RWCA is risk-weighted; baselines cannot compensate by getting easy questions right. CCB-RareCritical has exactly one rare-critical witness per 20 sessions; only systems that track it across 200 sessions score above 0.70.

3. **Budget-scaling discriminates systems.** At unlimited budget, full-context LLM scores ≈ 1.0. The landmark shows what happens at bounded budgets — the regime that actually matters for long-running agents.

4. **One number.** ΔRWCA = RWCA(Level-3) − RWCA(best-baseline) at B/|H| = 0.10. Target: ΔRWCA ≥ 0.20.

---

## Experimental Setup

### Systems (8 baselines + Level-3)

| System | Budget mechanism |
|---|---|
| Level-3 ESWP (ai-knot v2 + S1–S6) | ΔF-write + Landauer-ODE + sheaf-section read |
| v2-Sprint1-placeholder | regret_charge = risk_severity, greedy plan |
| Recency-only | keep most recent B atoms by observation_time |
| Frequency-only | keep most accessed B atoms |
| Full-context-128K | keep all sessions up to 128K token limit, truncate oldest |
| MemGPT-style | two-tier: working (small B_w) + archive (B_a); LRU paging |
| Mem0-style | extract salient facts; keep top-B by salience score |
| LightMem-style | sleep-time summary compression; keep top-B summaries |
| Random | uniform random selection of B atoms |

For baselines without an adapter: approximate behavior with parameterized variants of the MemoryAdapter interface.

### Dataset: CCB-RareCritical-Landmark

- 4 domains × 5 seeds × 1 history per seed = 20 histories
- Each history: 200 sessions, 3 rare-critical witnesses (planted at sessions 20, 70, 140)
- Probe queries: at delays 50, 100, 150 after each plant → 9 probes per history
- Total probe queries: 20 × 9 = 180
- Ground truth: counterfactual twin for each history (as in CCB spec)

### Budget Sweep

Test at B/|H| ∈ {0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00}

where B = number of atoms in memory, |H| = total atoms extractable from all sessions.

At B/|H| = 1.00: all systems have full memory → should all score similarly (baseline sanity check).
At B/|H| = 0.03: severe budget → only Level-3 expected to pass.

### Metrics

Primary: RWCA at B/|H| = 0.10 (the "headline number")
Secondary: PCR (pack coverage) and ESR (extraction sufficiency) decomposition
Reproducibility: 5 seeds, report mean ± 1 std, paired t-test Level-3 vs. best-baseline

### Compute Budget

```
20 histories × 9 probes × 2 (original + counterfactual) × 9 budget points × 9 systems
= 20 × 9 × 2 × 9 × 9 = 29,160 reader calls (worst case, most are cached)

With caching (same pack → same render → same reader call):
estimated actual calls: ~8,000

At $0.001 per call (gpt-4o-mini): ~$8 total
Extraction probe adds same: ~$16 total for full landmark run
```

---

## Expected Result and Failure Modes

### Expected Figure Shape

```
RWCA
1.0 |-------- Level-3 ESWP (flat, ≥ 0.90 down to B/|H|=0.10) --------
    |
0.8 |  -------  Mem0-style (≥ 0.80 at B/|H|=0.20, drops at 0.10)
    |
0.6 |    ---------  LightMem-style (drops at B/|H|=0.15)
    |       ------ MemGPT-style
    |          ------  Recency-only / Frequency-only
0.4 |              -------  Sprint1-placeholder
    |                  ------  Random
    +-----------------------------------------------------------> B/|H|
    0    0.05  0.10  0.15  0.20  0.30  0.50  1.00
```

The landmark curve is Level-3 staying ≥ 0.90 from B/|H| = 0.10 to 1.00, with all baselines dropping below 0.70 at B/|H| ≤ 0.10.

### Failure Modes and What They Indicate

| Failure | What it means |
|---|---|
| Level-3 RWCA < 0.80 at B/|H|=0.10 | ΔF-write or Landauer-ODE not protecting rare-critical atoms; investigate PCR (if PCR < 0.70, write/forget failing; if PCR ≥ 0.70, read/extract failing) |
| Best baseline RWCA > 0.80 at B/|H|=0.10 | Landmark gap too small; may need richer rare-critical scenario (more sessions, lower plant frequency) |
| Curves don't spread until B/|H| < 0.05 | Budget sweep needs to go lower; try 0.01, 0.02 |
| Level-3 drops at high budget (B/|H|=0.50) | Regression — sheaf-section gluing or RG-consolidate over-pruning; investigate |
| ESR < 0.70 even when PCR > 0.80 | Reader-extraction failure dominant; extraction probe not improving packs |

### Dead-End Protocol

If the landmark cannot be achieved (Level-3 ΔRWCA < 0.10 at B/|H|=0.10 across 3 seeds):

1. **Report the survival curve itself as a benchmark contribution**: empirical measurement that rare-critical survival degrades non-linearly for all current systems. The lack of a positive result IS the scientific contribution.
2. **Publish CCB generator + scoring as standalone artifact**: enables other researchers to test future systems.
3. **Narrow claim to extraction-sufficiency only**: ESR difference between Level-3 and baselines may be significant even when RWCA is not, establishing ESWP's extraction-side contribution independently.

---

## Reproducibility Requirements

The landmark run must be fully reproducible from a single command:

```bash
.venv/bin/python src/ai_knot_v2/bench/ccb/runner.py \
  --domains MED SCH PID FIN \
  --seeds 0 1 2 3 4 \
  --n-sessions 200 \
  --budget-sweep 0.03 0.05 0.08 0.10 0.15 0.20 0.30 0.50 1.00 \
  --systems level3 sprint1 recency frequency fullcontext memgpt mem0 lightmem random \
  --output results/landmark_$(date +%Y%m%d).json
```

Output: JSON scorecard per system per domain per budget point; CSV for figure generation; `figures/landmark_curve.pdf` auto-generated.

All random seeds fixed; no LLM calls at generation time (only at evaluation time); results deterministic given same LLM provider + model.

---

## Relation to Paper

The landmark figure is **Figure 1** of the minimal publishable paper:
- X-axis: memory budget fraction B/|H|
- Y-axis: RWCA (risk-weighted counterfactual accuracy)
- Lines: Level-3 ESWP vs. 8 baselines
- Error bars: ±1 std across 5 seeds
- Caption: "ESWP Level-3 maintains ≥ 0.90 RWCA at 10% budget across all domains; all baselines collapse below 0.70."

Supporting figures:
- Figure 2: PCR vs. ESR decomposition (showing IWT vs. ESWP failure modes)
- Figure 3: History-length scaling (N=100 vs. N=200 — gap should widen)
- Table 1: Per-domain RWCA scores at B/|H| = 0.10

Single headline number for abstract: **ΔRWCA = 0.23** (target) at B/|H| = 0.10, 5 seeds, p < 0.01 paired t-test.
