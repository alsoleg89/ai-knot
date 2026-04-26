# Lexical Bridge Stage 1 Gate — FAIL

Run: `lexical3-proc-c0` (conv0, 199 Qs) + `lexical3-proc-c1` (conv1, 105 Qs)  
Branch: `feat/lexicon`, `AI_KNOT_LEXICAL_BRIDGE=1`, top_k=60, gpt-4.1-nano  
Date: 2026-04-26

---

## Accuracy vs Baseline

| Conv | Baseline | Lexical3 | Δ |
|------|----------|----------|---|
| conv0 | 135/199 = 67.8% | 107/199 = 53.8% | **-14.0pp** |
| conv1 | 65/105 = 61.9% | 48/105 = 45.7% | **-16.2pp** |
| **combined** | **200/304 = 65.8%** | **155/304 = 51.0%** | **-14.8pp** |

Per-category (combined):

| Cat | Baseline | Lexical3 | Δ |
|-----|----------|----------|---|
| cat1 single-hop | 17/43 = 39.5% | 10/43 = 23.3% | -16.3pp |
| cat2 multi-hop | 37/63 = 58.7% | 21/63 = 33.3% | **-25.4pp** |
| cat3 temporal | 10/13 = 76.9% | 11/13 = 84.6% | +7.7pp |
| cat4 open-ended | 90/114 = 78.9% | 71/114 = 62.3% | -16.7pp |
| cat5 adversarial | 46/71 = 64.8% | 42/71 = 59.2% | -5.6pp |

---

## PROC Diagnostics

| Metric | Baseline c0 | Lexical3 c0 | Baseline c1 | Lexical3 c1 |
|--------|-------------|-------------|-------------|-------------|
| PoolGoldRecall@K | 0.962 | 0.962 | 0.979 | 0.929 |
| PackGoldRecall@Budget | 0.547 | **0.372** | 0.490 | **0.417** |
| LLM-fail bucket | 5 | 38 | 9 | 37 |
| hard-miss bucket | 2 | **35** | 3 | 13 |
| partial-recall | 9 | 16 | 5 | 6 |

---

## Root Cause

The `work_career` frame expansion adds generic terms (`work, career, hire, employ,
role, position, company`) to ALL factual queries about jobs/careers. These
high-frequency terms score strongly in BM25 and displace specific dated gold
facts from the top-k pool.

Effect on cat2 "When did..." queries:
- Query: "When did Jon lose his job as a banker?"
- Expansion adds: `work(0.7), career(0.7), hire(0.6), employ(0.6), role(0.6), position(0.5), company(0.6)`
- Result: many generic career-related facts outrank the specific dated event fact
- cat2 conv1: **1/26 correct** (vs baseline 16/26)

**hard-miss spike**: c0 hard-miss went 2 → 35. Generic expansion terms push gold
facts below top-k=60 cutoff entirely — PoolGoldRecall stays high in aggregate
but per-question gold-in-pool drops to 0 for 35 questions.

**PackGoldRecall collapse**: -17.5pp c0, -7.3pp c1. Even when gold enters pool,
the expanded-term distractor facts crowd it out of the final pack.

---

## LexicalExpansionUplift

| Metric | c0 | c1 | Gate |
|--------|----|----|------|
| PoolGoldRecall delta | 0.0pp | **-5.0pp** | ≥ +2pp required |
| PackGoldRecall delta | **-17.5pp** | **-7.3pp** | not applicable |
| Accuracy delta | -14.0pp | -16.2pp | not applicable |

**LexicalExpansionUplift = -5.0pp average** → Gate FAIL (required ≥ +2pp).

---

## Decision

**Stage 1 FAIL** — per plan §3 decision tree: "P1 fail → halt; pivot."

Stop-and-revert rule triggered: regression > 2pp on both convs.

**Pivot: Phase B (Evidence Pack V2)**

Rationale from baseline PROC:
- PackGoldRecall baseline = 84.0% → 15% pack loss is the structural bottleneck
- LLM-fail = 57.7% of wrong answers → pack quality directly drives LLM success
- Lexical bridge addresses retrieval coverage but baseline PoolGoldRecall is already 98.7%
- Phase B addresses the right bottleneck; Phase A addresses the wrong one

feat/lexicon branch: do NOT merge. Park for redesign (stricter weight caps,
TEMPORAL/NAVIGATIONAL exclusion extended to specific-event queries).
