# Bench Decision Log

Append one entry per experiment. Every commit that touches product code or
bench settings must have a paired entry here before it lands on the branch.

**Decisions:** ACCEPT | REVERT | PARK (PARK = keep branch alive, diagnosis in progress)

---

## Template

```
## YYYY-MM-DD — <title>

**Commit:** `<sha>` on `<branch>`
**Baseline:** `baselines/latest_2conv.json` (label: `<label>`, cat1-4 = X %)
**Run:** `data/runs/<run-id>/report.json` (cat1-4 = Y %, delta = ±Z pp)
**Config deviations:** none | [list of knob diffs from canonical.json]
**Decision:** ACCEPT | REVERT | PARK
**Reason:** one-line summary
**Next baseline update:** yes | no (insufficient improvement) | no (gate failed)
```

---

## 2026-04-21 — Decision pipeline bootstrap

**Commit:** `(pipeline PRs 1–4)` on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1 manually recorded, 59.2 % cat1-4 (unverified in registry)
**Run:** n/a — infra-only PRs, no bench run
**Config deviations:** n/a
**Decision:** ACCEPT
**Reason:** Process infra; no product code changed; cannot regress bench numbers
**Next baseline update:** no (infra only)

---

## 2026-04-21 — Moves A+B+C+D (MMR, fallback gate, conditional RRF, debug trace)

**Commit:** `3d3752b` + `06f30f5` + `2251e1d` on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-06f30f5-2conv-drift` — drift run (ollama:qwen2.5:7b), NOT canonical
**Config deviations:** answer=ollama:qwen2.5:7b, judge=ollama:qwen2.5:7b; .env bug caused all prior runs to use ollama instead of OpenAI
**Decision:** PARK — drift run; awaiting canonical gpt-4o-mini gate `gate-23cd897-2conv`
**Reason:** .env was not sourced by tsx (Node); fixed in commit `20d25fd`; canonical gate now running
**Drift-run numbers (informational only, not comparable to baseline):**
  - cat1-4 aggregate: 60.1 % (+0.9 pp vs pf3 baseline of 59.2 %)
  - cat1: 18.6 % (−11.6 pp — model artifact: qwen much weaker on set-valued list-all questions)
  - cat2: 57.1 % (+11.1 pp — confirms MMR session diversity working)
  - cat3: 53.8 % (−7.7 pp — within noise on qwen)
  - cat4: 78.1 % (+0.9 pp)
**Next baseline update:** pending canonical run result

---

## 2026-04-21 — Moves A+C+E canonical gate (Move B reverted)

**Commit:** `5f522c1` (Moves A+B+C+D+E) → `68ae0ea` (revert Move B) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-5f522c1-2conv` — canonical (gpt-4o-mini), COMPLETED
**Config deviations:** none
**Decision:** REVERT (partial) — Move B fallback gate `<5` reverted to `<2`
**Reason:** gate-5f522c1 cat1 dropped -11.6pp (gate FAIL, threshold -8.2pp). Diagnosis: Move B widened BM25 fallback gate from <2→<5, adding noise claims for 3-4-bundle entities that then propagated into joint RRF (Move C) and polluted scalar/cat1 context. Aggregate only -1.3pp but per-cat FAIL. cat2 +6.4pp, cat3 +7.7pp (Moves A+C are helping — kept).
**gate-5f522c1 numbers (canonical):**
  - cat1-4: 57.9 % (-1.3 pp)
  - cat1: 18.6 % (-11.6 pp ⚠ GATE FAIL)
  - cat2: 52.4 % (+6.4 pp)
  - cat3: 69.2 % (+7.7 pp)
  - cat4: 74.6 % (-2.6 pp)
**Next baseline update:** no (gate failed)

---

## 2026-04-21 — Fallback gate fix (Move B reverted, gate-68ae0ea)

**Commit:** `68ae0ea` on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-68ae0ea-2conv` — killed before completion (stdout-pipe buffering bug; 79/304 answers recorded then idled)
**Config deviations:** none
**Decision:** SUPERSEDED — run did not complete; continuation is `gate-80200ff-2conv` after Move C revert
**Reason:** Move B revert alone insufficient to diagnose cat1 without also isolating Move C. Re-run below.

---

## 2026-04-21 — Move C revert (joint RRF removed, gate-80200ff)

**Commit:** `80200ff` on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-80200ff-2conv` — canonical (gpt-4o-mini), COMPLETED
**Config deviations:** none
**Decision:** PARK — aggregate PASSED (+1.3pp), per-cat cat1 FAILED (-9.3pp, threshold -8.2pp); diagnosis in progress on F1
**gate-80200ff numbers (canonical):**
  - cat1-4: 60.5 % (+1.3 pp) ✓ aggregate gate passed
  - cat1: 20.9 % (-9.3 pp ⚠ GATE FAIL)
  - cat2: 50.8 % (+4.8 pp ✓)
  - cat3: 69.2 % (+7.7 pp ✓)
  - cat4: 79.8 % (+2.6 pp ✓)
**Reason:** Move C revert alone did not recover cat1 — the cat1 drop vs pf3-phase1 is caused by something OTHER than joint RRF. Failure analysis on log.jsonl shows 27/35 cat1 WRONG flagged "evidence-missing-gold" (R-type); manual inspection reveals two sub-patterns: (a) wrong-entity window contamination — `search_episodes_by_entities` entity-filters centers but 3-turn prev/next expansion pulls in counterparty turns that dominate `render_top_k`; (b) claim-subject misattribution — 240 of 288 extracted claims have non-person subjects like "Dance", "The kids", "That photo", "Your store" because the grammatical subject is taken verbatim instead of being resolved to the speaker. Move F1 (centers-first window expansion) targets sub-pattern (a); Move G (extractor subject normalization) would target sub-pattern (b).
**Next baseline update:** no (gate failed per-cat)

---

## 2026-04-21 — F1-alone (Z reverted, Move F1 centers-first expansion)

**Commit:** `0b8f778` (revert of Move Z `fdb62dd`) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-0b8f778-2conv` — canonical (gpt-4o-mini), COMPLETED
**Config deviations:** none
**Decision:** ACCEPT (no promote) — aggregate PASS, but cat1 -2.3pp so not strictly better than baseline
**Reason:** F1 (centers-first window expansion) alone produced aggregate +0.9pp (60.1% vs 59.2% baseline) with gate PASS on all per-cat thresholds. However cat1 slipped 27.9% vs 30.2% baseline — retrieval bottleneck on single-hop factoids persists. cat2 gained +4.8pp (50.8%) confirming F1's value for multi-hop windowing. Previous stacked moves (G+Y crashed MCP subprocess; Z exhaustive entity union regressed cat1 to 21% at mid-run) both reverted.
**gate-0b8f778 numbers (canonical):**
  - cat1-4: 60.1 % (+0.9 pp) ✓ aggregate gate passed
  - cat1: 27.9 % (-2.3 pp, within threshold >-8.2pp)
  - cat2: 50.8 % (+4.8 pp ✓)
  - cat3: 61.5 % (+0.0 pp)
  - cat4: 77.2 % (-0.0 pp)
**Next baseline update:** no — cat1 regression makes this not strictly better; branch kept for further moves

---

## 2026-04-21 — Move M RM3 pseudo-relevance feedback

**Commit:** `e4488e7` → `59d139f` (revert) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** pf3-phase1-2conv, 59.2 % cat1-4 (gpt-4o-mini)
**Run:** `gate-e4488e7-2conv` — canonical (gpt-4o-mini), COMPLETED
**Config deviations:** none
**Decision:** REVERT — per `feedback_regression_stop_rule` (cat1 -4.6pp > 2pp threshold)
**Reason:** RM3 two-pass expansion (first top_k=10, extract top-8 content terms, second full top_k) intended to replicate SmartSearch's published +9pp cat1 gain. Instead regressed cat1 -4.6pp. Failure mode: when first-pass top-10 already misses the fact-turn (which is the defining failure mode of LOCOMO cat1 R-type errors), RM3 expansion terms come from noise neighborhood and amplify mis-ranking. cat3 did gain +7.7pp (inference benefits from context diversification) but not worth the cat1 hit.
**gate-e4488e7 numbers (canonical):**
  - cat1-4: 59.7 % (+0.5 pp) ✓ aggregate gate passed
  - cat1: 25.6 % (-4.6 pp ⚠ exceeds regression stop threshold 2pp)
  - cat2: 50.8 % (+4.8 pp)
  - cat3: 69.2 % (+7.7 pp ✓ inference context diversification)
  - cat4: 76.3 % (-0.9 pp)
**Next baseline update:** no (reverted)

---

## Known bad artifacts

### `data/runs/ddsa-off/`

`report.json` records `gpt-4o-mini` for both models.
`checkpoint.json` records `ollama:qwen2.5:7b` for both.
Root cause: checkpoint saved model from run-start env; report read model
from end-of-run CLI state after a manual model swap mid-run.
Fix shipped in Phase 2 (run_config.json written once at start, never mutated).
Run numbers from this dir are **not comparable** to any baseline.

### `data/runs/phase2-2conv/`, `phase3-2conv/`, `phase5-2conv/`

All three used `ollama:qwen2.5:7b` for answer and judge (local dev config).
pf3 baselines used `gpt-4o-mini`. Numbers are **not comparable**.
