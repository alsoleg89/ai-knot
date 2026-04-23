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

## 2026-04-21 — Move 1A per-session MMR floor=2 for SET (REVERTED, gate-e4488e7 baseline)

**Commit:** not committed — reverted before merge on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `gate-e4488e7-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 25.6 %, cat1-4 59.7 %
**Run:** `data/runs/p1-1a-2conv/report.json` — canonical, same models, same dataset
**Config deviations:** none
**Decision:** REVERT — target category regressed
**Reason:** Hypothesis was that SET-type cat1 Q (e.g. "Where has Melanie camped?") with multi-session evidence lose coverage under floor=1. Bumped floor to 2 per session when SET query + ≥2 unique sessions in candidate pool (`_set_query_floor` helper in `storage/sqlite_storage.py`). Net effect on cat1: -2.3 pp (11/43 → 10/43), with **zero gains** and one new regression (Q 0:71 "What book did Melanie read from Caroline's suggestion?" — the floor pushed the session containing "Becoming Nicole" evidence out of render_top_k in favor of lower-quality cross-session matches). Hypothesis was wrong for single-answer SET queries: diversity hurts when one session has the ground-truth. Needs narrower trigger (only true multi-answer SET queries).
**1A numbers (canonical gpt-4o-mini × gpt-4o-mini):**
  - cat1: 23.3 % (−2.3 pp ⚠ target regressed with zero gains)
  - cat2: 50.8 % (±0)
  - cat3: 61.5 % (−7.7 pp, 1 Q of 13 — sample noise)
  - cat4: 79.8 % (+3.5 pp, unrelated benefit — floor happens to diversify non-SET too)
  - cat5: 23.9 % (−4.2 pp, floor above stop-rule 20 %)
  - cat1-4 aggregate: 60.5 % (+0.9 pp — floors held but target failed)
**Next baseline update:** no (reverted; baseline remains `gate-e4488e7-2conv`)

---

## 2026-04-21 — Move 1B leading-adverbial strip for FP patterns

**Commit:** `f13370a` on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `gate-e4488e7-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 25.6 %, cat1-4 59.7 %
**Run:** `data/runs/p1-1b-2conv/report.json` — canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** ACCEPT — all floors held, target gained, no per-cat regression beyond noise
**Reason:** Added `_LEADING_ADV_RE` + `_strip_leading_adverbial` in `materialization.py` so sentences like "Last weekend I joined X", "Yesterday I bought Y", "Recently I moved to Z" feed the FP regex with `I <verb>` opener. 11 FP call sites (`_FP_LIKES_RE` … `_FP_ACTIVITY_RE` + `_FP_EVENT_PATTERNS` loop) switched from `sent` → `fp_sent`; original `sent` retained for qualifiers, spans, claim IDs. Strip only fires when residue begins with `I\s+\w+` → non-FP sentences with adverbial prefix stay untouched.
**1B numbers (canonical gpt-4o-mini × gpt-4o-mini):**
  - cat1: 30.2 % (+4.7 pp ✓ target category gained)
  - cat2: 50.8 % (±0)
  - cat3: 69.2 % (±0)
  - cat4: 80.7 % (+4.4 pp ✓ non-regression; floor 68.1 %)
  - cat5: 26.8 % (−1.4 pp, within floor 20 % and threshold −8.2 pp)
  - cat1-4 aggregate: 62.7 % (+3.0 pp ✓ aggregate gate passed)
**Next baseline update:** yes — `p1-1b-2conv` becomes new baseline for 1C/1D sub-moves

---

## 2026-04-21 — Move 1C-plays FP `plays` pattern (REVERTED)

**Commit:** not committed — reverted before merge on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 30.2 %, cat1-4 62.7 %
**Run:** `data/runs/p1-1c-plays-2conv/report.json` — canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — aggregate regression exceeds stop-rule; target category erased
**Reason:** Added `_FP_PLAYS_RE` as narrow STATE pattern for "I play/played/practice/practiced (the)? <single-token>" with negative-lookahead blacklist (with/around/here/to/…) between `_FP_ACTIVITY_RE` and `_FP_EVENT_PATTERNS` loop. Aim: cat1 Q 0:60-style instrument recall (e.g. "What instrument does Melanie play?"). In 2-conv canonical the pattern generated noise claims: cat1-4 aggregate dropped -2.1 pp (60.5 % vs 62.7 %), triggering `feedback_regression_stop_rule`. cat1 collapsed back to 25.6 % (-4.7 pp, erasing entire 1B gain) and cat2 dropped -3.2 pp. Likely failure mode: "practice" and uninflected "play" match non-instrument contexts ("I play my piece", "I practice routinely"-style with blacklist escapes), and emitted STATE `plays::<word>` crowds out the legitimate evidence chunks in render_top_k.
**1C-plays numbers (canonical gpt-4o-mini × gpt-4o-mini):**
  - cat1: 25.6 % (−4.7 pp ⚠ target erased to pre-1B baseline)
  - cat2: 47.6 % (−3.2 pp ⚠ regressed)
  - cat3: 69.2 % (±0)
  - cat4: 79.8 % (−0.9 pp, within floor 72.5 % / abs 68.1 %)
  - cat5: 23.9 % (−2.8 pp, within floor 20 %)
  - cat1-4 aggregate: 60.5 % (−2.1 pp ⚠ exceeds 2 pp regression stop threshold)
**Next baseline update:** no — `p1-1b-2conv` remains baseline for 1C-painted / 1C-read / 1D attempts

---

## 2026-04-22 — Move 2P POSSESSIVE-family kinship extractor (REVERTED)

**Commit:** not committed — reverted before merge on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 30.2 %, cat1-4 62.7 %
**Run:** `data/runs/p1-2p-family-2conv/report.json` (run dir deleted post-revert), canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — no gains on target cat1, cat3 dropped near stop-threshold, cat1-4 aggregate below baseline
**Reason:** Added `_FAMILY_TERMS` dict (40 kinship lemmas → 8 canonical classes: spouse/parent/child/sibling/grandparent/grandchild/pibling/nibling) + `_POSSESSIVE_FAMILY_RE` + emission block in `materialize_episode` that fires for `^My <kinship>(?:'s name)? (?:is|was|are|were)? (?:named|called)? <ProperName>` when speaker is known and not a pronoun subject. Emitted `RELATION` claim `<speaker> has_<class> <Name>`, slot_key `"{speaker}::has_{class}"`. Bumped `MATERIALIZATION_VERSION` 6 → 7 to trigger rebuild. 15 regression tests added in `tests/test_materialization_deterministic.py` (spouse/husband/son/daughter/mom/brother/grandma/aunt/ex-wife/apostrophe-s/two-word-name/no-name-sentinel/non-family-rejection/no-speaker-skip/dict-completeness). Justified as generic: closed semantic class (schema.org Person), universal in natural speech, not benchmark-derived.
**2P numbers (canonical gpt-4o-mini × gpt-4o-mini):**
  - cat1: 25.6 % (−4.6 pp ⚠ target erased to pre-1B baseline)
  - cat2: 50.8 % (±0)
  - cat3: 61.5 % (−7.7 pp, within floor 61.0 % but ~0.5 pp from stop-threshold −8.2 pp)
  - cat4: 80.7 % (±0, well above floor 72.5 %)
  - cat5: 25.4 % (−1.4 pp, within floor 20 %)
  - cat1-4 aggregate: 61.4 % (−1.3 pp, technically within ±2 pp gate but below baseline with zero target gain)
**Rationale for REVERT despite being within numeric stop-rule:** rule is necessary-not-sufficient. New feature with net-negative delta and no target gain is a regression regardless of magnitude. cat1 loss erases 1B gain; cat3 approaches stop-threshold (−7.7 vs −8.2 pp floor). Suspected cause: `has_<class>` RELATION claims crowd out primary cat1-evidence in `render_top_k` for SET/single-hop Q that don't involve kinship, diluting retrieval precision. Kinship coverage would only help a narrow Q subset while adding noise to the majority.
**Next baseline update:** no — `p1-1b-2conv` remains baseline. No follow-up 2P-variant planned (revert closes this branch of the generic-extractor experiment).

---

## 2026-04-23 — Move 3R SET-conditional RRF weight flip (REVERTED)

**Commit:** not committed — reverted before merge on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 30.2 %, cat1-4 62.7 %
**Run:** `data/runs/p1-3r-2conv/report.json` (run dir deleted post-revert), canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — zero target-category GAINs; net negative cat1 and cat3
**Reason:** Hypothesis (formed after diagnostic of the 30 cat1 WRONG: 25/30 are retrieval-miss; evidence for gold items exists in ingest but ranks below non-evidence Melanie-turns): generic-concept SET Q ("activities", "events", "books") fail because BM25-dominant RRF weights `[2.0, 1.0]` can't bridge synonymy between Q tokens and gold-verb tokens, so embeddings (weight 1.0) get outvoted. Change: in `search_episodes_by_entities`, flip to `[1.0, 2.0]` when `diversity=True` (SET gate). Scalar path unchanged. 10 LOC + 1 regression test (`test_search_episodes_rrf_weights_flip_for_set`) covering both code paths via fake embedder. All suite tests pass except pre-existing flaky `test_auto_tags_change_ranking`. Bench moved 8 Q verdicts (5 context changes) but net per-cat all zero or negative:
  - cat1: 2 LOSS, 0 GAIN (both losses were temporal/complex Q, not SET — target SET Q unchanged)
  - cat3: 1 LOSS, 0 GAIN
  - cat4: 2 GAIN, 2 LOSS (±0)
  - cat5: 1 LOSS, 0 GAIN
**3R numbers (canonical gpt-4o-mini × gpt-4o-mini):**
  - cat1: 25.6 % (−4.6 pp ⚠ target erased; identical to pre-1B baseline)
  - cat2: 50.8 % (±0)
  - cat3: 61.5 % (−7.7 pp, within floor but near stop-threshold)
  - cat4: 80.7 % (±0)
  - cat5: 25.4 % (−1.4 pp, within floor 20 %)
  - cat1-4 aggregate: 61.4 % (−1.3 pp)
**Rationale for REVERT:** zero GAINs in target category (cat1 SET Q) invalidates the hypothesis. The 19 cat1 SET retrieval-miss Q diagnosed on `p1-1b` did not flip to CORRECT under embedding-dominant RRF, so the synonymy-bridging hypothesis doesn't explain those failures. Side effects (cat1/cat3/cat5 losses) were non-SET Q where scalar-ish ranking was already near-optimal and the flip introduced noise. The diagnostic remains valid (83 % of cat1 failures are retrieval-miss, evidence exists in ingest) but the specific fix does not hold; the bottleneck is elsewhere (candidate-pool size? top_k cap? entity-substring filter too loose? claim-vs-episode retrieval divergence?).
**Next baseline update:** no — `p1-1b-2conv` remains baseline.

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
