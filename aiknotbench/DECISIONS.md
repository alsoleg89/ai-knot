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

## 2026-04-23 — Move 4 implicit-SET aux widening (REVERTED)

**Commit:** `5c99ac6` (reverted by `e453aef`) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical (gpt-4o-mini × gpt-4o-mini), cat1 30.2 %, cat1-4 62.7 %, cat4 76.3 %, cat5 28.2 %
**Run:** `data/runs/m4-2conv/report.json`, canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — cat1-4 aggregate drop exceeds stop-rule; target category (cat1) regressed by full 1B gain
**Reason:** Hypothesis (formed after re-analysis of the 25 cat1 retrieval-miss Q diagnosed on `p1-1b`: 15/25 are classified `AnswerSpace.DESCRIPTION` by `_detect_geometry` because the implicit-SET gate at `query_contract.py:385` fires only for `what/which + {has, have} + NOUN_HEAD`). Change: widen the aux set to the full closed English auxiliary class `{has, have, does, do, did, is, are, was, were}`, keeping `_SET_NOUN_HEADS` filter intact. 10 LOC + 2 regression-test classes (positive: 6 Q with non-`has/have` aux correctly route as SET; negative: 6 singular/bool Q unchanged). All suite tests pass except pre-existing flaky `test_auto_tags_change_ranking` and `test_scoped_recall_finds_all_entity_mentions` (both fail on HEAD).
**m4 numbers (canonical gpt-4o-mini × gpt-4o-mini, 2-conv):**
  - cat1: 25.6 % (−4.7 pp ⚠ erases 1B gain; back to pre-1B level)
  - cat2: 49.2 % (−1.6 pp)
  - cat3: 69.2 % (±0)
  - cat4: 78.9 % (+2.6 pp, above floor 68.1 %)
  - cat5: 25.4 % (−2.8 pp, above floor 20 %)
  - cat1-4 aggregate: 60.5 % (−2.2 pp, trips >2 pp stop-rule)
**Rationale for REVERT:** `feedback_regression_stop_rule` fires on cat1-4 aggregate (−2.2 pp > 2 pp threshold) and on target-category collapse (cat1 −4.7 pp = full 1B gain erased). cat4 +2.6 pp is the only directional positive; cat5 regression (−2.8 pp) plus cat2 regression (−1.6 pp) plus target-cat collapse make the move net-negative. Mechanistic read: widening the SET gate routes single-answer cat1/cat5 Q ("What is Alice's focus?" is safe only because "focus" singular; but many cat1/cat5 target Q have `_SET_NOUN_HEADS`-qualifying nouns paired with broader aux verbs) into the SET pipeline — which applies MMR diversity and per-session floor. For genuinely single-answer Q, diversity re-ranking demotes the top-correct episode and the floor forces cross-session spread that admits off-topic context. The classifier bottleneck diagnosis remains valid but the fix is wrong: SET-widening must be coupled with a mechanism that retains single-answer precision when the Q is list-shaped but the evidence concentrates in one session.
**Next baseline update:** no — `p1-1b-2conv` remains baseline.

---

## 2026-04-23 — Move 5 universal balanced-profile cap widening (REVERTED)

**Commit:** `fedb3d7` (reverted by `2749df4`) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical, cat1 30.2 %, cat2 50.8 %, cat3 69.2 %, cat4 76.3 %, cat5 28.2 %, cat1-4 62.7 %
**Run:** `data/runs/m5-2conv/report.json`, canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — cat3 dropped −15.4 pp (7.2 pp below 8.2 pp floor); cat1-4 aggregate dropped −3.9 pp (exceeds 2 pp stop-rule)
**Reason:** Hypothesis (formed from read-only retrieval trace on `p1-1b-2conv` knot.db, documented in `memory/project_locomo_cat1_retrieval_diagnostic.md`): cat1 retrieval-miss Q have gold evidence ranking at raw ranks 13–21, just outside the balanced profile's `render_top_k=12` / `raw_search_top_k=20`. The SET-only widening in `_caps_for_contract` (render 18, raw 28) wasn't load-bearing as a gate — the asymmetry between DESCRIPTION (12/20) and SET (18/28) was causing non-SET cat1 Q to lose gold at ranks 16–18. Change: widen balanced to `(raw=28, window=32, collect=22, render=18, budget=30 000)` unconditionally. `_caps_for_contract` becomes a no-op on balanced, still widens narrow for SET. 2 files, 24 insertions. Tests updated (`test_set_caps_widened_vs_scalar` now asserts monotone widening rather than strict widening on balanced base).
**m5 numbers (canonical gpt-4o-mini × gpt-4o-mini, 2-conv):**
  - cat1: 27.9 % (−2.3 pp)
  - cat2: 47.6 % (−3.2 pp)
  - cat3: 53.8 % (−15.4 pp ⚠⚠⚠ catastrophic)
  - cat4: 77.2 % (+0.9 pp)
  - cat5: 32.4 % (+4.2 pp, the only winner)
  - cat1-4 aggregate: 58.8 % (−3.9 pp, trips 2 pp stop-rule)
**Rationale for REVERT:** cat3 collapse (9/13 → 7/13, lost 2 of 13 Q, 7.4 pp below the absolute floor) forces immediate revert under `feedback_regression_stop_rule` per-cat guard. cat1-4 aggregate regression reinforces. Only cat5 (open-ended description) benefited — consistent with the thesis that widening context helps description-style Q and hurts scalar/hop Q. Mechanistic read: cat3 (hop/aggregate) depends on precise evidence alignment; more context at render_top_k=18 dilutes the signal with near-neighbor turns that confuse the LLM's multi-step reasoning. Cat2 (multi-hop) regressed for the same reason. cat1 target regression means the diagnostic insight (gold at ranks 13–21) didn't actually convert to CORRECTs — the widened render window brought more noise than gold. The diagnostic remains valid, but universal widening is the wrong lever. Future direction: **rank-time precision** (e.g. narrowing the entity filter at source via speaker-prefix signal or Q-token pre-filter) rather than **truncation-width widening**, since the latter adds noise at the same rate as signal.
**Next baseline update:** no — `p1-1b-2conv` remains baseline.

---

## 2026-04-23 — Move 2S1 SSP instrumentation / mention graph (REVERTED, not committed)

**Commit:** none — change applied in working tree only, reverted before commit on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical, cat1 30.2 %, cat2 50.8 %, cat3 69.2 %, cat4 76.3 %, cat5 28.2 %, cat1-4 62.7 %
**Run:** `data/runs/p-2s1-2conv/report.json`, canonical (deviations=[]), same models
**Config deviations:** none
**Decision:** REVERT — cat1-4 aggregate dropped −3.0 pp (exceeds 2 pp stop-rule). Attributed to LLM stochasticity, not real regression (see rationale)
**Reason:** Plan Phase 2 stage 2S1 — first architectural prep for Synaptic Subject Projection. Hypothesis: build a mention-context graph (MCG) per (agent_id, session_id), project non-named-subject claims onto candidate entity hypotheses with edge weights (same-sentence-named 1.00, speaker 0.70, adjacent-turn 0.55, session-wide 0.25, MIN 0.40), attach as `qualifiers["ssp_hypotheses"]` on each claim as instrumentation-only scaffolding. Scaffolding should be retrieval-neutral (no operator / renderer / index reads the new qualifier) and enable 2S2/2S3 shadow-claim emission later. Change: new `src/ai_knot/mention_graph.py` (~205 LOC: `MentionGraph` dataclass, `project_for_turn`, `attach_subject_hypotheses_to_claims` post-pass) + single post-pass call in `rebuild_claims_from_raw` + 15 unit tests in `tests/test_mention_graph.py`. Verified code-path neutrality: `grep "qualifiers\." src/ai_knot/query_runtime.py src/ai_knot/query_operators.py src/ai_knot/support_bundles.py` returned only `date_token`, `time_anchor`, `relative_time` — `ssp_hypotheses` is never consumed downstream.
**p-2s1-2conv numbers (canonical gpt-4o-mini × gpt-4o-mini, 2-conv):**
  - cat1: 23.3 % (10/43; −7.0 pp, 43-Q sample lost 3 Q)
  - cat2: 49.2 % (31/63; −1.6 pp, lost 1 Q)
  - cat3: 69.2 % (9/13; unchanged)
  - cat4: 78.1 % (89/114; +1.8 pp, gained 2 Q)
  - cat5: 23.9 % (17/71; −4.3 pp, lost 3 Q)
  - cat1-4 aggregate: 59.7 % (139/233; −3.0 pp, trips 2 pp stop-rule)
**Rationale for REVERT:** Aggregate −3.0 pp exceeds the 2 pp stop-rule by itself, so revert is mandatory. But *why* a retrieval-neutral change moved the bench is the important finding. The new qualifier is never read anywhere in retrieval, rendering, or indexing. The episode→claim set is identical module the opaque qualifier string. Identical render buses reach the LLM. The delta is gpt-4o-mini stochasticity on small per-category samples: n=43 cat1, n=13 cat3, n=71 cat5 — at temp=0 the sampler still differs across runs when any upstream float (score, tie-break, iteration order) varies, and ±3 Q per category is within the observed 2-conv noise band. cat3 unchanged supports this (cat3 is the most noise-sensitive with n=13, and it would move first under a real retrieval shift). **New operational lesson:** the 2-conv gate cannot discriminate instrumentation-level moves from LLM noise — the plan's "cat1 ±0.5 pp" gate for 2S1 was unrealistic at this sample size. Any future code-neutral scaffolding (including 2S2 dry-run, 2S3 shadow emission, 2S4 folded storage) must be validated on either full-10 (n=233) or repeated-2-conv (≥3 runs averaged) or a targeted micro-bench on the Q family the instrumentation enables. Chaining more 2-conv gate moves on this baseline will just produce more reverts. Documented in `memory/project_locomo_phase1_retrieval_exhausted.md` as the 6th consecutive revert on cat1-targeting work.
**Next baseline update:** no — `p1-1b-2conv` remains baseline. Working tree restored: `mention_graph.py` and `tests/test_mention_graph.py` deleted; `rebuild_claims_from_raw` post-pass call removed; materialization tests (54 passed, 1 skipped) confirm clean state.

---

## 2026-04-23 — Move 6A speaker-prefix BM25 boost (REVERTED, not-a-cat1-fix)

**Commit:** `ca29ad8` (reverted by `d1ea055`) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — canonical, cat1 30.2 %, cat2 50.8 %, cat3 69.2 %, cat4 **80.7 %**, cat5 **26.8 %**, cat1-4 **62.66 %**. (Note: prior DECISIONS entries referenced cat4=76.3% / cat5=28.2% / cat1-4=59.7% — those were the `gate-e4488e7-2conv` pre-1B baseline; this entry uses the real post-1B baseline from `p1-1b-2conv/report.json`.)
**Runs:** `data/runs/m6a-run1-2conv`, `m6a-run2-2conv`, `m6a-run3-2conv`/report.json — 3× canonical (gpt-4o-mini × gpt-4o-mini)
**Config deviations:** none
**Decision:** REVERT — not a cat1 fix; avg cat1 −2.33 pp while cat1 is the sole goal. Aggregate stayed inside 2 pp stop-rule (−0.43 pp) and per-category floors held, so formal stop-rule did not trigger — but accepting this move would hide the actual cat1 bottleneck.
**Reason:** Diagnosis from `memory/project_locomo_phase1_retrieval_exhausted.md`: (A) pool composition — `raw_text LIKE '%<name>%'` returns ~280 candidates per major entity in 2-conv; BM25+embedding RRF cannot separate fact-bearing speaker-turns from counterparty mentions. New observation this session: `raw_episodes.speaker` column stores *role* (`user`/`assistant`), **not** name; but 100 % of LOCOMO turns prefix `raw_text` with `<Name>: …` — single source of truth for who is actually speaking. Change: added `_speaker_prefix_boost(raw_text, entities) → float` returning `AI_KNOT_SPEAKER_PREFIX_BOOST` (default 1.5) when `raw_text.startswith(f"{entity}:")` for any focus entity; no down-demotion; applied to `max(bm25(center), bm25(window))` in the ranking sort key of `search_episodes_by_entities`. 1 src file + 1 new test file (10 regression tests), 237 LOC total (mostly test scaffolding). Pre-flight BM25-only simulation on `p1-1b-2conv/knot.db` for the target Q "Which cities has Jon visited?" showed Rome-trip moves from rank 187 → 135 — still far outside top-60 — confirming ahead of bench that speaker-boost would NOT address semantic-gap cat1 misses like Rome-trip (BM25 has zero overlap with Q token "visited" / "cities"; gold turn only contains "trip" / "Rome"). Bench was still run to measure effect on Q-families the boost *could* help.
**m6a 3-run averaged (canonical gpt-4o-mini × gpt-4o-mini, 2-conv):**
  - cat1: 27.91 % avg (12, 11, 13 / 43; **−2.33 pp**)
  - cat2: 55.03 % avg (34, 35, 35 / 63; **+4.23 pp** — real cross-noise gain)
  - cat3: 66.67 % avg (9, 9, 8 / 13; −2.56 pp — single-run dip in r3, noise-floor territory at n=13)
  - cat4: 78.65 % avg (90, 89, 90 / 114; −2.05 pp — consistent drop across 3 runs)
  - cat5: 24.88 % avg (18, 18, 17 / 71; −1.88 pp — above 20 % floor)
  - cat1-4 agg: 62.23 % avg (145, 144, 146 / 233; **−0.43 pp** — within noise)
**Rationale for REVERT:** Asymmetric trade-off. Speaker-boost helps cat2 (multi-hop benefits from anchoring claims to speakers) but hurts cat1 (single-hop recall often needs counterparty-turn evidence like "Caroline mentioned Melanie's pottery") and cat4 (adversarial Q become over-confident when counterparty signal is suppressed). The only category where boost helps is cat2, which is not the plan goal. Cat1, the sole target, regressed by −1 Q average — inside the 2-conv noise band, but consistent (never gained in any of 3 runs). Accepting this move as a cat2+cat4-split-trade would add complexity without moving the cat1 target and would pollute the baseline for the next real cat1 work (materializer rewrite, Phase B). Keeping `p1-1b-2conv` clean preserves diagnostic clarity. Pattern from `memory/project_locomo_phase1_retrieval_exhausted.md` holds: ranking-level moves (even architectural ones like prefix-aware pool weighting) don't address bottleneck B (23 % raw→claim materialization rate). Next cat1 move must attack materializer coverage directly.
**Next baseline update:** no — `p1-1b-2conv` remains baseline.
**Implementation note for future:** the `_speaker_prefix_boost` helper itself is a clean utility (correct on Jonathan!=Jon edge case, case-sensitive, env-overridable). If a future move wants speaker-awareness without the cat1/cat4 trade-off, the hook is easy to restore and apply selectively (e.g. only for SET-answer Q via `AnswerSpace.SET` gate, or only for multi-hop frames). Not pursuing that now because cat1 remains the primary blocker.

---

## 2026-04-23 — Phase 1E: relative-time + speaker-as-subject EVENT fallback (REVERTED)

**Commit:** `887a318` (reverted by `e380f4a`) on `feature/configurable-mcp-env-v0.9.4`
**Baseline:** `p1-1b-2conv` — cat1 30.23 %, cat2 50.79 %, cat3 69.23 %, cat4 80.70 %, cat5 26.76 %, cat1-4 62.66 %.
**Run:** `data/runs/p1e-2conv/report.json` — canonical (gpt-4o-mini × gpt-4o-mini, 2-conv)
**Config deviations:** none
**Decision:** REVERT — zero signal, six negative Q-flips; formal stop-rule thresholds did not trigger, but the move has no redeeming outcome on the target.
**Reason:** Addresses bottleneck B from `memory/project_locomo_cat1_rank_dilution_is_materializer.md`: 19/30 cat1 WRONG on `p1-1b-2conv` are materializer under-emission (Jon-Rome, Melanie-painting-yesterday, etc. — first-person past-tense narratives that don't match FP `^I\s+verb` because the speaker prefix was already stripped, and don't match `_DATE_RE` because the time anchor is relative). Change: (1) `_RELATIVE_DATE_RE` covering `yesterday|today|last X|recently|N days ago|this morning|over the weekend`; (2) `_resolve_relative_date(token, session_date)` → fixed offsets (yesterday=−1d, last week=−7d, last month=−30d, last year=−365d, recently=−3d, few days ago=−5d, etc.); (3) `_PAST_VERB_OPENERS` frozenset (~40 irregular English past tenses: Took, Went, Saw, Met, Bought, Made, …) + `-ed` suffix heuristic for regulars; (4) EVENT fallback path widened to accept `_RELATIVE_DATE_RE` alongside `_DATE_RE` and replace first-word past-verb subjects with `speaker`. 398 LOC (src + 31 new unit tests in `test_materialization_relative_time.py`). `MATERIALIZATION_VERSION` bumped 6 → 7. Full suite: 1237 passed + 2 skipped, no regressions.
**p1e-2conv numbers:**
  - cat1: 23.26 % (10/43; **−6.97 pp**)
  - cat2: 50.79 % (32/63; = baseline)
  - cat3: 61.54 % (8/13; **−7.69 pp** — within n=13 noise)
  - cat4: 80.70 % (92/114; = baseline)
  - cat5: 23.94 % (17/71; −2.82 pp — above 20 % floor)
  - cat1-4 agg: 60.94 % (142/233; −1.72 pp — inside 2 pp threshold)
**Q-level diff vs p1-1b-2conv:** 0 WRONG → CORRECT, 6 CORRECT → WRONG. The six regressions:
  1. `[0:14 cat3]` "Would Caroline still want to pursue counseling…" — reasoning Q; new answer hedges around gold "Likely no".
  2. `[0:55 cat1]` "What subject have Caroline and Melanie both painted?" gold "Sunsets" — new answer bloats to "nature, including sunsets and animals" → judged wrong.
  3. `[0:76 cat1]` "When did Melanie go on a hike after the roadtrip?" gold "19 October 2023" — new answer "October 18, 2023" → **off-by-1 day caused by the fixed-offset relative-date resolver** ("yesterday" → session_date − 1d, not the actual utterance time). This is the direct cost of anchoring on session-date rather than utterance-date.
  4. `[0:170 cat5]` "What does Caroline say running has been great for?" gold "Her mental health" — new answer "boost your mood" → judged wrong.
  5. `[1:5 cat1]` "What Jon thinks the ideal dance studio should look like?" gold "By the water, with natural light and Marley flooring" — new answer omits "By the water" → rank dilution by new EVENT claims.
  6. `[1:97 cat5]` "Where is Gina's HR internship?" gold "fashion department…" — new answer denies the internship exists → strong regression.
**Target-Q status:** Jon-Rome turn (*"Took a short trip last week to Rome"*) and Melanie-painting-yesterday turn — the two hand-selected regression targets the fix was built for — did **not** flip to CORRECT. The materializer now emits EVENT claims for these turns, but the event_time is wrong (session_date − offset ≠ real utterance date) and the claims are never retrieved in time for the answering step because BM25+embedding ranking still cannot match Q-tokens "cities" / "visited" to raw tokens "took" / "trip" / "Rome".
**Root-cause of the negative outcome:**
  - **Session-date anchor is not utterance-date anchor.** Fixing `event_time = session_date − relative_offset` creates off-by-N-day errors ([0:76] is the direct manifestation) and corrupts `narrative_cluster_render` time-buckets.
  - **More EVENT claims = more rank noise.** Non-target first-person sentences with `-ed` endings or ambiguous relative-time phrases ("earlier", "recently") now emit EVENT claims that compete with fact-bearing evidence in `render_top_k=12`.
  - **Speaker-fallback is correct in isolation, wrong in aggregate.** Replacing a first-word past-verb subject with `speaker` when the sentence has a relative-time anchor is semantically right, but adds claims that the ranking layer then pushes ahead of the real fact-turns.
**Architectural consequence:** The cheap ~50-LOC patch predicted in `memory/project_locomo_cat1_rank_dilution_is_materializer.md` was too optimistic. Relative-time anchoring needs *utterance-date* resolution (per-turn timestamp, not session-level), and that requires either (a) a real temporal parser (dateparser/Duckling dependency) or (b) skipping time-resolution entirely and emitting EVENT with `event_time=None` — but `_collect_evidence_episode_ids` uses `event_time` for narrative ordering, so (b) would break cat2 temporal Q. Neither fits within the spaCy-free Phase-1 envelope.
**Implication for Phase plan:** Seven consecutive REVERTs on this branch (Moves 4, 5, 1A, 1C-plays, 2S1, 6A, 1E). The 2-conv gate is exhausted as a discriminator. The `memory/project_locomo_phase1_retrieval_exhausted.md` conclusion is now confirmed twice: ranking-level moves are done, and the first materializer-level move collapsed because the 2-conv gate can't tell signal from noise at n=43 cat1 (±3 Q = ±7 pp noise floor). Remaining options:
  1. **Accept 30.2 % as terminal** on the regex-only materializer and close Phase 1.
  2. **spaCy POS/lemma rewrite** (800 LOC + new dep) to fix bottleneck B structurally — this is the path from `memory/project_locomo_cat1_rank_dilution_is_materializer.md` Option A. Requires user sign-off for dependency cost.
  3. **3-run averaging or full-10 gate** for any next move, to clear the ±7 pp cat1 noise floor. Requires user sign-off for bench cost (~33 min for 3×2-conv, ~55 min for 1×10-conv).
**Next baseline update:** no — `p1-1b-2conv` remains baseline.

---

## 2026-04-23 — Claims-first additive promotion + render_top_k bump (REVERTED, stop-rule tripped)

**Commit:** `3203620` (reverted by `9dc7a8e`) on `feature/configurable-mcp-env-v0.9.4`. Replay harness restored separately as `461ca71`.
**Baseline:** `p1-1b-2conv` — cat1 30.23 %, cat2 50.79 %, cat3 69.23 %, cat4 80.70 %, cat5 26.76 %, cat1-4 62.66 %.
**Run:** `data/runs/p1-promo-2conv/report.json` — canonical (gpt-4o-mini × gpt-4o-mini, 2-conv)
**Config deviations:** none (env default `AIKNOT_CLAIMS_FIRST_PROMOTION=1` active)
**Decision:** REVERT — aggregate drop **−3.86 pp** exceeds the 2 pp stop-rule threshold by itself; cat4 **−4.39 pp** hurts the 8.2 pp per-category floor margin (80.7 → 76.3 %, 5 Q lost).
**Reason:** Addresses the 19/30 retrieval-bucket finding in `memory/project_locomo_cat1_bottleneck_audit_20260423.md`. Change in `src/ai_knot/query_runtime.py` (step 7c): (1) `_claim_source_promotions` picks up to 6 episode IDs whose atomic claims have subject matching a focus entity, ranked by `overlap(value_tokens, q_tokens) × confidence × salience` with SET / DESCRIPTION exempting the overlap requirement; (2) `_merge_promoted_first` additive merge — `PROTECT_K=6` existing leaders preserved, only NEW IDs inserted; (3) `execute_query` bumps `collect_cap` and `render_top_k` by `n_new_promoted` so promoted raws don't displace existing top-K past the render cutoff. 122 LOC src + 4 regression tests + replay harness `scripts/cat1_retrieval_replay.py` (192 LOC). Gated by env var `AIKNOT_CLAIMS_FIRST_PROMOTION` (default on).
**Pre-bench signal (offline replay on 30 cat1 WRONG):**
  - flag=0: 7/30 gold-in-context
  - flag=1: 9/30 gold-in-context (+2 Q: [0:24] destress, [0:60] instruments; 0 losses)
  - CORRECT cat1 (13): 9/13 unchanged — no regressions seen in the replay check
  Replay predicted net +2 Q cat1 CORRECT. **Bench delivered −2 Q cat1 CORRECT.** Divergence is 4 Q on cat1 alone and +9 Q aggregate degradation across cat2/cat4/cat5 that the replay never measured.
**p1-promo-2conv bench numbers:**
  - cat1: 25.58 % (11/43; **−4.65 pp**, lost 2 Q)
  - cat2: 47.62 % (30/63; **−3.17 pp**, lost 2 Q)
  - cat3: 69.23 % (9/13; = baseline)
  - cat4: 76.32 % (87/114; **−4.39 pp**, lost 5 Q)
  - cat5: 25.35 % (18/71; −1.41 pp)
  - cat1-4 agg: 58.80 % (137/233; **−3.86 pp**, trips 2 pp stop-rule)
**Q-level diff vs p1-1b-2conv:** 16 CORRECT → WRONG, 6 WRONG → CORRECT (net −10 across all categories).
  - CORRECT → WRONG: [0:35 cat2 camping date], [0:45 cat2 pride parade date], [0:49 cat2 pride festival date], [0:55 cat1 both painted], [0:70 cat1 trans events], [0:72 cat2 friend adoption date], [0:76 cat1 hike-after-roadtrip date], [0:90 cat4 marriage duration], [0:95 cat4 camping activities], [1:53 cat4 Gina customer experience], [1:59 cat4 Gina clothing+dance], [1:62 cat4 Gina confidence], [1:75 cat4 Gina grand opening], [1:79 cat5 Jon temp job], [1:89 cat5 Jon store feeling], [1:97 cat5 Gina HR internship].
  - WRONG → CORRECT: [0:0 cat2 LGBTQ support date], [0:9 cat2 friends meetup date], [0:32 cat1 LGBTQ+ events], [0:135 cat4 Melanie October setback], [0:152 cat5 charity race realization], [0:165 cat5 counseling motivation].
  - Replay-predicted gains [0:24] and [0:60] BOTH still WRONG on bench — the gold tokens entered context but the LLM answered differently (likely rank-position or extra-context interference). Replay's "gold-in-context" metric has a large false-positive rate as a bench predictor.
**Root-cause of the negative outcome:**
  1. **`render_top_k` + `collect_cap` bump enlarges the context itself.** Every Q where promotion fired got ~1 extra raw rendered (avg `n_new_promoted`≈1-3 per Q). For Q where the gold raw was already in context, the extra raws add *competing* evidence that the LLM has to disambiguate. cat4 adversarial Q (Gina-dance, Gina-customer-experience) are the worst affected: more evidence increases the chance the LLM extracts a confident-but-wrong answer instead of saying "not in context".
  2. **`_claim_source_promotions` ranking fires on too many cat2 temporal Q.** cat2 date-asking Q have `Melanie`/`Caroline` as focus entity; any Melanie-subject claim gets promoted. The date-bearing raw is often a different raw from the promoted ones, so promoted raws push the date-bearing raw out of the collect window even with the bump (char_budget is NOT bumped — we only bumped render_top_k). 4 cat2 date-Q flipped to WRONG this way.
  3. **`char_budget` not bumped = silent truncation under expanded render_top_k.** We added up to 6 raws per Q but kept `char_budget=22_000` for balanced profile. At ~1200 char/turn, 15-18 turns already saturates. Bumping render_top_k past 12 without a matching char_budget bump forces mid-raw truncation, cutting off whichever turn happens to cross the byte limit — which is non-deterministic with respect to gold content.
  4. **Offline replay gold-in-context is a poor predictor of bench CORRECT.** Replay only measured token-presence; it did not measure LLM-extraction fidelity under enlarged context, and it did not sample cat2 / cat4 / cat5 at all. Predicted +2 Q cat1; bench delivered −2 Q cat1, net −10 across cat1-4. A retrieval change that moves the context envelope must be validated either against CORRECT Q of all cat types (not just the target WRONG bucket) or with the actual answering LLM in the loop.
**Architectural consequence:**
  - Adding retrieval signal ≠ adding answer quality. The replay harness is still useful to confirm a change CAN surface gold tokens, but a bench prediction requires running a second lane that also simulates LLM extraction under the new context — not just gold-containment. Without that, replay wins are phantom wins.
  - The move was designed additively (`PROTECT_K=6`, merge-new-only) specifically to avoid displacement, and it succeeded at that. The damage was **expansion, not displacement** — growing the context hurts answering precision. The cat4 cluster of losses (5 Q, all on Gina-related adversarial Q from conv1) confirms this: those Q were CORRECT on baseline because context was tight and the LLM could say "not in context"; after promotion the extra Gina-subject claims gave the LLM false confidence to extract a plausible-sounding wrong answer.
  - **Stop-rule was the right gate.** Aggregate −3.86 pp is well past the 2 pp threshold and cat4 −4.39 pp crosses into the 8.2 pp floor margin territory. No partial-accept path.
**Implication for next moves:** Eight consecutive REVERTs on this branch (Moves 4, 5, 1A, 1C-plays, 2S1, 6A, 1E, promo). Two new lessons:
  1. **Context-expanding retrieval changes need cat1-4 replay or answer-stage simulation before bench.** Gold-in-context on WRONG Q is not enough.
  2. **`_caps_for_contract` widening must be done on ALL answer-spaces jointly.** Increasing render_top_k for one answer-space (SET via `_caps_for_contract`) while leaving balanced at 12 creates a discontinuity — promoted raws going to a balanced Q break while the same code path works for a SET Q.
**Next baseline update:** no — `p1-1b-2conv` remains baseline. Replay harness stays in tree as `scripts/cat1_retrieval_replay.py` for future use, with the known limitation now documented.

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
