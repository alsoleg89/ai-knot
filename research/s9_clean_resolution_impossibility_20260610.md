# S9 conflict_resolution — clean deterministic close is impossible (2026-06-10)

Follow-up to `research/s9_s26_closure_deep_dive_20260610.md` PR2 and
`research/ma_metric_calibration_20260610.md` §4.  The deep-dive's PR2 prototype
reached `conflict_resolution 0.333 → 1.0` using a **claim-frame value lexicon**
(`support_status = {supported, deprecated, …}`, `coverage_window = {weekdays,
24/7, …}`).  That lexicon is hand-derived from the S9 fixture's own queries, so
it is benchmark cherry-pick (each value looks generic, but the *selection* is
driven by which queries fail).  This note records why the **clean, no-lexicon**
alternative cannot close Q2/Q3, so the gate is read honestly (conflict_resolution
is advisory / resolver-dependent) and the close is routed to the optional
semantic-resolver seam.

## The structural reason (measured)

For a value-conflict query, the rivals all describe the **same subject**, so the
tokens that *identify* that subject are shared by every candidate → **low IDF**.
The tokens that *diverge* are the conflicting values → **high IDF**.  IDF-overlap
clustering keys on shared high-IDF tokens, of which there are none.  Probe over
the real S9 fixture (`_build_idf` / `_idf_weighted_overlap` from `canonical.py`):

**Q2 — "Is the REST collector endpoint still supported?"**
stale `A1` ("…supports both gRPC and REST endpoints for backward compatibility")
vs current `C1` ("REST collector endpoint has been officially deprecated…").

| | shared tokens | IDF |
|---|---|---|
| subject (shared) | collector | 1.000 |
| | rest | 1.405 |
| | endpoint | 1.405 |
| discriminators A1 (high-IDF) | both, api, backward, support, compatibility | 2.10 |
| discriminators C1 (high-IDF) | notion, migration, guide, official, since | 2.10 |

`idf_weighted_overlap(A1,C1) = 0.215` (floor 0.35). **High-IDF shared subject
tokens: NONE.**

**Q3 — "When is incident commander coverage required?"**
stale `A2` ("…rotation covers weekdays…") vs current `B2` ("…expanded to 24/7…").
shared = {incident 1.000, command 1.288}; discriminators differ (A2: cover, during,
rotation, weekday; B2: expand, february, weekend, 7, outage). overlap `0.167`.
**High-IDF shared subject tokens: NONE.**

## Why no clean threshold works

- Lowering the overlap floor to ~0.20 to catch A1/C1 also grabs `B1` ("REST
  collector endpoint introduced throttling…", a **complementary** throughput fact
  that must survive) and the two `Collector config changes …` facts — they share
  the same low-IDF subject tokens.  This is exactly the floor-lowering that
  over-fired on S21's legitimate historical "deprecated/replaced" facts and was
  reverted under the stop-rule in #79.
- Gating the merge on shared **high-IDF** subject tokens fails too: there are none.
- Recency/stem demotion (demote an older stem-less fact when a newer stem-bearing
  fact shares its query-subject tokens) drops the complementary `B1` and risks the
  same S21 over-fire — it cannot tell "supersede the stale support claim" from
  "keep the complementary throttling claim" without value semantics.

Deciding that `supports` and `deprecated` are opposite values of one dimension
requires either a value lexicon (cherry-pick) or semantic judgement (LLM).

## Outcome

- **Q1** (numeric SLA) is already closed deterministically by the CAS slot +
  canonical resolver (#79).
- **Q2/Q3** are genuine semantic value-conflicts → handled by the opt-in
  `SemanticConflictResolver` seam (`canonical.py`, off by default, never imports
  an LLM in core).  Behavior is locked by `TestSemanticConflictSeam` in
  `tests/test_shared_pool.py` (default keeps both; injected resolver drops the
  stale one).
- The gate marks `conflict_resolution ≥ 0.80` **advisory / resolver-dependent**
  (deterministic ceiling ≈ 0.33); the hard S9 gate is `correct_at_3` (the system
  reliably surfaces the correct answer — baseline 1.00).  The honest competitive
  line vs Zep/Mem0: they pay an LLM call for *every* conflict; ai-knot resolves
  slotted/lexically-near conflicts deterministically and reserves the LLM seam for
  the semantically-divergent tail.
