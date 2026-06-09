# Multi-agent benchmark baseline — post-#75 (2026-06-09)

Locked baseline snapshot of the S8–S26 multi-agent suite, run on the
event_time-persistence foundation (PR #75). Every later multi-agent change
(governance spine, retrieval tail) is measured as a delta against this table.
The earlier assessment ran on stale `main` (`a9240ab`); these numbers confirm
the same failures persist on post-#75 code — they are structural, not an
artifact of an old tree.

## Reproduce

```bash
# from the worktree on feat/ma-benchmark-scorecard (off feat/event-time-persistence)
PYTHONPATH=src /Users/alsoleg/Documents/github/ai-knot/.venv/bin/python \
  -m tests.eval.benchmark.runner --multi-agent --mock-judge --ma-storage sqlite \
  --output /tmp/ma_report.md --raw-output /tmp/ma_raw.json
```

- Backend: `ai_knot_multi_agent` only (protocol scenarios have no cross-system analog).
- Judge: `--mock-judge`. All MA metrics below are **deterministic structural counts**,
  not LLM-judged — reproducible bit-for-bit.
- Storage: sqlite (`atomic_update` via `BEGIN EXCLUSIVE`).
- Dense retrieval fell back to BM25-only ("Embedding batch failed") on the
  embedding-dependent scenarios (e.g. S24) — no embedding endpoint configured.
  Documented, not hidden: the gate is computed in this degraded-dense mode.

## Protocol Correctness — all green (the moat works)

| Scenario | Metric | Value |
|---|---|---|
| S10 MESI CAS | cas_correctness | 1.00 |
| S11 MESI Sync | delta_correctness | 1.00 |
| S13 Concurrent Writers | no_lost_updates | 1.00 |
| S17 Self-Correction | correction_surfaced | 1.00 |
| S20 Belief Revision | final_consensus | 1.00 |
| S25 Conflict Resolution (slotted) | resolution_correctness / canonical_coverage | 1.00 |

## Retrieval & Behavior

| Scenario | Headline metric | Value | Status |
|---|---|---:|---|
| S8 Isolation | overlap_coverage | 1.00 | ✅ |
| **S9 Pool Publish** | **conflict_resolution** | **0.00** | ❌ target ≥0.80 |
| S9 Pool Publish | precision_at_3 | 0.44 | ❌ target ≥0.70 |
| S9 Pool Publish | supersession_propagation | 1.00 | ✅ |
| S12 Topic Gating | triage_precision | 1.00 | ✅ |
| S12 Topic Gating | escalation_recall | 0.75 | ⚠️ sub-metric |
| S14 Trust Drift | trust_floor_reached | 1.00 | ✅ |
| S15 Topic Leakage | channel_precision | 1.00 | ✅ |
| S15 Topic Leakage | shared_term_isolation | 0.67 | ⚠️ sub-metric |
| S16 Knowledge Relay | chain_depth / layer recall | 1.00 | ✅ |
| S18 Trust Calibration | trust_calibration | 1.00 | ✅ |
| **S19 Incident Reconstruction** | **evidence_precision** | **0.57** | ❌ target ≥0.70 |
| S19 Incident Reconstruction | evidence_recall | 1.00 | ✅ |
| S21 Partial Assembly | coverage | 1.00 | ✅ |
| S21 Partial Assembly | cross_agent_recall | 0.80 | ✅ |
| S22 Temporal Staleness | freshness_recall / staleness_rejection | 1.00 | ✅ |
| **S23 Adversarial Noise** | **free_standing_suppression** | **0.60** | ❌ target ≥0.85 |
| S23 Adversarial Noise | slot_suppression | 1.00 | ✅ |
| **S23 Adversarial Noise** | **trust_penalty** | **0.00** | ❌ target >0.50 |
| S24 Onboarding | pool_retrieval_recall / kb_absorption | 1.00 | ✅ |
| **S26 Sparse Assembly** | **target_shard_recall_at_1000** | **0.13** | ❌ target ≥0.60 |
| S26 Sparse Assembly | target_shard_recall_at_10 / at_100 | 0.53 | ❌ target ≥0.60 |
| S26 Sparse Assembly | distractor_rate_at_1000 | 0.96 | ❌ target ≤0.50 |
| S26 Sparse Assembly | all_shards_covered_at_{10,100,1000} | 0.00 | ❌ |
| S26 Sparse Assembly | p95_retrieve_ms_at_1000 | 83.61 | ✅ (≤150ms) |

## The four real failures (drive Phase 1/2)

1. **S9 conflict_resolution = 0.00** (supersession_propagation = 1.00).
   Free-standing competing claims conflict via differing *values*, not via the
   hardcoded `_CONFLICT_SIGNAL_STEMS` vocabulary, so `ClaimFamilyResolver` keeps
   both. Clean fix = claim_key→CAS at publish (Phase 2 C1). NOT new verbs.
2. **S23 trust_penalty = 0.00, free_standing_suppression = 0.60.**
   Trust penalty only accrues from `quick_inv_count`, raised solely on *slot*
   supersession; free-standing noise never slot-supersedes. Evidence-before-belief
   gate (Phase 1 B1) + abstention (B6) are the clean levers.
3. **S19 evidence_precision = 0.57.** Red-herring noise leaks into the evidence
   set; evidence gate (B1) should tighten this.
4. **S26 target_shard_recall@1000 = 0.13, distractor@1000 = 0.96.**
   No facet→shortlist→per-shard harvest, no diversity floor, no near-miss
   penalty. Hardest item; same recall wall as LOCOMO cat1 (Phase 2 C2).

## Acceptance gate (Phase 0 A2 will encode these as pass/fail thresholds)

| Metric | Baseline | Target |
|---|---:|---:|
| S9 conflict_resolution | 0.00 | ≥ 0.80 |
| S9 precision_at_3 | 0.44 | ≥ 0.70 |
| S19 evidence_precision | 0.57 | ≥ 0.70 |
| S23 free_standing_suppression | 0.60 | ≥ 0.85 |
| S23 trust_penalty | 0.00 | > 0.50 |
| S26 target_shard_recall_at_1000 | 0.13 | ≥ 0.60 |
| S26 distractor_rate_at_1000 | 0.96 | ≤ 0.50 |
| S26 p95_retrieve_ms_at_1000 | 83.61 | ≤ 150 |
| Protocol correctness (S10/11/13/17/20/25) | 1.00 | = 1.00 (no regression) |

Raw JSON for this run: `/tmp/ma_raw.json` (regenerate with the command above).
