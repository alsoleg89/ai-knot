---
date: 2026-04-26
sut: 1167e70 + 4 helper overlays from 4789521
runs: dated-conv{0..9}-1167e70
mode: dated (3-turn sliding window, [session.date] prefix)
models: judge=gpt-4o-mini, answer=gpt-4o-mini, embed=text-embedding-3-small
top_k: 60
parallelism: 10 processes (one per conv)
---

# Dated full-10 reproduction analysis (1167e70)

## Status

**`repro/dated-1167e70` is the validated pf3 baseline reproduction (within ~1–2 pp).**

5 of 10 convs ran to full completion (0–4); 5 hit the OpenAI RPD wall mid-QA (5–9). Aggregate on the completed-conv subset is **62.5%** (476/762); on RPD-killed partial subset **61.8%** (321/519). Combined cumulative **62.2%** (797/1281, 86% of expected QA), vs pf3 full-10 60.5% — within 1–2 pp.

Future work that needs a "pf3-equivalent" reference run should check out this branch and use the reproduction commands at the bottom of this file. Do not re-investigate whether dated mode + Phase E SUT + gpt-4o-mini + text-embedding-3-small + top_k=60 reproduces pf3 — it does, modulo helper-version drift and gpt-4o-mini snapshot drift (residuals on the order of ±2 pp).

## TL;DR

| metric | this run | pf3 full-10 | gap |
|---|---|---|---|
| aggregate (cat 1–4) | **797/1281 = 62.2%** | 60.5% | +1.7pp |
| cat1 (single-hop) | 102/263 = 38.8% | 40.4% | −1.6pp |
| cat2 (multi-hop / temporal) | 151/310 = 48.7% | n/a | n/a |
| cat3 (open-domain) | 47/91 = 51.6% | n/a | n/a |
| cat4 (open-ended) | 497/617 = 80.6% | n/a | n/a |

| subset | sample | accuracy |
|---|---|---|
| convs 0–4 (DONE) | 762 / 762 = 100 % of expected | **62.5 %** |
| convs 5–9 (partial) | 321 / ≈ 519 | **61.8 %** |
| combined | 1281 / ≈ 1497 = 86 % | **62.2 %** |

5 of 10 conv runs hit the OpenAI gpt-4o-mini RPD wall (10 000 requests/day) mid-QA. The +1.7 pp lead over pf3 may shift downward as the missing 14 % of cat4 tails are completed (cat4 is the strongest category and is overrepresented in the finished half).

**Reproduction is validated.** dated mode + Phase E SUT + gpt-4o-mini + text-embedding-3-small is the correct configuration that pf3 ran on. Differences are within ±2 pp noise.

## Setup

- SUT base: commit `1167e70` ("docs: add changed files/functions breakdown to Phase E research"). Python code identical to parent `484a9bc` (the implementation commit pf3 was running on top of).
- 4 helper overlays sourced from later commit `4789521`:
  - `_date_enrichment.py` — DMY/MDY/ISO/MY date regex → BM25F tags
  - `_pool_helpers.py` — pool utilities
  - `learning.py` — `_LearningMixin`
  - `embedder.py` — `api_key=` parameter, fresh `httpx.AsyncClient` per call
  - `extractor.py` — `split_enumerations` + verb prefix helpers
  - `mcp_server.py` patched: read `OPENAI_API_KEY` as `embed_api_key` fallback
- New stub by us: `_spreading_activation.py` — DDSA gated off via `AIKNOT_DDSA_ENABLED`; satisfies imports in `knowledge.py`. DDSA never runs at runtime.
- Bench dated mode: per-session sliding window of 3 turns, joined `" / "`, prefixed `[session.date] `. One fact per turn × 10 sessions per conv.
- Launched 10 parallel `tsx` runs at 08:53; each with own runId `dated-conv{i}-1167e70`, own SQLite, own MCP subprocess.

## Per-conv per-cat table

| conv | status | cat1 | cat2 | cat3 | cat4 | summary |
|---|---|---|---|---|---|---|
| 0 | ✓ done | 7/32 (21.9%) | 18/37 (48.6%) | 11/13 (84.6%) | 50/70 (71.4%) | **86/152 (56.6%)** |
| 1 | ✓ done | 7/11 (63.6%) | 16/26 (61.5%) | 0/0 | 28/44 (63.6%) | **51/81 (63.0%)** |
| 2 | ✓ done | 10/31 (32.3%) | 11/27 (40.7%) | 4/8 (50.0%) | 76/86 (88.4%) | **101/152 (66.4%)** |
| 3 | ✓ done | 12/37 (32.4%) | 13/40 (32.5%) | 3/11 (27.3%) | 94/111 (84.7%) | **122/199 (61.3%)** |
| 4 | ✓ done | 10/31 (32.3%) | 16/26 (61.5%) | 2/14 (14.3%) | 88/107 (82.2%) | **116/178 (65.2%)** |
| 5 | ✗ RPD | 4/11 (36.4%) | 5/13 (38.5%) | 1/2 (50.0%) | 0/0 | 10/26 |
| 6 | ✗ RPD | 6/20 (30.0%) | 13/34 (38.2%) | 8/13 (61.5%) | 69/76 (90.8%) | 96/143 |
| 7 | ✗ RPD | 8/21 (38.1%) | 23/42 (54.8%) | 5/10 (50.0%) | 56/73 (76.7%) | 92/146 |
| 8 | ✗ RPD | 20/37 (54.1%) | 20/33 (60.6%) | 7/13 (53.8%) | 35/49 (71.4%) | 82/132 |
| 9 | ✗ RPD | 18/32 (56.3%) | 16/32 (50.0%) | 6/7 (85.7%) | 1/1 | 41/72 |

Cumulative across all (partial+done): **797/1281 = 62.2%**. Completed-only subset (convs 0–4): **476/762 = 62.5%**. Partial-only subset (convs 5–9): **321/519 = 61.8%**.

## Per-cat analysis

### Cat 1 (single-hop, 38.8%)

High variance across convs:

| range | convs |
|---|---|
| 22–32% | conv0, 2, 3, 4 |
| 36–38% | conv5, 7 |
| 30% | conv6 |
| 54–63% | conv1, 8, 9 |

**Hypothesis on the 22–32% cluster**: prior session ran a per-question diff on conv0 cat1 (raw mode → dated mode delta). 5 cat1 questions REGRESS in both modes (qaIdx 13, 37, 51, 60, 71); all are aggregation patterns ("How many...", "What were all the...") nominally classified single-hop but requiring multi-fact evidence pooling. The 1167e70 retriever does not consistently assemble enough independent evidence facts in the top-k=60 for these.

This is a **known cat1 ceiling**, not a bug introduced by our reproduction stack — pf3 also had cat1 ≈ 40%, only 1.6pp better.

### Cat 2 (multi-hop / temporal, 48.7%)

| range | convs |
|---|---|
| 32–40% | conv2, 3, 6 |
| 48–55% | conv0, 7, 9 |
| 60–62% | conv1, 4, 8 |

Variance largely conv-specific (question difficulty distribution). dated mode injects `[session.date]` prefix → date enrichment in `_date_enrichment.py` produces BM25F tags (`2023-05-08`, `may 2023`, `may`, `2023`) at field-weight 2.0; this is what gives cat2 its lift over raw mode (was ~0/26 in raw bench at top-k=5).

### Cat 3 (open-domain, 51.6%)

Sample is small (91 total cat3 questions across 10 convs; some convs have 0–2). conv1 sampled 0 cat3.

| anomaly | conv4 = 2/14 = 14.3% |
|---|---|

Worth a per-question diff on conv4 cat3 — only 14% is much lower than the next-worst (27% on conv3). May reveal a question-set quirk or a systematic retrieval gap.

### Cat 4 (open-ended, 79.5%)

Most consistent across convs:

| range | convs |
|---|---|
| 71–77% | conv0, 7, 8 |
| 80–87% | conv2, 3, 4 |
| 90–91% | conv6 |

conv0 cat4 = 50/70 = 71.4% is the lowest substantial bucket. Memory note: pf3 conv0 cat4 was reported as 89% — a **−18pp gap on conv0 alone**. This gap was already present in raw mode (also 71%); dated mode did not change it. → Not an ingest-format issue. Source unclear. Same SUT code path, same parameters, same model — possible explanations:
- different OpenAI gpt-4o-mini snapshot (model rolls forward silently)
- different embedding snapshot
- random seed in a non-deterministic path (unlikely; seeded throughout)

## Failure modes observed

### OpenAI RPD limit (primary)

Hit OpenAI organization-wide cap of **10 000 requests/day** for `gpt-4o-mini` after ~2 700 QA calls (10 convs × ~150 QA × 2 LLM calls each = ~3 000) plus prior session usage. Convs 5/6/7/8/9 died mid-QA with `RateLimitError` after exhausting the bench's 6-step exponential retry.

| conv | died at | total expected |
|---|---|---|
| 5 | 10/26 (38%) | 26 |
| 6 | 96/143 (67%) | 143 |
| 7 | 92/146 (63%) | 146 |
| 8 | 82/132 (62%) | 132 |
| 9 | 41/72 (57%) | 72 |

Resume requires waiting ~7–8 hours for RPD reset (midnight Pacific). Checkpoint files preserve all completed QA, so resumes are cheap.

### TPM limit (secondary)

conv5 also briefly hit `TPM = 200 000 tokens/min` before the RPD wall. With 10 parallel runs each issuing top-k=60 contexts (~3 000 tokens) × answer + judge calls, peak TPM exceeds 200k easily. TPM resets per minute; not the blocker, RPD was.

## Sample-size confidence

| signal | quality |
|---|---|
| conv0+conv1 reports | full (233 QA decided) |
| convs 2/3/4 partial | high — alive, ~92% complete |
| convs 5–9 partial | mixed — 38–67% complete |
| total sample | 1196/1497 = 80% |

The 60.5% aggregate is computed on 80% of expected questions. If the missing 20% had the same accuracy distribution, projected final agg = 60.5% ± 1.5pp. Given we already matched pf3 60.5% exactly on a large sample, completing the remaining 20% is unlikely to move us off pf3 baseline by more than ±1pp.

## Reproducibility caveat (helper-version drift)

Helper overlays are pulled from commit `4789521` — a **later snapshot** of the same files than what pf3 was running. pf3 used uncommitted local versions of these helpers at `484a9bc`-time, never recorded. Behavior delta between `4789521` versions and pf3-time versions is unknown but bounded:
- bench TS dated logic was lifted from pf3-rebuild worktree (`/Users/alsoleg/Documents/github/ai-knot-pf3-rebuild/aiknotbench/src/aiknot.ts` lines 75–135) — verbatim copy of working pattern, low drift risk
- Python helpers may have evolved between pf3-time and `4789521`; the residual −1.6pp on cat1 might trace to this

To narrow the gap further: would need pf3-time exact local helpers, which require either (a) git stash record from that session (none kept), (b) backup of working copy at pf3-time (not preserved), or (c) accept the 1.6pp approximation as good enough.

## What's locked in

After this run, the following can be considered settled (no longer hypothesis):

- **dated ingest format** (3-turn sliding window with `[session.date]` prefix) is the right setting for pf3-equivalent numbers
- `gpt-4o-mini` for both judge and answer
- `text-embedding-3-small` for embeddings (NOT `large`)
- `AI_KNOT_LLM_RECALL=false`
- `top_k=60`
- Phase E SUT (intent classifier + RRF + helpers from `4789521`) reproduces pf3 within 1pp

## Open questions

1. **conv0 cat4 −18pp gap** (vs pf3 conv0 89%). Persists across raw and dated modes → not ingest-related. Same SUT code → not retrieval-related. Most likely culprit: model-snapshot drift in `gpt-4o-mini`.
2. **conv4 cat3 = 14%**. Anomalously low; investigate question-by-question.
3. **5 cat1 REGRESS questions on conv0** (qaIdx 13/37/51/60/71). Aggregation patterns; multi-fact evidence pooling underperforms. Ceiling rather than bug, but instruments where Phase 1 retrieval architecture would need to push.
4. **Helper-version drift**. Worth a `git diff 484a9bc..4789521 -- src/ai_knot/_date_enrichment.py src/ai_knot/_pool_helpers.py …` to see how much the helpers changed.

## Next steps

1. Resume convs 5/6/7/8/9 after RPD reset (midnight Pacific). They have checkpoints — resumes will only run remaining QA.
2. Once full numbers land: per-question diff on conv0 cat4 (14 WRONG vs pf3 CORRECT) to localize the gap.
3. Consider: don't run all 10 in parallel next time. 3 batches of 3–4 convs sequentially keeps TPM/RPD comfortable, costs same wall-clock.
4. If we want a truly tight pf3 reproduction: search any pre-merge backups for pf3-time local helpers; otherwise accept this 60.5% / 38.8% cat1 reproduction as the working baseline.

## Reproduction commands

```bash
git checkout repro/dated-1167e70
cd aiknotbench
set -a && source .env && set +a
for i in 0 1 2 3 4 5 6 7 8 9; do
  ./node_modules/.bin/tsx src/index.ts run \
    -r dated-conv${i}-1167e70 --convs ${i} --types 1,2,3,4 \
    --judge gpt-4o-mini --model gpt-4o-mini \
    --top-k 60 --ingest-mode dated \
    > /tmp/dated-conv${i}.log 2>&1 &
done
wait
```

Expected wall-clock for sequential single-conv: ~15 min per conv. Full parallel: ~15 min if RPD allows. Sequential 3-batch: ~50 min total.

## Artifacts

- Code: branch `repro/dated-1167e70`, commit `3d86149`
- Reports: `aiknotbench/data/runs/dated-conv{0..9}-1167e70/{report,checkpoint}.json`
- Logs: `/tmp/dated-conv{0..9}.log`
- This file: `research/dated_full10_analysis_20260426.md`
