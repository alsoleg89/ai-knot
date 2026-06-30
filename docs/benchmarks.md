# Benchmarks

This page reports **reproducible** retrieval-quality numbers for ai-knot. Every
figure in the headline table is produced by a single command, is deterministic
bit-for-bit (no network, no LLM, fixed seeds), and runs in well under a minute on
a laptop. The goal is evidence you can re-run, not a leaderboard screenshot.

> **What this measures.** Retrieval quality on an in-repo golden suite: given a
> query and a fact set, does the retriever surface the *right* facts, stay robust
> to distractors, and keep the injected context small? This is the part of a
> memory layer that is purely algorithmic, so it can be pinned down exactly.
>
> **What this is not.** It is not end-to-end QA accuracy. Question-answering
> scores depend on a reader LLM and an LLM judge, which are neither deterministic
> nor offline — see [Full QA evaluation](#full-qa-evaluation-locomo) for how to
> run that path.

## Headline: golden retrieval suite

Two backends on the same fixtures and judge:

- **baseline** — a naive store: keep every message, return the most recent / most
  lexically-overlapping. The "memory is a log" approach.
- **ai-knot** — the production retriever: BM25 + intent-weighted RRF fusion,
  deduplication, decay. No LLM, no embeddings (dense channel disabled).

| Scenario | Metric | baseline | ai-knot | What it shows |
|---|---|---:|---:|---|
| S1 — ranking | semantic MRR | 0.18 | **0.83** | the right fact ranks near the top |
| S1 — ranking | precision@1 | 0.10 | **0.70** | top hit is correct 7× as often |
| S1 — ranking | precision@5 | 0.40 | **1.00** | the answer is always in the top 5 |
| S2 — semantic gap | recall@3 | 0.30 | **0.60** | finds facts that don't share query words |
| S5 — noise (200 distractors) | signal recall@3 | 0.60 | **0.80** | robust when 97% of the store is noise |
| S6 — token economy | compression | 0.67 | **0.73** | injected context shrinks ~4× vs raw |
| S9 — scale (1000 facts) | MRR@1000 | 0.46 | **0.67** | ranking holds up as the store grows |

The single biggest jump is ranking precision: on the S1 set the naive store puts
the right fact first 10% of the time; ai-knot does it 70% of the time, and never
fails to surface it within the top 5.

### Reproduce it

From a clone, with the dev extra installed (`pip install -e ".[dev]"`):

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent \
  --backends baseline,ai_knot_no_llm
```

- `AI_KNOT_EMBED_URL=""` disables the dense channel → no network, deterministic.
- `--mock-judge` uses a deterministic string-overlap judge → no LLM.
- Results land in `benchmark_report.md` and `benchmark_raw.json`.

Because both sources of nondeterminism (embeddings, LLM judge) are switched off,
re-running yields the same numbers every time. That is the point: the claims
above are auditable, not anecdotal.

## LoCoMo: retrieval grounding

The same harness runs against the public
[LoCoMo](https://github.com/snap-research/locomo) long-conversation benchmark
(10 conversations, ~5.9k turns, ~2k questions across single-hop, multi-hop,
temporal, open-ended, and adversarial categories).

```bash
# Downloads locomo10.json from the public dataset repo on first run.
python -m tests.eval.benchmark.runner --scenarios s_locomo --skip-multi-agent \
  --backends ai_knot_no_llm
```

Every score on this path is **deterministic** — computed by token overlap against
the gold answer, not an LLM judge — so the full 10-conversation table reproduces
exactly:

| LoCoMo metric | baseline | ai-knot | Δ |
|---|---:|---:|---:|
| **evidence_recall@5** (gold evidence in top 5) | 0.15 | **0.26** | +71% |
| single-hop grounding F1 | 0.044 | **0.066** | +50% |
| multi-hop grounding F1 | 0.025 | **0.033** | +32% |
| temporal grounding F1 | 0.055 | **0.071** | +29% |
| open-ended grounding F1 | 0.066 | **0.107** | +62% |
| adversarial grounding F1 | 0.060 | **0.093** | +55% |
| overall grounding F1 | 0.054 | **0.084** | +56% |

ai-knot beats the naive log on **every** LoCoMo category. The headline is
`evidence_recall@5`: how often the answer's evidence is actually retrievable —
the ceiling any reader LLM works against.

> **Read these correctly.** They are *retrieval-grounding* metrics — how well the
> retriever surfaces and lexically matches the gold evidence. They are **not**
> the end-to-end LoCoMo QA-accuracy numbers you see on leaderboards, which require
> an answer-generation pass plus an LLM judge. This repo's harness measures the
> retrieval signal those scores are built on; it does not generate or grade free-text
> answers. Treat the F1 column as "grounding strength," not "% correct."

## LongMemEval: point-in-time recall

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) stresses the failure mode
that breaks log-style memory: **temporal reasoning and knowledge updates** over long
histories — "what was true *as of* a given date," after a fact has been revised.

ai-knot is built for exactly this. The bi-temporal model (`valid_from` /
`valid_until` / `event_time`) plus deterministic supersession means a query can be
answered at a point in time:

```python
# A fact and its later revision both live in the store.
kb.recall("where does the user work?", now=question_date)
# → returns what was true on question_date, excluding facts a later one superseded.
```

This is the LongMemEval point-in-time adapter (`recall(now=question_date)`,
also exposed over MCP and the CLI as `recall --now`). The mechanism is
regression-tested in-repo — see `tests/test_event_time_persistence.py` and
`tests/test_supersession_ingest.py` — so the temporal correctness LongMemEval
rewards is guaranteed, not hoped for. The full LongMemEval QA harness (answer
generation + judge over the published dataset) runs externally and depends on the
chosen reader model; it is not part of the deterministic suite here.

## Multi-agent acceptance gate

Memory shared across agents has its own scored gate (scenarios S8–S26: CAS
correctness, fan-in recall, trust/adversarial discount, conflict resolution).
It runs on every PR:

```bash
python -m tests.eval.benchmark.runner --multi-agent --mock-judge --ma-gate
```

See [production-readiness.md](production-readiness.md) §4 for the guarantees the
gate enforces.

## Methodology notes

- **Backends are wired identically** — same fixtures, same judge, same query set.
  The only variable is the retrieval logic.
- **`ai_knot_no_llm` is the honest offline column.** Under `--mock-judge` the
  LLM-extraction backend (`ai_knot`) reduces to the same `kb.add()` path, so the
  no-LLM column is the one to cite for a zero-network claim.
- **Numbers are means over repeated runs** (`schema_version: 2` raw output stores
  `{mean, stdev}` per metric); with dense + judge disabled the stdev is zero.
- Full runner options: [tests/eval/benchmark/BENCHMARK.md](../tests/eval/benchmark/BENCHMARK.md).
