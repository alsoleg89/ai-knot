# Benchmarks

The agent-memory field reports two benchmarks above all others — **LoCoMo** and
**LongMemEval** — as an LLM-judged QA-accuracy percentage, broken down by question
category. This page does that, with two differences that matter:

1. Every number names its **reader model and judge** (an LLM-judged score is
   meaningless without them).
2. Alongside the QA number, ai-knot reports a **deterministic, one-command
   retrieval number that cannot move** — because the leaderboard numbers
   notoriously do.

## Why the reproducible number is the anchor

LoCoMo/LongMemEval leaderboard numbers are in a credibility crisis:

- **Zep** reported **84%** on LoCoMo from a single run; an independent
  re-evaluation that restricted scoring to the four validated categories, aligned
  the prompts, and averaged ten runs put it at **58.44% ± 0.20** — a ~25-point gap
  traced to a scoring-denominator error (cat-5 answers counted in the numerator
  but not the denominator). Zep disputes the re-run and claims **75.14%** with its
  own config. ([getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5))
- Across the field, published LoCoMo claims now span **~55% to >90%** (Mem0,
  ByteRover, Memori, MemMachine, …), with the leading vendors openly contesting
  each other's methodology. The number you see depends on whose harness ran it.
  ([mem0.ai](https://mem0.ai/blog/ai-memory-benchmarks-in-2026))

An LLM-judged score is a function of the reader, the judge, the prompts, the run
count, and which categories you score. So ai-knot leads with a number that has
none of those degrees of freedom (see [the deterministic suite](#deterministic-retrieval-suite)),
then reports the QA accuracy with every knob named.

---

## LoCoMo — QA accuracy

Full 10-conversation [LoCoMo](https://github.com/snap-research/locomo) run.
**Reader: gpt-4.1 · Judge: gpt-4o · retrieval: BM25 + RRF + dense (text-embedding-3-small),
top_k=30.** Categories 1–4 scored; **category 5 (adversarial) excluded**, per the
dataset authors (this is the exact step Zep got wrong).

**Overall: 78.0%** (1201/1540)

| Category | Accuracy |
|---|---:|
| cat1 — single-hop | 60.6% (171/282) |
| cat2 — multi-hop | 67.6% (217/321) |
| cat3 — temporal | 63.5% (61/96) |
| cat4 — open-ended | **89.4%** (752/841) |
| **Overall (cat1–4)** | **78.0%** (1201/1540) |

It is not one lucky conversation — accuracy is **74–84% on every one of the 10**:

| conv | cat1 | cat2 | cat3 | cat4 | overall |
|---:|---|---|---|---|---|
| 1 | 53% | 73% | 77% | 90% | 77% |
| 2 | 55% | 81% | — | 91% | 83% |
| 3 | 65% | 74% | 88% | 94% | 84% |
| 4 | 49% | 75% | 27% | 86% | 74% |
| 5 | 52% | 50% | 43% | 90% | 74% |
| 6 | 63% | 67% | 29% | 92% | 76% |
| 7 | 55% | 59% | 77% | 94% | 79% |
| 8 | 67% | 71% | 60% | 86% | 79% |
| 9 | 70% | 61% | 85% | 85% | 76% |
| 10 | 75% | 62% | 86% | 90% | 81% |

For context (vendor/independent-reported, different readers/judges, different
harnesses — treat as landscape, **not** a controlled head-to-head): Zep 84%
claimed → 58.4% corrected → 75.1% Zep's rebuttal; Mem0 high-60s (own paper) and
~55% in ByteRover's re-run; newer entrants claim 80–92%. The spread *is* the
point — none of these is comparable to another without the same reader, judge,
prompts, and category set.

> **Reproduce it.** From the `aiknotbench` harness with an OpenAI key:
> `run -r locomo --model gpt-4.1 --judge gpt-4o --types 1,2,3,4 --top-k 30`.
> The number depends on the reader/judge you choose — that is true of every number
> in the landscape above, which is exactly why the [deterministic suite](#deterministic-retrieval-suite)
> exists.

---

## LongMemEval — QA accuracy

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) Oracle set (500 questions,
evidence sessions provided so the score isolates **memory + reader quality**).
**Reader: gpt-4.1 · Judge: gpt-4o.**

**Overall: 59.6%** (298/500)

| Question type | Accuracy |
|---|---:|
| single-session-assistant | **98.2%** (55/56) |
| single-session-user | **95.7%** (67/70) |
| abstention (false-premise) | **90.0%** (27/30) |
| knowledge-update | 62.8% (49/78) |
| single-session-preference | 50.0% (15/30) |
| multi-session | 49.6% (66/133) |
| temporal-reasoning | 34.6% (46/133) |
| **turn-level recall** | **96.2%** |
| **session-level recall** | **94.0%** |

ai-knot is near-perfect on information extraction (single-session 95–98%) and
**abstention** — it declines false-premise questions 90% of the time, the failure
mode that makes memory systems confabulate. The retrieval is essentially solved here
(94–96% recall); the remaining gap is reader reasoning on temporal and multi-session
questions. Peer band for deterministic-retrieval + small-reader systems on this
benchmark: Mem0 49%, Zep 63.8%, LightMem 67.8%, TiMem 76.9% (the published >84%
numbers use a frontier reader over a fully in-context log — a different category).

---

## Deterministic retrieval suite

The QA numbers above use an LLM reader and judge — necessarily model-dependent.
These do not: same fixtures, same scoring, fixed seeds, **no network, no LLM**.
Re-run and you get identical numbers.

| Scenario | Metric | naive log | ai-knot |
|---|---|---:|---:|
| ranking | semantic MRR | 0.18 | **0.83** |
| ranking | precision@1 | 0.10 | **0.70** |
| ranking | precision@5 | 0.40 | **1.00** |
| noise (200 distractors) | signal recall@3 | 0.60 | **0.80** |
| token economy | compression | 0.67 | **0.73** |
| scale (1000 facts) | MRR@1000 | 0.46 | **0.67** |
| **LoCoMo** | `evidence_recall@5` | 0.15 | **0.26** (+71%) |

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

`evidence_recall@5` is the deterministic LoCoMo retrieval metric — how often the
gold evidence lands in the top 5. It is the ceiling the QA accuracy above is built
on, and unlike the QA number it cannot drift.

If you want a **shareable side-by-side scorecard** instead of a raw harness run,
use [competitor-bench-pack.md](competitor-bench-pack.md) and:

```bash
python scripts/run_competitor_bench_pack.py --profile offline
```

---

## LongMemEval point-in-time (the bi-temporal lever)

LongMemEval's hardest axes — **knowledge-update** and **temporal-reasoning** — are
exactly what ai-knot's bi-temporal model targets. A revised fact is answered *as of*
a point in time, deterministically:

```python
kb.recall("where does the user work?", now=question_date)
# → what was true on question_date; facts a later one superseded are excluded.
```

This is the `recall(now=…)` adapter (also on the CLI as `recall --now` and over MCP).
The temporal correctness it depends on is regression-tested in-repo
(`tests/test_event_time_persistence.py`, `tests/test_supersession_ingest.py`).

---

## Multi-agent acceptance gate

Shared-pool memory has its own scored gate (S8–S26: CAS correctness, fan-in recall,
trust/adversarial discount, conflict resolution), run on every PR:

```bash
python -m tests.eval.benchmark.runner --multi-agent --mock-judge --ma-gate
```

See [production-readiness.md](production-readiness.md) §4.

---

## Methodology notes

- **Reader / judge named on every QA number.** LoCoMo and LongMemEval QA accuracy
  use gpt-4.1 (reader) + gpt-4o (judge, LongMemEval's official auto-judge).
- **LoCoMo scores cat1–4; cat5 (adversarial) is excluded** per the dataset authors.
- **LongMemEval here is the Oracle variant** (evidence sessions provided) — it
  isolates memory + reader quality from retrieval over a 40-session haystack.
- **Deterministic suite has zero degrees of freedom** — no embeddings, no LLM, fixed
  seeds; `schema_version: 2` raw output stores `{mean, stdev}` (stdev = 0 here).
- Full runner options: [tests/eval/benchmark/BENCHMARK.md](../tests/eval/benchmark/BENCHMARK.md).
