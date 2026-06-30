# Benchmarks

The agent-memory field reports two benchmarks above all others — **LoCoMo** and
**LongMemEval** — and it reports them as an *LLM-judged QA-accuracy percentage*,
broken down by question category, in a table against competitors. This page does
that, with one deliberate difference: **every ai-knot number here is deterministic
and reproducible from a single command.** That difference is the point, and the
rest of this section explains why.

## Why reproducibility is the headline

LoCoMo/LongMemEval leaderboard numbers are in a credibility crisis. The two most
public examples:

- **Zep** reported **84%** on LoCoMo. An independent re-evaluation put the real
  figure at **58.44%** — a 25-point gap traced to including the adversarial
  category that LoCoMo specifies should be *excluded*, modifying system prompts in
  Zep's favour, and reporting a single run instead of an average.
  ([getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5))
- **Mem0**'s own materials cite **91.6%** on LoCoMo and **94.8%** on LongMemEval;
  independent reproductions land closer to **58–66%**.
  ([mem0.ai benchmarks](https://mem0.ai/blog/ai-memory-benchmarks-in-2026))

The lesson is not "everyone is lying" — it is that an LLM-judged number is a
function of the reader model, the judge model, the prompts, the run count, and
which categories you score. Change any of those and the headline moves 20+ points.
So ai-knot leads with numbers that **cannot** move: no LLM on the path, fixed
seeds, multiple runs, the full category set scored, one command to reproduce.

## How the field reports these benchmarks

For reference, the conventional framing this page mirrors:

**LoCoMo** — ~2k QA pairs over 10 long multi-session conversations. Metrics: **J**
(LLM-as-judge accuracy, the headline), **F1**, **BLEU-1**. Categories: single-hop,
multi-hop, temporal, open-domain, and adversarial — where *adversarial (category 5)
is excluded from the score* per the dataset authors.

**LongMemEval** — 500 questions, judged for QA correctness by GPT-4o. Six question
types: single-session-user, single-session-assistant, single-session-preference,
temporal-reasoning, knowledge-update, multi-session — plus abstention variants.
Variants: `_S` (~115k tokens), `_M` (~500 sessions), Oracle. The hard categories
everyone regresses on are **multi-session** and **knowledge-update**.

Vendor-reported landscape (numbers vary by model/judge — treat as context, not
ground truth):

| System | LoCoMo (J, reported) | LongMemEval (acc, reported) |
|---|---|---|
| Mem0 (own materials) | 91.6 / indep. ~58–66 | 94.8 / indep. ~66 |
| Zep | 84 → corrected 58.44 | 63.8–75.1 |
| LangMem | ~78 | — |
| Memori | ~82 | — |

*Sources: [mem0.ai](https://mem0.ai/blog/ai-memory-benchmarks-in-2026),
[zep-papers#5](https://github.com/getzep/zep-papers/issues/5). The spread is the
story.*

## ai-knot: reproducible retrieval quality

ai-knot's public, one-command numbers measure **retrieval quality** — the signal a
reader LLM's QA accuracy is built on. On an in-repo golden suite, against a naive
recency/lexical log baseline:

| Scenario | Metric | baseline | ai-knot |
|---|---|---:|---:|
| ranking | semantic MRR | 0.18 | **0.83** |
| ranking | precision@1 | 0.10 | **0.70** |
| ranking | precision@5 | 0.40 | **1.00** |
| noise (200 distractors) | signal recall@3 | 0.60 | **0.80** |
| token economy | compression | 0.67 | **0.73** |
| scale (1000 facts) | MRR@1000 | 0.46 | **0.67** |

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

No network, no LLM, fixed seeds — re-run it and the numbers are identical.

### LoCoMo (retrieval grounding, deterministic)

Run against the public [LoCoMo](https://github.com/snap-research/locomo) dataset,
scored deterministically by token overlap against the gold answer (not an LLM
judge), so the table reproduces exactly. ai-knot beats the naive log in **every**
category:

| LoCoMo category | baseline | ai-knot | Δ |
|---|---:|---:|---:|
| **evidence_recall@5** | 0.15 | **0.26** | +71% |
| single-hop (grounding F1) | 0.044 | **0.066** | +50% |
| multi-hop (grounding F1) | 0.025 | **0.033** | +32% |
| temporal (grounding F1) | 0.055 | **0.071** | +29% |
| open-domain (grounding F1) | 0.066 | **0.107** | +62% |
| adversarial (grounding F1) | 0.060 | **0.093** | +55% |

```bash
python -m tests.eval.benchmark.runner --scenarios s_locomo \
  --skip-multi-agent --backends baseline,ai_knot_no_llm
```

> **Honest labelling.** These are *retrieval-grounding* metrics — how well the
> retriever surfaces and matches the gold evidence — **not** the LLM-judged J score
> the leaderboards report. `evidence_recall@5` is the ceiling any reader can work
> against; the F1 column is grounding strength, not "% correct." To produce a J
> score, add a reader+judge pass (below) — and inherit all of its model-dependence.

### LongMemEval (point-in-time, knowledge-update, temporal)

LongMemEval's two hardest categories — **knowledge-update** and **temporal-
reasoning** — are exactly the failure mode ai-knot's bi-temporal model targets.
When a fact is revised, ai-knot answers *as of* a point in time deterministically:

```python
kb.recall("where does the user work?", now=question_date)
# → what was true on question_date; facts a later one superseded are excluded.
```

This is the LongMemEval point-in-time pattern (`recall(now=…)`, also on the CLI as
`recall --now` and over MCP). The temporal correctness it depends on is
regression-tested in-repo (`tests/test_event_time_persistence.py`,
`tests/test_supersession_ingest.py`), so the behaviour LongMemEval's knowledge-
update / temporal categories reward is guaranteed, not curve-fit. The full
LongMemEval QA harness (answer generation + GPT-4o judge over the 500-question set)
runs externally and is model-dependent, like every number in the landscape table.

## Running the LLM-judged J score yourself

The harness supports a reader+judge pass for those who want a leaderboard-style
number — with the caveat that it is no longer deterministic and depends on your
model choices:

```bash
# Requires a reachable Ollama (judge + reader); see BENCHMARK.md for OpenAI wiring.
python -m tests.eval.benchmark.runner --scenarios s_locomo --backends ai_knot
```

If you publish a J score, follow what the Zep/Mem0 dispute taught the field:
average ≥3 runs with stddev, keep prompts and retrieval templates identical across
systems, and **exclude adversarial category 5** from the LoCoMo score.

## Multi-agent acceptance gate

Shared-pool memory has its own scored gate (S8–S26: CAS correctness, fan-in
recall, trust/adversarial discount, conflict resolution), run on every PR:

```bash
python -m tests.eval.benchmark.runner --multi-agent --mock-judge --ma-gate
```

See [production-readiness.md](production-readiness.md) §4.

## Methodology notes

- **Identical wiring** — same fixtures, same judge, same query set across backends;
  the only variable is the retrieval logic.
- **`ai_knot_no_llm` is the honest offline column** — under `--mock-judge` the
  extraction backend reduces to the same `kb.add()` path, so the no-LLM column is
  the one to cite for a zero-network claim.
- **Means over repeated runs** — raw output stores `{mean, stdev}` per metric
  (`schema_version: 2`); with the dense channel and LLM judge disabled, stdev is 0.
- Full runner options: [tests/eval/benchmark/BENCHMARK.md](../tests/eval/benchmark/BENCHMARK.md).
