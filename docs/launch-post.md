# Stop shipping your agent's chat log as its memory

*The launch story for ai-knot — deterministic, self-hosted long-term memory for AI agents.*

Updated: **July 5, 2026**

> This is the canonical announcement piece. It seeds the GitHub Release notes, the
> "Show & Tell" discussion, and the long-form Dev.to / blog article. For the message
> house it must agree with, see [positioning.md](positioning.md); for channel-specific
> copy and the calendar, see [launch-plan.md](launch-plan.md).

---

## The problem everyone has already felt

Most agent stacks still treat memory as a transcript problem: save every message,
summarize occasionally, and replay a large slice of history back into the next prompt.
It works in the demo and fails in production. Three failures show up on schedule:

- **Cost.** You pay to re-send the same background knowledge on every turn.
- **Context rot.** The more history you stuff back in, the more irrelevant text
  competes with the three facts that actually matter.
- **No auditability.** When recall is a chain of summaries, embeddings, rerankers, and
  model calls, you can't explain *why* one fact surfaced and another didn't — or write a
  test that pins it down.

A conversation is evidence. Memory is the small set of knowledge that should survive it.

## What ai-knot is

**ai-knot stores facts instead of transcripts, recalls only the few a turn needs, and
keeps the read path deterministic — no LLM on recall.** It's self-hosted, runs over
SQLite / PostgreSQL / YAML, and ships the same `add / search / list / delete` loop across
Python, TypeScript, a CLI, an MCP server, an HTTP sidecar, and adapters for the agent
frameworks people already use.

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")
kb.add("User deploys APIs with Docker and Kubernetes")
kb.add("User prefers Go and avoids Java")

kb.search("what stack does the user use?")   # alias: kb.recall(...)
# [1] User deploys APIs with Docker and Kubernetes
# [2] User prefers Go and avoids Java
```

## The 30-second proof

No signup, no key, no cloud:

```bash
pip install ai-knot
ai-knot demo
```

Starting from Node instead:

```bash
npm install ai-knot
npx ai-knot-demo
```

The demo runs the whole `add → search → list → get → delete` loop against temporary local
storage and prints every step. If there's no embedding endpoint, it falls back to BM25-only
retrieval and says so — a supported, fully-deterministic mode.

## Why it's built this way

### 1. Determinism is a feature, not a compromise

Recall uses deterministic retrieval and rank fusion. The same stored state and the same
query return the same result — every time. That buys four things a probabilistic read path
can't: lower latency, lower cost, auditable behavior, and **regression tests against memory
itself**. You can lock recall into CI and notice the day it changes.

### 2. Facts, not messages

The unit is a fact — semantic, procedural, or episodic — with importance, tags, event time,
provenance, and validity windows. Because facts have a lifecycle, correction is explicit:
`learn` for extract-on-write, `add_resolved` for structured update/delete, `valid_until`
for supersession, and `lineage` for the audit trail. Memory stops being a black box you
overwrite and starts being a store you can inspect.

### 3. Bi-temporal recall

Facts separate *when you learned something* from *when it was true*. So point-in-time
queries work:

```python
kb.recall("where does the user work?", now=question_date)
# → what was true on question_date; facts a later one superseded are excluded.
```

This is the axis most memory systems get wrong on knowledge-update and temporal questions.

### 4. Multi-agent memory needs governance, not a shared table

When several agents publish into one memory, retrieval is the easy half. The hard half is
governance. ai-knot's shared pool adds provenance-aware publishing, evidence-before-belief
gating, per-agent visibility scopes, trust penalties that survive flooding and laundering
attempts, and fan-in recall that reconstructs an answer scattered across many agents'
shards. This is the sharpest edge versus "memory for one chatbot."

## The part the category is getting wrong: benchmarks

Agent-memory leaderboard numbers are in a credibility crisis, and it's worth being blunt
about it because it shapes how ai-knot reports its own results.

- **Zep** published an **84%** LoCoMo score. An independent re-evaluation that restricted
  scoring to the four validated categories, aligned the prompts, and averaged ten runs put
  it at **58.44% ± 0.20** — a ~25-point gap traced to counting Category-5 answers in the
  numerator but not the denominator. Zep's rebuttal claims **75.14%** with its own config.
  The thread closed unresolved.
  ([getzep/zep-papers#5](https://github.com/getzep/zep-papers/issues/5))
- **Mem0's** own LoCoMo self-report climbed from **~67%** (2025 paper) to **92.5%** (2026
  blog) — a ~25-point jump attributed to a "token-efficient algorithm" that no independent,
  neutral re-run has reproduced.
- Across the field, published LoCoMo claims now span **~58% to >92%**, and the leading
  vendors openly contest each other's methodology. The number you see is a function of the
  reader model, the judge model, the prompt wording, the run count, and which categories
  you score.

So ai-knot leads with a number that has **none of those degrees of freedom**:

| Scenario | Metric | naive log | ai-knot |
|---|---|---:|---:|
| ranking | semantic MRR | 0.18 | **0.83** |
| ranking | precision@5 | 0.40 | **1.00** |
| scale (1000 facts) | MRR@1000 | 0.46 | **0.67** |
| LoCoMo | `evidence_recall@5` | 0.15 | **0.26** (+71%) |

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

Same fixtures, fixed seeds, no network, no LLM — re-run it and you get identical numbers.

Then, **with every knob named**, ai-knot reports the LLM-judged QA numbers too:
**LoCoMo 78.0%** (cat1–4, `gpt-4.1` reader / `gpt-4o` judge) and **LongMemEval 59.6%**
(Oracle). Those aren't the biggest headline in the category — they're the most *legible*
one. On LongMemEval, ai-knot is near-perfect on information extraction (95–98%) and declines
false-premise questions **90%** of the time, which is exactly the failure mode that makes
memory systems confabulate. Full methodology, caveats, and per-conversation tables:
[docs/benchmarks.md](benchmarks.md).

## Where it fits, honestly

ai-knot is not trying to be every product in the 2026 memory category. Choose it when you
want self-hosted deterministic fact memory you can test and inspect, storage you control,
explicit correction loops, and real multi-agent governance. **Don't** choose it first if
you want a fully-managed hosted memory SaaS, a graph-first relational reasoning engine, or a
full agent runtime. Stating the boundaries is part of earning the trust. The full
buyer-facing breakdown against Mem0, Zep, Letta, Cognee, LangMem, and Memori lives in
[comparison.md](comparison.md).

## Try it in your stack in under a minute

| Starting from… | Install | Run first |
|---|---|---|
| Python | `pip install ai-knot` | `ai-knot demo` |
| Node / TypeScript | `npm install ai-knot` | `npx ai-knot-demo` |
| Claude / OpenClaw / any MCP host | `pip install "ai-knot[mcp]"` | `ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config` |
| CrewAI · LangGraph · LlamaIndex · PydanticAI · OpenAI Agents · AutoGen · Vercel AI SDK | `pip install "ai-knot[...]"` | the matching `examples/*_surface_demo.py` |
| Re-run the benchmark | — | the deterministic command above |

- **Repo:** <https://github.com/alsoleg89/ai-knot>
- **Docs:** [usage](usage.md) · [integrations](integrations.md) · [memory commands](memory-commands.md) · [benchmarks](benchmarks.md) · [comparison](comparison.md) · [FAQ](faq.md)

The next generation of agent memory won't win on bigger context. It'll win on **which
context it keeps, how reliably it retrieves it, and how credibly it proves it.** That's the
bet ai-knot is making.
