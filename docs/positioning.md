# ai-knot positioning

Updated: **July 1, 2026**

This is the message house for ai-knot. If copy, docs, release notes, or outbound
posts disagree with this page, this page wins.

---

## One-liner

**Long-term memory for AI agents — no LLM on read or write. Store facts, not transcripts;
recall only what matters, in milliseconds; for one agent or a whole team; drop it into your
stack in one line.**

## Short pitch

Most agent stacks still treat memory as a log: save every message, replay too much
history, pay a token tax forever, and hope the model notices the right detail.
ai-knot turns memory into a **self-hosted knowledge layer**. It extracts or accepts
facts, stores them in SQLite / PostgreSQL / YAML, and retrieves only the few facts
the next turn needs. The hot path is deterministic: no LLM on recall.

## ICP

1. **AI engineers building assistants or coding agents**
   They are already paying the cost of replaying chat history and want something
   they can `pip install`, test, and self-host.
2. **Teams building multi-agent systems**
   They need shared state, provenance, trust, and visibility rules, not just one
   more vector store.
3. **Regulated, local-first, or air-gapped deployments**
   They cannot rely on a hosted memory SaaS or an LLM call in the retrieval path.

## Main pain solved

Agents either:

- forget everything between sessions, or
- replay too much history into every prompt, which makes them slower, more expensive,
  and less reliable over time.

ai-knot solves the replay-tax problem by keeping **knowledge**, not the raw log.

## Why now

1. **Agent adoption is up, but memory quality still lags.**
   Most frameworks improved orchestration faster than they improved memory.
2. **Benchmark trust is weak.**
   Memory leaderboard claims move dramatically with the reader, judge, prompts,
   and category filters. A reproducible stance is now a differentiator.
3. **MCP made memory pluggable.**
   Claude Desktop / Claude Code / OpenClaw and the rise of HTTP-capable MCP
   hosts gave developers immediate integration surfaces for bring-your-own
   memory tools.
4. **Self-hosted is back in demand.**
   Teams want more control over privacy, latency, and infrastructure spend.

## Why ai-knot is different

### 1. No LLM required — on read *or* write

Recall is deterministic. That means lower latency and cost, reproducible outputs, auditable
behavior, and testable regressions.

But the sharper, rarer claim is about the **write** path. Several competitors already keep
the LLM off *recall* (Zep/Graphiti, LangMem, Memori). What almost none of them do is let you
populate memory without an LLM: Mem0, Zep/Graphiti, Letta, Cognee, LangMem, and Memori all
use a model to extract facts or build a graph on ingestion. In `ai-knot`, direct fact
insertion (`add` / `add_resolved`) needs **no model call**, and `learn()` extraction is
opt-in — so the *entire pipeline* can run with zero LLM calls. That is what makes a truly
air-gapped, fully-reproducible deployment possible.

> **Message discipline / honesty guardrail.** Do **not** claim ai-knot is the *only*
> no-LLM-on-recall system — it isn't, and a developer audience will catch it. Lead with
> **"no LLM on read *or* write,"** **multi-agent governance**, and **one-line integration** — the
> combination that is genuinely defensible. See [comparison.md](comparison.md) for the
> checked, per-competitor claims.

### 2. It stores facts, not just embeddings or logs

The product narrative is not "another retrieval layer." It is "a memory system
that distills noisy interaction history into a smaller, queryable knowledge base."

### 3. It is self-hosted and storage-pluggable

SQLite, PostgreSQL, and YAML all expose the same API. YAML is human-readable and
git-trackable; PostgreSQL supports shared deployments; SQLite is the low-friction
production default.

### 4. It has a real multi-agent governance story

Shared memory is not just a common database:

- provenance-aware publishing,
- evidence-before-belief gating,
- visibility scopes,
- trust and quick-invalidation penalties,
- fan-in recall across multiple agents.

### 5. The benchmark stance is part of the product

ai-knot ships both:

- named-reader QA numbers, and
- deterministic, rerunnable retrieval numbers.

That makes reproducibility part of the product promise, not just a footnote.

## Proof points to repeat

- **LoCoMo:** 78.0% QA accuracy (cat1-4, gpt-4.1 reader / gpt-4o judge)
- **LongMemEval:** 59.6% QA accuracy (Oracle)
- **Deterministic retrieval suite:** MRR 0.83 vs 0.18 naive
- **Backends:** SQLite / PostgreSQL / YAML
- **Surfaces:** Python, plain function-calling Python helpers, npm/TypeScript, Vercel AI SDK, MCP over stdio or Streamable HTTP, HTTP sidecar + browser inspector, CrewAI, LlamaIndex, OpenAI Agents SDK, PydanticAI, LangGraph tool helpers, LangChain retriever/chat-memory
- **Read path:** no LLM required

## Message pillars

### Pillar 1: Stop replaying the whole transcript

Lead with the pain everyone already feels: prompt bloat, context rot, latency, cost.

### Pillar 2: Determinism is a feature, not a compromise

The right buyer values reproducibility and auditability more than "the model guesses
semantic relevance somewhere in the loop."

### Pillar 3: Self-hosted without lock-in

The right buyer wants a memory layer they can run in their own stack, inspect, back up,
and migrate.

### Pillar 4: Multi-agent memory needs governance

This is where ai-knot is most defensible versus "memory for one chatbot."

## What ai-knot is not

Do not position ai-knot as:

- a hosted memory platform,
- a full agent runtime,
- a graph-first relational reasoning engine,
- the fastest way to get started if someone already decided on LangGraph-only memory.

That honesty helps trust.

## Competitive wedge by buyer

| Buyer | Why they choose ai-knot |
|---|---|
| Indie builder | cheaper, simpler, self-hosted memory without adopting a new platform |
| Platform team | deterministic retrieval, explicit storage control, regression-friendly behavior |
| Multi-agent builder | trust, provenance, visibility, fan-in recall |
| Regulated team | no LLM on recall, private infrastructure, inspectable state |

## CTA ladder

1. **Primary CTA:** run `ai-knot demo` (or `npx ai-knot-demo` for the npm bridge)
2. **Secondary CTA:** try the CLI memory loop: `ai-knot add`, `ai-knot search`, `ai-knot list`, `ai-knot delete`
3. **Tertiary CTA:** inspect the benchmark command and methodology
4. **Surface CTA:** choose a surface: CrewAI, LlamaIndex, PydanticAI, LangGraph, MCP (`setup` or `serve-mcp`), TypeScript, OpenAI Agents SDK, HTTP
5. **Contributor CTA:** ask for the next adapter or backend

## What to say in 30 seconds

> ai-knot is a self-hosted memory layer for agents that stores facts instead of
> replaying whole transcripts. Retrieval is deterministic, so recall is cheap,
> auditable, and testable. It works over SQLite/Postgres/YAML, ships MCP over
> stdio or Streamable HTTP, framework adapters including LlamaIndex,
> PydanticAI, and the OpenAI Agents SDK, LangGraph memory helpers, and a
> TypeScript client, and publishes benchmark numbers you can actually re-run.
