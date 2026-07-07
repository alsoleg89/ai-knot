# Comparison guide

Updated: **July 5, 2026**

This is the buyer-facing comparison page: honest and specific about where
`ai-knot` fits against the OSS agent-memory projects that matter in 2026.

Every capability claim below is checked against each project's public docs and repo. Where
a competitor shares a strength with `ai-knot`, this page says so — an honest comparison is
worth more than a flattering one, and the whole category has a
[credibility problem](benchmarks.md) that rewards restraint.

---

## The projects

| Project | What it is | License / hosting |
|---|---|---|
| **ai-knot** | Deterministic, self-hosted fact memory; no LLM required on read *or* write | MIT, self-hosted |
| **Mem0** | The most-adopted "memory layer"; LLM-driven extraction + retrieval | Apache-2.0 OSS + hosted platform |
| **Zep / Graphiti** | Temporal knowledge-graph memory; Graphiti is the OSS engine, Zep the SaaS | Graphiti Apache-2.0; Zep proprietary |
| **Letta** (ex-MemGPT) | Stateful-agent platform where the LLM manages its own memory | Apache-2.0 OSS + cloud |
| **Cognee** | LLM + ontology + knowledge-graph memory pipeline (ECL) | Apache-2.0 OSS + cloud |
| **LangMem** | LangGraph-native memory library | MIT, library (LangGraph-coupled) |
| **Memori** | SQL-native memory; structured retrieval, no vector DB required | Apache-2.0 OSS + cloud |

> Maturity, stated plainly: Mem0 (~60k★), Graphiti (~28k★), Cognee (~27k★), Letta (~24k★),
> Memori (~15.5k★), and LangMem (~1.5k★) are all older and more adopted than `ai-knot`.
> `ai-knot` is the newcomer with a narrow, defensible wedge — not the incumbent. If you
> want the largest ecosystem and community today, several of these win on that axis alone.

## One-sentence difference

- **ai-knot** — deterministic, self-hosted fact memory that needs no LLM on the read path
  *or* the write path, with real multi-agent governance and 78% LoCoMo QA accuracy.
- **Mem0** — a broad, popular memory layer that uses an LLM to extract facts on write and to
  help select what to retrieve; vector-store (optionally graph) backed.
- **Zep / Graphiti** — builds an LLM-extracted *temporal knowledge graph*; recall is
  graph + hybrid search (no LLM at query time), but ingestion is LLM-heavy and needs a graph DB.
- **Letta** — the LLM *is* the memory manager: retrieval happens through the agent's own tool
  calls, which is powerful but the opposite of deterministic.
- **Cognee** — an ontology-and-graph pipeline where LLMs build and query the knowledge graph.
- **LangMem** — the native memory library for LangGraph; Python-only, LLM-formed memories.
- **Memori** — SQL-native structured memory; like `ai-knot` it keeps the LLM off recall, but
  uses an LLM to populate the store and has no YAML/graph-free story of its own.

## Feature comparison

| Capability | ai-knot | Mem0 | Zep / Graphiti | Letta | Cognee | LangMem | Memori |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Self-hosted OSS core path | ✅ | ✅ | ✅¹ | ✅ | ✅ | ✅ | ✅ |
| **No LLM required on recall** | ✅ | ❌ | ✅ | ❌ | ◑ | ✅ | ✅ |
| **No LLM required on write / ingest** | ✅² | ◑ | ❌ | ◑ | ❌ | ❌ | ❌ |
| No knowledge graph / graph DB required | ✅ | ◑ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Human-readable, git-trackable store (YAML) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ◑³ |
| Deterministic, reproducible recall | ✅ | ❌ | ◑⁴ | ❌ | ❌ | ◑ | ✅ |
| Bi-temporal point-in-time recall (`now=…`) | ✅ | ◑ | ✅ | ❌ | ◑ | ❌ | ◑ |
| Multi-agent trust / visibility / provenance | ✅ | ❌ | ◑⁵ | ◑ | ◑ | ❌ | ❌ |
| Python **and** TypeScript client | ✅ | ✅ | ◑⁶ | ✅ | ✅ | ❌⁷ | ✅ |
| MCP server | ✅ | ◑ | ✅ | ◑ | ✅ | ❌ | ✅ |
| Reproducible, no-LLM published benchmark | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

Legend: ✅ first-class · ◑ partial / not the center of the product · ❌ not offered.

1. Graphiti (the engine) is Apache-2.0 and self-hostable; **Zep** itself is a proprietary
   hosted service built on top of it.
2. `ai-knot`'s default `add` / `add_resolved` writes are direct fact insertion with **no
   model call**. LLM extraction (`learn`) is an *optional* convenience, not a requirement —
   so the entire pipeline, write and read, can run with zero LLM calls. Most of the others use an
   LLM to populate the store by default; a couple expose a raw-insert escape hatch (e.g. Mem0's
   `add(infer=False)`, Letta archival inserts), but it bypasses their own extraction — zero-model
   *structured* writes are ai-knot's default path, not a fallback.
3. Memori is SQL-native and exportable to SQLite (queryable/auditable), but has no
   human-first YAML store.
4. Graphiti's *retrieval* is deterministic hybrid search, but the graph it queries is
   built by an LLM, so the end-to-end result depends on non-deterministic ingestion.
5. Zep/Graphiti has the strongest *temporal provenance* story of the group (bi-temporal,
   fact invalidation), but explicit trust-scoring and visibility-scope governance is not a
   documented OSS feature.
6. Graphiti is Python-first; TypeScript/Go clients are via the Zep platform.
7. LangMem is Python-only and coupled to LangGraph; it is a library, not a standalone service.

### The two rows that matter most

Several projects here already keep the LLM **off the recall path** — Graphiti, LangMem, and
Memori all qualify, so "no LLM on recall" is not, by itself, unique to `ai-knot`.

The rarer property is the **write** row: `ai-knot` is the only project in this table that
does not *require* an LLM to populate memory. Direct fact insertion is the default; `learn()`
extraction is opt-in. That's what makes a genuinely **zero-LLM, air-gappable** deployment
possible — the deterministic suite in [benchmarks.md](benchmarks.md) backs it: same fixtures,
fixed seeds, no network, no model, identical numbers on every run.

## When ai-knot wins clearly

Choose `ai-knot` if you care about:

- a memory layer that can run with **no LLM anywhere** — read or write — for cost,
  latency, air-gapped, or reproducibility reasons
- **deterministic recall you can lock into a regression test**
- self-hosted storage you control directly (SQLite / PostgreSQL / YAML), including a
  human-readable, git-trackable option
- explicit correction and audit loops (`learn`, `add_resolved`, `valid_until`, `lineage`)
  instead of "just overwrite the memory"
- shared memory across several agents with **trust, provenance, and visibility rules**, not
  just a common table
- deterministic retrieval numbers, in a category where headline QA numbers swing 25+
  points between vendors

## When not to choose ai-knot

Be honest with yourself — pick another project first if:

- you want a **fully managed hosted memory SaaS** today → Mem0, Zep, or Letta Cloud
- you want an **LLM-built knowledge graph** and relational/graph reasoning as the core
  product → Zep/Graphiti or Cognee
- you want the **LLM to autonomously manage its own memory inside the agent loop** → Letta
- you're all-in on **LangGraph** and want the most native store with zero extra surface →
  LangMem
- you want the **biggest headline LoCoMo number** — that is not `ai-knot`'s game, and
  [benchmarks.md](benchmarks.md) explains why that number is the least trustworthy thing in
  the category right now
- you want the **largest community and ecosystem** this quarter → Mem0 leads by a wide
  margin

## Practical buyer guide

### "I just want my agent to remember users across sessions."

`ai-knot` if you want deterministic, inspectable, self-hosted fact memory with no model on
the hot path. Mem0 if you want the largest ecosystem and don't mind an LLM in the loop.

### "I need memory for a coding agent, self-hosted."

`ai-knot` for deterministic recall, storage control, and multi-agent governance. Letta if you
want the agent itself to reason about and manage its memory. Memori if a pure SQL store with
zero new infra is the priority.

### "I need a real knowledge graph and relational reasoning."

Zep/Graphiti or Cognee are the natural fits — that *is* their product. `ai-knot` deliberately
does not build a graph; it trades graph reasoning for determinism and a smaller operational
surface (no Neo4j/FalkorDB/Neptune to run).

### "I already use LangGraph or CrewAI."

Those are host ecosystems, not competitors. `ai-knot` ships adapters for each
([integrations.md](integrations.md)); use it when you want deterministic, self-hosted
long-term memory *under* the runtime you already have. LangMem is the most native choice if
you never leave LangGraph and don't need portability, MCP, or a graph-free deterministic store.

### "I already use LlamaIndex."

LlamaIndex is the host runtime here, not the competitor. `ai-knot` fits the same
`memory=...` seam through `AiKnotLlamaIndexMemory`, so you keep the LlamaIndex runtime and
swap only the long-term memory layer for a deterministic, self-hosted one.

### "The benchmark numbers confuse me — who's actually best?"

Nobody can answer that from the public numbers, and that's the point. Published LoCoMo claims
span ~58% to >92%, the same system (Zep) has been reported at 84%, 58%, and 75%, and vendors
openly dispute each other's methodology. `ai-knot` reports 78% with the reader and judge named, plus a deterministic retrieval number with zero degrees of freedom — see [benchmarks.md](benchmarks.md).

## The honest wedge

`ai-knot` is not trying to be everything in the category, and it is not the most adopted
project on this page. Its wedge is narrow and defensible:

> **A self-hosted memory layer that needs no LLM on the read *or* write path, treats
> multi-agent memory as a governance problem, not just a shared table, and backs its
> retrieval with a benchmark that can't drift.**

That is narrower than the graph systems and younger than the incumbents, but it is easier to
trust, easier to test, and easier to explain.
