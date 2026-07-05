<!-- mcp-name: io.github.alsoleg89/ai-knot -->
<div align="center">

# 🪢 ai-knot

### The deterministic memory layer for AI agents.

**No LLM on the read path _or_ the write path.** ai-knot keeps a self-hosted, MCP-native knowledge store, recalls only the few facts each turn needs, and does it deterministically — cheap, reproducible, and testable. Temporal, multi-agent, and air-gappable by default.

[![CI](https://github.com/alsoleg89/ai-knot/actions/workflows/ci.yml/badge.svg)](https://github.com/alsoleg89/ai-knot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ai-knot)](https://pypi.org/project/ai-knot/)
[![npm](https://img.shields.io/npm/v/ai-knot)](https://www.npmjs.com/package/ai-knot)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

[**Fastest proof**](#fastest-proof-30-seconds) · [**Basic memory commands**](#basic-memory-commands) · [**Integrations**](docs/integrations.md) · [**Examples**](examples/README.md) · [**Benchmarks**](docs/benchmarks.md) · [**Comparison**](docs/comparison.md) · [**Open in Codespaces**](https://codespaces.new/alsoleg89/ai-knot)

**No LLM on recall · no LLM on write · a benchmark you can re-run · Python · TypeScript · MCP · HTTP**

`pip install ai-knot && ai-knot demo` — 30 seconds, no signup, no API key

**Works with:** Claude · OpenClaw · CrewAI · LangGraph · LlamaIndex · AutoGen · OpenAI Agents SDK · PydanticAI · Vercel AI SDK

**Backends:** SQLite · PostgreSQL · YAML  |  **Surfaces:** Python · TypeScript · CLI · MCP · HTTP · Browser inspector

<img src="docs/assets/hero-demo.gif" alt="ai-knot demo: store facts once, recall only what matters, with deterministic memory persisted across restarts" width="1200" />

</div>

> **Why another memory library?** Every agent-memory benchmark is unreproducible — the same
> system has been publicly reported at **84%, 58%, and 75%** on LoCoMo, and claims across the
> field span ~58–92%. ai-knot ships a retrieval number that **can't drift** — deterministic
> ranking **MRR 0.83 vs 0.18** for naive BM25, same fixed seeds, no network, no LLM — and runs
> the whole pipeline, read *and* write, with zero model calls. [Re-run it yourself →](docs/benchmarks.md)

*Self-hosted OSS — no cloud tier, no signup, no API key. New and pre-1.0: if the reproducibility-first approach resonates, a ⭐ helps others find it, and questions are welcome in [Discussions](https://github.com/alsoleg89/ai-knot/discussions).*

---

## At a glance

| If you care about… | ai-knot gives you… |
|---|---|
| recall that stays cheap and reproducible | deterministic `search` / `recall` with no LLM on the read path |
| storage you can inspect and migrate | SQLite for the default path, PostgreSQL for shared deployments, YAML for human-readable state |
| one memory product across stacks | Python, TypeScript, CLI, MCP, HTTP sidecar, Browser inspector, and framework adapters |
| memory you can debug and correct | `list`, `get`, `delete`, `lineage`, `learn`, `add_resolved`, and `valid_until` |
| more than one agent writing to memory | shared memory with trust, provenance, visibility, and fan-in recall |

```text
conversation / tool output -> add or learn -> ai-knot -> SQLite | PostgreSQL | YAML
next question              -> search / recall -> 3-5 relevant facts -> next prompt
```

## The 2026 memory problem

First-generation memory layers (2023–2024) fixed transcript replay by throwing *more* LLM at it — a model to extract facts on write, a model to pick what to retrieve, sometimes a model to build a graph. That trades one problem for three: cost on every call, non-determinism you can't test, and benchmark numbers nobody can reproduce.

| ❌ First-gen (2023–2024) | ✅ ai-knot (2026) |
|---|---|
| LLM on extraction, retrieval, and ranking | no LLM on the read path *or* the write path |
| token + latency cost on every turn | recall is cheap and testable in CI |
| benchmark scores nobody can re-run | one deterministic number that can't drift |
| a black box you overwrite blindly | lineage, supersession, and audit built in |

`ai-knot` is the 2026 take: memory as a **deterministic, self-hosted, MCP-native layer** you can inspect, test, and run air-gapped — the same `add / search / list / delete` loop across Python, TypeScript, CLI, MCP, and HTTP, with direct paths for Claude, OpenClaw, CrewAI, LangGraph, LlamaIndex, AutoGen, OpenAI Agents SDK, and PydanticAI.

## Fastest proof (30 seconds)

If you want the single shortest proof that the product works, use this path:

```bash
pip install ai-knot
ai-knot demo
```

If you are starting from Node / TypeScript instead:

```bash
npm install ai-knot
npx ai-knot-demo
```

The raw Python API is just as small:

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")

fact = kb.add("User deploys APIs with Docker and Kubernetes")
kb.add("User prefers Go and avoids Java")
kb.add("Team standup is at 10am")

print(kb.search("what stack does the user use?"))  # alias: kb.recall(...)
# [1] User deploys APIs with Docker and Kubernetes
# [2] User prefers Go and avoids Java

print(kb.list())
print(kb.get(fact.id))
# kb.delete(fact.id)  # alias: kb.forget(...)
```

That is the core promise: persist facts to your own storage, then pull back only the 3-5 facts the next turn needs.

## Basic memory commands

If you only remember one product loop, make it this:

```bash
ai-knot add    assistant "User deploys APIs with Docker and Kubernetes"
ai-knot search assistant "what does the user deploy with?"   # alias: ai-knot recall
ai-knot list   assistant                                      # alias: ai-knot show
ai-knot delete assistant <fact_id>                            # alias: ai-knot forget
```

The same loop exists across every major surface:

| Surface | Add | Search | List | Delete |
|---|---|---|---|---|
| Core Python | `kb.add(...)` | `kb.search(...)` / `kb.recall(...)` | `kb.list()` / `kb.list_facts()` | `kb.delete(id)` / `kb.forget(id)` |
| TypeScript / npm | `await kb.add(...)` | `await kb.search(...)` / `await kb.recall(...)` | `await kb.list()` / `await kb.listFacts()` | `await kb.delete(id)` / `await kb.forget(id)` |
| CLI | `ai-knot add ...` | `ai-knot search ...` / `ai-knot recall ...` | `ai-knot list ...` / `ai-knot show ...` | `ai-knot delete ...` / `ai-knot forget ...` |
| MCP | `add` | `search` / `recall` | `list` / `list_facts` | `delete` / `forget` |
| HTTP sidecar | `POST /v1/facts` | `POST /v1/search` | `GET /v1/facts` | `DELETE /v1/facts/{fact_id}` |

When you need a deeper correction or audit loop:

- use `get` when you already have a `fact_id`
- use `lineage` when the history of one fact matters
- use `learn` when you want extract-on-write from raw text
- use `add_resolved` when you want explicit `update` / `delete` semantics with `valid_until`

For the full command map, including `include_inactive`, `lineage`, structured correction, and cross-surface equivalents, use [docs/memory-commands.md](docs/memory-commands.md).

If you're wiring Claude or OpenClaw, keep setup separate from the memory verbs:

```bash
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot setup claude   --agent-id assistant --storage sqlite --write-default-config
ai-knot doctor --json
```

Once the MCP server is live, use the same `add` / `search` / `list` / `delete` loop inside the client.

## Start here

Find your stack, install it, and run one command to see the full memory loop:

| If you're starting from… | Install | Run this first | What it proves |
|---|---|---|---|
| Core Python | `pip install ai-knot` | `ai-knot demo` | the end-to-end `add` / `search` / `list` / `get` / `delete` loop against temporary local storage |
| Node / TypeScript | `npm install ai-knot` | `npx ai-knot-demo` | the same built-in proof through the packaged Node bridge |
| Any function-calling Python agent | `pip install ai-knot` | `python examples/function_calling_surface_demo.py` | plain Python memory callables for runtimes that register ordinary tools |
| CrewAI | `pip install "ai-knot[crewai]"` | `python examples/crewai_surface_demo.py` | root memory plus scoped agent memory without a real model call |
| LangGraph tool-style memory | `pip install "ai-knot[langgraph]"` | `python examples/langgraph_surface_demo.py` | explicit `add` / `search` / `list` / `delete` tools on a LangGraph-shaped path |
| LlamaIndex | `pip install "ai-knot[llamaindex]" "llama-index-llms-openai"` | `python examples/llamaindex_surface_demo.py` | the `memory=...` seam on a zero-network path |
| PydanticAI | `pip install "ai-knot[pydanticai]"` | `python examples/pydanticai_surface_demo.py` | per-run instruction injection with deterministic memory |
| Claude / OpenClaw / any MCP client | `pip install "ai-knot[mcp]"` | `ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config` | one-command config merge plus MCP sanity check |
| HTTP sidecar | `pip install "ai-knot[server]"` | `python examples/http_sidecar_surface_demo.py` | `/v1/facts`, `/v1/search`, `GET /v1/facts`, and delete over the HTTP JSON surface |
| Vercel AI SDK | `npm install ai-knot ai @ai-sdk/openai` | `cd npm && npm run example:vercel-ai-sdk` | memory injection into a mainstream TypeScript app path |
| No local install | none | follow [docs/codespaces-quickstart.md](docs/codespaces-quickstart.md) | install-free first run in Codespaces |

Need the repo-native CLI transcript first? Run `python examples/cli_memory_loop.py`.

Need the visual debug surface first? Run `python examples/browser_inspector_demo.py` for the Browser inspector.

Need the real model-backed LlamaIndex path? Run `OPENAI_API_KEY=... python examples/llamaindex_integration.py`.

Need every runnable path in one place? Use [examples/README.md](examples/README.md).

TypeScript note: the npm package uses the Python engine underneath, so keep Python `3.11+` on `PATH` and use `npx ai-knot-doctor --json` if the bridge looks wrong.

If the bridge or environment looks suspicious:

```bash
ai-knot doctor --json
npx ai-knot-doctor --json
```

## What it looks like in your stack

You should be able to spot your existing runtime in under a minute.

| Stack | Integration seam |
|---|---|
| Core Python | `KnowledgeBase(...)` |
| Plain function-calling Python runtimes | `create_basic_memory_functions(...)` |
| LangChain / LangGraph tools | `create_basic_memory_tools(...)` |
| CrewAI | `AiKnotCrewAIMemory` |
| LlamaIndex | `AiKnotLlamaIndexMemory` |
| AutoGen | `AiKnotAutoGenMemory` |
| OpenAI Agents SDK | `AiKnotAgentsMemory` |
| PydanticAI | `AiKnotPydanticAIMemory` |
| TypeScript / Node | `KnowledgeBase` |
| TypeScript over HTTP | `HttpKnowledgeBase` |
| Claude / OpenClaw / any MCP host | `ai-knot-mcp` via `setup` or `serve-mcp` |

### Any function-calling Python agent

```python
from ai_knot.integrations import create_basic_memory_functions

functions = create_basic_memory_functions(kb, top_k=5, include_get=True)
```

### CrewAI

```python
from ai_knot.integrations.crewai import AiKnotCrewAIMemory

memory = AiKnotCrewAIMemory(kb, top_k=5)
crew = Crew(agents=[researcher, writer], tasks=[task], memory=memory)
```

### LangGraph / LangChain

```python
from langgraph.prebuilt import create_react_agent
from ai_knot.integrations.langchain import create_basic_memory_tools

tools = create_basic_memory_tools(kb, top_k=5)
agent = create_react_agent(model, tools=tools)
```

### LlamaIndex

```python
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory

memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=5)
```

### TypeScript / Vercel AI SDK

```typescript
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { AiKnotAISDKMemory, KnowledgeBase } from "ai-knot";

const kb = new KnowledgeBase({ agentId: "assistant", storage: "sqlite" });
const memory = new AiKnotAISDKMemory(kb, { topK: 5 });
const system = await memory.buildSystem("Write a deployment checklist.", {
  baseSystem: "You are a concise staff engineer.",
});

const { text } = await generateText({
  model: openai("gpt-5"),
  system,
  prompt: "Write a deployment checklist.",
});
```

### TypeScript / HTTP sidecar

```typescript
import { HttpKnowledgeBase } from "ai-knot";

const kb = new HttpKnowledgeBase({
  baseUrl: "http://127.0.0.1:8000",
  token: process.env.AI_KNOT_SERVER_TOKEN,
});
```

### Claude / OpenClaw / any MCP client

```bash
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot serve-mcp assistant --port 8765
# MCP endpoint: http://127.0.0.1:8765/mcp
```

For the full surface map, use [docs/integrations.md](docs/integrations.md). If you want the assistant itself to know these surfaces before editing your repo, use [skills/README.md](skills/README.md). For a shareable landing page, use [docs/site/index.html](docs/site/index.html).

## Why teams pick ai-knot

- **Deterministic recall.** No LLM on the read path, so the hot path is cheaper, faster, and regression-testable.
- **Storage you control.** SQLite for the default path, PostgreSQL for shared deployments, YAML when you want human-readable state.
- **Fact lifecycle, not blind overwrite.** `learn`, `add_resolved`, `valid_until`, and `lineage` make memory correction explicit.
- **Framework breadth without lock-in.** Python, TypeScript, MCP, HTTP, and adapters for the agent stacks people already use.
- **Multi-agent governance.** Shared memory with provenance, trust, visibility, and fan-in recall instead of "everyone writes to one bag of vectors."
- **Benchmark credibility.** Named-reader QA numbers plus deterministic retrieval metrics you can actually rerun.

## Built for teams of agents

The single-agent path is the easy part. The harder problem is shared memory that does not turn into noise.

- fan-in recall across multiple agents
- evidence-before-belief gating
- per-agent visibility control
- deterministic conflict handling
- trust penalties that survive flooding or laundering attempts

If multi-agent memory is part of the product, not a future nice-to-have, this is one of `ai-knot`'s sharpest edges. Details live in [docs/usage.md](docs/usage.md#multi-agent).

## What you can build

| You are building… | ai-knot gives you… |
|---|---|
| A chatbot that remembers users | persistent per-user facts without replaying the full chat log |
| A coding agent | project, tooling, and preference memory that comes back in milliseconds |
| A team of agents | shared memory with trust and visibility rules, not just a shared database |
| A regulated or air-gapped system | self-hosted storage and no LLM on recall |
| A product that must be testable | deterministic retrieval you can lock into regression tests |

## How it compares

`ai-knot` is the newcomer in a crowded 2026 memory landscape, not the incumbent. Several projects are far more adopted — Mem0 (~60k★), Graphiti (~28k★), Cognee (~27k★), Letta (~24k★), Memori (~15.5k★). `ai-knot`'s wedge is narrow on purpose.

| If you want… | Best fit |
|---|---|
| memory that needs no LLM on read **or** write (air-gapped, reproducible) | **ai-knot** |
| the largest, most-adopted general memory layer | **Mem0** |
| an LLM-built temporal knowledge graph | **Zep / Graphiti** |
| the LLM to manage its own memory inside the agent loop | **Letta** (ex-MemGPT) |
| an ontology + knowledge-graph memory pipeline | **Cognee** |
| the most native memory for a LangGraph-only stack | **LangMem** |
| SQL-native structured memory with no vector DB | **Memori** |

Several of these also keep the LLM off *recall* (Graphiti, LangMem, Memori). What's rarer: `ai-knot` needs no LLM on the **write** path either — direct fact insertion is the default and `learn()` extraction is optional — so the whole pipeline can run with zero model calls.

The honest wedge: **self-hosted deterministic memory with no LLM required on read or write, a benchmark you can re-run, and real multi-agent governance.**

For the full, checked feature matrix versus each project, use [docs/comparison.md](docs/comparison.md).

## Benchmarks you can rerun

Memory claims in this category swing hard depending on the reader model, judge model, prompts, and scoring rules. `ai-knot` publishes both named-reader QA numbers and deterministic retrieval metrics.

| Benchmark | Metric | ai-knot |
|---|---|---:|
| Golden suite | ranking MRR, deterministic, no LLM | **0.83** vs 0.18 naive |
| LoCoMo | `evidence_recall@5`, deterministic, no LLM | **0.26** vs 0.15 naive |
| LoCoMo | QA accuracy, cat1-4, `gpt-4.1` reader / `gpt-4o` judge | **78.0%** |
| LongMemEval | QA accuracy, Oracle, `gpt-4.1` / `gpt-4o` | **59.6%** |

The first two rows are the anchor: deterministic, no LLM, fixed seeds — re-run and you get the same numbers. The QA rows are LLM-judged (reader + judge named), so they move with the models; the deterministic rows don't.

**LoCoMo by category — nothing cherry-picked.** Reader `gpt-4.1`, judge `gpt-4o`; categories 1–4 scored (category 5 is adversarial and excluded per the dataset authors — the exact step the field's headline benchmark dispute got wrong):

| cat1 single-hop | cat2 multi-hop | cat3 temporal | cat4 open-ended | overall |
|:---:|:---:|:---:|:---:|:---:|
| 60.6% | 67.6% | 63.5% | **89.4%** | **78.0%** |

Strong on open-ended synthesis, honestly mid on single-hop and temporal — and 74–84% on every one of the 10 conversations, not one lucky run.

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

Full methodology, caveats, and per-conversation tables: [docs/benchmarks.md](docs/benchmarks.md).

## Performance

In-process BM25 recall, measured with `pytest-benchmark`:

| Facts in memory | recall p50 | p95 |
|---|---:|---:|
| 100 | ~1 ms | ~3 ms |
| 1,000 | ~8 ms | ~25 ms |
| 10,000 | ~80 ms | ~200 ms |

MCP tool round-trip over stdio is roughly `add` ~15 ms / `recall` ~20 ms p50. For the fuller picture — methodology, per-conversation tables, and the deterministic suite you can re-run — use [docs/benchmarks.md](docs/benchmarks.md). Continuous `pytest-benchmark` history is published to GitHub Pages once Pages is enabled (see [docs/RELEASE.md](docs/RELEASE.md)).

## Documentation

- Start with [docs/usage.md](docs/usage.md), [docs/memory-commands.md](docs/memory-commands.md), and [docs/integrations.md](docs/integrations.md).
- For runnable proofs, use [examples/README.md](examples/README.md).
- For positioning and buyer-facing material, use [docs/positioning.md](docs/positioning.md), [docs/comparison.md](docs/comparison.md), and [docs/faq.md](docs/faq.md).
- For benchmark methodology, use [docs/benchmarks.md](docs/benchmarks.md).
- For long-form material, use [docs/whitepaper.md](docs/whitepaper.md), [docs/developer-article.md](docs/developer-article.md), and [docs/site/index.html](docs/site/index.html).
- For productized assistant instructions, use [skills/README.md](skills/README.md).
- For deployment and repo internals, use [docs/deployment.md](docs/deployment.md), [docs/production-readiness.md](docs/production-readiness.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [docs/README.md](docs/README.md).

## Contributing

PRs are welcome, especially around storage backends, framework adapters, benchmark coverage, and docs. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

## License

MIT. If a stack you use is missing, open an issue with the integration surface you want next.
