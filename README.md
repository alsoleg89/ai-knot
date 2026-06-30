# ai-knot

![CI](https://github.com/alsoleg89/ai-knot/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/ai-knot)
![npm](https://img.shields.io/npm/v/ai-knot)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

**Deterministic memory for AI agents — with benchmark numbers you can actually reproduce.**

Most agent frameworks treat memory as a log: store every message, replay it, pay to inject six
months of history into a prompt that needs three sentences of it. ai-knot treats memory as a
*knowledge base* — it distills conversations into structured facts, retrieves only what the next
turn needs, and forgets the rest. No LLM on the retrieval path. Pluggable storage. Self-hosted.
And for teams of agents, a shared memory pool with trust, governance, and bi-temporal supersession.

📚 [Benchmarks](docs/benchmarks.md) · [Usage guide](docs/usage.md) · [Deployment](docs/deployment.md) · [Production readiness](docs/production-readiness.md) · [Architecture](ARCHITECTURE.md) · [Launch post](docs/launch-post.md)

```
1000 messages (~400k tokens)  ──extract+verify──▶  ~12 facts (~300 tokens)  ──BM25+RRF──▶  3–5 facts injected
```

---

## Benchmarks you can reproduce

The agent-memory field reports LoCoMo and LongMemEval as an LLM-judged accuracy %. Those numbers
are also notoriously irreproducible — Zep's **84%** on LoCoMo became **58%** on an independent
re-run; Mem0's cited **91.6%** reproduces at **~58–66%**. An LLM-judged score moves 20+ points with
the reader model, the judge, and the prompts.

So ai-knot leads with a deterministic, one-command retrieval number that **cannot** move,
and reports the LLM-judged QA accuracy alongside it (with the reader model named, as everyone's
should be).

| Benchmark | Metric | naive log | ai-knot |
|---|---|---:|---:|
| **Golden retrieval suite** | ranking MRR (deterministic) | 0.18 | **0.83** |
| **LoCoMo** | `evidence_recall@5` (deterministic) | 0.15 | **0.26** (+71%) |
| **LoCoMo** | QA accuracy (LLM-judged) | — | see [benchmarks.md](docs/benchmarks.md) |
| **LongMemEval** | QA accuracy (LLM-judged) | — | see [benchmarks.md](docs/benchmarks.md) |

The deterministic numbers reproduce bit-for-bit:

```bash
AI_KNOT_EMBED_URL="" python -m tests.eval.benchmark.runner \
  --mock-judge --skip-multi-agent --backends baseline,ai_knot_no_llm
```

Full tables, the field landscape, and the methodology stance: **[docs/benchmarks.md](docs/benchmarks.md)**.

---

## Install

```bash
pip install ai-knot                 # core
pip install "ai-knot[openai]"       # + LLM extraction
pip install "ai-knot[postgres]"     # + PostgreSQL backend
pip install "ai-knot[mcp]"          # + MCP server (Claude Desktop / Code)
npm install ai-knot                 # Node / TypeScript (needs Python 3.11+ in PATH)
```

## Quickstart (30 seconds)

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="my_agent")

# Add facts directly...
kb.add("User is a senior backend developer at Acme Corp", type="semantic", importance=0.95)
kb.add("User prefers Python, dislikes async code", type="procedural", importance=0.85)

# ...or extract them from a conversation
from ai_knot import ConversationTurn
kb.learn([ConversationTurn(role="user", content="I deploy everything in Docker")],
         provider="openai", api_key="sk-...")

# At inference time, get only what matters — recall never calls an LLM
context = kb.recall("how should I write this deployment script?")
# -> "[procedural] User prefers Python, dislikes async code
#     [semantic]   User deploys everything in Docker"
```

Full API: **[docs/usage.md](docs/usage.md)**.

---

## Why ai-knot

- **Deterministic by design.** Retrieval, dedup, conflict resolution, and bi-temporal supersession
  run with no LLM on the read path — reproducible, cheap, auditable. The dense channel is optional
  and degrades gracefully.
- **Signal, not noise.** LLM extraction + ATC verification + power-law forgetting (Wixted & Ebbesen
  1997) keep ~12 facts instead of 1000 messages. Stale facts fade; reinforced facts persist.
- **No vendor lock-in.** SQLite / PostgreSQL / YAML behind one API. The YAML store is human-readable
  and Git-trackable. Six LLM providers for extraction. Self-hosted, MIT.
- **Bi-temporal.** Every fact knows when it was learned *and* when its event happened. Ask
  "what was true as of the incident?" — `recall(now=…)` answers it; superseded facts are excluded,
  not deleted.
- **MCP-native.** Ships an `ai-knot-mcp` server — Claude Desktop and Claude Code call it as tools
  with zero glue code.

## Built for teams of agents

The same store becomes a `SharedMemoryPool` — the part that's genuinely hard to find elsewhere:

- **Fan-in recall** — answers scattered across agents are set-covered, not crowded out.
- **Evidence-before-belief** — a claim with no provenance pointer is withheld, not published.
- **Per-agent visibility** — keep facts out of the global pool or share them selectively.
- **Laundering-resistant trust** — known-bad publishers are discounted even on wide recall; flooding
  can't wash out a quick-invalidation penalty.
- **Deterministic conflict resolution** — slotted supersession + monotonic CAS, with an *optional*
  LLM seam for the semantic tail (off by default).

Enforced by a scored acceptance gate (scenarios S8–S26) that runs on every PR. Details in
[docs/usage.md](docs/usage.md#multi-agent) and [production-readiness.md](docs/production-readiness.md).

---

## Why not just use Mem0 / Zep / LangMem?

| | ai-knot | Mem0 | Zep | LangMem |
|---|---|---|---|---|
| Self-hosted / no cloud required | Yes | Partial | Partial | Yes |
| Pluggable + human-readable store | Yes | No | No | No |
| No LLM on the retrieval path | Yes | No | No | No |
| Reproducible benchmark numbers | Yes | Disputed | Disputed | — |
| Type-aware power-law forgetting | Yes | No | No | No |
| Deterministic conflict resolution | Yes | No | No | No |
| Bi-temporal supersession | Yes (deterministic) | No | Yes (LLM) | No |
| Multi-agent trust + governance | Yes | No | No | No |
| Fan-in recall across agents | Yes | No | No | No |
| MCP server | Yes | No | No | No |
| Free forever | Yes (MIT) | No | No | Yes |

*Fair is fair:* Mem0 is a mature product with a hosted offering and a large community; Zep ships a
genuine temporal knowledge graph; LangMem integrates tightly with LangGraph. ai-knot's bet is a
narrow one — **deterministic, dependency-light, self-hosted memory with a real multi-agent
governance model, and numbers you can reproduce.** Full positioning and trade-offs:
[docs/launch-post.md](docs/launch-post.md).

---

## Performance

In-process BM25 recall, measured with `pytest-benchmark`:

| Facts in memory | recall p50 | p95 |
|----------------|-----|-----|
| 100 | ~1 ms | ~3 ms |
| 1 000 | ~8 ms | ~25 ms |
| 10 000 | ~80 ms | ~200 ms |

MCP tool round-trip (stdio): `add` ~15 ms / `recall` ~20 ms p50. Use `storage="sqlite"` for
lower variance at scale. [Full benchmark history →](https://alsoleg89.github.io/ai-knot/dev/bench/)

---

## Contributing

PRs welcome — especially storage backends, framework adapters, and retrieval strategies.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [DEVELOPMENT.md](DEVELOPMENT.md).

## License

MIT. Found a bug or a missing backend? Open an issue. Built something with it? We'd like to hear.
