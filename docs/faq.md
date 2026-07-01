# FAQ and objections

Updated: **June 30, 2026**

Use this document for README follow-ups, launch replies, investor or partner
questions, and objection handling in public threads.

---

## FAQ

### What does ai-knot actually do?

It turns agent memory into a knowledge base. Instead of replaying full chat logs
into every prompt, it stores facts and recalls only the few that matter for the
next turn.

### Does recall call an LLM?

No. The read path is deterministic. LLMs can be used during `learn()` extraction
or optional semantic helpers, but recall itself does not require an LLM call.

### Why is that important?

It makes recall cheaper, lower-latency, reproducible, auditable, and easier to
regression-test.

### Is this only for Python?

No. Python is the core package, but ai-knot also ships:

- an npm / TypeScript client,
- an MCP server for Claude Desktop / Claude Code,
- a FastAPI HTTP sidecar,
- a CrewAI adapter,
- an OpenAI Agents SDK adapter,
- LangChain / LangGraph adapters.

### Why not just store the transcript?

Because a transcript is not the same thing as useful memory. Most of a long
conversation is irrelevant to the next turn. Storing facts keeps context smaller
and more focused.

### Why not just use a vector database?

Vector search can help, but ai-knot is not just a search wrapper. It combines
structured facts, deterministic recall, conflict handling, forgetting, and
multi-agent governance. A vector DB alone does not give you that product surface.

### What is the best storage backend to start with?

- `YAML` for demos and inspection
- `SQLite` for local or single-server production
- `PostgreSQL` for shared, multi-process deployments

### Does ai-knot support shared memory across agents?

Yes. The `SharedMemoryPool` adds provenance, trust, visibility scopes,
evidence-gated publishing, and fan-in recall across multiple agents.

### Is ai-knot good for regulated or air-gapped environments?

Yes, relative to the rest of the category. It is self-hosted, the recall path can
run without an LLM, and the storage layer is under your control.

### How should I benchmark it?

Use both:

1. the named-reader QA benchmarks in `docs/benchmarks.md`, and
2. the deterministic retrieval command in the same document.

The deterministic suite is the fastest sanity check.

### What is the fastest CLI path to try it?

Start with:

- `ai-knot add assistant "User deploys in Docker"`
- `ai-knot search assistant "what does the user deploy with?"`
- `ai-knot show assistant`
- `ai-knot forget assistant <fact_id>`

If you want ai-knot to extract facts from raw text rather than add them one by
one, use `ai-knot learn assistant "raw note here"` with a configured provider.

### When should I not choose ai-knot?

Do not choose it if:

- you want a hosted memory SaaS right now,
- you already committed deeply to a graph-memory architecture,
- you want a full agent runtime rather than a memory layer,
- your main goal is framework-native convenience inside one ecosystem and you do
  not care about portability.

---

## Objections you should expect

### "Deterministic means less capable than LLM-based memory."

Sometimes, yes. That trade-off is real. ai-knot optimizes for reproducibility,
cost, and auditability on the hot path, and keeps an optional seam for semantic
tail cases instead of making the whole read path probabilistic.

### "Why would I trust your benchmark if the whole field is noisy?"

Because ai-knot does not ask you to trust a single headline. It publishes the
reader and judge for QA metrics **and** ships a deterministic retrieval suite you
can rerun locally.

### "This feels niche compared with Mem0 or Letta."

It is narrower by design. ai-knot is not trying to be a hosted memory platform or
an agent runtime. Its wedge is self-hosted, deterministic memory with real
multi-agent governance.

### "Could I build this myself with a vector DB and some prompts?"

You could build parts of it. The point is the productized combination: structured
facts, forgetting, conflict handling, bi-temporal recall, shared-pool trust, MCP,
TypeScript, docs, and reproducible benchmarks.

### "Why do I care about multi-agent governance?"

Because shared memory becomes unreliable very quickly when multiple agents can
publish into it without provenance, trust, or visibility rules.

### "Why not just use LangMem if I am already on LangGraph?"

If your only goal is the most native LangGraph memory surface, that may be the
easiest choice. If you want portability across stacks, storage control, MCP, or
deterministic recall, ai-knot has a different value proposition.

### "Is YAML storage serious?"

YAML is for inspection and low-friction dev workflows. Serious production paths
are SQLite or PostgreSQL. YAML is a feature, not the entire story.

### "Is this only valuable for chatbots?"

No. The strongest use cases are often coding agents, internal copilots, regulated
assistants, and multi-agent systems that need shared state.

---

## Short answers for public threads

- **What is it?** Self-hosted deterministic memory for AI agents.
- **Why is it different?** No LLM on recall, reproducible benchmarks, multi-agent governance.
- **Why now?** Agent usage is up and memory benchmarks are in a credibility crisis.
- **What should I try first?** README quickstart, then the deterministic benchmark command.
