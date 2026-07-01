# Comparison guide

Updated: **June 30, 2026**

This is the buyer-facing comparison page: short, practical, and explicit about
where ai-knot fits versus adjacent open-source options.

---

## One-sentence difference

- **ai-knot**: deterministic, self-hosted agent memory that stores facts instead of transcripts
- **Mem0**: broad memory platform with strong ecosystem reach and hosted posture
- **Graphiti**: graph-shaped memory for relational and temporal reasoning
- **Letta**: stateful-agent platform, not just a memory primitive
- **LangMem**: native memory for the LangGraph ecosystem

## Quick matrix

| Need | Best default choice |
|---|---|
| Self-hosted memory with no LLM on recall | **ai-knot** |
| Hosted / SaaS-friendly memory layer | **Mem0** |
| Knowledge-graph-first memory | **Graphiti** |
| Full stateful-agent platform | **Letta** |
| Tightest LangGraph-native path | **LangMem** |

## Feature comparison

| Capability | ai-knot | Mem0 | Graphiti | Letta | LangMem |
|---|:---:|:---:|:---:|:---:|:---:|
| Self-hosted core path | ✅ | ◑ | ✅ | ✅ | ✅ |
| No LLM on retrieval path | ✅ | ❌ | ❌ | ❌ | ❌ |
| SQLite / Postgres / YAML control | ✅ | ❌ | ❌ | ❌ | ❌ |
| Human-readable store option | ✅ | ❌ | ❌ | ❌ | ❌ |
| MCP surface | ✅ | ❌ | ✅ | ❌ | ❌ |
| TypeScript client | ✅ | ✅ | ✅ | ✅ | ❌ |
| Bi-temporal recall | ✅ | ❌ | ✅ | ◑ | ❌ |
| Multi-agent governance | ✅ | ❌ | ❌ | ◑ | ❌ |
| Framework-agnostic | ✅ | ✅ | ✅ | ◑ | ❌ |
| Hosted path | ❌ | ✅ | ✅ | ✅ | ◑ |

## When ai-knot wins clearly

Choose ai-knot if you care about:

- deterministic recall you can regression-test
- self-hosted infrastructure with storage you control
- air-gapped or regulated deployments
- shared memory across agents with trust, provenance, and visibility rules
- benchmark claims you can re-run, not just quote

## When not to choose ai-knot

Do not choose ai-knot first if:

- you want a managed hosted product
- you need graph-shaped reasoning as the center of the design
- you want an end-to-end agent runtime rather than a memory layer
- you are already committed to LangGraph and prefer the most native option over portability

## Practical buyer guide

### "I just want my agent to remember users across sessions."

Pick ai-knot if you want simple self-hosted control and deterministic recall.
Pick Mem0 if you prefer a broader commercial/platform path.

### "I need memory for several cooperating agents."

ai-knot is the strongest fit here because the shared pool includes governance
mechanics, not only retrieval.

### "I need temporal + relational memory."

Graphiti is the most natural comparison if graph reasoning is the actual product
requirement. ai-knot is stronger if you value determinism and lower operational
complexity more than graph expressiveness.

### "I already use LangGraph."

LangMem has the most native posture. ai-knot is still the better choice if you
want portability across stacks or tighter storage control.

## The honest wedge

ai-knot is not trying to be everything in the category. Its wedge is:

> **self-hosted deterministic memory with reproducible benchmarks and multi-agent governance**

That is narrower than the broadest platforms, but it is also easier to trust and
easier to explain.
