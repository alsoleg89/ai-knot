# Comparison guide

Updated: **July 4, 2026**

This is the buyer-facing comparison page: short, practical, and explicit about
where `ai-knot` fits against the product shapes that actually matter in 2026.

---

## One-sentence difference

- **ai-knot**: deterministic, self-hosted fact memory for agents with no LLM on recall
- **OpenViking**: context database with layered retrieval and a filesystem-style context model
- **MemPalace**: local-first verbatim conversation memory with semantic search
- **Engram**: persistent memory server for coding agents, packaged as a single binary
- **codebase-memory-mcp**: code intelligence and codebase knowledge graph, not general long-term user memory

## Quick matrix

| Need | Best default choice |
|---|---|
| Self-hosted memory with no LLM on recall | **ai-knot** |
| Layered context database / context filesystem | **OpenViking** |
| Verbatim transcript memory | **MemPalace** |
| Coding-agent memory server / one-binary install | **Engram** |
| Codebase graph memory for code navigation | **codebase-memory-mcp** |

## Feature comparison

| Capability | ai-knot | OpenViking | MemPalace | Engram | codebase-memory-mcp |
|---|:---:|:---:|:---:|:---:|:---:|
| Self-hosted core path | ✅ | ✅ | ✅ | ✅ | ✅ |
| No LLM on retrieval path | ✅ | ❌ | ✅ | ✅ | ✅ |
| SQLite / Postgres / YAML control | ✅ | ◑ | ❌ | ❌ | ❌ |
| Human-readable store option | ✅ | ◑ | ❌ | ❌ | ❌ |
| MCP surface | ✅ | ✅ | ✅ | ✅ | ✅ |
| HTTP surface | ✅ | ✅ | ❌ | ✅ | ❌ |
| TypeScript client | ✅ | ◑ | ❌ | ❌ | ❌ |
| Browser / visual inspection path | ✅ | ✅ | ❌ | ✅ | ✅ |
| Fact lifecycle and correction (`lineage`, structured update/delete) | ✅ | ◑ | ❌ | ❌ | ❌ |
| Multi-agent governance | ✅ | ◑ | ❌ | ❌ | ❌ |
| Verbatim transcript focus | ❌ | ❌ | ✅ | ◑ | ❌ |
| Codebase-graph-first product shape | ❌ | ❌ | ❌ | ❌ | ✅ |

`◑` means "present, but not the center of the product story."

## When ai-knot wins clearly

Choose `ai-knot` if you care about:

- deterministic recall you can regression-test
- self-hosted infrastructure with storage you control directly
- memory that behaves like a product primitive, not a black box service
- explicit correction and audit loops instead of "just overwrite the memory"
- shared memory across several agents with trust, provenance, and visibility rules

## When not to choose ai-knot

Do not choose `ai-knot` first if:

- you want a larger context-database system with heavier hierarchical retrieval and context management
- you want verbatim transcripts to remain the primary stored artifact
- you want a coding-agent memory server as a single standalone binary first and foremost
- your real problem is codebase graph exploration rather than user / agent long-term memory

## Practical buyer guide

### "I just want my agent to remember users across sessions."

Pick `ai-knot` if you want deterministic, inspectable fact memory.
Pick `MemPalace` if verbatim conversation storage is the actual requirement.

### "I need memory for a coding agent."

Pick `ai-knot` if you want broader storage control, framework surfaces, and
multi-agent governance.
Pick `Engram` if your priority is a one-binary coding-agent memory server.

### "I need a broader context system, not just fact memory."

`OpenViking` is the more natural comparison when the real requirement is a
context database with layered retrieval and a richer context operating model.
`ai-knot` is stronger if you value determinism and lower operational surface
area more than a bigger context-management framework.

### "I need code intelligence across a repo, not user memory."

`codebase-memory-mcp` is the more natural fit there. It is a code graph and
structural code-intelligence product. `ai-knot` can store agent memory about a
project, but it is not trying to replace a codebase graph.

### "I already use LangGraph."

`ai-knot` still fits well if you want portable memory across stacks or tighter
storage control. The repo has LangGraph-shaped helpers for both the explicit
`create_basic_memory_tools(...)` loop, the optional
`create_get_memory_tool(...)` inspect helper, and the compact
`create_manage_memory_tool(...)` / `create_search_memory_tool(...)` surface.

### "I already use LlamaIndex."

LlamaIndex is not the competitor here; it is the host ecosystem. If you want a
native-feeling seam there, `ai-knot` fits the same `memory=...` shape through
`AiKnotLlamaIndexMemory`. Choose that path when you want deterministic,
self-hosted long-term memory without giving up the LlamaIndex runtime you
already use.

## The honest wedge

`ai-knot` is not trying to be everything in the category. Its wedge is:

> **self-hosted deterministic fact memory with reproducible benchmarks and multi-agent governance**

That is narrower than the broadest context systems, but it is also easier to
trust, easier to test, and easier to explain.
