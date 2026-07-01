# Submission pack

Updated: **July 1, 2026**

Use this file when listing ai-knot in directories, awesome-lists, MCP indexes,
GitHub metadata, and launch posts that need a shorter blurb than the full copy in
[announce.md](announce.md).

## Repo metadata

### Suggested GitHub repository description

> Deterministic, self-hosted memory for AI agents: store facts instead of transcripts, recall only what matters.

### Suggested GitHub topics

Use up to 20. Recommended set:

- `ai`
- `agents`
- `agent-memory`
- `ai-memory`
- `memory-layer`
- `context-engineering`
- `llm`
- `llmops`
- `rag`
- `mcp`
- `model-context-protocol`
- `claude`
- `openclaw`
- `openai-agents`
- `langchain`
- `langgraph`
- `python`
- `typescript`
- `sqlite`
- `postgresql`

## Short descriptions by length

### 50-60 chars

> Deterministic memory for AI agents

### 80-100 chars

> Self-hosted agent memory that stores facts, not transcripts, and recalls only what matters.

### 120-140 chars

> Deterministic, self-hosted memory for AI agents with SQLite/Postgres/YAML, MCP support, and reproducible benchmarks.

### 160-180 chars

> ai-knot is a self-hosted memory layer for AI agents: it stores facts instead of replaying transcripts, retrieves only relevant context, and ships reproducible benchmarks.

## Listing blurbs

### Awesome-list / directory default

> ai-knot is a deterministic, self-hosted memory layer for AI agents. It stores facts instead of transcripts, retrieves only the few facts the next turn needs, and ships reproducible benchmark numbers.

### MCP directory

> ai-knot ships an MCP server for persistent agent memory. It gives Claude/Desktop-style clients self-hosted recall over SQLite, PostgreSQL, or YAML, with no LLM on the retrieval path.

### OpenClaw ecosystem / app directory

> ai-knot adds persistent, self-hosted memory to OpenClaw through one MCP config block, with deterministic recall over SQLite, PostgreSQL, or YAML.

### Claude / MCP directory

> ai-knot adds persistent, self-hosted memory to Claude Desktop / Claude Code through MCP, with deterministic recall over SQLite, PostgreSQL, or YAML.

### Agent-memory directory

> ai-knot focuses on deterministic recall and auditability: store facts, not logs; recall only relevant context; keep the hot path cheap, reproducible, and self-hosted.

### CrewAI ecosystem / framework note

> ai-knot can plug into CrewAI's native memory surface through `Crew(memory=...)`
> and `Agent(memory=memory.scope(...))`, while keeping long-term storage and
> retrieval deterministic and self-hosted.

### Infra / LLM tooling directory

> Self-hosted memory infrastructure for AI agents with deterministic retrieval, bi-temporal recall, SQLite/Postgres/YAML backends, and framework/MCP integration surfaces.

## Awesome-list PR snippets

### One-line bullet

> - [ai-knot](https://github.com/alsoleg89/ai-knot) - Deterministic, self-hosted memory for AI agents with MCP support and reproducible benchmarks.

### Slightly longer bullet

> - [ai-knot](https://github.com/alsoleg89/ai-knot) - Stores facts instead of replaying transcripts, recalls only relevant context, runs self-hosted over SQLite/Postgres/YAML, and publishes deterministic benchmark numbers you can rerun.

## MCP-specific listing text

### 1 sentence

> ai-knot is an MCP server for persistent, self-hosted agent memory with deterministic recall.

### 2 sentences

> ai-knot adds persistent memory to MCP-compatible clients like Claude Desktop / Claude Code. It stores facts instead of transcripts and recalls only the relevant context from SQLite, PostgreSQL, or YAML.

## Comparison-safe summary

Use this when a listing asks "how is it different?":

> Unlike broader memory platforms, ai-knot optimizes for deterministic recall, self-hosted storage control, and reproducible benchmarks rather than a hosted or LLM-heavy read path.

## CTA variants

### Trial CTA

> Start with the README quickstart or open the repo in Codespaces.

### Benchmark CTA

> Re-run the deterministic benchmark command in `docs/benchmarks.md`.

### Integration CTA

> Try the surface that matches your stack: CrewAI, MCP, OpenClaw, OpenAI Agents SDK, LangChain, TypeScript, or HTTP.

## Best links by context

| Context | Link to use |
|---|---|
| Launch landing page (when GitHub Pages is enabled) | `docs/site/index.html` / the repo Pages URL |
| First-touch developer audience | repo root / README |
| Skeptical technical audience | `docs/benchmarks.md` |
| Product / strategy audience | `docs/whitepaper.md` |
| Builders who want code quickly | `docs/usage.md` |
| Launch threads and comments | `docs/faq.md` |
| People asking "where should I start?" | README quickstart table |

## What not to say

Avoid these patterns in listings:

- "best memory for agents"
- "beats every competitor"
- "fully autonomous agent platform"
- "knowledge graph"
- "hosted memory"

They either over-claim or misclassify the product.
