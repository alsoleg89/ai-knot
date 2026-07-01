# README and integration pattern audit

Updated: **July 1, 2026**

This document looks narrowly at one adoption question:

1. how winning memory projects route a new visitor into the product from the README;
2. which integration surfaces they surface first;
3. which of those patterns `ai-knot` should copy, counter, or ignore.

---

## What the best READMEs do above the fold

The strongest projects in this category do not begin with architecture depth.
They begin with routing.

Common pattern:

1. **Name the category and wedge fast.**
   `Mem0` says memory layer. `Graphiti` says temporal context graph. `Letta`
   says stateful agents. `LangMem` says long-term memory for agents.
2. **Split the first-run path by surface.**
   Library vs CLI vs cloud, MCP vs REST, or CLI vs API.
3. **Show the host-framework object names.**
   `Crew(memory=...)`, `AssistantAgent(memory=[...])`, `create_manage_memory_tool(...)`,
   not just abstract claims about compatibility.
4. **Give one proof hook early.**
   Benchmarks, graph visuals, desktop shell, or MCP/app screenshots.
5. **Minimize "what do I install?" ambiguity.**
   The install line follows the surface.

That is the core reason these READMEs convert curiosity into trials.

---

## Mem0

### README shape

Mem0's README starts with a benchmark/research hook, then routes visitors into:

- library,
- self-hosted server,
- cloud platform,
- CLI,
- agent skills,
- integrations and demos.

It does not force one default path for everyone.

### Integration shape

Mem0 exposes many entry points directly in the README:

- Python SDK,
- npm SDK,
- self-hosted stack,
- cloud signup,
- CLI,
- framework integrations,
- coding-assistant skills.

### What works

- The market instantly sees breadth.
- The project looks trialable from multiple ecosystems.
- Skills turn coding assistants into a distribution channel, not just docs readers.

### What `ai-knot` should copy

- surface-specific starts,
- exact install lines by surface,
- assistant-facing packaging,
- clear "core loop" example plus ecosystem examples.

---

## Graphiti

### README shape

Graphiti starts with a strong concept label, then quickly moves into:

- a visible MCP callout,
- comparison tables,
- installation by backend/provider,
- quickstart,
- MCP server,
- REST service.

### Integration shape

Graphiti's README makes the available surfaces concrete:

- Python library,
- MCP server,
- REST API,
- graph-database backend choices,
- managed counterpart narrative through Zep.

### What works

- The project is easy to position against alternatives.
- MCP is visible before the reader gets lost in implementation detail.
- Comparison tables reduce mental load.

### What `ai-knot` should copy

- keep MCP/app paths near the top,
- compare category choices clearly,
- separate "simple deterministic memory" from graph-heavy infrastructure.

---

## Letta

### README shape

Letta splits onboarding into two distinct starts:

- get started in the CLI,
- get started with the API.

That is simple, but powerful. It matches the two main jobs people come for.

### Integration shape

Letta makes the product shell itself part of onboarding:

- terminal/desktop agent experience,
- hosted API,
- Python SDK,
- TypeScript SDK,
- channel/app integrations in the broader product story.

### What works

- The README answers "do I use this myself, or embed it in my app?"
- The product feels demoable, not just programmable.
- A platform story can appeal beyond infra specialists.

### What `ai-knot` should copy

- keep "human-operated assistant" and "embedded app memory" as separate routes,
- keep MCP/app setup commands visible,
- keep the repo usable even when the reader is not starting from Python.

---

## LangMem

### README shape

LangMem goes straight from one-liner to a working agent example. The first
serious code block is not a low-level store primitive; it is an agent with
memory tools wired in.

### Integration shape

LangMem is strongest where it is most native:

- LangGraph store integration,
- memory tools in the hot path,
- background memory manager for asynchronous consolidation.

### What works

- Framework users see exactly where the library fits.
- The project avoids generic "works with anything" fuzziness.
- The framework object names are front and center.

### What `ai-knot` should copy

- show native adapter names in README,
- keep framework-specific examples short and runnable,
- route by "what stack are you already on?" before deep API detail.

---

## What changed in `ai-knot` because of this review

This pass applied the market patterns directly:

1. The root [README.md](../README.md) now includes a `What it looks like in your stack`
   section with native snippets for CrewAI, AutoGen, the OpenAI Agents SDK, and
   MCP clients.
2. The integrations index now includes an assistant-facing skill surface.
3. The repo now ships a reference skill in [../skills/ai-knot/SKILL.md](../skills/ai-knot/SKILL.md)
   plus a [skills/README.md](../skills/README.md) entry point.
4. The HTTP sidecar now includes a lightweight browser inspector, closing the
   earlier "product shell" gap for demos and debugging.

---

## Remaining adoption gap

The strongest remaining gap is not an onboarding surface anymore; it is public
distribution state. The branch now has README routing, assistant-facing
packaging, and a lightweight browser inspector, but the public `main` branch and
npm latest tag still need to catch up before the launch can be considered fully
executed.

---

## Sources

- Mem0 README: https://github.com/mem0ai/mem0
- Mem0 skills index: https://github.com/mem0ai/mem0/tree/main/skills
- Graphiti README: https://github.com/getzep/graphiti
- Letta README: https://github.com/letta-ai/letta
- LangMem README: https://github.com/langchain-ai/langmem
