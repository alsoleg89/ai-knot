# LangGraph case study / proof asset

Updated: **July 2, 2026**

Use this file when you want one concrete `ai-knot` integration story that
starts from the LangChain / LangGraph ecosystem rather than from MCP or a
framework-neutral Python snippet.

Official references:

- LangGraph repo: https://github.com/langchain-ai/langgraph
- LangGraph docs: https://docs.langchain.com/oss/python/langgraph/overview
- LangMem repo: https://github.com/langchain-ai/langmem
- LangMem docs: https://langchain-ai.github.io/langmem/

LangGraph is a real framework distribution surface, even though LangMem remains
the most native memory path inside that ecosystem. For `ai-knot`, the point is
not to beat LangMem at being LangMem; it is to meet the ecosystem with the same
search-shaped tool flow while keeping memory deterministic and portable.

---

## The angle

Do not pitch this as "LangMem but different." Pitch it as:

> **Keep the LangGraph agent/tool flow, add deterministic long-term memory.**

That means:

- your LangGraph agent can still call memory as tools,
- ai-knot now exposes both the explicit `add/search/list/delete` tool loop and
  the familiar compact `manage + search` shape,
- retrieval stays deterministic and self-hosted,
- you do not have to move memory into LangGraph's own store layer to get
  persistent facts.

The honest framing matters here: if someone wants the most native possible
LangGraph memory story and nothing else, LangMem is still the path of least
resistance. `ai-knot` wins when the team cares about portability, storage
control, and deterministic recall more than single-ecosystem purity.

---

## Fastest proof paths

### Zero-network LangGraph tool proof

Runs without LangGraph and without an API key:

```bash
python examples/langgraph_surface_demo.py
```

What it proves:

- the tool-style seam exists now,
- `create_manage_memory_tool(...)` and `create_search_memory_tool(...)` are easy
  to inspect,
- memory add/search/list/delete stays local before any real model call.

### Retriever / chat-memory path

Runs with only `ai-knot` installed:

```bash
python examples/langchain_integration.py
```

What it proves:

- `AiKnotRetriever` works as a Runnable-style retriever,
- `AiKnotChatMemory` works as a conversational-memory drop-in,
- the LangChain / LangGraph surface stays dependency-light.

### Real LangGraph install path

```bash
pip install "ai-knot[langgraph]"
```

Use `ai-knot[langchain]` instead when you only need the retriever / chat-memory
surface and do not want the full LangGraph runtime.

---

## What to emphasize

### Problem

LangGraph users often want durable memory, but the market's default answers are
either "adopt the ecosystem-native store story" or "move memory into another
hosted/vector-heavy layer."

### What ai-knot adds

- LangGraph-shaped memory tools for both the explicit `add/search/list/delete`
  loop and the compact `manage + search` path,
- deterministic recall with no LLM on the read path,
- self-hosted storage (SQLite / PostgreSQL / YAML),
- a framework-agnostic memory core that also works outside LangGraph,
- retriever and chat-memory adapters in the same surface area.

### What not to claim

- Do not say it replaces LangGraph.
- Do not say it is more native than LangMem inside LangGraph.
- Do not hide the trade-off: LangMem is tighter to the ecosystem; ai-knot is
  broader, more portable, and more storage-controlled.

---

## Copy blocks

### GitHub discussion / follow-up comment

> LangGraph path for `ai-knot` is ready.
>
> The repo now has tool-style helpers for both the explicit
> `create_basic_memory_tools(kb)` loop and the familiar compact
> `create_manage_memory_tool(kb)` + `create_search_memory_tool(kb)` surface.
>
> That means you can keep a LangGraph agent/tool workflow and add deterministic,
> self-hosted long-term memory underneath.
>
> Fastest proof:
> - `python examples/langgraph_surface_demo.py`
> - `python examples/langchain_integration.py`

### X / LinkedIn

> LangGraph users: `ai-knot` now has tool-style memory helpers.
>
> Use `create_basic_memory_tools(kb)` when you want the literal
> `add/search/list/delete` loop, or keep
> `create_manage_memory_tool(kb)` + `create_search_memory_tool(kb)` when you
> want the compact LangMem-shaped surface. Either way, recall stays
> deterministic and storage stays self-hosted.
>
> Shortest proof: `python examples/langgraph_surface_demo.py`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you are already on LangGraph, the honest answer is that LangMem is still
> the most native option. The `ai-knot` angle is different: tool-shaped memory
> helpers plus deterministic recall, storage control, and portability outside
> LangGraph. Start with `python examples/langgraph_surface_demo.py`.

---

## Recommended CTA

Lead with one of these, in order:

1. `python examples/langgraph_surface_demo.py`
2. `python examples/langchain_integration.py`
3. [integrations.md](integrations.md)

Do not send LangGraph users to the whitepaper first. Send them to the shortest
proof.
