# Integrations

The fastest way to pick the right `ai-knot` entry point for your stack.

If you're evaluating the project, start here, then jump into the full API
reference in [usage.md](usage.md). For the same `add -> search -> list -> delete`
loop mapped across every surface, use [memory-commands.md](memory-commands.md).

Across surfaces, the recognizable memory loop stays the same: store with
`add`/`learn`, retrieve with `search`/`recall`, inspect with `list`, and remove
with `delete`/`forget`.

## First-run memory loop

Keep one first-run loop in mind across every surface:

`add -> search -> list -> delete`

For agent-memory wording, use `learn`, `recall`, and `forget`. When
onboarding Claude, Claude Code, OpenClaw, or another MCP client, keep
the operator verbs separate: `setup` and `doctor` connect the client, while the
memory loop inside the client stays `add/search/list/delete`.

For the user-facing inspect path, CLI / MCP / npm / HTTP list surfaces now
default to the **current active memory**. Reach for
`--include-inactive` / `include_inactive=true` / `includeInactive: true` when
you want superseded history for audit/debug work.

| Surface | Add | Search | List | Delete |
|---|---|---|---|---|
| Core Python | `kb.add(...)` | `kb.search(...)` / `kb.recall(...)` | `kb.list()` / `kb.list_facts()` | `kb.delete(id)` / `kb.forget(id)` |
| TypeScript / npm | `await kb.add(...)` | `await kb.search(...)` / `await kb.recall(...)` | `await kb.list()` / `await kb.listFacts()` | `await kb.delete(id)` / `await kb.forget(id)` |
| CLI | `ai-knot add ...` | `ai-knot search ...` / `ai-knot recall ...` | `ai-knot list ...` / `ai-knot show ...` | `ai-knot delete ...` / `ai-knot forget ...` |
| MCP | `add` | `search` / `recall` | `list` / `list_facts` | `delete` / `forget` |
| HTTP sidecar | `POST /v1/facts` | `POST /v1/search` | `GET /v1/facts` | `DELETE /v1/facts/{fact_id}` |

If you already have a `fact_id`, the main product surfaces also keep one
precise inspect verb available before you delete or supersede anything:
`kb.get(fact_id)`, `await kb.get(factId)`, `ai-knot get ...`, MCP `get`, and
`GET /v1/facts/{fact_id}` on the HTTP sidecar.
When the memory has versions and you want the by-id audit trail instead of only
one record, use `kb.lineage(...)`, `await kb.lineage(...)`,
`ai-knot lineage ...`, MCP `memory_lineage`, the HTTP sidecar lineage endpoint,
or `memory.lineage(...)` on the OpenClaw Python adapter.

## Structured correction surfaces

For more than a by-id delete, the structured correction
seam stays consistent across transports:

- Core Python: `kb.add_resolved([Fact(..., op=MemoryOp.UPDATE)])`
- TypeScript / npm: `await kb.addResolved([{ ..., op: "update" }])`
- CLI: `ai-knot add-resolved ... --op update|delete|noop`
- MCP: `add_resolved` with `facts[].op`
- HTTP sidecar: `POST /v1/facts/resolved` with `facts[].op`

Use this path when a fact changes over time and you want lineage
(`valid_until`, `lineage(...)`) instead of mutating or wiping memory blindly.
Keep the default `add -> search -> list -> delete` loop for first-run
evaluation; reach for structured correction when the memory lifecycle matters.

| Surface | Best for | Install | Start here |
|---|---|---|---|
| Core Python API | chatbots, coding agents, custom app logic | `pip install ai-knot` | [`examples/quickstart.py`](../examples/quickstart.py) |
| Generic function-calling Python agent | runtimes that register ordinary Python callables as tools | `pip install ai-knot` | [`examples/function_calling_surface_demo.py`](../examples/function_calling_surface_demo.py) |
| CrewAI | native `Crew(memory=...)` / `Agent(memory=...)` wiring | `pip install "ai-knot[crewai]"` | [`examples/crewai_integration.py`](../examples/crewai_integration.py) |
| AutoGen | `AssistantAgent(memory=[...])` with persistent facts | `pip install "ai-knot[autogen]"` | [`examples/autogen_integration.py`](../examples/autogen_integration.py) |
| OpenAI Agents SDK | `RunConfig` hook into existing SDK runs | `pip install "ai-knot[agents]"` | [`examples/openai_agents_integration.py`](../examples/openai_agents_integration.py) |
| PydanticAI | per-run `instructions=` memory injection on an existing `Agent` | `pip install "ai-knot[pydanticai]"` | [`examples/pydanticai_integration.py`](../examples/pydanticai_integration.py) |
| LlamaIndex | native `memory=...` seam for chat engines and agents | `pip install "ai-knot[llamaindex]"` | [`examples/llamaindex_surface_demo.py`](../examples/llamaindex_surface_demo.py) |
| OpenClaw | MCP-backed desktop/app memory or Python-side provider compatibility | `pip install "ai-knot[mcp]"` | [`examples/openclaw_integration.py`](../examples/openclaw_integration.py) |
| LangChain / LangGraph | tool-style memory helpers, retriever, or chat-memory drop-in | `pip install "ai-knot[langgraph]"` | [`examples/langgraph_surface_demo.py`](../examples/langgraph_surface_demo.py) |
| Vercel AI SDK | prepend recalled facts to `generateText()` / `streamText()` inputs | `npm install ai-knot ai @ai-sdk/openai` | [`npm/examples/vercel-ai-sdk.ts`](../npm/examples/vercel-ai-sdk.ts) |
| MCP server | Claude Desktop / Claude Code / any MCP client, including HTTP-capable MCP hosts | `pip install "ai-knot[mcp]"` | [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server) |
| Skills / coding assistants | teach Codex / Claude Code / OpenClaw-style tools ai-knot's surfaces up front | `npx skills add https://github.com/alsoleg89/ai-knot --skill ai-knot` | [../skills/README.md](../skills/README.md) |
| TypeScript / npm | Node apps with the Python sidecar/client path or the HTTP sidecar client | `npm install ai-knot` | [../npm/README.md](../npm/README.md) / `cd npm && npm run example:basic-memory-loop` |
| HTTP sidecar | polyglot services, remote agent runtimes, and browser inspection | `pip install "ai-knot[server]"` | [`examples/http_sidecar_surface_demo.py`](../examples/http_sidecar_surface_demo.py) |

---

## Recommended routes

### CrewAI

Use `AiKnotCrewAIMemory` when you want ai-knot to look like a native CrewAI
memory object and plug directly into `Crew(memory=...)` or an agent-scoped
`memory.scope(...)` view.

```bash
pip install "ai-knot[crewai]"
```

If you want ai-knot itself to do LLM-backed `extract_memories()` from raw CrewAI
task output, combine it with a provider extra such as
`pip install "ai-knot[crewai,openai]"`.

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.crewai import AiKnotCrewAIMemory

kb = KnowledgeBase("assistant", provider="openai")
kb.add("User prefers Python")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotCrewAIMemory(kb, top_k=5)
crew = Crew(agents=[researcher, writer], tasks=[task], memory=memory)
```

What it does:

- lets CrewAI call ai-knot through the native `remember` / `recall` surface,
- supports scoped views via `memory.scope("/agent/researcher")`,
- preserves CrewAI's tool ergonomics while keeping storage and retrieval in ai-knot,
- uses ai-knot's own extraction path for `extract_memories()` when the `KnowledgeBase`
  has a default provider configured.

Try one of these next:

- zero-network adapter demo: [`examples/crewai_surface_demo.py`](../examples/crewai_surface_demo.py)
- full Crew wiring example: [`examples/crewai_integration.py`](../examples/crewai_integration.py)
- distribution-ready proof asset: [crewai-case-study.md](crewai-case-study.md)
- full API notes: [usage.md#crewai](usage.md#crewai)

### AutoGen

Use `AiKnotAutoGenMemory` when you already have an AutoGen `AssistantAgent` and
want long-term memory without replacing AutoGen's own short-term context flow.

Note: as of **July 1, 2026**, the official AutoGen README marks the framework
as **maintenance mode** and points new users toward Microsoft Agent Framework.
That makes this a strong surface for **existing AutoGen users**, not the main
greenfield launch wedge.

```bash
pip install "ai-knot[autogen]"
```

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.autogen import AiKnotAutoGenMemory

kb = KnowledgeBase("assistant")
kb.add("User prefers Python")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotAutoGenMemory(kb, top_k=5)
agent = AssistantAgent(name="assistant", model_client=model_client, memory=[memory])
```

What it does:

- extracts the latest user turn from AutoGen's model context,
- recalls only the most relevant facts from ai-knot,
- injects them as a `SystemMessage`,
- keeps the adapter dependency-light and self-hosted.

Try next:

- zero-network surface proof: [`examples/autogen_surface_demo.py`](../examples/autogen_surface_demo.py)
- real integration example: [`examples/autogen_integration.py`](../examples/autogen_integration.py)
- installed-base proof asset: [autogen-case-study.md](autogen-case-study.md)
- full API notes: [usage.md#autogen](usage.md#autogen)

### OpenAI Agents SDK

Use `AiKnotAgentsMemory` when you're already on the OpenAI Agents SDK and want
the long-term memory hook inside `RunConfig`.

```bash
pip install "ai-knot[agents]"
```

```python
memory = AiKnotAgentsMemory(kb, top_k=5)
run_config = memory.build_run_config()
```

What it does:

- recalls only the facts relevant to the current user prompt,
- appends them through the SDK's native `RunConfig` / `call_model_input_filter` seam,
- keeps sessions, tracing, tools, and handoffs inside the SDK's own runtime,
- stays dependency-light: importing the adapter does not require `openai-agents`.

Try next:

- zero-network surface proof: [`examples/openai_agents_surface_demo.py`](../examples/openai_agents_surface_demo.py)
- real integration example: [`examples/openai_agents_integration.py`](../examples/openai_agents_integration.py)
- distribution-ready proof asset: [openai-agents-case-study.md](openai-agents-case-study.md)
- full API notes: [usage.md#openai-agents-sdk](usage.md#openai-agents-sdk)

### PydanticAI

Use `AiKnotPydanticAIMemory` when you already have a PydanticAI `Agent` and
want ai-knot to append query-relevant long-term facts through the framework's
runtime `instructions=` hook on each run.

```bash
pip install "ai-knot[pydanticai]"
```

```python
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory

memory = AiKnotPydanticAIMemory(kb, top_k=5)
result = memory.run_sync(
    agent,
    "Write a deployment checklist.",
    instructions="You are a concise staff engineer.",
)
```

What it does:

- recalls only the facts relevant to the current user prompt,
- appends them under `## Agent Memory` to the same runtime `instructions` surface PydanticAI already uses,
- stays dependency-light: importing the adapter does not require `pydantic-ai`,
- works with sync, async, and streaming run methods that accept `instructions=...`.

Try next:

- zero-network surface proof: [`examples/pydanticai_surface_demo.py`](../examples/pydanticai_surface_demo.py)
- real integration example: [`examples/pydanticai_integration.py`](../examples/pydanticai_integration.py)
- distribution-ready proof asset: [pydanticai-case-study.md](pydanticai-case-study.md)
- full API notes: [usage.md#pydanticai](usage.md#pydanticai)

### Vercel AI SDK

Use `AiKnotAISDKMemory` when you already have a Vercel AI SDK app and want
ai-knot to fill the `system` or `messages` surface with recalled long-term
facts instead of replaying whole transcripts.

```bash
npm install ai-knot ai @ai-sdk/openai
```

```typescript
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { AiKnotAISDKMemory, KnowledgeBase } from "ai-knot";

const kb = new KnowledgeBase({ agentId: "assistant", storage: "sqlite" });
const memory = new AiKnotAISDKMemory(kb, { topK: 4 });
const system = await memory.buildSystem("Write a deploy checklist.", {
  baseSystem: "You are a concise staff engineer.",
});

const { text } = await generateText({
  model: openai("gpt-5"),
  system,
  prompt: "Write a deploy checklist.",
});
```

What it does:

- recalls only the facts relevant to the current user input,
- composes them into the exact `system` string shape AI SDK apps already use,
- keeps model choice, streaming, and route/UI wiring inside your own AI SDK code,
- also supports `buildMessages()` if your app already operates on message arrays.

Try next:

- zero-network repo proof: `cd npm && npm run example:vercel-ai-sdk-surface`
- repo-native surface file: [`../npm/examples/vercel-ai-sdk-surface.ts`](../npm/examples/vercel-ai-sdk-surface.ts)
- real wiring example: `cd npm && OPENAI_API_KEY=... npm run example:vercel-ai-sdk`
- bridge triage if the Python subprocess path is unclear: `cd npm && npm run doctor`
- sidecar-based TypeScript path when you already run `ai-knot serve`: [`../npm/examples/http-sidecar.ts`](../npm/examples/http-sidecar.ts)
- repo-native example file: [`../npm/examples/vercel-ai-sdk.ts`](../npm/examples/vercel-ai-sdk.ts)
- npm package docs: [../npm/README.md](../npm/README.md)
- distribution-ready proof asset: [vercel-ai-sdk-case-study.md](vercel-ai-sdk-case-study.md)

### LlamaIndex

Use `AiKnotLlamaIndexMemory` when you're already on LlamaIndex and want the
most familiar seam possible: a `memory=...` object that keeps short-term chat
history in the framework while ai-knot injects only the relevant long-term
facts on `get(...)`.

```bash
pip install "ai-knot[llamaindex]"
```

For a real model-backed run, add the LLM package too:

```bash
pip install "ai-knot[llamaindex]" "llama-index-llms-openai"
```

```python
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory

memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=5)

chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
# or: await FunctionAgent(...).run("Write a deployment checklist.", memory=memory)
```

What it does:

- mirrors the same `memory=...` shape LlamaIndex already documents for chat engines and agents,
- keeps chat history in a primary short-term memory while ai-knot recalls only the relevant facts for the next turn,
- works without a hard LlamaIndex dependency at import time, so the adapter is still easy to inspect and test,
- optionally supports `extract_on_write=True` when you want write-time fact extraction through `kb.learn(...)` instead of raw-message storage.

Try next:

- zero-network surface proof: [`examples/llamaindex_surface_demo.py`](../examples/llamaindex_surface_demo.py)
- real integration example: [`examples/llamaindex_integration.py`](../examples/llamaindex_integration.py)
- distribution-ready proof asset: [llamaindex-case-study.md](llamaindex-case-study.md)
- full API notes: [usage.md#llamaindex](usage.md#llamaindex)

### OpenClaw

Use the MCP config path if you're integrating with the OpenClaw app. Use the
Python adapter only if you need OpenClaw-style provider compatibility inside
your own runtime.

```bash
ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-default-config
# or, for a non-default config path:
ai-knot setup openclaw --agent-id my_agent --storage sqlite --write-config ~/.openclaw/openclaw.json
```

Try one of these next:

- zero-network proof of both flows: [`examples/openclaw_integration.py`](../examples/openclaw_integration.py)
- app/MCP distribution angle: [openclaw-case-study.md](openclaw-case-study.md)
- full API notes: [usage.md#openclaw](usage.md#openclaw)

After a successful config write, restart the client and run `ai-knot doctor --json`.
The next-turn memory verbs inside OpenClaw stay the same `add/search/list/delete`
loop shown in the README.

If you need the Python provider-compat path instead of the MCP app path, the
adapter keeps the same semantics with one small naming difference:

```python
from ai_knot import Fact, KnowledgeBase
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter

kb = KnowledgeBase("my_agent")
memory = OpenClawMemoryAdapter(kb)
created = memory.add([{"role": "user", "content": "Deploy on Fridays"}])
memory.search("deployment schedule")
memory.get_all()                  # active memories by default
structured = kb.add_resolved([
    Fact(content="User works at Acme", entity="user", attribute="employer", value_text="Acme")
])[0]
current = memory.update(structured.id, "User now works at Globex")
memory.lineage(current["id"])          # newest -> oldest supersession chain
memory.get_all(include_inactive=True)   # optional audit trail
memory.forget(created["results"][0]["id"])   # alias: memory.delete(...)
```

`OpenClawMemoryAdapter` now also exposes `recall()`, `list()`, `lineage()`,
and `forget()` aliases, so the same cross-surface `add/search/list/delete`
loop still reads naturally if you prefer ai-knot's own verbs. Structured
`update()` calls now preserve supersession lineage when the target fact already
has slot/entity metadata, `memory.lineage(current_id)` gives the direct by-id
audit trail, and `get_all()` / `list()` show current active memories by default
with `include_inactive=True` available for history views.

### Claude Desktop / Claude Code

Use the MCP path when you want Claude to call ai-knot as a tool with no custom
Python glue in your project.

```bash
ai-knot setup claude --agent-id my_agent --storage sqlite --write-default-config
# or, for a non-default config path:
ai-knot setup claude --agent-id my_agent --storage sqlite --write-config ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

Try one of these next:

- zero-network setup proof: [`examples/claude_mcp_setup.py`](../examples/claude_mcp_setup.py)
- Claude/MCP distribution angle: [claude-mcp-case-study.md](claude-mcp-case-study.md)
- deployment notes: [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server)

After a successful config write, restart Claude and run `ai-knot doctor --json`.
Inside the client, the memory loop is still `add/search/list/delete`.

### HTTP-capable MCP hosts

Use `serve-mcp` when your MCP host supports remote Streamable HTTP instead of a
local stdio subprocess.

```bash
ai-knot serve-mcp assistant --port 8765
```

Try next:

- deployment notes: [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server)
- release/discovery path: [../server.json](../server.json)

### Skills / coding assistants

If your coding tool supports the skills standard, install the repo-native
`ai-knot` skill when you want the assistant to have ai-knot-specific setup,
adapter, and troubleshooting patterns in context before it edits code.

```bash
npx skills add https://github.com/alsoleg89/ai-knot --skill ai-knot
```

This surface is most useful when:

- you want an assistant to route between core Python, MCP, CrewAI, LlamaIndex, AutoGen, the OpenAI Agents SDK, and PydanticAI without guessing,
- you want the assistant to know the exact ai-knot object names (`KnowledgeBase`, `AiKnotCrewAIMemory`, `AiKnotLlamaIndexMemory`, `create_basic_memory_functions`, `create_basic_memory_tools`, `create_get_memory_tool`, `create_manage_memory_tool`, `create_search_memory_tool`, `AiKnotAutoGenMemory`, `AiKnotAgentsMemory`, `AiKnotPydanticAIMemory`),
- you want first-run troubleshooting (`ai-knot doctor --json`) loaded as part of the integration path.

See also: [../skills/README.md](../skills/README.md)

### Generic function-calling Python runtimes

Use `create_basic_memory_functions(...)` when your agent runtime accepts
ordinary Python callables directly and you do **not** want LangChain-style tool
objects in the middle.

```bash
pip install ai-knot
```

```python
from ai_knot.integrations import create_basic_memory_functions

functions = create_basic_memory_functions(kb, top_k=5, include_get=True)
# Register these callables with your runtime's tool surface.
```

What it does:

- exposes the literal `add/search/list/delete` loop as plain Python callables,
- can also add by-id inspection through `include_get=True`,
- keeps the integration dependency-light because the runtime only needs normal
  functions, not LangChain objects,
- gives custom supervisors and function-calling agents the same correction loop
  the README and CLI already teach.

Try next:

- zero-network callable proof: [`examples/function_calling_surface_demo.py`](../examples/function_calling_surface_demo.py)
- if your runtime specifically expects LangChain/LangGraph tool objects, jump to
  the next section instead of wrapping these yourself

### LangChain / LangGraph

Use `create_basic_memory_tools(...)` when you want the clearest
`add/search/list/delete` tool loop. Use `create_manage_memory_tool(...)` +
`create_search_memory_tool(...)` when you want the closest LangMem-style
agent/tool seam. Use `AiKnotRetriever` for RAG or graph nodes, and
`AiKnotChatMemory` when you want a conversational-memory shape.
If you want the same loop as plain Python callables instead of tool objects, use
`create_basic_memory_functions(...)`.
If your agent already has a `fact_id` and needs targeted inspection, add
`create_get_memory_tool(...)` or call
`create_basic_memory_tools(..., include_get=True)`.

```bash
pip install "ai-knot[langgraph]"
```

If you only need the retriever / chat-memory surface and do not want the full
LangGraph runtime, use:

```bash
pip install "ai-knot[langchain]"
```

```python
from langgraph.prebuilt import create_react_agent
from ai_knot.integrations.langchain import create_basic_memory_tools

tools = create_basic_memory_tools(kb, top_k=5)
agent = create_react_agent(model, tools=tools)
```

What it does:

- can expose the literal `add/search/list/delete` loop to the agent runtime,
- can also expose by-id inspection through `create_get_memory_tool(...)` or
  `include_get=True` once your correction loop already has a `fact_id`,
- still keeps the familiar LangMem-style `manage + search` path when you want it,
- keeps persistent storage and deterministic recall in ai-knot instead of a
  LangGraph-only memory backend,
- still exposes `AiKnotRetriever` and `AiKnotChatMemory` when tools are not the
  right seam,
- keeps the adapters dependency-light and import-safe even without LangChain installed.

Try next:

- zero-network LangGraph proof: [`examples/langgraph_surface_demo.py`](../examples/langgraph_surface_demo.py)
- retriever / chat-memory example: [`examples/langchain_integration.py`](../examples/langchain_integration.py)
- distribution-ready proof asset: [langgraph-case-study.md](langgraph-case-study.md)
- full API notes: [usage.md#langchain--langgraph](usage.md#langchain--langgraph)

### HTTP sidecar and browser inspector

Use the HTTP sidecar when you need a polyglot surface or a quick browser view
of what ai-knot has stored.

```bash
pip install "ai-knot[server]"
ai-knot --storage sqlite serve my_agent --port 8000
```

Then:

- JSON API: `POST /v1/facts`, `POST /v1/search`, `GET /v1/facts`, `DELETE /v1/facts/{fact_id}`, `GET /v1/stats`
- `POST /v1/recall` is still available when you want the agent-memory wording instead of `search`
- zero-network JSON-loop proof: [`examples/http_sidecar_surface_demo.py`](../examples/http_sidecar_surface_demo.py)
- Node / TypeScript client for the same sidecar: `HttpKnowledgeBase({ baseUrl, token })`
- browser inspector: open `http://127.0.0.1:8000/inspect`
- repo-native Node proof: `cd npm && npm run example:http-sidecar`
- zero-network seeded demo: [`examples/browser_inspector_demo.py`](../examples/browser_inspector_demo.py)

See also: [deployment.md#browser-inspector](deployment.md#browser-inspector)

---

## Status

The main stack-specific surfaces are now in-repo: CrewAI, LlamaIndex, AutoGen,
OpenAI Agents SDK, PydanticAI, OpenClaw, LangChain / LangGraph, Vercel AI SDK,
MCP, assistant skills, TypeScript, and the HTTP sidecar/browser inspector.
