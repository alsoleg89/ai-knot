# Integrations

The fastest way to pick the right `ai-knot` entry point for your stack.

If you're evaluating the project, start here first, then jump into the full API
reference in [usage.md](usage.md).

Across surfaces, the recognizable memory loop stays the same: store with
`add`/`learn`, retrieve with `search`/`recall`, inspect with `list`, and remove
with `delete`/`forget`.

| Surface | Best for | Install | Start here |
|---|---|---|---|
| Core Python API | chatbots, coding agents, custom app logic | `pip install ai-knot` | [`examples/quickstart.py`](../examples/quickstart.py) |
| CrewAI | native `Crew(memory=...)` / `Agent(memory=...)` wiring | `pip install "ai-knot[crewai]"` | [`examples/crewai_integration.py`](../examples/crewai_integration.py) |
| AutoGen | `AssistantAgent(memory=[...])` with persistent facts | `pip install "ai-knot[autogen]"` | [`examples/autogen_integration.py`](../examples/autogen_integration.py) |
| OpenAI Agents SDK | `RunConfig` hook into existing SDK runs | `pip install "ai-knot[agents]"` | [`examples/openai_agents_integration.py`](../examples/openai_agents_integration.py) |
| PydanticAI | per-run `instructions=` memory injection on an existing `Agent` | `pip install "ai-knot[pydanticai]"` | [`examples/pydanticai_integration.py`](../examples/pydanticai_integration.py) |
| OpenClaw | MCP-backed desktop/app memory or Python-side provider compatibility | `pip install "ai-knot[mcp]"` | [`examples/openclaw_integration.py`](../examples/openclaw_integration.py) |
| LangChain / LangGraph | retriever or chat-memory drop-in | `pip install ai-knot` | [`examples/langchain_integration.py`](../examples/langchain_integration.py) |
| Vercel AI SDK | prepend recalled facts to `generateText()` / `streamText()` inputs | `npm install ai-knot ai @ai-sdk/openai` | [`npm/examples/vercel-ai-sdk.ts`](../npm/examples/vercel-ai-sdk.ts) |
| MCP server | Claude Desktop / Claude Code / any MCP client | `pip install "ai-knot[mcp]"` | [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server) |
| Skills / coding assistants | teach Codex / Claude Code / OpenClaw-style tools ai-knot's surfaces up front | `npx skills add https://github.com/alsoleg89/ai-knot --skill ai-knot` | [../skills/README.md](../skills/README.md) |
| TypeScript / npm | Node apps with the Python sidecar/client path | `npm install ai-knot` | [../npm/README.md](../npm/README.md) |
| HTTP sidecar | polyglot services, remote agent runtimes, and browser inspection | `pip install "ai-knot[server]"` | [deployment.md#11-http-sidecar](deployment.md#11-http-sidecar) |

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

See also: [usage.md#autogen](usage.md#autogen)

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

See also: [usage.md#openai-agents-sdk](usage.md#openai-agents-sdk)

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
- repo-native example file: [`../npm/examples/vercel-ai-sdk.ts`](../npm/examples/vercel-ai-sdk.ts)
- npm package docs: [../npm/README.md](../npm/README.md)
- distribution-ready proof asset: [vercel-ai-sdk-case-study.md](vercel-ai-sdk-case-study.md)

### OpenClaw

Use the MCP config path if you're integrating with the OpenClaw app. Use the
Python adapter only if you need OpenClaw-style provider compatibility inside
your own runtime.

```bash
ai-knot setup openclaw --agent-id my_agent --storage sqlite
```

Try one of these next:

- zero-network proof of both flows: [`examples/openclaw_integration.py`](../examples/openclaw_integration.py)
- app/MCP distribution angle: [openclaw-case-study.md](openclaw-case-study.md)
- full API notes: [usage.md#openclaw](usage.md#openclaw)

### Claude Desktop / Claude Code

Use the MCP path when you want Claude to call ai-knot as a tool with no custom
Python glue in your project.

```bash
ai-knot setup claude --agent-id my_agent --storage sqlite
```

Try one of these next:

- zero-network setup proof: [`examples/claude_mcp_setup.py`](../examples/claude_mcp_setup.py)
- Claude/MCP distribution angle: [claude-mcp-case-study.md](claude-mcp-case-study.md)
- deployment notes: [deployment.md#4-run-the-mcp-server](deployment.md#4-run-the-mcp-server)

### Skills / coding assistants

If your coding tool supports the skills standard, install the repo-native
`ai-knot` skill when you want the assistant to have ai-knot-specific setup,
adapter, and troubleshooting patterns in context before it edits code.

```bash
npx skills add https://github.com/alsoleg89/ai-knot --skill ai-knot
```

This surface is most useful when:

- you want an assistant to route between core Python, MCP, CrewAI, AutoGen, the OpenAI Agents SDK, and PydanticAI without guessing,
- you want the assistant to know the exact ai-knot object names (`KnowledgeBase`, `AiKnotCrewAIMemory`, `AiKnotAutoGenMemory`, `AiKnotAgentsMemory`, `AiKnotPydanticAIMemory`),
- you want first-run troubleshooting (`ai-knot doctor --json`) loaded as part of the integration path.

See also: [../skills/README.md](../skills/README.md)

### LangChain / LangGraph

Use `AiKnotRetriever` for RAG or graph nodes, and `AiKnotChatMemory` when you
want a conversational-memory shape.

```python
retriever = AiKnotRetriever(kb, top_k=3)
docs = retriever.invoke("what language does the user use?")
```

See also: [usage.md#langchain--langgraph](usage.md#langchain--langgraph)

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
- browser inspector: open `http://127.0.0.1:8000/inspect`
- zero-network seeded demo: [`examples/browser_inspector_demo.py`](../examples/browser_inspector_demo.py)

See also: [deployment.md#browser-inspector](deployment.md#browser-inspector)

---

## Status

The stack-specific surfaces that matter most for a first launch are now in-repo:
CrewAI, AutoGen, OpenAI Agents SDK, PydanticAI, OpenClaw, LangChain / LangGraph,
Vercel AI SDK, MCP, assistant skills, TypeScript, and the HTTP sidecar/browser
inspector. The
remaining gaps are less about adapters and more about public distribution:
publishing the updated branch, npm parity, and turning the prepared proof assets
into public posts.
