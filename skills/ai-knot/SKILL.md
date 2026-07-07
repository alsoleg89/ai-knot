---
name: ai-knot
description: >
  Deterministic, self-hosted long-term memory for AI agents. Trigger when the
  user wants persistent memory, MCP-backed assistant memory, or wants to
  integrate ai-knot into Python, CrewAI, LlamaIndex, AutoGen, the OpenAI
  Agents SDK, PydanticAI,
  LangChain/LangGraph, TypeScript, or HTTP services. This is the default
  ai-knot skill for product and integration usage.
license: MIT
metadata:
  author: alsoleg89
  version: "0.11.0"
  category: ai-memory
  tags: "memory, ai-agents, mcp, crewai, llamaindex, autogen, langgraph, python, typescript"
---

# ai-knot

`ai-knot` is a deterministic memory layer for agents. It stores structured
facts and recalls only the few relevant ones for the next turn. Recall never
needs an LLM; LLMs are optional only on the write/extraction path.

## Choose the right surface

| Surface | Install | Use this shape |
|---|---|---|
| Core Python | `pip install ai-knot` | `KnowledgeBase.add(...)`, `KnowledgeBase.search(...)` / `KnowledgeBase.recall(...)` |
| `learn()` extraction | `pip install "ai-knot[openai]"` | `KnowledgeBase(..., provider="openai")` |
| Generic function-calling agent | `pip install ai-knot` | `create_basic_memory_functions(...)` |
| CrewAI | `pip install "ai-knot[crewai]"` | `AiKnotCrewAIMemory` via `Crew(memory=...)` |
| LangGraph tools | `pip install "ai-knot[langgraph]"` | `create_basic_memory_tools(...)`, `create_get_memory_tool(...)`, or `create_manage_memory_tool(...)` + `create_search_memory_tool(...)` |
| LangChain retriever / chat memory | `pip install "ai-knot[langchain]"` | `AiKnotRetriever(...)` / `AiKnotChatMemory(...)` |
| LlamaIndex | `pip install "ai-knot[llamaindex]"` | `AiKnotLlamaIndexMemory.from_defaults(..., knowledge_base=kb)` |
| AutoGen | `pip install "ai-knot[autogen]"` | `AiKnotAutoGenMemory` via `AssistantAgent(memory=[...])` |
| OpenAI Agents SDK | `pip install "ai-knot[agents]"` | `AiKnotAgentsMemory(...).build_run_config()` |
| PydanticAI | `pip install "ai-knot[pydanticai]"` | `AiKnotPydanticAIMemory(...).run_sync(agent, prompt, instructions=...)` |
| Claude / OpenClaw / MCP clients | `pip install "ai-knot[mcp]"` | `ai-knot setup claude ...` or `ai-knot setup openclaw ...` |
| HTTP sidecar | `pip install "ai-knot[server]"` | `ai-knot serve` + `HttpKnowledgeBase({ baseUrl, token })` |
| Node / TypeScript | `npm install ai-knot` | `KnowledgeBase(...)`, `HttpKnowledgeBase(...)`, `npx ai-knot-doctor --json` |

## Default memory loop

All integrations reduce to one first-run loop:

`add -> search -> list -> delete`

Keep the aliases straight:

- use `learn(...)` when you want ai-knot to extract facts from raw text with an LLM;
- use `recall(...)` as an alias for `search(...)`;
- use `forget(...)` as an alias for `delete(...)`.
- when you already have a `fact_id`, use `get(...)` for targeted inspection before deleting or superseding anything.

For user-facing inspect flows, CLI / MCP / npm / HTTP list surfaces default to
the current active memory. Ask for history explicitly with
`--include-inactive`, `include_inactive=true`, or `includeInactive: true` when
you need superseded facts for audit/debug work.

For Claude / OpenClaw / MCP-backed assistant flows, keep setup verbs separate:
`setup` and `doctor` get the client connected, then the memory loop inside the
client is still `add/search/list/delete`.

Minimal example:

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")
fact = kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

facts = kb.search("what stack does the user use?")  # alias: kb.recall(...)
print(kb.get(fact.id))
print(kb.list())
kb.delete(fact.id)  # alias: kb.forget(...)
```

### CLI / MCP mental model

```bash
ai-knot add assistant "User deploys APIs with Docker and Kubernetes"
ai-knot search assistant "what does the user deploy with?"
ai-knot list assistant
ai-knot get assistant <fact_id>
ai-knot delete assistant <fact_id>
```

For MCP clients, the same verbs surface as `add`, `search`, `list`, `get`, and
`delete`. Setup remains separate:

```bash
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot doctor --json
```

On supported platforms, `--write-default-config` will merge the default client
config for you. Use `--write-config <path>` for a non-default plain-JSON path.

## Framework-native entry points

### CrewAI

```python
from ai_knot.integrations.crewai import AiKnotCrewAIMemory

memory = AiKnotCrewAIMemory(kb, top_k=5)
crew = Crew(agents=[researcher, writer], tasks=[task], memory=memory)
scoped_agent = Agent(..., memory=memory.scope("/agent/researcher"))
```

### AutoGen

```python
from ai_knot.integrations.autogen import AiKnotAutoGenMemory

memory = AiKnotAutoGenMemory(kb, top_k=5)
agent = AssistantAgent(name="assistant", model_client=model_client, memory=[memory])
```

### OpenAI Agents SDK

```python
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory

run_config = AiKnotAgentsMemory(kb, top_k=5).build_run_config()
result = Runner.run_sync(agent, "Write a deployment checklist.", run_config=run_config)
```

### PydanticAI

```python
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory

memory = AiKnotPydanticAIMemory(kb, top_k=5)
result = memory.run_sync(
    agent,
    "Write a deployment checklist.",
    instructions="You are a concise staff engineer.",
)
```

### LangGraph

```python
from langgraph.prebuilt import create_react_agent
from ai_knot.integrations.langchain import (
    create_basic_memory_tools,
)

tools = create_basic_memory_tools(kb, top_k=5)
agent = create_react_agent(model, tools=tools)
```

If you want the more compact LangMem-shaped surface, keep
`create_manage_memory_tool(kb)` + `create_search_memory_tool(kb, top_k=5)`.
If the agent already has a `fact_id`, add `create_get_memory_tool(kb)` or use
`create_basic_memory_tools(kb, top_k=5, include_get=True)` for targeted
inspection.

### Generic function-calling agent

```python
from ai_knot.integrations import create_basic_memory_functions

functions = create_basic_memory_functions(kb, top_k=5, include_get=True)
```

Use this path when the host runtime registers ordinary Python callables as
tools and you do not want LangChain-style tool objects in the middle.

### LlamaIndex

```python
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory

memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=5)
chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
```

### Claude / OpenClaw / any MCP client

```bash
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
```

The setup commands can merge the default client config automatically on
supported platforms.

## First troubleshooting step

If install or setup fails, start with:

```bash
ai-knot doctor --json
```

For the npm bridge path, start with:

```bash
npx ai-knot-doctor --json
```

If that command path is unavailable, use:

```bash
python -m ai_knot.cli doctor --json
```

## Repo-native references

Read only what you need:

- surface routing: [../../docs/integrations.md](../../docs/integrations.md)
- full API and adapters: [../../docs/usage.md](../../docs/usage.md)
- MCP / HTTP deployment: [../../docs/deployment.md](../../docs/deployment.md)
- first-run troubleshooting: [../../docs/troubleshooting.md](../../docs/troubleshooting.md)
- examples: [../../examples/quickstart.py](../../examples/quickstart.py),
  [../../examples/function_calling_surface_demo.py](../../examples/function_calling_surface_demo.py),
  [../../examples/crewai_integration.py](../../examples/crewai_integration.py),
  [../../examples/langgraph_surface_demo.py](../../examples/langgraph_surface_demo.py),
  [../../examples/langchain_integration.py](../../examples/langchain_integration.py),
  [../../examples/llamaindex_surface_demo.py](../../examples/llamaindex_surface_demo.py),
  [../../examples/autogen_integration.py](../../examples/autogen_integration.py),
  [../../examples/openai_agents_integration.py](../../examples/openai_agents_integration.py),
  [../../examples/pydanticai_integration.py](../../examples/pydanticai_integration.py),
  [../../examples/pydanticai_surface_demo.py](../../examples/pydanticai_surface_demo.py),
  [../../examples/openclaw_integration.py](../../examples/openclaw_integration.py)
