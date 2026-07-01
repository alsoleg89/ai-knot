---
name: ai-knot
description: >
  Deterministic, self-hosted long-term memory for AI agents. Trigger when the
  user wants persistent memory, MCP-backed assistant memory, or wants to
  integrate ai-knot into Python, CrewAI, AutoGen, the OpenAI Agents SDK,
  PydanticAI,
  LangChain/LangGraph, TypeScript, or HTTP services. This is the default
  ai-knot skill for product and integration usage.
license: MIT
metadata:
  author: alsoleg89
  version: "0.11.0"
  category: ai-memory
  tags: "memory, ai-agents, mcp, crewai, autogen, langgraph, python, typescript"
---

# ai-knot

`ai-knot` is a deterministic memory layer for agents. It stores structured
facts and recalls only the few relevant ones for the next turn. Recall never
needs an LLM; LLMs are optional only on the write/extraction path.

## Choose the right surface

| Surface | Install | Use this shape |
|---|---|---|
| Core Python | `pip install ai-knot` | `KnowledgeBase.add(...)`, `KnowledgeBase.recall(...)` |
| `learn()` extraction | `pip install "ai-knot[openai]"` | `KnowledgeBase(..., provider="openai")` |
| CrewAI | `pip install "ai-knot[crewai]"` | `AiKnotCrewAIMemory` via `Crew(memory=...)` |
| AutoGen | `pip install "ai-knot[autogen]"` | `AiKnotAutoGenMemory` via `AssistantAgent(memory=[...])` |
| OpenAI Agents SDK | `pip install "ai-knot[agents]"` | `AiKnotAgentsMemory(...).build_run_config()` |
| PydanticAI | `pip install "ai-knot[pydanticai]"` | `AiKnotPydanticAIMemory(...).run_sync(agent, prompt, instructions=...)` |
| Claude / OpenClaw / MCP clients | `pip install "ai-knot[mcp]"` | `ai-knot setup claude ...` or `ai-knot setup openclaw ...` |
| HTTP sidecar | `pip install "ai-knot[server]"` | `ai-knot serve` |
| Node / TypeScript | `npm install ai-knot` | use the npm wrapper / Python sidecar path |

## Core loop

All integrations reduce to the same loop:

1. store facts with `add(...)` or `learn(...)`;
2. call `recall(...)` for the next turn;
3. inject only the recalled facts into the model context.

Minimal example:

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

facts = kb.recall("what stack does the user use?")
```

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

### Claude / OpenClaw / any MCP client

```bash
ai-knot setup claude --agent-id assistant --storage sqlite
ai-knot setup openclaw --agent-id assistant --storage sqlite
```

The setup commands print a paste-ready MCP config.

## First troubleshooting step

If install or setup fails, start with:

```bash
ai-knot doctor --json
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
  [../../examples/crewai_integration.py](../../examples/crewai_integration.py),
  [../../examples/autogen_integration.py](../../examples/autogen_integration.py),
  [../../examples/openai_agents_integration.py](../../examples/openai_agents_integration.py),
  [../../examples/pydanticai_integration.py](../../examples/pydanticai_integration.py),
  [../../examples/pydanticai_surface_demo.py](../../examples/pydanticai_surface_demo.py),
  [../../examples/openclaw_integration.py](../../examples/openclaw_integration.py)
