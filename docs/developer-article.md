# Stop replaying the whole transcript

## Adding deterministic memory to an agent in under 30 minutes

Updated: **July 2, 2026**

---

Most agent memory systems still start from the transcript. The conversation grows,
the prompt grows with it, and sooner or later you are paying to re-send months of
history so the model can recover three facts it actually needs.

ai-knot takes a simpler view: memory should look more like a knowledge base than a
chat log. Store facts. Recall the right few. Keep the read path deterministic so it
is cheap and testable.

## What you get

- no LLM on the retrieval path
- self-hosted storage: SQLite, PostgreSQL, or YAML
- MCP server for Claude Desktop / Claude Code / OpenClaw plus remote Streamable HTTP MCP
- TypeScript client for Node apps over MCP or the HTTP sidecar
- HTTP sidecar + browser inspector
- Vercel AI SDK adapter
- CrewAI adapter
- LlamaIndex adapter
- AutoGen adapter
- OpenAI Agents SDK adapter
- PydanticAI adapter
- LangChain / LangGraph adapters
- multi-agent shared memory with trust and provenance controls

## 1. Start with the smallest possible loop

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys services with Docker and Kubernetes")

context = kb.search("what stack should I use?")  # alias: kb.recall(...)
print(context)
```

That is the hot path: `add` or `learn`, then `search` / `recall`. For persistent
memory, the full first-run loop should stay just as obvious: `add -> search -> list -> delete`.

If you want to prove the same loop without opening Python first:

```bash
ai-knot add    assistant "User prefers Python over Java"
ai-knot learn  assistant "User deploys in Docker and uses PostgreSQL"
ai-knot search assistant "what language does the user prefer?"
ai-knot list   assistant
ai-knot delete assistant <fact_id>
```

If you want that same loop mapped across Python, TypeScript, CLI, MCP, and HTTP,
use [memory-commands.md](memory-commands.md).

## 2. Why this is better than replaying history

The goal of memory is not to preserve every sentence. The goal is to preserve the
few facts that matter later:

- preferences,
- durable user facts,
- prior decisions,
- operational context.

Everything else is often noise.

## 3. Use LLMs where they help, not everywhere

If you want ai-knot to extract facts from a conversation, give it a provider during
`learn()`. If you only need recall, no LLM is required.

```python
from ai_knot import ConversationTurn, KnowledgeBase

kb = KnowledgeBase(agent_id="assistant", provider="openai", api_key="sk-...")
turns = [
    ConversationTurn(role="user", content="I deploy everything in Docker"),
    ConversationTurn(role="assistant", content="Noted"),
]
kb.learn(turns)
print(kb.search("how should I deploy this service?"))
```

That split matters. Extraction can be probabilistic. Retrieval does not have to be.

## 4. Pick the surface that matches your stack

### Python agent

Start with the direct `KnowledgeBase` API.

### Claude Desktop / Claude Code / OpenClaw

Use `ai-knot-mcp` for stdio MCP clients, or `ai-knot serve-mcp assistant --port 8765`
when the host supports remote Streamable HTTP MCP. The same memory loop inside
the client stays `add/search/list/delete`.

On supported platforms, the shortest stdio setup path is now one command:

```bash
ai-knot setup claude --agent-id assistant --storage sqlite --write-default-config
ai-knot setup openclaw --agent-id assistant --storage sqlite --write-default-config
ai-knot doctor --json
```

### Node / TypeScript

Install `npm install ai-knot` and use the TypeScript client over the same MCP tools.
If the local Python bridge is unclear, start with `npx ai-knot-doctor --json`.
If a sidecar is already running, use `HttpKnowledgeBase({ baseUrl, token })`
instead of the local MCP subprocess path.
That sidecar path now also keeps `learn([...])` and `addResolved([...])`, so
the no-spawn Node route still has extract-on-write and structured supersession.

### Vercel AI SDK

Use `AiKnotAISDKMemory` when you want deterministic recalled facts to fill the
same `system` / `messages` surface your app already uses.

### CrewAI

Use `AiKnotCrewAIMemory` when you want a native `Crew(memory=...)` or
`Agent(memory=memory.scope(...))` path with ai-knot behind it.

### LlamaIndex

Use `AiKnotLlamaIndexMemory` when you want the familiar `memory=...` seam for
`SimpleChatEngine`, `FunctionAgent`, or `ReActAgent`, but with deterministic
long-term memory under that seam.

### OpenAI Agents SDK

Use `AiKnotAgentsMemory` to inject recalled long-term facts into `RunConfig`
without replacing the SDK's own session history.

### LangChain / LangGraph

Use `create_basic_memory_tools(...)` when you want the clearest
`add/search/list/delete` tool flow, `create_manage_memory_tool(...)` /
`create_search_memory_tool(...)` when you want the compact LangMem-shaped
surface, `AiKnotRetriever` for retrieval flows, or `AiKnotChatMemory` for
conversational memory. When an agent already has a `fact_id` from a list/debug
step, add `create_get_memory_tool(...)` or use
`create_basic_memory_tools(..., include_get=True)` for targeted inspection.
If your runtime accepts plain Python callables directly, step down one layer and
start with `create_basic_memory_functions(...)` instead of wrapping everything
as LangChain-style tools.

### PydanticAI

Use `AiKnotPydanticAIMemory` when you want per-run `instructions=...` memory
injection without replacing the host `Agent`.

### AutoGen

Use `AiKnotAutoGenMemory` when you want to keep `AssistantAgent(memory=[...])`
and add deterministic long-term memory underneath.

### HTTP-first environments

Run the FastAPI sidecar and call `/v1/learn`, `/v1/search`, `/v1/facts`, or
`/v1/facts/resolved`, or open `/inspect` for a lightweight browser view of the
same store. From Node / TypeScript, the same sidecar now also maps directly to
`HttpKnowledgeBase`, including `learn([...])` and structured `addResolved([...])`
writes.
Fastest repo-native proof of that JSON surface:
`python examples/http_sidecar_surface_demo.py`. It exercises `/health`,
`/v1/facts`, `/v1/search`, `GET /v1/facts/{fact_id}`, and delete without
binding a real port.

## 5. The multi-agent part is where it gets interesting

Single-agent memory is already useful, but the harder problem is shared state across
agents. ai-knot's `SharedMemoryPool` adds:

- fan-in recall across agents,
- evidence-aware publishing,
- visibility scopes,
- trust penalties for bad publishers.

That makes it more than "one database table several agents can write to."

## 6. Why the benchmark stance matters

Agent-memory benchmarks are noisy because the reader model, judge model, prompts,
and category filters can all move the number. ai-knot therefore publishes:

- named-reader QA results for standard benchmarks, and
- a deterministic retrieval suite that you can rerun locally.

The second number is the faster credibility test. If a memory project cannot show
you a stable retrieval gain without a model in the loop, it is harder to know what
the product itself is contributing.

## 7. When to use ai-knot

Use it if you want:

- self-hosted memory,
- deterministic recall,
- a smaller context pack than a full transcript,
- a shared memory layer for several agents,
- storage you can inspect and control.

Do not use it if your main requirement is a managed cloud memory platform or a
full agent runtime.

## 8. What to try next

1. Run `ai-knot demo`
   If you want the raw Python API immediately after that, run `python examples/quickstart.py`.
2. Try the zero-network surface that matches your stack:
   `examples/crewai_surface_demo.py`, `examples/pydanticai_surface_demo.py`,
   `examples/langgraph_surface_demo.py`, `examples/llamaindex_surface_demo.py`,
   `examples/openai_agents_surface_demo.py`, or `examples/autogen_surface_demo.py`
   If you're evaluating the npm path first, use `cd npm && npm run doctor`.
3. Try the real integration example for that surface
   If a sidecar is already running, use `cd npm && npm run example:http-sidecar`.
4. Wire the stdio or remote MCP path into Claude or OpenClaw
5. Re-run the deterministic benchmark command in `docs/benchmarks.md`

The practical takeaway is simple: the next generation of agent memory should not be
"more transcript." It should be **better selected knowledge**.
