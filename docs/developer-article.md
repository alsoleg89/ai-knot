# Stop replaying the whole transcript

## Adding deterministic memory to an agent in under 30 minutes

Most agent memory systems still start from the transcript. The conversation grows,
the prompt grows with it, and sooner or later you are paying to re-send months of
history so the model can recover three facts it actually needs.

ai-knot takes a simpler view: memory should look more like a knowledge base than a
chat log. Store facts. Recall the right few. Keep the read path deterministic so it
is cheap and testable.

## What you get

- no LLM on the retrieval path
- self-hosted storage: SQLite, PostgreSQL, or YAML
- MCP server for Claude Desktop / Claude Code
- TypeScript client for Node apps
- CrewAI adapter
- OpenAI Agents SDK adapter
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

That is the core product loop: `add` or `learn`, then `search` / `recall`.

If you want to prove the same loop without opening Python first:

```bash
ai-knot add    assistant "User prefers Python over Java"
ai-knot learn  assistant "User deploys in Docker and uses PostgreSQL"
ai-knot search assistant "what language does the user prefer?"
ai-knot list   assistant
```

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

### Claude Desktop / Claude Code

Run `ai-knot-mcp` and register it as an MCP server.

### Node / TypeScript

Install `npm install ai-knot` and use the TypeScript client over the same MCP tools.

### CrewAI

Use `AiKnotCrewAIMemory` when you want a native `Crew(memory=...)` or
`Agent(memory=memory.scope(...))` path with ai-knot behind it.

### OpenAI Agents SDK

Use `AiKnotAgentsMemory` to inject recalled long-term facts into `RunConfig`
without replacing the SDK's own session history.

### LangChain / LangGraph

Use `AiKnotRetriever` for retrieval flows or `AiKnotChatMemory` for conversational memory.

### HTTP-first environments

Run the FastAPI sidecar and call `/v1/recall` / `/v1/facts`, or open `/inspect`
for a lightweight browser view of the same store.

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

1. Run `python examples/quickstart.py`
2. Try the CrewAI or LangChain example in `examples/crewai_integration.py` or `examples/langchain_integration.py`
3. Wire the MCP server into Claude
4. Re-run the deterministic benchmark command in `docs/benchmarks.md`

The practical takeaway is simple: the next generation of agent memory should not be
"more transcript." It should be **better selected knowledge**.
