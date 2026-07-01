# ai-knot usage guide

The complete API reference. For the quick pitch and benchmarks, see the
[README](../README.md); for production guarantees, [production-readiness.md](production-readiness.md).
For a surface-by-surface routing guide, see [integrations.md](integrations.md).

- [Initialization — storage + provider](#initialization)
- [Memory types](#memory-types)
- [Conflict resolution](#conflict-resolution)
- [Forgetting (power-law decay)](#forgetting)
- [Retrieval with scores](#retrieval-with-scores)
- [Snapshots](#snapshots)
- [Batch insertion](#batch-insertion)
- [Async API](#async-api)
- [`learn()` options](#learn-options)
- [LLM providers](#llm-providers)
- [LLM-enhanced features](#llm-enhanced-features)
- [Multilingual (Russian)](#multilingual)
- [Clock injection and RRF](#clock-injection-and-rrf)
- [CLI](#cli)
- [Knowledge on disk (YAML)](#knowledge-on-disk)
- [MCP server](#mcp-server)
- [OpenClaw](#openclaw)
- [LangChain / LangGraph](#langchain--langgraph)
- [CrewAI](#crewai)
- [OpenAI Agents SDK](#openai-agents-sdk)
- [AutoGen](#autogen)
- [OpenAI integration](#openai-integration)
- [Multi-agent: shared pool](#multi-agent)
- [Bi-temporal `event_time`](#bi-temporal-event_time)
- [Examples](#examples)

---

## Initialization

Storage is set once on `KnowledgeBase`. The LLM provider can be set at init (recommended
for production) or passed per `learn()` call. They are independent and combine freely.

```python
from ai_knot import KnowledgeBase, ConversationTurn
from ai_knot.storage import SQLiteStorage

# Option A: configure provider once at init (recommended)
kb = KnowledgeBase(
    agent_id="assistant",
    storage=SQLiteStorage(db_path="./agent.db"),
    provider="openai",
    api_key="sk-...",          # or reads OPENAI_API_KEY from env if omitted
)
kb.learn(turns)                # no credentials needed per call

# Option B: pass provider per call (legacy, still supported)
kb = KnowledgeBase(agent_id="assistant", storage=SQLiteStorage(db_path="./agent.db"))
kb.learn(turns, provider="openai")                           # reads OPENAI_API_KEY
kb.learn(turns, provider="anthropic", api_key="sk-ant-...")  # Claude
kb.learn(turns, provider="openai-compat",                    # any compatible API
         api_key="...", base_url="http://localhost:8000/v1")

context = kb.recall("how should I deploy this?")   # recall never calls the LLM
```

### Pluggable storage backends

No vendor lock-in. Swap backends with one line — same API, same code.

```python
from ai_knot.storage import YAMLStorage, SQLiteStorage, create_storage

kb = KnowledgeBase("bot", storage=YAMLStorage())                       # dev, zero infra
kb = KnowledgeBase("bot", storage=SQLiteStorage(db_path="/data/a.db")) # single server
storage = create_storage("postgres", dsn="postgresql://u:p@db:5432/ai-knot")  # multi-process
```

> **Cross-process safety (SQLite):** `SQLiteStorage` implements `AtomicUpdateCapable` —
> `SharedMemoryPool.publish()` wraps the whole load→merge→save in a `BEGIN EXCLUSIVE`
> transaction, preventing lost updates when processes share a `.db` file.

---

## Memory types

| Type | When to use | Example |
|---|---|---|
| `semantic` | Stable facts about the user or world | "User works at Sber" |
| `procedural` | How the user wants things done | "Always use type hints" |
| `episodic` | Specific past events with time context | "Deploy failed last Tuesday" |

Default `semantic` covers most cases. Use `procedural` for preferences and rules;
`episodic` for dated events you might want to forget sooner.

---

## Conflict resolution

`learn()` cross-checks new facts against everything stored before inserting. If a new fact
is semantically similar to an existing one (Jaccard ≥ 0.7 by default), the existing fact is
reinforced (importance +0.05, capped at 1.0) instead of a duplicate being created.

```python
kb.add("User deploys on Docker")
kb.add("User deploys with Docker Compose")          # reinforced, not duplicated
kb.learn(turns, provider="openai", conflict_threshold=0.8)   # control per call
```

---

## Forgetting

Accumulating everything makes agents **worse** — irrelevant facts pollute the context
window (*context rot*). ai-knot uses a **power-law decay curve** (Wixted & Ebbesen, 1997),
empirically superior to exponential decay (R²=98.9% vs 96.3%):

```
retention(t) = (1 + t / (9 × stability))^(-decay_exp)
stability = 336h × importance × (1 + ln(1 + access_count))
decay_exp = { semantic: 0.8, procedural: 1.0, episodic: 1.3 }
```

Decay is applied automatically inside every `recall()`. For facts never recalled, run
`kb.decay()` in a daily cron. Custom exponents:

```python
kb = KnowledgeBase("agent", decay_config={"semantic": 0.5, "episodic": 2.0})
```

---

## Retrieval with scores

`recall_facts_with_scores()` returns each fact with a hybrid relevance score (BM25 + retention
+ importance):

```python
scored = kb.recall_facts_with_scores("Docker deployment", top_k=5)
for fact, score in scored:
    print(f"[{score:.2f}] [{fact.type.value}] {fact.content}")
relevant = [f for f, s in scored if s >= 0.5]
```

Use `recall_facts()` for plain Fact objects; `recall_facts_with_scores()` when scores matter.

---

## Snapshots

Save and restore the complete state of a knowledge base at any point:

```python
kb.snapshot("before_refactor")
kb.add("User switched to Go")
kb.restore("before_refactor")            # atomic rollback
kb.list_snapshots()                      # ["before_refactor"]
diff = kb.diff("before_refactor", "current")
print(diff.added, diff.removed)
```

Both YAML (`.ai_knot/{agent_id}/snapshots/`) and SQLite (same DB file) support snapshots.

---

## Batch insertion

`add_many()` inserts multiple pre-extracted facts in one storage round-trip, no LLM call:

```python
kb.add_many(["User deploys on Fridays", "Stack: Python + FastAPI"])
kb.add_many([
    {"content": "Senior backend engineer", "type": "semantic", "importance": 0.95},
    {"content": "Always use type hints", "type": "procedural", "importance": 0.8},
])
```

---

## Async API

Every blocking op has an `async` variant that runs in a thread-pool, keeping the event loop free:

| Sync | Async |
|------|-------|
| `kb.learn(...)` | `await kb.alearn(...)` |
| `kb.recall(q)` | `await kb.arecall(q)` |
| `kb.recall_facts(q)` | `await kb.arecall_facts(q)` |

```python
async def handle_message(turns):
    await kb.alearn(turns)
    return await kb.arecall("current topic")
```

---

## `learn()` options

```python
kb.learn(turns, provider="openai", timeout=10.0)    # abort slow LLM calls (default 30s)
kb.learn(turns, provider="openai", batch_size=10)   # chunk long convs (default 20)
```

---

## LLM providers

Six providers for fact extraction:

| Provider | Name | Env var |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` |
| GigaChat | `gigachat` | `GIGACHAT_API_KEY` |
| Yandex GPT | `yandex` | `YANDEX_API_KEY` |
| Qwen | `qwen` | `QWEN_API_KEY` |
| Any OpenAI-compatible | `openai-compat` | `LLM_API_KEY` |

---

## LLM-enhanced features

When a provider is configured, extra capabilities activate (auto-tagging during `learn()`,
opt-in query expansion at recall). Without an LLM, everything still works — tags via
`add(tags=[...])`, raw BM25 queries.

```python
kb = KnowledgeBase("agent", provider="openai", api_key="sk-...", llm_recall=True)
kb.learn(turns)                # → tags=["python", "preferences"]
kb.recall("what database?")    # → expansion synonyms added at weight 0.4
```

---

## Multilingual

Zero-dependency Russian stemmer, auto-detected from Cyrillic script — no config:

```python
kb.add("Пользователь предпочитает Python для бэкенда")
kb.recall("запрещённые слова")   # morphological variants match
```

---

## Clock injection and RRF

All recall/decay methods accept `now` for deterministic testing and time-travel. RRF weights
`(bm25, importance, retention, recency)` are configurable:

```python
from datetime import datetime, UTC
kb.recall("deployment", now=datetime(2026, 12, 1, tzinfo=UTC))
kb = KnowledgeBase("agent", rrf_weights=(5.0, 2.0, 2.0, 1.0))
```

---

## CLI

```bash
ai-knot add     my_agent "fact"
ai-knot learn   my_agent "raw text to distill into facts"
ai-knot search  my_agent "query"                                 # alias for recall
ai-knot recall  my_agent "query"
ai-knot recall  my_agent "query" --now 2025-01-01T00:00:00   # point-in-time recall
ai-knot list    my_agent                                      # alias for show
ai-knot show    my_agent                                      # list stored facts + IDs
ai-knot forget  my_agent <fact_id>                            # single-fact delete
ai-knot delete  my_agent <fact_id>                            # alias for forget
ai-knot lineage my_agent <fact_id>                           # supersession audit trail
ai-knot doctor --json                                        # install / integration triage
ai-knot stats   my_agent
ai-knot clear   my_agent                                      # wipe the whole namespace
ai-knot decay   my_agent
ai-knot export  my_agent out.yaml
ai-knot import  my_agent in.yaml
ai-knot setup claude --agent-id bot --storage sqlite         # paste-ready MCP config
ai-knot setup openclaw --agent-id bot --storage sqlite       # OpenClaw config
ai-knot serve   my_agent --port 8000                         # HTTP sidecar + browser inspector
```

The baseline human-operated loop is `add -> search -> list -> delete`. That is
the most recognizable shape if you come from tools like Mem0 or from CRUD-style
CLIs. `ai-knot` also preserves the agent-memory language: `search` and `recall`
are the same underlying command, `list` and `show` are the same listing
surface, and `delete` / `forget` both remove a single fact by ID.

If you want the CLI to do LLM-backed extraction instead of manual `add()` calls:

```bash
export AI_KNOT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
ai-knot learn my_agent "User writes Go, deploys in Docker, and avoids Java."
```

`learn` accepts `--provider`, `--api-key`, `--model`, `--role`, and `--base-url`
for explicit control, but it will also honor `AI_KNOT_PROVIDER`,
`AI_KNOT_API_KEY`, and the provider-specific env vars (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and so on).

If CLI install or integration setup behaves unexpectedly, start with
`ai-knot doctor --json` and [troubleshooting.md](troubleshooting.md).

---

## Knowledge on disk

```yaml
# .ai_knot/my_agent/knowledge.yaml — readable, editable, Git-trackable
a1b2c3:
  content: "User is a senior backend developer at Acme Corp"
  type: semantic
  importance: 0.95
  slot_key: "user::role"              # slot-addressed (set automatically)
d4e5f6:
  content: "User prefers Python, dislikes async patterns"
  type: procedural
  topic_channel: devops               # optional — route to a named channel
  visibility_scope: local             # local = private; default global
```

---

## MCP server

ai-knot ships a native MCP server. Claude can call `add`, `recall`, `forget`, `snapshot`,
etc. as tools — no Python on your end.

```bash
pip install "ai-knot[mcp]"
```

```json
{
  "mcpServers": {
    "ai-knot": {
      "command": "ai-knot-mcp",
      "env": {
        "AI_KNOT_AGENT_ID": "myagent",
        "AI_KNOT_STORAGE": "sqlite",
        "AI_KNOT_DB_PATH": "/absolute/path/to/memory.db"
      }
    }
  }
}
```

**Tools:** `add`, `recall`, `recall_json`, `learn`, `forget`, `list_facts`, `stats`,
`snapshot`, `restore`, `list_snapshots`, `health`, `capabilities`.

| Variable | Default | Description |
|---|---|---|
| `AI_KNOT_AGENT_ID` | `default` | Agent namespace |
| `AI_KNOT_STORAGE` | `sqlite` | `sqlite` (recommended) or `yaml` |
| `AI_KNOT_DATA_DIR` | `.ai_knot` | Base dir for file backends (absolute path) |
| `AI_KNOT_DB_PATH` | — | Full path to SQLite file |

> **TypeScript agents:** always use `recall_json` — returns a stable JSON array.
> **Claude Desktop** launches from a non-interactive shell; always set an absolute `AI_KNOT_DATA_DIR`/`AI_KNOT_DB_PATH`.
> Want the shortest setup proof? Run [`examples/claude_mcp_setup.py`](../examples/claude_mcp_setup.py).
> For channel-ready copy, see [claude-mcp-case-study.md](claude-mcp-case-study.md).

---

## OpenClaw

| Situation | Solution |
|---|---|
| OpenClaw TypeScript app | `ai-knot setup openclaw ...` or `generate_mcp_config()` |
| Python agent (LangChain, LangGraph, CrewAI) | `OpenClawMemoryAdapter(kb)` |

```bash
ai-knot setup openclaw --agent-id my_agent --storage sqlite
```

Copy the printed JSON into:

- macOS / Linux: `~/.openclaw/openclaw.json`
- Windows: `%APPDATA%\OpenClaw\openclaw.json`

```python
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter
memory = OpenClawMemoryAdapter(KnowledgeBase("my_agent"))
memory.add([{"role": "user", "content": "Deploy on Fridays"}])
results = memory.search("deployment schedule")
memory.update(results[0]["id"], "Deploy on Thursdays")
```

> `add()` stores only the last user message (warns on multi-turn) — use `kb.learn(turns, api_key=...)`
> for full extraction. `update()` assigns a new ID.
> Want both flows in one place? Run [`examples/openclaw_integration.py`](../examples/openclaw_integration.py).
> For distribution copy and the app-first message angle, see [openclaw-case-study.md](openclaw-case-study.md).

---

## LangChain / LangGraph

Two thin adapters let a LangChain or LangGraph agent use ai-knot as long-term
memory. ai-knot takes **no** hard dependency on LangChain — if `langchain_core`
is installed the retriever yields real `Document` objects, otherwise a shim with
the same `page_content` / `metadata` surface.

**As a retriever** (RAG chains, LangGraph nodes):

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.langchain import AiKnotRetriever

kb = KnowledgeBase("my_agent")
kb.add("User ships in Go and avoids Java")

retriever = AiKnotRetriever(kb, top_k=3)
docs = retriever.invoke("what language does the user use?")   # or .get_relevant_documents(...)
print(docs[0].page_content)        # "User ships in Go and avoids Java"
print(docs[0].metadata["score"])   # fusion relevance score
```

**As conversational memory** (drop-in for `BaseChatMemory`):

```python
from ai_knot.integrations.langchain import AiKnotChatMemory

memory = AiKnotChatMemory(kb)                                  # memory_key="history"
memory.save_context({"input": "I deploy everything in Docker"}, {"output": "Noted."})
memory.load_memory_variables({"input": "how should I deploy?"})
# {"history": "[1] I deploy everything in Docker"}
```

`save_context` distills the user turn into a stored fact; `load_memory_variables`
recalls only the facts relevant to the current input — so the prompt carries
durable knowledge, not the whole transcript. `clear()` forgets every fact.

---

## CrewAI

`AiKnotCrewAIMemory` lets ai-knot plug into CrewAI's native memory surface.
Use it when you want `Crew(memory=...)` or `Agent(memory=memory.scope(...))`
to keep persistent facts in ai-knot instead of CrewAI's default unified memory
backend.

There is **no hard dependency** on `crewai`. Importing the adapter is safe
without CrewAI installed; the real CrewAI package is only needed for actual
`Crew` / `Agent` runs.

```bash
pip install "ai-knot[crewai]"
```

If you want ai-knot itself to do LLM-backed `extract_memories()` from raw CrewAI
task output, combine it with a provider extra such as
`pip install "ai-knot[crewai,openai]"`.

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.crewai import AiKnotCrewAIMemory

kb = KnowledgeBase(
    agent_id="assistant",
    provider="openai",  # enables LLM-backed extract_memories() for CrewAI task outputs
)
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotCrewAIMemory(kb, top_k=5)

researcher = Agent(
    role="Researcher",
    goal="Find stack constraints",
    backstory="Keeps durable implementation notes",
    memory=memory.scope("/agent/researcher"),
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    memory=memory,
)
```

What the adapter does:

- exposes ai-knot through CrewAI's native `remember` / `recall` shape,
- returns CrewAI-style memory records and matches,
- supports hierarchical scope views via `memory.scope(...)`,
- stores CrewAI scope/categories/metadata alongside ai-knot facts,
- uses ai-knot's extraction path for `extract_memories()` when the `KnowledgeBase`
  has a default provider configured, else falls back to a simple line/sentence split.

This is the right surface when you're already on CrewAI but want deterministic,
self-hosted ai-knot retrieval instead of CrewAI's default memory backend.

> Runnable examples:
> [`examples/crewai_surface_demo.py`](../examples/crewai_surface_demo.py) for a
> zero-network memory-surface proof, and
> [`examples/crewai_integration.py`](../examples/crewai_integration.py) for a real
> Crew / Agent wiring path.

---

## OpenAI Agents SDK

`AiKnotAgentsMemory` adds ai-knot's long-term fact recall to an OpenAI Agents SDK
run. It uses the SDK's `call_model_input_filter` hook to append relevant memory
to the model instructions immediately before each model call.

There is **no hard dependency** on `openai-agents`. Importing the adapter is safe
without the SDK installed; the import is required only when you build a real
`RunConfig` or execute a run.

```bash
pip install "ai-knot[agents]"
```

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory

kb = KnowledgeBase("assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotAgentsMemory(kb, top_k=5)
run_config = memory.build_run_config()

# Pass run_config into Runner.run(...) / Runner.run_sync(...)
```

What the adapter does:

- extracts the latest user text from Responses-style input items,
- recalls the most relevant long-term facts from ai-knot,
- appends them to the model instructions under `## Agent Memory`,
- composes with an existing `call_model_input_filter` if you already have one.

This complements the SDK's session history. It does **not** replace short-term
conversation state; it adds a durable fact layer on top.

> Runnable example: [`examples/openai_agents_integration.py`](../examples/openai_agents_integration.py)

---

## AutoGen

`AiKnotAutoGenMemory` implements the async memory shape used by AutoGen's
`AssistantAgent(memory=[...])` path. ai-knot stays responsible for long-term
fact storage and ranked recall; AutoGen stays responsible for the agent loop and
short-term context.

There is **no hard dependency** on AutoGen. Importing the adapter is safe
without `autogen-core` / `autogen-agentchat`; those packages are needed only for
real AutoGen runs.

```bash
pip install "ai-knot[autogen]"
```

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.autogen import AiKnotAutoGenMemory

kb = KnowledgeBase("assistant")
kb.add("User prefers Python over Java")
kb.add("User deploys APIs with Docker and Kubernetes")

memory = AiKnotAutoGenMemory(kb, top_k=5)
agent = AssistantAgent(
    name="coding_assistant",
    model_client=model_client,
    memory=[memory],
)
```

What the adapter does:

- reads the latest user turn from AutoGen's `model_context`,
- recalls the top relevant facts from ai-knot,
- converts them into AutoGen `MemoryContent` objects for `query()`,
- injects a `SystemMessage` block during `update_context()`,
- supports `add()`, `clear()`, and `close()` for protocol compatibility.

This is the right surface when you want AutoGen-native memory wiring but do not
want to move your persistent memory into another hosted or vector-only store.

> Runnable example: [`examples/autogen_integration.py`](../examples/autogen_integration.py)

---

## OpenAI integration

```python
from ai_knot.integrations.openai import MemoryEnabledOpenAI

client = MemoryEnabledOpenAI(knowledge_base=kb)
enriched = client.enrich_messages([{"role": "user", "content": "Write a deploy script"}])
response = openai.OpenAI().chat.completions.create(model="gpt-4o", messages=enriched)
```

---

## Multi-agent

When several agents share one store, it becomes a `SharedMemoryPool` with fan-in recall,
evidence-gated publishing, per-agent visibility, auto-trust, and a deterministic conflict
resolver. The full pipeline (`ai_knot.multi_agent`) is deterministic — no LLM, no graph.

```python
from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage import SQLiteStorage

storage = SQLiteStorage(db_path="./team.db")
pool = SharedMemoryPool(storage=storage)
pool.register("researcher"); pool.register("writer")
researcher = KnowledgeBase("researcher", storage=storage)

fact = researcher.add("API rate limit is 100 req/s", importance=0.9)
pool.publish("researcher", [fact.id], kb=researcher)

results = pool.recall("rate limits", "writer", top_k=3)
for fact, score in results:
    print(f"[{score:.2f}] (trust={pool.get_trust(fact.origin_agent_id):.2f}) {fact.content}")
```

### Publish gating, channels, delta sync

```python
kb.publish(pool, utility_threshold=0.3)                  # gate low-signal facts
kb.add("Deploy uses Helm 3", topic_channel="devops")     # domain routing
deltas = pool.sync_slot_deltas("writer")                 # only changed slots (<15% volume)
```

### Fan-in recall

When an answer is scattered across many agents, a flat top-k misses it. `pool.recall()` routes
the query → splits into facets → retrieves per facet → set-covers the facets (returns `< top_k`
rather than padding with weak fillers) → resolves competing claims.

### Governance — evidence, visibility, abstention

```python
pool.publish("researcher", ids, kb=researcher, require_evidence=True)   # evidence-before-belief
pool.publish("researcher", [secret.id], kb=researcher, visibility_scope="legal")
pool.grant_read("counsel", "legal")                                    # per-agent read projection
results = pool.recall("did we sign the Acme contract?", "writer", top_k=5)
if pool.last_recall_abstains():
    answer = "I don't have enough evidence to answer that."
```

### Trust integrity

- **Monotonic CAS** rejects stale-replay re-supersession.
- **Laundering-resistant trust** accrues penalty over publish *events*, not volume.
- **Malicious discount** — an agent below trust 0.2 is down-weighted even in wide recall.

### Optional semantic conflict resolver

The deterministic resolver collapses slotted/lexically-near conflicts. Value conflicts that
share a subject but diverge in wording need semantic judgement — an **opt-in seam**, off by default:

```python
from ai_knot.integrations.semantic_resolver_llm import LLMSemanticConflictResolver
pool = SharedMemoryPool(storage=storage, semantic_resolver=LLMSemanticConflictResolver(complete))
```

The core ships no model and imports none — the resolver is parameterized by your
`complete(prompt) -> str` callable and runs only on candidates the deterministic pass left standing.

---

## Bi-temporal `event_time`

Every fact carries two clocks: when ai-knot *learned* it (ingest time) and when the event it
describes *happened* (`event_time`, persisted across all three backends). So "what did we believe
on May 3?" and "what was true as of the incident?" are different, answerable questions. Superseding
closes the old fact (`valid_until`) instead of deleting it — history stays queryable.
`kb.add_resolved()` is the dependency-free seam to push pre-structured updates through supersession
without an LLM extraction call.

---

## Examples

### Manual add + recall (no LLM)
```python
from ai_knot import KnowledgeBase, MemoryType
kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python", type=MemoryType.PROCEDURAL, importance=0.9)
kb.add("User deploys with Docker", importance=0.85)
context = kb.recall("how to deploy?")
```

### Per-customer knowledge (support agent)
```python
def handle_ticket(customer_id: str, message: str) -> str:
    kb = KnowledgeBase(agent_id=f"customer_{customer_id}")
    return kb.recall(message)   # past issues, preferences, tier — per customer
```

### Coding agent with project context
```python
kb = KnowledgeBase(agent_id="project", storage=YAMLStorage(".ai_knot"))
kb.add("Stack: FastAPI + PostgreSQL + Docker", importance=1.0)
kb.add("No unittest — use pytest only", type=MemoryType.PROCEDURAL, importance=0.9)
# Commit .ai_knot/ to Git — new team members clone the context
```

See [`examples/`](../examples/) for runnable scripts (`quickstart.py`, `shared_pool.py`,
`coding_agent.py`, `openai_integration.py`, `multilingual.py`).
