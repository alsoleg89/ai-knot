# ai-knot

![CI](https://github.com/alsoleg89/ai-knot/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/ai-knot)
![npm](https://img.shields.io/npm/v/ai-knot)
![License](https://img.shields.io/badge/license-MIT-green)

**Agent knowledge layer — distills conversations into structured facts, retrieves what matters, forgets the rest.**

Most agent frameworks treat memory as a log. ai-knot treats it as a knowledge base.
It extracts facts from conversations, scores them by importance, and retrieves only what's
relevant when building the next prompt. Pluggable storage, six LLM providers, no vendor lock-in.

---

## The problem

Most frameworks store everything — messages, tool calls, system prompts, the whole log.
That's fine until you're paying to inject six months of conversation history
into every request, most of which has nothing to do with what the user asked.

The log grows to 400k tokens. The model needs maybe 300 of those for the next turn —
but there's no obvious way to know which ones without reading all of them first.
ai-knot solves this by keeping a distilled knowledge base instead of a raw log.

```
1000 messages  (~400k tokens)
    ↓ LLM extraction
~12 facts      (~300 tokens)
    ↓ TF-IDF retrieval
3–5 facts injected into the next prompt
```

---

## Install

**Python:**
```bash
pip install ai-knot

# With OpenAI for LLM extraction:
pip install "ai-knot[openai]"

# With PostgreSQL backend:
pip install "ai-knot[postgres]"

# With MCP server (Claude Desktop / Claude Code):
pip install "ai-knot[mcp]"
```

**Node.js / TypeScript (requires Python 3.11+ in PATH):**
```bash
npm install ai-knot
```

---

## Quickstart (30 seconds)

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="my_agent")

# Add facts manually
kb.add("User is a senior backend developer at Acme Corp",
       type="semantic", importance=0.95)
kb.add("User prefers Python, dislikes async code",
       type="procedural", importance=0.85)

# Or extract automatically from a conversation
from ai_knot import ConversationTurn
turns = [
    ConversationTurn(role="user",      content="I deploy everything in Docker"),
    ConversationTurn(role="assistant", content="Got it, I'll use Docker examples"),
]
kb.learn(turns, provider="openai", api_key="sk-...")  # LLM extracts + stores relevant facts

# At inference time — get what matters
context = kb.recall("how should I write this deployment script?")
# -> "[procedural] User prefers Python, dislikes async code
#     [semantic]   User deploys everything in Docker
#     [semantic]   User is a senior backend developer at Acme Corp"

# Inject into your prompt
response = openai_client.chat(...,
    system=f"You are a helpful assistant.\n\n{context}")
```

---

## Performance

Benchmarks run on Ubuntu (`ubuntu-latest`, GitHub Actions).
[Full benchmark history →](https://alsoleg89.github.io/ai-knot/dev/bench/)

### Retrieval latency (TF-IDF, in-process)

Measured with `pytest-benchmark`, `pedantic` mode, 20 rounds after 3 warmups.
`TFIDFRetriever.search()` is O(n) — IDF is recomputed on every call.

| Facts in memory | p50 | p95 | QPS |
|----------------|-----|-----|-----|
| 100 | ~1 ms | ~3 ms | ~800 |
| 1 000 | ~8 ms | ~25 ms | ~100 |
| 10 000 | ~80 ms | ~200 ms | ~12 |

> Numbers are indicative. Run `pytest tests/test_performance.py -m slow --benchmark-only` locally for hardware-accurate results.

### Full-stack recall latency (storage I/O + decay + TF-IDF)

`KnowledgeBase.recall()` reads storage on every call. YAML adds ~10–50 ms I/O overhead; SQLite is lower-variance at scale.

| Backend | Facts | p50 | p95 |
|---------|-------|-----|-----|
| YAML | 1 000 | ~30 ms | ~80 ms |
| SQLite | 1 000 | ~20 ms | ~50 ms |

### MCP tool call round-trip (stdio transport)

Measured end-to-end: Python subprocess spawn is one-time; per-call overhead is JSON serialization + tool execution.

| Tool | Facts | p50 | p95 |
|------|-------|-----|-----|
| `add` | — | ~15 ms | ~80 ms |
| `recall` | 50 | ~20 ms | ~100 ms |
| `stats` | — | ~5 ms | ~20 ms |

> Context: pure MCP stdio JSON-RPC overhead is ~10 ms P95 with no tool execution
> ([tmdevlab MCP benchmark](https://www.tmdevlab.com/mcp-server-performance-benchmark.html)).
> Anthropic recommends keeping agent memory tool latency under 100 ms
> ([Reduce Latency docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency)).
> Use `storage="sqlite"` for lower variance at scale.

---

## What ai-knot keeps — and what it drops

Pass it a conversation and it calls your LLM to figure out what's worth keeping —
preferences, recurring patterns, explicit facts. Greetings, clarifications, filler — those don't make the cut.

```
What happened in the conversation:         What ai-knot stores:
---                                        ---
"hey"                                      X skipped
"thanks"                                   X skipped
"ok got it"                                X skipped
"I really hate working with async"         -> "User dislikes async code"
"by the way we deploy on kubernetes"       -> "User deploys on Kubernetes"
"can you make it shorter please"           -> "User prefers concise responses"
```

**Signal, not noise.** Importance scores, retention decay, deduplication — built in.

---

## Conflict resolution — no more duplicate facts

`learn()` cross-checks new facts against everything already stored before inserting anything.
If a new fact is semantically similar to an existing one (Jaccard similarity ≥ 0.7 by default),
the existing fact is reinforced instead of a duplicate being created:

```python
kb.add("User deploys on Docker")
kb.add("User deploys with Docker Compose")  # similar enough -> reinforced, not duplicated

# Control the threshold per call:
kb.learn(turns, provider="openai", conflict_threshold=0.8)
```

Importance is bumped by 0.05 (capped at 1.0) each time reinforcement fires — the knowledge base
naturally weights well-confirmed facts higher over time.

---

## Snapshots — version your knowledge base

Save and restore the complete state of a knowledge base at any point in time:

```python
kb.add("User prefers Python")
kb.add("User deploys on Docker")

kb.snapshot("before_refactor")      # save current state

kb.add("User switched to Go")       # state changes
kb.forget(some_fact_id)

kb.restore("before_refactor")       # atomically roll back

# List all saved snapshots:
names = kb.list_snapshots()         # ["before_refactor"]

# Compare two snapshots (or current state):
diff = kb.diff("before_refactor", "current")
print(f"Added: {[f.content for f in diff.added]}")
print(f"Removed: {[f.content for f in diff.removed]}")
```

Both YAML and SQLite backends support snapshots. YAML stores them under
`.ai_knot/{agent_id}/snapshots/`. SQLite stores them in the same database file.

---

## MCP server — use ai-knot from Claude Desktop or Claude Code

ai-knot ships a native MCP server. Install it and Claude can call `add`, `recall`,
`forget`, and `snapshot` as tools — without any Python code on your end:

```bash
pip install "ai-knot[mcp]"
```

Add to your `claude_desktop_config.json`:

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

**Available tools:** `add`, `recall`, `recall_json`, `forget`, `list_facts`, `stats`, `snapshot`, `restore`, `list_snapshots`.

> **TypeScript agents:** always use `recall_json` — it returns a stable JSON array (`[]` when empty).
> `recall` returns a plain string and `"No relevant facts found."` on empty — harder to parse reliably.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `AI_KNOT_AGENT_ID` | `default` | Agent namespace |
| `AI_KNOT_STORAGE` | `sqlite` | `sqlite` (recommended) or `yaml` |
| `AI_KNOT_DATA_DIR` | `.ai_knot` | Base dir for file backends (use absolute path) |
| `AI_KNOT_DB_PATH` | — | Full path to SQLite file (overrides `DATA_DIR` for sqlite) |

> **Note:** Claude Desktop launches processes from a non-interactive shell where `cwd` is
> undefined. Always set `AI_KNOT_DATA_DIR` or `AI_KNOT_DB_PATH` to an absolute path.

---

## OpenClaw integration

**Which path should I use?**

| Situation | Solution |
|---|---|
| OpenClaw TypeScript app (recommended) | `generate_mcp_config()` → paste into `~/.openclaw/openclaw.json` |
| Python agent (LangChain, LangGraph, CrewAI) | `OpenClawMemoryAdapter(kb)` |

ai-knot works as an OpenClaw memory backend via MCP. Two steps:

```bash
pip install "ai-knot[mcp]"   # installs the ai-knot-mcp entry point
```

> **Note:** `ai-knot` (without `[mcp]`) does not install `ai-knot-mcp`.
> The config will be generated but OpenClaw won't find the command.

Generate the config snippet:

```python
import json
from ai_knot.integrations.openclaw import generate_mcp_config

print(json.dumps(generate_mcp_config("my_agent"), indent=2))
```

Paste the output into your OpenClaw config file:

- **macOS / Linux:** `~/.openclaw/openclaw.json`
- **Windows:** `%APPDATA%\OpenClaw\openclaw.json`

Your agent will have access to all ai-knot tools: `add`, `recall`, `recall_json`,
`forget`, `list_facts`, `list_snapshots`, `stats`, `snapshot`, `restore`.

For Python-native agents (LangChain, LangGraph, CrewAI), use the adapter class instead:

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.openclaw import OpenClawMemoryAdapter

kb = KnowledgeBase("my_agent")
memory = OpenClawMemoryAdapter(kb)

memory.add([{"role": "user", "content": "Deploy on Fridays"}])
results = memory.search("deployment schedule")
memory.update(results[0]["id"], "Deploy on Thursdays")
memory.delete(results[0]["id"])
```

> **Multi-turn extraction:** `add()` stores only the last user message and emits a warning
> if the list has more than one user message. For extracting multiple facts from a full
> conversation, use `kb.learn(turns, api_key=...)` directly.

> **`update()` assigns a new ID.** The old fact is deleted and a new one is created.
> If you need to hold a stable reference, call `delete()` + `add()` yourself and record
> the returned ID.

---

## Pluggable storage backends

No vendor lock-in. Swap backends with one line:

```python
from ai_knot import KnowledgeBase
from ai_knot.storage import YAMLStorage, SQLiteStorage

# Development — zero infra:
kb = KnowledgeBase(agent_id="bot", storage=YAMLStorage())

# Production — single server:
kb = KnowledgeBase(agent_id="bot",
    storage=SQLiteStorage(db_path="/data/agent.db"))
```

Same API. Same code. Different storage.

---

## Initialization — storage + LLM provider together

Storage is set once on `KnowledgeBase`. The LLM provider can be set at init (recommended
for production) or passed per `learn()` call. They are independent and combine freely:

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
kb.learn(more_turns)           # same provider reused

# Option B: pass provider per call (legacy, still supported)
kb = KnowledgeBase(agent_id="assistant", storage=SQLiteStorage(db_path="./agent.db"))
turns = [
    ConversationTurn(role="user", content="I deploy everything in Docker"),
    ConversationTurn(role="assistant", content="Got it!"),
]
kb.learn(turns, provider="openai")           # reads OPENAI_API_KEY from env
kb.learn(turns, provider="openai",    api_key="sk-...")      # explicit key
kb.learn(turns, provider="anthropic", api_key="sk-ant-...")  # Claude
kb.learn(turns, provider="openai-compat",                    # any compatible API
         api_key="...", base_url="http://localhost:8000/v1")

# Recall never calls the LLM — no provider needed
context = kb.recall("how should I deploy this?")
```

Mix and match: any storage backend with any LLM provider.

---

## Retrieval with relevance scores

`recall_facts_with_scores()` returns each fact together with its numeric relevance score.
The score is a **hybrid value** combining TF-IDF similarity to the query, Ebbinghaus
retention, and the fact's importance — higher is more relevant.

Use it when you need to filter or rank facts programmatically rather than inject them
directly into a prompt:

```python
scored = kb.recall_facts_with_scores("Docker deployment", top_k=5)
for fact, score in scored:
    print(f"[{score:.2f}] [{fact.type.value}] {fact.content}")
# [0.87] [procedural] User deploys everything in Docker
# [0.61] [episodic] Deploy failed last Tuesday at 3 PM

# Keep only highly relevant facts
relevant = [fact for fact, score in scored if score >= 0.5]
```

**vs `recall_facts()`** — use `recall_facts()` when you just need the Fact objects;
use `recall_facts_with_scores()` when scores matter for downstream logic.

---

## Memory types

| Type | When to use | Example |
|---|---|---|
| `semantic` | Stable facts about the user or world | "User works at Sber", "Stack is Python + FastAPI" |
| `procedural` | How the user wants things done | "Always use type hints", "Prefer pytest over unittest" |
| `episodic` | Specific past events with time context | "Deploy failed last Tuesday at 3 PM", "User approved the v2 design on Monday" |

Not sure which type to use? Default `semantic` covers most cases. Use `procedural` for
preferences and rules; `episodic` for dated events you might want to forget sooner.

---

## Forgetting (why it matters)

Accumulating everything makes agents **worse**, not better.
Irrelevant facts pollute the context window — this is called **context rot**.

ai-knot uses an Ebbinghaus-based decay curve:

```
retention = e^(-time / stability)

stability = 336h × importance × (1 + ln(1 + access_count))
-> high importance + frequently accessed = remembered for months
-> low importance + never accessed     = forgotten in days
```

Facts accessed often get **reinforced**. Stale facts **fade automatically**.

**Do you need to call `decay()` manually?** No — decay is applied automatically inside
every `recall()` call. For facts that are never recalled (e.g. background knowledge your
agent doesn't actively query), run `kb.decay()` in a daily cron job to keep retention
scores current.

---

## CLI

```bash
ai-knot show   my_agent            # list all stored facts
ai-knot recall my_agent "query"    # test retrieval
ai-knot add    my_agent "fact"     # add a fact
ai-knot stats  my_agent            # counts, avg importance, retention
ai-knot decay  my_agent            # apply forgetting curve
ai-knot clear  my_agent            # wipe knowledge base
ai-knot export my_agent out.yaml   # backup to file
ai-knot import my_agent in.yaml    # restore from backup
```

---

## How knowledge looks on disk (YAML backend)

```yaml
# .ai_knot/my_agent/knowledge.yaml — readable, editable, Git-trackable

a1b2c3:
  content: "User is a senior backend developer at Acme Corp"
  type: semantic
  importance: 0.95
  retention_score: 0.91
  access_count: 12
  created_at: '2026-03-01T10:00:00+00:00'
  last_accessed: '2026-03-27T09:00:00+00:00'
  tags: [user_profile, work]

d4e5f6:
  content: "User prefers Python, dislikes async patterns"
  type: procedural
  importance: 0.85
  retention_score: 0.88
  access_count: 34
```

Edit it by hand. Commit it to Git. Roll back when needed.

---

## Batch fact insertion — `add_many()`

Insert multiple pre-extracted facts in a single storage round-trip, without any LLM call:

```python
# Plain strings — use method-level defaults for type/importance/tags
kb.add_many(["User deploys on Fridays", "User uses Docker", "Stack: Python + FastAPI"])

# Dicts for full control per fact
kb.add_many([
    {"content": "User is a senior backend engineer", "type": "semantic", "importance": 0.95},
    {"content": "Always use type hints", "type": "procedural", "importance": 0.8},
    {"content": "Sprint demo went well", "type": "episodic", "importance": 0.6},
])

# Mix strings and dicts — strings use method defaults
kb.add_many(
    ["Quick fact"],
    type=MemoryType.PROCEDURAL,
    importance=0.7,
)
```

Useful when facts come from an external source, are pre-processed by another tool, or
the LLM extraction step is handled upstream.

---

## Async API

All blocking operations have `async` variants that run in a thread-pool executor,
keeping the asyncio event loop free during LLM HTTP calls:

| Sync | Async |
|------|-------|
| `kb.learn(turns, ...)` | `await kb.alearn(turns, ...)` |
| `kb.recall(query)` | `await kb.arecall(query)` |
| `kb.recall_facts(query)` | `await kb.arecall_facts(query)` |

```python
import asyncio
from ai_knot import KnowledgeBase, ConversationTurn

kb = KnowledgeBase(agent_id="bot", provider="openai", api_key="sk-...")

# FastAPI handler — never blocks the event loop
async def handle_message(turns: list[ConversationTurn]) -> str:
    await kb.alearn(turns)
    return await kb.arecall("current topic")

# Concurrent extraction for multiple agents
kb_a = KnowledgeBase(agent_id="a", provider="openai", api_key="sk-...")
kb_b = KnowledgeBase(agent_id="b", provider="openai", api_key="sk-...")
results = await asyncio.gather(
    kb_a.alearn(turns_a),
    kb_b.alearn(turns_b),
)
```

---

## `learn()` options: timeout and batch_size

```python
# Abort slow LLM calls after 10 seconds (default: 30 s)
kb.learn(turns, provider="openai", api_key="sk-...", timeout=10.0)

# Split long conversations into chunks of 10 turns per LLM call (default: 20)
# Prevents silent fact loss when the LLM truncates a large JSON response
kb.learn(turns, provider="openai", api_key="sk-...", batch_size=10)
```

---

## LLM providers

ai-knot ships with 6 providers for fact extraction:

| Provider | Name | Env var |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` |
| GigaChat | `gigachat` | `GIGACHAT_API_KEY` |
| Yandex GPT | `yandex` | `YANDEX_API_KEY` |
| Qwen | `qwen` | `QWEN_API_KEY` |
| Any OpenAI-compatible | `openai-compat` | `LLM_API_KEY` |

```python
kb.learn(turns, provider="anthropic")  # uses ANTHROPIC_API_KEY from env
kb.learn(turns, provider="gigachat", api_key="...")
kb.learn(turns, provider="openai-compat", api_key="...", base_url="http://localhost:8080/v1")
```

---

## OpenAI integration

```python
from ai_knot import KnowledgeBase
from ai_knot.integrations.openai import MemoryEnabledOpenAI

kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python")
kb.add("User deploys on Docker")

client = MemoryEnabledOpenAI(knowledge_base=kb)

import openai
openai_client = openai.OpenAI()

messages = [{"role": "user", "content": "Write me a deployment script"}]
enriched = client.enrich_messages(messages)

response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=enriched,
)
```

---

## Architecture

```
Conversation Turns
       |
[ Extractor ]         LLM-based distillation -> structured facts
       |
[ KnowledgeBase ]     importance scoring + deduplication + decay
       |
[ Storage Adapter ]   YAML / SQLite / PostgreSQL (Mongo, Qdrant planned)
       |
[ Retriever ]         TF-IDF (zero deps) + Embeddings (planned)
       |
Context String        injected into agent system prompt
```

**Why TF-IDF instead of embeddings?** Embeddings need either an API call or a 500 MB local model just
to recall which language the user prefers. For knowledge bases up to ~10k facts, TF-IDF with hybrid
scoring (keyword match + retention + importance) is fast, deterministic, and requires zero extra setup.
Semantic embeddings are on the roadmap for larger knowledge bases where keyword overlap isn't reliable.

**Known limitation:** extraction quality depends on the LLM. GPT-4o extracts nuanced procedural
facts reliably; smaller models (gpt-3.5-turbo, haiku) occasionally miss implicit preferences or
conflate episodic events with semantic facts. When accuracy matters, use a capable model for `learn()`.

---

## Examples

### 1. Manual add + recall (no LLM required)

```python
from ai_knot import KnowledgeBase, MemoryType

kb = KnowledgeBase(agent_id="assistant")
kb.add("User prefers Python",          type=MemoryType.PROCEDURAL, importance=0.9)
kb.add("User deploys with Docker",     importance=0.85)
kb.add("Deploy failed last Tuesday",   type=MemoryType.EPISODIC,   importance=0.4)

context = kb.recall("how to deploy?")
# -> "[procedural] User prefers Python
#     [semantic]   User deploys with Docker"
```

### 2. SQLite + OpenAI

```python
from ai_knot import KnowledgeBase, ConversationTurn
from ai_knot.storage import SQLiteStorage

kb = KnowledgeBase(agent_id="bot", storage=SQLiteStorage(db_path="./bot.db"))
turns = [ConversationTurn(role="user", content="I work with Python and FastAPI")]
kb.learn(turns, provider="openai")          # reads OPENAI_API_KEY from env
context = kb.recall("what stack does user use?")
```

### 3. YAML storage + Anthropic (Claude)

```python
from ai_knot import KnowledgeBase, ConversationTurn
from ai_knot.storage import YAMLStorage

kb = KnowledgeBase(agent_id="bot", storage=YAMLStorage(base_dir=".ai_knot"))
turns = [ConversationTurn(role="user", content="Always write tests with pytest")]
kb.learn(turns, provider="anthropic", api_key="sk-ant-...")
# Facts are saved to .ai_knot/bot/knowledge.yaml — readable, Git-trackable
```

### 4. PostgreSQL + any OpenAI-compatible endpoint

```python
from ai_knot import KnowledgeBase, ConversationTurn
from ai_knot.storage import create_storage

storage = create_storage("postgres", dsn="postgresql://user:pass@db:5432/ai-knot")
kb = KnowledgeBase(agent_id="assistant", storage=storage)
turns = [ConversationTurn(role="user", content="Prefer concise answers")]
kb.learn(turns, provider="openai-compat",
         api_key="...", base_url="http://localhost:8000/v1")
```

### 5. Per-customer knowledge (support agent)

```python
from ai_knot import KnowledgeBase

def handle_ticket(customer_id: str, message: str) -> str:
    kb = KnowledgeBase(agent_id=f"customer_{customer_id}")
    context = kb.recall(message)
    # Agent sees: past issues, preferences, tier — specific to this customer
    return context
```

### 6. Coding agent with project context

```python
from ai_knot import KnowledgeBase, MemoryType
from ai_knot.storage import YAMLStorage

kb = KnowledgeBase(agent_id="project", storage=YAMLStorage(".ai_knot"))
kb.add("Stack: FastAPI + PostgreSQL + Docker",  importance=1.0)
kb.add("No unittest — use pytest only",         type=MemoryType.PROCEDURAL, importance=0.9)
kb.add("All endpoints require JWT auth",        importance=0.95)
# Commit .ai_knot/ to Git — new team members clone the context
```

### 7. Shared knowledge across multiple agents

```python
from ai_knot import KnowledgeBase
from ai_knot.storage import SQLiteStorage

storage = SQLiteStorage(db_path="./team.db")
researcher = KnowledgeBase(agent_id="team_alpha", storage=storage)
writer     = KnowledgeBase(agent_id="team_alpha", storage=storage)

researcher.add("API rate limit is 100 req/s")
context = writer.recall("rate limits")  # sees researcher's facts instantly
```

### 8. Stats and forgetting curve

```python
from ai_knot import KnowledgeBase

kb = KnowledgeBase(agent_id="assistant")
kb.add("User likes dark mode")
kb.add("User timezone is UTC+3")

stats = kb.stats()
print(f"Facts: {stats['total_facts']}")
print(f"Avg importance: {stats['avg_importance']:.2f}")
print(f"By type: {stats['by_type']}")

kb.decay()  # apply Ebbinghaus forgetting curve — stale facts lose retention score
```

---

## Roadmap

- [x] Core KnowledgeBase (extraction + storage + retrieval)
- [x] Ebbinghaus forgetting curve
- [x] YAML + SQLite backends
- [x] OpenAI integration
- [x] CLI
- [x] PostgreSQL backend
- [x] Conflict resolution in `learn()` (cross-session deduplication)
- [x] Snapshots (`snapshot`, `restore`, `diff`)
- [x] MCP server (Claude Desktop / Claude Code)
- [x] npm package (`npm install ai-knot`)
- [x] OpenClaw integration (`OpenClawMemoryAdapter` + `generate_mcp_config`)
- [x] Scored retrieval (`recall_facts_with_scores`)
- [ ] MongoDB backend
- [ ] Qdrant + Weaviate backends
- [ ] Semantic embeddings (sentence-transformers / OpenAI)
- [ ] LangChain / CrewAI integrations
- [ ] Web UI knowledge inspector
- [ ] REST API / sidecar mode

---

## Why not just use Mem0 / Zep / LangMem?

| | ai-knot | Mem0 | Zep | LangMem |
|---|---|---|---|---|
| Self-hosted | Yes | Partial | Yes | Yes |
| No cloud required | Yes | No | No | Yes |
| Pluggable storage | Yes | No | No | No |
| Human-readable store | Yes | No | No | No |
| Setup time | 30 sec | 10 min | 30 min | 5 min |
| Framework-agnostic | Yes | Partial | Partial | LangGraph only |
| Forgetting curve | Yes | No | No | No |
| Snapshots + diff | Yes | No | No | No |
| MCP server | Yes | No | No | No |
| Free forever | Yes (MIT) | No | No | Yes |

---

## Contributing

PRs welcome. Especially looking for: storage backend implementations,
integration adapters, retrieval strategies.

---

## License

MIT

---

Found a bug or a missing backend? Open an issue. Built something with it? We'd like to hear.
