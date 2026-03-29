# agentmemo

![CI](https://github.com/alsoleg89/agentmemo/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/agentmemo)
![npm](https://img.shields.io/npm/v/@alsoleg89/agentmemo)
![License](https://img.shields.io/badge/license-MIT-green)

**Agent knowledge layer — distills conversations into structured facts, retrieves what matters, forgets the rest.**

Most agent frameworks treat memory as a log. agentmemo treats it as a knowledge base.
It extracts facts from conversations, scores them by importance, and retrieves only what's
relevant when building the next prompt. Pluggable storage, six LLM providers, no vendor lock-in.

---

## The problem

Most frameworks store everything — messages, tool calls, system prompts, the whole log.
That's fine until you're paying to inject six months of conversation history
into every request, most of which has nothing to do with what the user asked.

The log grows to 400k tokens. The model needs maybe 300 of those for the next turn —
but there's no obvious way to know which ones without reading all of them first.
agentmemo solves this by keeping a distilled knowledge base instead of a raw log.

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
pip install agentmemo

# With OpenAI for LLM extraction:
pip install "agentmemo[openai]"

# With PostgreSQL backend:
pip install "agentmemo[postgres]"

# With MCP server (Claude Desktop / Claude Code):
pip install "agentmemo[mcp]"
```

**Node.js / TypeScript (requires Python 3.11+ in PATH):**
```bash
npm install @alsoleg89/agentmemo
```

---

## Quickstart (30 seconds)

```python
from agentmemo import KnowledgeBase

kb = KnowledgeBase(agent_id="my_agent")

# Add facts manually
kb.add("User is a senior backend developer at Acme Corp",
       type="semantic", importance=0.95)
kb.add("User prefers Python, dislikes async code",
       type="procedural", importance=0.85)

# Or extract automatically from a conversation
from agentmemo import ConversationTurn
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

## What agentmemo keeps — and what it drops

Pass it a conversation and it calls your LLM to figure out what's worth keeping —
preferences, recurring patterns, explicit facts. Greetings, clarifications, filler — those don't make the cut.

```
What happened in the conversation:         What agentmemo stores:
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
`.agentmemo/{agent_id}/snapshots/`. SQLite stores them in the same database file.

---

## MCP server — use agentmemo from Claude Desktop or Claude Code

agentmemo ships a native MCP server. Install it and Claude can call `add`, `recall`,
`forget`, and `snapshot` as tools — without any Python code on your end:

```bash
pip install "agentmemo[mcp]"
```

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agentmemo": {
      "command": "agentmemo-mcp",
      "env": {
        "AGENTMEMO_AGENT_ID": "myagent",
        "AGENTMEMO_STORAGE": "sqlite",
        "AGENTMEMO_DB_PATH": "/absolute/path/to/memory.db"
      }
    }
  }
}
```

**Available tools:** `add`, `recall`, `forget`, `list_facts`, `stats`, `snapshot`, `restore`.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `AGENTMEMO_AGENT_ID` | `default` | Agent namespace |
| `AGENTMEMO_STORAGE` | `yaml` | `yaml` or `sqlite` |
| `AGENTMEMO_DATA_DIR` | `.agentmemo` | Base dir for YAML backend (use absolute path) |
| `AGENTMEMO_DB_PATH` | — | Full path to SQLite file (overrides `DATA_DIR` for sqlite) |

> **Note:** Claude Desktop launches processes from a non-interactive shell where `cwd` is
> undefined. Always set `AGENTMEMO_DATA_DIR` or `AGENTMEMO_DB_PATH` to an absolute path.

---

## Pluggable storage backends

No vendor lock-in. Swap backends with one line:

```python
from agentmemo import KnowledgeBase
from agentmemo.storage import YAMLStorage, SQLiteStorage

# Development — zero infra:
kb = KnowledgeBase(agent_id="bot", storage=YAMLStorage())

# Production — single server:
kb = KnowledgeBase(agent_id="bot",
    storage=SQLiteStorage(db_path="/data/agent.db"))
```

Same API. Same code. Different storage.

---

## Initialization — storage + LLM provider together

Storage is set once on `KnowledgeBase`. The LLM provider is passed per `learn()` call.
They are independent and combine freely:

```python
from agentmemo import KnowledgeBase, ConversationTurn
from agentmemo.storage import SQLiteStorage

# Step 1: pick a storage backend
storage = SQLiteStorage(db_path="./agent.db")

# Step 2: create the knowledge base
kb = KnowledgeBase(agent_id="assistant", storage=storage)

# Step 3: learn from a conversation — LLM provider goes here
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

## Memory types

| Type | Stores | Example | Default importance |
|---|---|---|---|
| `semantic` | Facts about the world / user | "User works at Acme Corp" | 0.8 |
| `procedural` | How the user wants things done | "Always use type hints" | 0.8 |
| `episodic` | Specific past events | "Deploy failed last Tuesday" | 0.8 |

---

## Forgetting (why it matters)

Accumulating everything makes agents **worse**, not better.
Irrelevant facts pollute the context window — this is called **context rot**.

agentmemo uses an Ebbinghaus-based decay curve:

```
retention = e^(-time / stability)

stability = 336h × importance × (1 + ln(1 + access_count))
-> high importance + frequently accessed = remembered for months
-> low importance + never accessed     = forgotten in days
```

Facts accessed often get **reinforced**. Stale facts **fade automatically**.
No manual cleanup needed.

---

## CLI

```bash
agentmemo show   my_agent            # list all stored facts
agentmemo recall my_agent "query"    # test retrieval
agentmemo add    my_agent "fact"     # add a fact
agentmemo stats  my_agent            # counts, avg importance, retention
agentmemo decay  my_agent            # apply forgetting curve
agentmemo clear  my_agent            # wipe knowledge base
agentmemo export my_agent out.yaml   # backup to file
agentmemo import my_agent in.yaml    # restore from backup
```

---

## How knowledge looks on disk (YAML backend)

```yaml
# .agentmemo/my_agent/knowledge.yaml — readable, editable, Git-trackable

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

## LLM providers

agentmemo ships with 6 providers for fact extraction:

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
from agentmemo import KnowledgeBase
from agentmemo.integrations.openai import MemoryEnabledOpenAI

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
from agentmemo import KnowledgeBase, MemoryType

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
from agentmemo import KnowledgeBase, ConversationTurn
from agentmemo.storage import SQLiteStorage

kb = KnowledgeBase(agent_id="bot", storage=SQLiteStorage(db_path="./bot.db"))
turns = [ConversationTurn(role="user", content="I work with Python and FastAPI")]
kb.learn(turns, provider="openai")          # reads OPENAI_API_KEY from env
context = kb.recall("what stack does user use?")
```

### 3. YAML storage + Anthropic (Claude)

```python
from agentmemo import KnowledgeBase, ConversationTurn
from agentmemo.storage import YAMLStorage

kb = KnowledgeBase(agent_id="bot", storage=YAMLStorage(base_dir=".agentmemo"))
turns = [ConversationTurn(role="user", content="Always write tests with pytest")]
kb.learn(turns, provider="anthropic", api_key="sk-ant-...")
# Facts are saved to .agentmemo/bot/knowledge.yaml — readable, Git-trackable
```

### 4. PostgreSQL + any OpenAI-compatible endpoint

```python
from agentmemo import KnowledgeBase, ConversationTurn
from agentmemo.storage import create_storage

storage = create_storage("postgres", dsn="postgresql://user:pass@db:5432/agentmemo")
kb = KnowledgeBase(agent_id="assistant", storage=storage)
turns = [ConversationTurn(role="user", content="Prefer concise answers")]
kb.learn(turns, provider="openai-compat",
         api_key="...", base_url="http://localhost:8000/v1")
```

### 5. Per-customer knowledge (support agent)

```python
from agentmemo import KnowledgeBase

def handle_ticket(customer_id: str, message: str) -> str:
    kb = KnowledgeBase(agent_id=f"customer_{customer_id}")
    context = kb.recall(message)
    # Agent sees: past issues, preferences, tier — specific to this customer
    return context
```

### 6. Coding agent with project context

```python
from agentmemo import KnowledgeBase, MemoryType
from agentmemo.storage import YAMLStorage

kb = KnowledgeBase(agent_id="project", storage=YAMLStorage(".agentmemo"))
kb.add("Stack: FastAPI + PostgreSQL + Docker",  importance=1.0)
kb.add("No unittest — use pytest only",         type=MemoryType.PROCEDURAL, importance=0.9)
kb.add("All endpoints require JWT auth",        importance=0.95)
# Commit .agentmemo/ to Git — new team members clone the context
```

### 7. Shared knowledge across multiple agents

```python
from agentmemo import KnowledgeBase
from agentmemo.storage import SQLiteStorage

storage = SQLiteStorage(db_path="./team.db")
researcher = KnowledgeBase(agent_id="team_alpha", storage=storage)
writer     = KnowledgeBase(agent_id="team_alpha", storage=storage)

researcher.add("API rate limit is 100 req/s")
context = writer.recall("rate limits")  # sees researcher's facts instantly
```

### 8. Stats and forgetting curve

```python
from agentmemo import KnowledgeBase

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
- [x] npm package (`npm install agentmemo`)
- [ ] MongoDB backend
- [ ] Qdrant + Weaviate backends
- [ ] Semantic embeddings (sentence-transformers / OpenAI)
- [ ] LangChain / CrewAI integrations
- [ ] Web UI knowledge inspector
- [ ] REST API / sidecar mode

---

## Why not just use Mem0 / Zep / LangMem?

| | agentmemo | Mem0 | Zep | LangMem |
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
