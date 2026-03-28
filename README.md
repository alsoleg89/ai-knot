# agentmemo

![CI](https://github.com/alsoleg89/Agentmemo/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/agentmemo)
![License](https://img.shields.io/badge/license-MIT-green)

**Agent Knowledge Layer. Extract. Store. Retrieve. Any backend.**

Your agents don't need a memory — they need a knowledge base.
agentmemo distills conversations into structured facts, finds the right ones
when needed, and forgets the irrelevant. One line of code. Any storage backend.

---

## The problem (honestly)

Storing full conversation logs is easy. Everyone does it.
The hard part is answering one question at inference time:

> **"Out of 10,000 messages — what does my agent need to know RIGHT NOW?"**

A log can't answer that. agentmemo can.

```
Raw conversation (1000 messages, ~400k tokens)
          |  LLM distillation
Structured facts (12 entries, ~300 tokens)
          |  Embeddings + vector search
Relevant context (3 facts, injected into prompt)
```

---

## Install

```bash
pip install agentmemo

# With OpenAI for LLM extraction:
pip install "agentmemo[openai]"

# With PostgreSQL backend:
pip install "agentmemo[postgres]"
```

---

## Quickstart (30 seconds)

```python
from agentmemo import KnowledgeBase

kb = KnowledgeBase(agent_id="my_agent")

# Add facts manually
kb.add("User works at Sber as Operations Director",
       type="semantic", importance=0.95)
kb.add("User prefers Python, dislikes async code",
       type="procedural", importance=0.85)

# Or extract automatically from a conversation
from agentmemo import ConversationTurn
turns = [
    ConversationTurn(role="user",      content="I deploy everything in Docker"),
    ConversationTurn(role="assistant", content="Got it, I'll use Docker examples"),
]
kb.learn(turns, api_key="sk-...")  # LLM extracts + stores relevant facts

# At inference time — get what matters
context = kb.recall("how should I write this deployment script?")
# -> "[procedural] User prefers Python, dislikes async code
#     [semantic]   User deploys everything in Docker
#     [semantic]   User works at Sber as Operations Director"

# Inject into your prompt
response = openai_client.chat(...,
    system=f"You are a helpful assistant.\n\n{context}")
```

---

## What gets stored (and what doesn't)

agentmemo is **not a conversation log.** It's a structured knowledge extractor.

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

## Memory types

| Type | Stores | Example | Default importance |
|---|---|---|---|
| `semantic` | Facts about the world / user | "User works at Sber" | 0.8 |
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
  content: "User works at Sber as Operations Director"
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
| GigaChat (Sber) | `gigachat` | `GIGACHAT_API_KEY` |
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

messages = [{"role": "user", "content": "Write me a deployment script"}]
enriched = client.enrich_messages(messages)
# System prompt now includes relevant memory context.
# Pass `enriched` to your OpenAI client.
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
[ Storage Adapter ]   YAML / SQLite (Postgres, Mongo, Qdrant planned)
       |
[ Retriever ]         TF-IDF (zero deps) + Embeddings (planned)
       |
Context String        injected into agent system prompt
```

---

## Real-world use cases

### Personal AI assistant
Agent remembers your name, role, preferences, tech stack — across all sessions.

### Customer support (per-customer knowledge)
```python
def handle_ticket(customer_id: str, message: str):
    kb = KnowledgeBase(agent_id=f"customer_{customer_id}")
    context = kb.recall(message)
    # Agent knows: VIP status, past issues, preferred language
```

### Coding agent with project context
```python
kb = KnowledgeBase(agent_id="project", storage=YAMLStorage(".agentmemo"))
kb.add("Stack: FastAPI + PostgreSQL + Docker", importance=1.0)
kb.add("No unittest — use pytest only", importance=0.9)
# New team member clones repo -> agent already knows the project
```

### Multi-agent knowledge sharing
```python
researcher = KnowledgeBase(agent_id="team_alpha")
writer     = KnowledgeBase(agent_id="team_alpha")  # same store
# researcher.add() -> writer.recall() — zero extra setup
```

---

## Roadmap

- [x] Core KnowledgeBase (extraction + storage + retrieval)
- [x] Ebbinghaus forgetting curve
- [x] YAML + SQLite backends
- [x] OpenAI integration
- [x] CLI
- [ ] PostgreSQL + pgvector backend
- [ ] MongoDB backend
- [ ] Qdrant + Weaviate backends
- [ ] Semantic embeddings (sentence-transformers / OpenAI)
- [ ] MCP server support
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
| Free forever | Yes (MIT) | No | No | Yes |

---

## Contributing

PRs welcome. Especially looking for: storage backend implementations,
integration adapters, retrieval strategies.

---

## License

MIT

---

**If agentmemo made your agents smarter, give it a star.**
