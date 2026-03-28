# Skill: User Guide — Applying agentmemo to Your Project

## Who this is for

You are a developer integrating agentmemo into your own AI agent or chatbot project.
This guide takes you from zero to a working memory layer in 30 seconds,
then covers everything you need to go production.

---

## Step 1: Install

```bash
pip install agentmemo

# If you want LLM-powered automatic fact extraction:
pip install "agentmemo[openai]"
```

---

## Step 2: Create a knowledge base

```python
from agentmemo import KnowledgeBase

# agent_id is your namespace — one KB per agent / user / context
kb = KnowledgeBase(agent_id="my_assistant")
```

By default, facts are stored in `.agentmemo/my_assistant/knowledge.yaml` —
human-readable, editable, Git-trackable.

---

## Step 3: Add facts manually

```python
from agentmemo import MemoryType

kb.add("User works at Sber as Operations Director",
       importance=0.95)                              # semantic by default

kb.add("User prefers Python, dislikes async code",
       type=MemoryType.PROCEDURAL, importance=0.85)

kb.add("Deploy failed last Tuesday",
       type=MemoryType.EPISODIC, importance=0.40)    # low importance — will fade
```

### Memory types
| Type | Use for | Example |
|---|---|---|
| `SEMANTIC` (default) | Facts about the world/user | “User works at Sber” |
| `PROCEDURAL` | How the user wants things done | “Always use pytest” |
| `EPISODIC` | Specific past events | “Deploy failed last Tuesday” |

---

## Step 4: Recall at inference time

```python
# Pass the user’s current message
context = kb.recall("how should I write this deployment script?")
print(context)
# [procedural] User prefers Python, dislikes async code
# [semantic]   User deploys everything in Docker
# [semantic]   User works at Sber as Operations Director

# Inject into your LLM prompt
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"You are a helpful assistant.\n\n{context}"},
        {"role": "user",   "content": user_message},
    ],
)
```

`recall()` automatically:
1. Applies the Ebbinghaus forgetting curve (stale facts rank lower)
2. Uses TF-IDF to find the most relevant facts for the query
3. Increments `access_count` on returned facts (reinforcement)
4. Returns at most `top_k=5` facts as a ready-to-inject string

---

## Step 5: Extract facts from conversations automatically

```python
import os
from agentmemo import ConversationTurn

turns = [
    ConversationTurn(role="user",      content="I deploy everything in Docker"),
    ConversationTurn(role="assistant", content="Got it!"),
    ConversationTurn(role="user",      content="I hate async code"),
]

# Extracts facts via LLM, deduplicates, and stores
new_facts = kb.learn(
    turns,
    api_key=os.environ["OPENAI_API_KEY"],   # or ANTHROPIC_API_KEY + provider="anthropic"
)
print(f"Learned {len(new_facts)} facts")
```

The extractor skips filler (“thanks”, “ok”, “got it”) and returns only meaningful facts.

---

## Choosing a storage backend

### YAML (default) — development, small projects
```python
from agentmemo.storage import YAMLStorage

kb = KnowledgeBase(
    agent_id="bot",
    storage=YAMLStorage(base_dir=".agentmemo"),  # default
)
```
- Human-readable files, editable by hand
- Git-trackable — commit your agent’s memory with your code
- Best for: development, per-project agents, small teams

### SQLite — production, single server
```python
from agentmemo.storage import SQLiteStorage

kb = KnowledgeBase(
    agent_id="bot",
    storage=SQLiteStorage(db_path="/data/agentmemo.db"),
)
```
- Zero-server production storage
- Better for: multi-agent setups, higher write throughput
- Same API as YAML — swap anytime

---

## OpenAI integration (automatic memory injection)

```python
from agentmemo.integrations.openai import MemoryEnabledOpenAI

client = MemoryEnabledOpenAI(knowledge_base=kb)

# Your messages, as usual
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user",   "content": "Write me a deployment script"},
]

# Memory context injected automatically into the system prompt
enriched = client._enrich_messages(messages)

# Pass enriched to your real OpenAI client
import openai
response = openai.OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=enriched,
)
```

---

## Multi-user / multi-agent setup

```python
# Each user/agent gets their own namespace — no data leaks
def get_kb(user_id: str) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id=f"user_{user_id}",
        storage=SQLiteStorage(db_path="/data/agentmemo.db"),
    )

# In your request handler:
kb = get_kb(current_user.id)
context = kb.recall(user_message)
```

Shared storage, isolated namespaces. Zero extra setup.

---

## CLI — inspect and manage memory

```bash
# List all facts
agentmemo show my_assistant

# Test retrieval
agentmemo recall my_assistant "how should I deploy this?"

# Add a fact manually
agentmemo add my_assistant "User prefers FastAPI" --importance 0.9 --type procedural

# View stats
agentmemo stats my_assistant

# Apply forgetting curve manually
agentmemo decay my_assistant

# Backup
agentmemo export my_assistant backup.yaml

# Restore
agentmemo import my_assistant backup.yaml

# Wipe
agentmemo clear my_assistant
```

---

## Forgetting — why it helps you

Accumulating everything makes your agent **worse**:
- Old, irrelevant facts pollute the context window
- The agent gets confused by outdated information

agentmemo uses an Ebbinghaus decay curve:
- Facts with **low importance** and **no recent access** fade automatically
- Facts with **high importance** or **frequent access** are reinforced
- You never need to manually clean up — `decay()` runs on every `recall()`

```python
# This happens automatically on recall(), but you can run it manually:
kb.decay()
```

---

## Integrating into an existing chatbot (pattern)

```python
from agentmemo import KnowledgeBase
from agentmemo.storage import SQLiteStorage
import os

# Initialize once, reuse across requests
STORAGE = SQLiteStorage(db_path=os.environ.get("AGENTMEMO_DB", ".agentmemo/bot.db"))

def chat(user_id: str, user_message: str) -> str:
    kb = KnowledgeBase(agent_id=f"user_{user_id}", storage=STORAGE)

    # 1. Get relevant context
    context = kb.recall(user_message, top_k=5)

    # 2. Build prompt
    system_prompt = "You are a helpful assistant."
    if context:
        system_prompt += f"\n\n## What I know about you:\n{context}"

    # 3. Call LLM
    response = call_your_llm(system_prompt, user_message)

    # 4. (Optional) learn from this exchange
    from agentmemo import ConversationTurn
    kb.learn(
        [ConversationTurn("user", user_message), ConversationTurn("assistant", response)],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    return response
```

---

## FAQ

**Q: Does agentmemo store conversation logs?**
No. It stores structured facts extracted from conversations. Logs are your business.

**Q: Can I run it without an LLM?**
Yes. `add()` works without any API key. `learn()` requires an LLM to extract facts automatically.

**Q: Is my data sent anywhere?**
Only when you call `learn()` with an API key — the conversation is sent to your chosen LLM provider.
Your stored facts stay on your storage (local files or your own DB).

**Q: Can multiple processes share one SQLite DB?**
SQLite supports multiple readers but only one writer at a time.
For high-concurrency production, use the planned PostgreSQL backend.

**Q: Why not use Mem0 / Zep / LangMem?**
See the comparison table in `README.md`.
Short answer: agentmemo is self-hosted, pluggable, human-readable, and free forever (MIT).

**Q: How do I migrate from YAML to SQLite?**
```bash
agentmemo export my_agent backup.yaml --data-dir .agentmemo
agentmemo import my_agent backup.yaml --data-dir /path/to/sqlite/dir
```
Or pass the new storage backend directly in Python and call `save()` with the loaded facts.
