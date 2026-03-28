# agentmemo — Architecture

## Design goals

| Goal | How |
|---|---|
| **Zero vendor lock-in** | Pluggable `StorageBackend` protocol |
| **Zero mandatory cloud deps** | Only `click`, `pyyaml`, `httpx` required |
| **Human-readable storage** | YAML files by default |
| **Signal over noise** | LLM distillation + Ebbinghaus decay |
| **Framework-agnostic** | Plain Python objects, no base classes to inherit |

---

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│  CLI (agentmemo.cli)          Integrations (*.integrations) │
├─────────────────────────────────────────────────────────────┤
│                  KnowledgeBase (agentmemo.knowledge)         │  ← public API
├──────────────┬──────────────────┬───────────────────────────┤
│  Extractor   │  TFIDFRetriever  │  apply_decay / forgetting  │
│  (LLM calls) │  (TF-IDF search) │  (Ebbinghaus curve)        │
├──────────────┴──────────────────┴───────────────────────────┤
│              StorageBackend (protocol)                        │
│  YAMLStorage          SQLiteStorage         (future: PG …)  │
├─────────────────────────────────────────────────────────────┤
│              Core types: Fact, MemoryType, ConversationTurn   │
└─────────────────────────────────────────────────────────────┘
```

### Dependency rules (no circular imports)

```
types  ←  storage  ←  forgetting
                  ←  retriever
                  ←  extractor
                  ←  knowledge  ←  cli
                                ←  integrations
```

`knowledge.py` is the top of the internal dependency graph.  
Nothing in `storage/`, `forgetting.py`, or `retriever.py` may import from `knowledge.py`.

---

## Core types (`agentmemo.types`)

### `Fact`
The atomic unit of knowledge.

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Human-readable knowledge string |
| `type` | `MemoryType` | `semantic` / `procedural` / `episodic` |
| `importance` | `float` | 0.0–1.0; controls decay speed |
| `retention_score` | `float` | Current Ebbinghaus score (updated on recall) |
| `access_count` | `int` | Times retrieved — increases stability |
| `tags` | `list[str]` | Optional labels |
| `id` | `str` | 8-char UUID hex |
| `created_at` | `datetime` | UTC |
| `last_accessed` | `datetime` | UTC |

### `MemoryType`
`SEMANTIC` — facts about the world/user  
`PROCEDURAL` — how the user wants things done  
`EPISODIC` — specific past events  

---

## Storage layer

### `StorageBackend` protocol

```python
def save(self, agent_id: str, facts: list[Fact]) -> None: ...
def load(self, agent_id: str) -> list[Fact]: ...
def delete(self, agent_id: str, fact_id: str) -> None: ...
def list_agents(self) -> list[str]: ...
```

`save` is a **full replace** — the caller always loads, mutates, then saves.

Backends are responsible for:
- Namespacing by `agent_id`
- Serialising all `Fact` fields including datetime with UTC timezone

### YAMLStorage
File layout: `{base_dir}/{agent_id}/knowledge.yaml`  
Key: `fact.id` (8-char hex)  
Human-readable, Git-trackable, editable by hand.

### SQLiteStorage
Single `.db` file. Schema:
```sql
CREATE TABLE facts (
    id TEXT, agent_id TEXT,
    content TEXT, type TEXT,
    importance REAL, retention REAL,
    access_count INTEGER, tags TEXT,
    created_at TEXT, last_accessed TEXT,
    PRIMARY KEY (agent_id, id)
)
```

---

## Forgetting curve (`agentmemo.forgetting`)

```
retention(t) = exp(−t / stability)

stability = 168h × importance × (1 + ln(1 + access_count))
```

- `t` = hours since `last_accessed`
- Base stability = 168 h (1 week) for `importance=1.0, access_count=0`
- Applied via `apply_decay(facts)` **before** every retrieval in `KnowledgeBase.recall()`
- After retrieval, `access_count` is incremented and `last_accessed` is updated → reinforcement

---

## Retrieval (`agentmemo.retriever`)

`TFIDFRetriever.search(query, facts, top_k)` — zero external dependencies.

Scoring:
```
hybrid_score = 0.6 × tfidf + 0.2 × retention_score + 0.2 × importance
```

TF-IDF is computed fresh per query (no index) — suitable for knowledge bases up to ~10k facts.

---

## Extraction (`agentmemo.extractor`)

`Extractor.extract(turns)` sends the conversation to an LLM and parses the returned JSON array of facts.  
Supports `openai` and `anthropic` providers via `httpx`.  
Deduplication: Jaccard word-similarity ≥ 0.8 → keep first occurrence.

---

## Extension points

| What | Where | Interface |
|---|---|---|
| New storage backend | `src/agentmemo/storage/` | `StorageBackend` protocol |
| New LLM provider | `Extractor._call_<provider>` | Returns `list[dict]` from LLM |
| New integration | `src/agentmemo/integrations/` | Import `KnowledgeBase`, wrap |
| New retriever | Replace `TFIDFRetriever` | `.search(query, facts, top_k)` |
