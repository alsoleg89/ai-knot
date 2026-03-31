# ai-knot — Architecture

## Design goals

| Goal | How |
|---|---|
| **Zero vendor lock-in** | Pluggable `StorageBackend` protocol |
| **Zero mandatory cloud deps** | Only `click`, `pyyaml`, `httpx` required |
| **Human-readable storage** | YAML files by default |
| **Signal over noise** | LLM distillation + ATC verification + power-law decay |
| **Framework-agnostic** | Plain Python objects, no base classes to inherit |

---

## Layer diagram

```
┌─────────────────────────────────────────────────────────────┐
│  CLI (ai_knot.cli)          Integrations (*.integrations) │
├─────────────────────────────────────────────────────────────┤
│                  KnowledgeBase (ai_knot.knowledge)         │  ← public API
├──────────────┬──────────────────┬───────────────────────────┤
│  Extractor   │  BM25Retriever   │  apply_decay / forgetting  │
│  (LLM+ATC)   │  (Okapi BM25)    │  (power-law curve)         │
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

## Core types (`ai_knot.types`)

### `Fact`
The atomic unit of knowledge.

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Human-readable knowledge string |
| `type` | `MemoryType` | `semantic` / `procedural` / `episodic` |
| `importance` | `float` | 0.0–1.0; controls decay speed |
| `retention_score` | `float` | Current retention score (updated on recall) |
| `access_count` | `int` | Times retrieved — increases stability |
| `tags` | `list[str]` | Optional labels |
| `id` | `str` | 8-char UUID hex |
| `created_at` | `datetime` | UTC |
| `last_accessed` | `datetime` | UTC |
| `source_snippets` | `list[str]` | Source text excerpts for provenance |
| `source_spans` | `list[str]` | Span references in source |
| `supported` | `bool` | Whether ATC ≥ threshold (default `True`) |
| `support_confidence` | `float` | Raw ATC score (0.0–1.0) |
| `verification_source` | `str` | `"atc"` / `"manual"` / `"legacy"` |

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

## Forgetting curve (`ai_knot.forgetting`)

Power-law decay (Wixted & Ebbesen, 1997):

```
retention(t) = (1 + t / (9 × stability))^(-1)

stability = 336h × importance × (1 + ln(1 + access_count))
```

- `t` = hours since `last_accessed`
- Base stability = 336 h (2 weeks) for `importance=1.0, access_count=0`
- Power-law has heavier tail than exponential — important facts persist for months
- Applied via `apply_decay(facts)` **before** every retrieval in `KnowledgeBase.recall()`
- After retrieval, `access_count` is incremented and `last_accessed` is updated → reinforcement

---

## Retrieval (`ai_knot.retriever`)

`BM25Retriever.search(query, facts, top_k)` — zero external dependencies.
Returns `list[tuple[Fact, float]]` — each tuple is `(fact, hybrid_score)`.

Okapi BM25 scoring (Robertson & Zaragoza, 2009):
```
BM25(q, d) = Σ_t  IDF(t) × (k1+1)×tf / (k1×(1-b+b×dl/avgdl) + tf)

IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
k1 = 1.5,  b = 0.75
```

P95-clip normalization: raw BM25 scores clipped to 95th percentile, then
normalized to [0, 1] before hybrid blending:
```
hybrid_score = 0.6 × bm25_normalized + 0.2 × retention_score + 0.2 × importance
```

BM25 is computed fresh per query (no index) — suitable for knowledge bases up to ~10k facts.
`TFIDFRetriever` is kept as a backward-compatible alias for `BM25Retriever`.

`KnowledgeBase.recall_facts_with_scores()` exposes `(Fact, float)` pairs to callers that need
relevance scores (e.g. integration adapters). `recall()` and `recall_facts()` unpack the scores
internally and work as before.

---

## Extraction (`ai_knot.extractor`)

`Extractor.extract(turns)` sends the conversation to an LLM and parses the returned JSON array of facts.
Supports `openai` and `anthropic` providers via `httpx`.
Deduplication: Jaccard word-similarity ≥ 0.8 → keep first occurrence.

**ATC verification** (Broder, 1997): after extraction, each fact is verified against
the source text via Asymmetric Token Containment:
```
ATC(snippet, source) = |tokens(snippet) ∩ tokens(source)| / |tokens(snippet)|
```
Facts with ATC < 0.6 are flagged `supported=False` with the exact confidence score
stored in `support_confidence`. This prevents hallucinated facts from entering the
knowledge base while preserving them for optional manual review.

---

## Extension points

| What | Where | Interface |
|---|---|---|
| New storage backend | `src/ai_knot/storage/` | `StorageBackend` protocol |
| New LLM provider | `Extractor._call_<provider>` | Returns `list[dict]` from LLM |
| New integration | `src/ai_knot/integrations/` | Import `KnowledgeBase`, wrap |
| New retriever | Replace `TFIDFRetriever` | `.search(query, facts, top_k)` |
