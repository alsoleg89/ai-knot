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

Power-law decay (Wixted & Ebbesen, 1997) with type-aware exponents (Tulving 1972):

```
retention(t) = (1 + t / (9 × stability))^(-decay_exp)

stability = 336h × importance × (1 + ln(1 + access_count))
decay_exp = { semantic: 0.8, procedural: 1.0, episodic: 1.3 }
```

- `t` = hours since `last_accessed`
- Base stability = 336 h (2 weeks) for `importance=1.0, access_count=0`
- Decay exponent varies by memory type: semantic facts decay slower, episodic faster
- Power-law has heavier tail than exponential — important facts persist for months
- Exponents are configurable via `KnowledgeBase(decay_config={"semantic": 0.5})`
- Applied via `apply_decay(facts)` **before** every retrieval in `KnowledgeBase.recall()`
- After retrieval, `access_count` is incremented and `last_accessed` is updated → reinforcement

---

## LLM-enhanced features

ai-knot follows a **base + enhanced** pattern: core features work without an LLM,
but when a provider is configured, additional capabilities activate automatically.

| Feature | Base (no LLM) | Enhanced (with LLM) |
|---------|---------------|---------------------|
| Tags | User-supplied via `add(tags=[...])` | Auto-generated during `learn()` |
| Decay config | Hardcoded defaults | `decay_config={}` (no LLM needed) |
| Query expansion | Raw query → BM25 | `llm_recall=True` expands with weighted synonyms |
| Stemming | English Porter subset | + Cyrillic (Russian) Snowball-lite |
| RRF weights | Default `(5.0, 2.0, 2.0, 1.0)` | `rrf_weights=(...)` tunable |
| Clock injection | `now=None` (real time) | `now=datetime(...)` for testing |

### Auto-tagging

The extraction prompt includes `"tags"` in the JSON schema. The LLM generates
1-3 domain tags per fact during `learn()`. Tags activate BM25F field weighting
(`_W_TAGS=2.0`). No extra LLM calls — piggybacks on the existing extraction call.

### Query expansion (`ai_knot.query_expander`)

`LLMQueryExpander.expand(query)` adds 2-4 synonyms before BM25 search.
Opt-in via `KnowledgeBase(llm_recall=True)`. LRU cache (128 entries) avoids
repeated calls.

v0.8 changes: expansion tokens now receive weight 0.4 via `expansion_weights`
(original query tokens keep 1.0). This prevents expansion from diluting the
original query signal. LLM expansion and PRF expansion are merged — LLM tokens
take priority for overlapping terms. The expansion prompt is multilingual
(keeps the same language as the input query).

### Cyrillic stemmer (`ai_knot.tokenizer`)

v0.8 adds a zero-dependency Russian stemmer using a Snowball-lite algorithm.
The tokenizer auto-detects script via Unicode block check (`\u0400`–`\u04ff`)
and dispatches to `_stem_ru()` (Cyrillic) or `_stem_en()` (Latin).

This provides **symmetric normalization** at both index and query time —
critical for BM25 to match morphological variants (e.g. "запрещённые" and
"запрещённых" → same stem "запреще").

---

## Retrieval (`ai_knot.retriever`)

`BM25Retriever.search(query, facts, top_k, expansion_weights)` — zero
external dependencies. Returns `list[tuple[Fact, float]]` — each tuple is
`(fact, rrf_score)`.

Okapi BM25 scoring (Robertson & Zaragoza, 2009):
```
BM25(q, d) = Σ_t  IDF(t) × (k1+1)×tf / (k1×(1-b+b×dl/avgdl) + tf)

IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)
k1 = 1.5,  b = 0.75
```

Scoring pipeline:
1. **BM25F** — dual-field scoring (content weight 1.0, tags weight 2.0).
   High-DF filter: terms in >70% of docs get zero IDF.
2. **PRF** (Pseudo-Relevance Feedback) — top-3 docs expand query with
   up to 5 feedback terms at weight 0.5. Skipped for corpora < 4 docs.
3. **LLM expansion** (optional) — merged with PRF; LLM tokens weight 0.4.
4. **RRF** (Reciprocal Rank Fusion) — combines 4 ranked lists:
   BM25, importance, retention, recency. Default weights `(5.0, 2.0, 2.0, 1.0)`,
   configurable via `BM25Retriever(rrf_weights=(...))`.

BM25 is computed fresh per query (inverted index rebuilt each call) —
suitable for knowledge bases up to ~10k facts.
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
