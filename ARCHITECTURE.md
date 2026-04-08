# ai-knot — Architecture

## Design goals

| Goal | How |
|---|---|
| **Zero vendor lock-in** | Pluggable `StorageBackend` protocol |
| **Zero mandatory cloud deps** | Only `click`, `pyyaml`, `httpx` required |
| **Human-readable storage** | YAML files by default; SQLite for production |
| **Signal over noise** | LLM distillation + ATC verification + power-law decay |
| **Framework-agnostic** | Plain Python objects, no base classes to inherit |

---

## Layer diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  CLI (ai_knot.cli)     MCP Server     Integrations (OpenClaw, …)    │
├──────────────────────────────────────────────────────────────────────┤
│  KnowledgeBase  ←→  SharedMemoryPool                                 │  ← public API
│  (learn/recall)      (publish/recall/promote/gc_pool)                │
├──────────┬───────────────────┬───────────────────────────────────────┤
│ Extractor│ Retrievers        │ Forgetting + ConflictPolicy            │
│ (LLM+ATC)│ BM25Retriever    │ apply_decay (power-law)                │
│ tri-surf │ DenseRetriever   │ SlotStateMachinePolicy (SEMANTIC)      │
│ slot norm│ HybridRetriever  │ ProcedureStabilityPolicy (PROCEDURAL)  │
│          │ Intent Planner   │ EpisodicTimelinePolicy (EPISODIC)      │
├──────────┴───────────────────┴───────────────────────────────────────┤
│  StorageBackend (protocol)        EvidenceStore (protocol)            │
│  YAML  SQLite  Postgres           InlineEvidenceStore (default)       │
│  TemporalStorageCapable ← AtomicUpdateCapable ← SnapshotCapable     │
├──────────────────────────────────────────────────────────────────────┤
│  Core types: Fact, MemoryType, MESIState, SlotDelta, Provenance,     │
│  ConflictPolicy, Evidence, _RecallMeta, _PoolQueryIntent             │
└──────────────────────────────────────────────────────────────────────┘
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

The atomic unit of knowledge. All fields are persisted by storage backends except `op`.

| Field | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | required | Human-readable knowledge string |
| `type` | `MemoryType` | `semantic` | `semantic` / `procedural` / `episodic` |
| `importance` | `float` | `0.8` | Criticality 0.0–1.0; controls decay speed |
| `retention_score` | `float` | `1.0` | Current memory strength after decay |
| `access_count` | `int` | `0` | Times retrieved — increases stability |
| `tags` | `list[str]` | `[]` | Domain labels; BM25F field weight ×2.0 |
| `id` | `str` | auto | 8-char UUID hex |
| `created_at` | `datetime` | now UTC | Immutable creation timestamp |
| `last_accessed` | `datetime` | now UTC | Updated on each recall |
| `source_snippets` | `list[str]` | `[]` | Raw excerpts that support this fact |
| `source_spans` | `list[str]` | `[]` | Location references for each snippet |
| `supported` | `bool` | `True` | ATC score ≥ 0.6 (fact is grounded) |
| `support_confidence` | `float` | `1.0` | Raw ATC score |
| `verification_source` | `str` | `"manual"` | `"atc"` / `"manual"` / `"legacy"` |
| `access_intervals` | `list[float]` | `[]` | Hours between consecutive accesses |
| `origin_agent_id` | `str` | `""` | Agent that created this fact |
| `visibility` | `str` | `"private"` | Storage namespace tag (internal) |
| `source_verbatim` | `str` | `""` | Exact original phrase before LLM normalisation |
| `valid_from` | `datetime` | now UTC | When this version became active |
| `valid_until` | `datetime \| None` | `None` | When superseded; `None` = currently active |
| `entity` | `str` | `""` | Subject of the fact, e.g. `"jordan lee"` (lowercase) |
| `attribute` | `str` | `""` | Property being described, e.g. `"salary"` (lowercase) |
| `version` | `int` | `0` | Monotonic counter; incremented on each supersession |
| `mesi_state` | `MESIState` | `E` | Cache coherence state (see `MESIState`) |
| `canonical_surface` | `str` | `""` | Normalised form for BM25 indexing |
| `witness_surface` | `str` | `""` | Verbatim evidence excerpt for grounding |
| `prompt_surface` | `str` | `""` | Compact form injected into LLM prompts |
| `slot_key` | `str` | `""` | `"{entity}::{attribute}"` — deterministic dedup key |
| `value_text` | `str` | `""` | Extracted value, e.g. `"95000"` or `"Python 3.12"` |
| `qualifiers` | `dict[str, str]` | `{}` | Temporal / conditional modifiers |
| `state_confidence` | `float` | `1.0` | Confidence this reflects the current state |
| `topic_channel` | `str` | `""` | Domain routing label for shared pool, e.g. `"devops"` |
| `visibility_scope` | `str` | `"global"` | `"global"` = all agents, `"local"` = owning agent only |
| `op` | `MemoryOp` | `ADD` | Extraction intent — not persisted by storage |
| `memory_tier` | `str` | `"private"` | `"private"` / `"pool"` / `"org"` — internal tier for multi-agent promotion |

### `MemoryType`

```python
SEMANTIC   — stable facts about the world or user
PROCEDURAL — how the user wants things done
EPISODIC   — specific past events with time context
```

Decay exponents by type: `semantic=0.8`, `procedural=1.0`, `episodic=1.3` (Tulving 1972).
Configurable via `KnowledgeBase(decay_config={"episodic": 2.0})`.

### `MESIState`

MESI cache coherence protocol adapted for shared pool invalidation:

```python
E (Exclusive) — single private owner, no coordination needed (default on add())
S (Shared)    — published to pool; no prior fact for this slot existed
M (Modified)  — replaced an existing active fact via slot-addressed CAS
I (Invalid)   — superseded; valid_until is set; not returned by active queries
```

### `MemoryOp`

Extraction intent set by `Extractor._parse_fact()`. Never persisted.

```python
ADD     — insert new fact (default)
UPDATE  — force supersede of existing slot value
DELETE  — close matching fact without inserting a replacement
NOOP    — conversation confirms existing knowledge; skip
```

### `SlotDelta`

Lightweight change record used by `sync_slot_deltas()`:

| Field | Type | Description |
|---|---|---|
| `slot_key` | `str` | `"{entity}::{attribute}"` |
| `version` | `int` | Version counter at time of change |
| `op` | `str` | `"new"` / `"supersede"` / `"invalidate"` |
| `fact_id` | `str` | ID of the new (or invalidated) fact |
| `content` | `str` | Human-readable content |
| `prompt_surface` | `str` | Compact prompt surface (may be empty) |

### `ConflictPolicy` (protocol)

Per-`MemoryType` strategy for conflict resolution and decay:

```python
class ConflictPolicy(Protocol):
    def should_supersede(new_fact, existing) -> bool   # Policy-driven CAS override
    def decay_immune(fact) -> bool                      # Skip retention decay?
    def ttl_seconds(fact) -> float | None               # Time-to-live, or None
```

| Policy | Type | Supersedes? | Decay immune? | TTL |
|---|---|---|---|---|
| `SlotStateMachinePolicy` | SEMANTIC | Yes (different value) | No | None |
| `ProcedureStabilityPolicy` | PROCEDURAL | Yes (different value) | **Yes** | None |
| `EpisodicTimelinePolicy` | EPISODIC | **Never** (coexist) | No | 7 days (high-importance exempt) |

Wired into `_resolve_phase()` (slot CAS policy override) and `apply_decay()` (procedural immunity).
Explicit `op=UPDATE` bypasses the policy (user intent is authoritative).

### `Provenance`

Immutable lineage record for a fact:

| Field | Type | Description |
|---|---|---|
| `origin_agent` | `str` | Agent that first created the fact |
| `origin_turn` | `int` | Conversation turn index (-1 if unknown) |
| `published_by` | `str` | Agent that published to pool |
| `promoted_by` | `str` | Agent that promoted to higher tier |
| `supersedes_id` | `str` | ID of the fact this one replaced via CAS |
| `consolidation_ids` | `tuple[str, ...]` | Episodic facts consolidated into this one |

### `Evidence`

Source material record linked to a Fact:

| Field | Type | Description |
|---|---|---|
| `fact_id` | `str` | Canonical fact this evidence supports |
| `snippets` | `list[str]` | Raw source text snippets |
| `spans` | `list[str]` | Character spans in original document |
| `verbatim` | `str` | Exact source quote |
| `support_confidence` | `float` | How strongly evidence supports the fact |

`EvidenceStore` protocol provides the seam for future physical separation.
Default `InlineEvidenceStore` reads evidence from Fact fields (zero cost).

---

## Storage layer

### `StorageBackend` protocol

Minimal interface all backends must implement:

```python
def save(agent_id, facts)  -> None    # full replace — load, mutate, save
def load(agent_id)         -> list[Fact]
def delete(agent_id, id)   -> None
def list_agents()          -> list[str]
```

`save` is a **full replace** — the caller always loads, mutates, then saves.
Backends namespace by `agent_id`; no two agents share rows.

### `TemporalStorageCapable` (optional)

Index-accelerated extensions used by `SharedMemoryPool` and `KnowledgeBase.recall()`:

```python
load_active(agent_id)                      # WHERE valid_until IS NULL
load_since_version(agent_id, since, excl)  # dirty pull for MESI sync
load_active_frontier(agent_id)             # latest active fact per slot_key
load_slot_deltas_since(agent_id, since, excl)  # lightweight delta pull
save_atomic(agent_id, facts)               # single-writer atomicity
```

### `AtomicUpdateCapable` (optional)

Cross-process safe load→transform→save:

```python
def atomic_update(agent_id, fn: Callable[[list[Fact]], list[Fact]]) -> None
```

`SharedMemoryPool._publish_locked()` checks `isinstance(storage, AtomicUpdateCapable)` at
runtime and dispatches to `atomic_update()` when available. `SQLiteStorage` implements this
using `BEGIN EXCLUSIVE` — the entire load→merge→save cycle holds a database-level exclusive
lock, preventing lost updates from concurrent processes sharing the same `.db` file.

YAML and PostgreSQL backends fall back to `save_atomic()` (statement-level only).

### `SnapshotCapable` (optional)

Named checkpoints for rollback and diff:

```python
save_snapshot(agent_id, name, facts)  -> None
load_snapshot(agent_id, name)         -> list[Fact]
list_snapshots(agent_id)              -> list[str]
delete_snapshot(agent_id, name)       -> None
```

### YAMLStorage

File layout: `{base_dir}/{agent_id}/knowledge.yaml`
Key: `fact.id` (8-char hex). Human-readable, Git-trackable, editable by hand.
Snapshots stored under `{base_dir}/{agent_id}/snapshots/{name}.yaml`.

Does **not** implement `TemporalStorageCapable` or `AtomicUpdateCapable` —
temporal filtering and atomic updates are done in Python, not SQL.

### SQLiteStorage

Single `.db` file shared by all agents. Schema (abbreviated):

```sql
CREATE TABLE facts (
    id                TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    content           TEXT NOT NULL,
    type              TEXT NOT NULL DEFAULT 'semantic',
    importance        REAL NOT NULL DEFAULT 0.8,
    retention         REAL NOT NULL DEFAULT 1.0,
    access_count      INTEGER NOT NULL DEFAULT 0,
    tags              TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL,
    last_accessed     TEXT NOT NULL,
    -- provenance
    origin_agent_id   TEXT NOT NULL DEFAULT '',
    visibility        TEXT NOT NULL DEFAULT 'private',
    source_verbatim   TEXT NOT NULL DEFAULT '',
    -- temporal validity
    valid_from        TEXT NOT NULL DEFAULT '',
    valid_until       TEXT,                          -- NULL = active
    -- structured addressing
    entity            TEXT NOT NULL DEFAULT '',
    attribute         TEXT NOT NULL DEFAULT '',
    version           INTEGER NOT NULL DEFAULT 0,
    mesi_state        TEXT NOT NULL DEFAULT 'E',
    -- tri-surface retrieval
    canonical_surface TEXT NOT NULL DEFAULT '',
    witness_surface   TEXT NOT NULL DEFAULT '',
    prompt_surface    TEXT NOT NULL DEFAULT '',
    -- slot addressing
    slot_key          TEXT NOT NULL DEFAULT '',
    value_text        TEXT NOT NULL DEFAULT '',
    qualifiers        TEXT NOT NULL DEFAULT '{}',
    state_confidence  REAL NOT NULL DEFAULT 1.0,
    -- routing
    topic_channel     TEXT NOT NULL DEFAULT '',
    visibility_scope  TEXT NOT NULL DEFAULT 'global',
    PRIMARY KEY (agent_id, id)
);

CREATE INDEX idx_facts_valid   ON facts(agent_id, valid_until);
CREATE INDEX idx_facts_entity  ON facts(agent_id, entity, attribute);
CREATE INDEX idx_facts_version ON facts(agent_id, version);

CREATE TABLE snapshots (
    agent_id  TEXT NOT NULL,
    name      TEXT NOT NULL,
    data      TEXT NOT NULL,  -- JSON array of serialised facts
    PRIMARY KEY (agent_id, name)
);
```

Implements `TemporalStorageCapable`, `AtomicUpdateCapable`, and `SnapshotCapable`.
`atomic_update()` opens a dedicated connection with `BEGIN EXCLUSIVE`, runs the
callback, then commits — independent of the pool's normal read connections.

### PostgresStorage

Stores facts in a single `"ai-knot_facts"` table (same schema as SQLite, adapted for PG types).
Implements `TemporalStorageCapable`. `save_atomic()` uses a single PostgreSQL transaction
(`READ COMMITTED` isolation — statement-level atomicity, not cross-process CAS).
For cross-process serializable writes, use advisory locks externally.

---

## SharedMemoryPool (`ai_knot.knowledge`)

`SharedMemoryPool` is a shared knowledge space that multiple `KnowledgeBase` agents
publish facts into and recall from. It lives in the same `StorageBackend` as the agents,
using the reserved namespace `"__shared__"`.

### Registration

```python
pool = SharedMemoryPool(storage=SQLiteStorage("shared.db"))
pool.register("agent_a")   # must be called before publish / recall
pool.register("agent_b")
```

### Publish and slot-addressed CAS

`pool.publish(agent_id, fact_ids, kb=..., utility_threshold=0.5)`:

1. Loads candidate facts from the agent's private KB.
2. Filters by utility gate: `state_confidence × importance ≥ utility_threshold`.
3. Acquires `_publish_lock` (in-process thread safety).
4. Dispatches to `_publish_locked()`:
   - If `storage` is `AtomicUpdateCapable` → `storage.atomic_update("__shared__", _merge)`.
   - Otherwise → load + `_merge()` + `save_atomic()` / `save()`.
5. `_merge(shared_facts)` performs slot-addressed CAS per fact:
   - **With `slot_key`**: look up active fact for that slot → close it (`valid_until=now`, `mesi_state=I`), increment `version`, set new `mesi_state=M` (modified) or `S` (new).
   - **Without `slot_key`**: fall back to ID-based dedup.
6. Applies optional `visibility_scope` and `topic_channel` filters.
7. Updates `_publish_count[agent_id]` and trust side-channel `_quick_inv_count`.

### Trust model

Auto-trust is computed from observed behaviour — no manual configuration:

```
trust(agent) = max(0.1, quality × (1 − inv_penalty))
quality      = min(1.0, (used + PRIOR_WEIGHT × 0.8) / (published + PRIOR_WEIGHT))
inv_penalty  = quick_inv_count / published
```

Bayesian prior (`PRIOR_WEIGHT=3`) prevents untested agents from starting at trust=0.1.
`_used_count` is incremented **only for facts in the final top_k result**, not for
overfetch candidates — preventing inflated denominators from wide `top_k` values.

### Tier-aware pool lifecycle

- **Memory tiers**: `"private"` → `"pool"` (via publish) → `"org"` (via promote/auto)
- **Auto-promotion**: facts consumed by ≥3 distinct agents auto-promote to `"org"` tier
- **Tier scoring**: `"org"` facts get 1.05× multiplicative boost in recall
- **Pool TTL**: `gc_pool()` expires `"pool"` facts unused for 30 days; `"org"` exempt
- **Coverage hook**: `_last_recall_meta` tracks coverage/intent (internal, not public API)

### Delta sync

`pool.sync_slot_deltas(agent_id)` returns lightweight `SlotDelta` records for slots
that changed since the agent's last sync version. Token transfer is typically < 15%
of a full fact sync. Agents apply deltas to their private KB via `apply_slot_deltas()`.

---

## Forgetting curve (`ai_knot.forgetting`)

Power-law decay (Wixted & Ebbesen, 1997) with type-aware exponents (Tulving 1972):

```
retention(t) = (1 + t / (9 × stability))^(-decay_exp)

stability = 336h × importance × (1 + ln(1 + access_count))
decay_exp = { semantic: 0.8, procedural: 1.0, episodic: 1.3 }
```

- `t` = hours since `last_accessed`
- Base stability = 336 h (2 weeks) at `importance=1.0, access_count=0`
- Power-law has a heavier tail than exponential — important facts persist for months
- Exponents configurable via `KnowledgeBase(decay_config={"episodic": 2.0})`
- Applied automatically before every `recall()` call; reinforces accessed facts

---

## Retrieval (`ai_knot.retriever`)

`BM25Retriever.search(query, facts, top_k, expansion_weights)` returns `list[tuple[Fact, float]]`.

### Scoring pipeline

1. **BM25F** — dual-field (content weight 1.0, tags weight 2.0). High-DF filter: terms in
   >70% of docs get zero IDF. Uses `canonical_surface` when non-empty.

2. **PRF** (Pseudo-Relevance Feedback) — top-3 BM25 docs expand the query with up to 5
   feedback terms at weight 0.5. Skipped for corpora < 4 documents.

3. **LLM expansion** (opt-in) — 2-4 LLM synonyms merged with PRF at weight 0.4.
   Original query tokens keep weight 1.0. LRU-cached (128 entries).

4. **Slot-exact ranker** — facts whose `slot_key` matches the query's inferred slot get a
   +1.0 bonus, ensuring the most recent value for a given attribute always surfaces first.

5. **Char-trigram ranker** — Jaccard similarity on character trigrams closes the semantic
   gap for paraphrases and misspellings without embeddings.

6. **RRF** (Reciprocal Rank Fusion) — combines 6 ranked lists: BM25, slot-exact, trigram,
   importance, retention, recency. Default weights `(5.0, 3.0, 2.0, 1.5, 1.5, 1.0)`,
   configurable via `KnowledgeBase(rrf_weights=(...))`.
   Per-call `rrf_weights` override supported for intent-aware retrieval.

### Dense and hybrid retrieval

`DenseRetriever` — cosine similarity search over precomputed embeddings. Call
`set_embeddings(vectors)` to load precomputed vectors (from `embedder.py`).

`HybridRetriever` — fuses BM25 and dense retrieval via RRF. Falls back to BM25-only
when no embeddings are available (seamless upgrade path). Default fusion weights:
BM25=2.0, dense=1.0.

### Intent-aware retrieval planner

`_classify_pool_query(query, active_facts)` classifies queries into four intents:

| Intent | RRF Override | Rerank | Post-filter |
|---|---|---|---|
| `ENTITY_LOOKUP` | slot-exact → 8.0 | default | — |
| `INCIDENT` | recency → 3.0 | recency=0.12 | — |
| `CROSS_DOMAIN` | default | default | agent cap ceil(top_k×0.6) |
| `GENERAL` | default | default | — |

Wired into both `SharedMemoryPool.recall()` and `KnowledgeBase._execute_recall()`.

### Tri-surface retrieval

Each fact carries three surfaces for separate concerns:

| Surface | Field | Purpose |
|---|---|---|
| `canonical_surface` | BM25 indexing | Normalised form for dedup and search |
| `witness_surface` | Provenance | Verbatim evidence grounding the fact |
| `prompt_surface` | Output | Compact text injected into system prompts |

`recall()` and `recall_facts_with_scores()` return `prompt_surface` when non-empty,
falling back to `content`. BM25 always indexes `canonical_surface`.

---

## Extraction (`ai_knot.extractor`)

`Extractor.extract(turns)` sends the conversation to an LLM and parses the JSON array.
Supports `openai`, `anthropic`, `gigachat`, `yandex`, `qwen`, `openai-compat` providers.

### ATC verification (Broder, 1997)

Every extracted fact is verified against the source text:

```
ATC(snippet, source) = |tokens(snippet) ∩ tokens(source)| / |tokens(snippet)|
```

Facts with ATC < 0.6 are flagged `supported=False`; `support_confidence` stores the raw
score. Hallucinated facts are preserved for manual review but do not pollute active recall.

### Tri-surface generation

The extraction prompt instructs the LLM to emit three surfaces per fact:
- `canonical_surface` — standardised phrasing for indexing
- `witness_surface` — verbatim excerpt from the conversation
- `prompt_surface` — one-line summary for system prompt injection

When surfaces are absent (older prompts, non-LLM insertion), `content` is used as fallback.

### Slot key normalisation

`_parse_fact()` forces `entity` and `attribute` to lowercase and stripped before computing
`slot_key = f"{entity}::{attribute}"`. This prevents case-sensitivity drift from LLM output
creating duplicate slots for the same conceptual attribute.

```python
entity    = entry.get("entity", "").strip().lower()   # "Jordan Lee" → "jordan lee"
attribute = entry.get("attribute", "").strip().lower() # "Salary" → "salary"
slot_key  = f"{entity}::{attribute}" if entity and attribute else ""
```

---

## Extension points

| What | Where | Interface |
|---|---|---|
| New storage backend | `src/ai_knot/storage/` | `StorageBackend` protocol; optionally also `TemporalStorageCapable`, `AtomicUpdateCapable`, `SnapshotCapable` |
| New evidence store | `ai_knot.types` | `EvidenceStore` protocol: `get_evidence()`, `save_evidence()`, `delete_evidence()` |
| New conflict policy | `ai_knot.types` | `ConflictPolicy` protocol + add to `CONFLICT_POLICIES` registry |
| New LLM provider | `Extractor._call_<provider>` | Returns `list[dict]` from LLM |
| New integration | `src/ai_knot/integrations/` | Import `KnowledgeBase`, wrap |
| Custom retriever | `HybridRetriever(bm25, dense)` | BM25 + dense fusion with graceful fallback |
| Custom decay | `KnowledgeBase(decay_config=...)` | Dict of `MemoryType.value → exponent` |
