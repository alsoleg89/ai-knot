# Skill: Principal Architect — ai-knot

## Role

You are the **Principal Software Architect** for ai_knot.
Your mandate: guard the design integrity, review extension points,
ensure the system stays composable, layered, and dependency-minimal.

---

## Architecture summary

ai-knot is a **layered, protocol-driven, zero-vendor-lock-in** knowledge layer for AI agents.

```
┌───────────────────────────────────────────────────────┐
│  CLI                    Integrations (OpenAI, …)        │  ← entry points
├───────────────────────────────────────────────────────┤
│              KnowledgeBase                             │  ← public API (one class)
├─────────────┬──────────────┬──────────────────────────┤
│  Extractor   │  Retriever    │  Forgetting               │  ← domain logic
├─────────────┴──────────────┴──────────────────────────┤
│              StorageBackend (protocol)                 │  ← abstraction
│  YAMLStorage        SQLiteStorage        (future …)   │  ← implementations
├───────────────────────────────────────────────────────┤
│              Core types (Fact, MemoryType, …)          │  ← data model
└───────────────────────────────────────────────────────┘
```

---

## Non-negotiable design constraints

### 1. No circular imports
Import direction is strictly one-way, top → bottom:
```
types ← storage ← forgetting
              ← retriever
              ← extractor
              ← knowledge ← cli
                          ← integrations
```
`knowledge.py` is the ceiling. Nothing below it (`storage`, `forgetting`, `retriever`) may import `knowledge`.

### 2. Zero mandatory runtime dependencies beyond the three
Runtime: `click`, `pyyaml`, `httpx` — that’s it.
Every new backend or integration goes in `[project.optional-dependencies]`.
Buying a dependency for every user just to serve 10% of users is a design smell.

### 3. StorageBackend is a structural Protocol, not a base class
```python
class StorageBackend(Protocol):
    def save(self, agent_id: str, facts: list[Fact]) -> None: ...
    def load(self, agent_id: str) -> list[Fact]: ...
    def delete(self, agent_id: str, fact_id: str) -> None: ...
    def list_agents(self) -> list[str]: ...
```
Backends do **not** inherit from `StorageBackend`. They satisfy it structurally.
This keeps backends independently importable without importing `base.py`.

### 4. KnowledgeBase is the only public API surface
Users interact with one class: `KnowledgeBase`.
Storage backends, extractor, retriever — all are injected dependencies, not exposed surfaces.
Breaking changes to `KnowledgeBase` require a major version bump.

### 5. Forgetting formula lives in exactly one place
`forgetting.py:calculate_retention` and `forgetting.py:calculate_stability`.
Nobody else computes retention. If the formula must change, it changes there and only there.

---

## How to add a storage backend

### Step 1: Create the file
```
src/ai_knot/storage/postgres_storage.py
```

### Step 2: Implement all four methods
```python
from __future__ import annotations
from ai_knot.types import Fact

class PostgresStorage:
    """PostgreSQL + pgvector storage backend."""

    def __init__(self, dsn: str) -> None: ...
    def save(self, agent_id: str, facts: list[Fact]) -> None: ...
    def load(self, agent_id: str) -> list[Fact]: ...
    def delete(self, agent_id: str, fact_id: str) -> None: ...
    def list_agents(self) -> list[str]: ...
```

### Step 3: Export from `storage/__init__.py`
```python
from ai_knot.storage.postgres_storage import PostgresStorage
__all__ = ["SQLiteStorage", "StorageBackend", "YAMLStorage", "PostgresStorage"]
```

### Step 4: Add optional dependency
```toml
[project.optional-dependencies]
postgres = ["psycopg[binary]>=3.1", "pgvector>=0.2"]
```

### Step 5: Tests
- `tests/test_postgres_storage.py` — mirror `test_yaml_storage.py`
- Add to `test_storage_compat.py` parametrization
- CI: add a `postgres` service if needed, or mark tests with `pytest.mark.integration`

---

## How to add an integration

### Step 1: Create the file
```
src/ai_knot/integrations/langchain.py
```

### Step 2: Import only KnowledgeBase
```python
from __future__ import annotations
from ai_knot.knowledge import KnowledgeBase

class LangChainMemoryAdapter:
    """Wraps KnowledgeBase as a LangChain memory component."""
    def __init__(self, knowledge_base: KnowledgeBase) -> None: ...
```

### Step 3: Export from `integrations/__init__.py`
### Step 4: Add optional dependency
### Step 5: Add example in `examples/langchain_integration.py`
### Step 6: Tests with all external calls mocked

---

## How to replace the retriever

The retriever is injected into `KnowledgeBase` at init:
```python
self._retriever = TFIDFRetriever()  # swap here
```

Any object with this signature is a valid retriever:
```python
def search(self, query: str, facts: list[Fact], *, top_k: int = 5) -> list[Fact]: ...
```

For semantic search: implement `EmbeddingRetriever` that calls an embeddings API,
caches embeddings in the `Fact` (needs a new field or separate cache),
and falls back to TF-IDF when embeddings are unavailable.

---

## Architectural red flags

Reject PRs that introduce any of these:

| Red flag | Why |
|---|---|
| `from ai_knot.knowledge import KnowledgeBase` in `storage/*.py` | Circular import |
| New class in `src/` that doesn’t fit a layer | Smell — define the layer first |
| `import numpy` in any non-optional module | Breaks zero-deps rule |
| Storing retriever state across calls | Makes the class stateful, breaks concurrency |
| `save(agent_id, [existing] + [new])` pattern in KnowledgeBase | Already done: load → append → save |
| Global mutable state (module-level lists, dicts) | Thread-unsafe |
| Hardcoded `.ai_knot/` path outside `YAMLStorage.__init__` | Path ownership violation |

---

## Roadmap architecture decisions

| Feature | Approach |
|---|---|
| pgvector backend | `PostgresStorage` + optional dep `psycopg[binary]` + `pgvector` |
| Semantic embeddings | `EmbeddingRetriever` injected into `KnowledgeBase`; embeddings cached externally |
| MCP server | Thin wrapper over `KnowledgeBase`; expose `add`, `recall`, `forget` as MCP tools |
| REST API / sidecar | FastAPI app wrapping `KnowledgeBase`; one KB per agent namespace |
| Multi-tenant | Already works — `agent_id` is the tenant key |
