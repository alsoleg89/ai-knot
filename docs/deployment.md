# Deployment Guide

How to run ai-knot in production: storage, configuration, the MCP server, the
multi-agent pool, observability, and security. For the design rationale see
[ARCHITECTURE.md](../ARCHITECTURE.md); for a feature-by-feature readiness view
see [production-readiness.md](production-readiness.md).

---

## 1. Install

```bash
pip install ai-knot                 # core (sqlite + yaml, in-process BM25)
pip install "ai-knot[postgres]"     # + PostgreSQL backend
pip install "ai-knot[mcp]"          # + MCP server (Claude Desktop / Claude Code)
pip install "ai-knot[openai]"       # + OpenAI provider for learn()/embeddings
pip install "ai-knot[server]"       # + HTTP sidecar
pip install "ai-knot[integrations]" # + CrewAI / AutoGen / OpenAI Agents SDK adapters
```

ai-knot has **no heavy ML dependencies** by default — retrieval is deterministic
BM25 + RRF fusion. Cross-encoder reranking is opt-in behind an extra and is OFF
by default (see `AI_KNOT_RERANK` below).

---

## 2. Choose a storage backend

| Backend | Use for | Durability | Concurrency |
|---|---|---|---|
| `yaml` | tests, demos, human-inspectable state | file | single-process |
| `sqlite` | single-server production, edge | file (WAL) | single-writer |
| `postgres` | multi-process / HA production | server | multi-writer |

```python
from ai_knot import KnowledgeBase
from ai_knot.storage import create_storage

# Development — zero infra:
kb = KnowledgeBase(agent_id="svc", storage=create_storage("sqlite", db_path="/var/lib/ai-knot/mem.db"))

# Production — PostgreSQL:
kb = KnowledgeBase(agent_id="svc", storage=create_storage("postgres", dsn="postgresql://user:pass@host:5432/aiknot"))
```

**PostgreSQL notes**
- All tables are namespaced with the `ai-knot_` prefix, so ai-knot can share a
  database with your application schema.
- Tables are created on first use (`CREATE TABLE IF NOT EXISTS`); no separate
  migration step is required for a fresh deploy.
- Provide the DSN via `AI_KNOT_DSN` (the MCP server requires it when
  `AI_KNOT_STORAGE=postgres`).

---

## 3. Configuration (environment variables)

The MCP server and CLI read configuration from the environment. Secrets (API
keys, DSN) are read but never logged.

### Core
| Variable | Default | Meaning |
|---|---|---|
| `AI_KNOT_STORAGE` | `sqlite` | `yaml` \| `sqlite` \| `postgres` |
| `AI_KNOT_AGENT_ID` | `default` | namespace for this process's facts |
| `AI_KNOT_DATA_DIR` | `.ai_knot` | base dir for file backends |
| `AI_KNOT_DB_PATH` | _(data_dir/mem.db)_ | explicit sqlite path |
| `AI_KNOT_DSN` | — | PostgreSQL DSN (**required** for `postgres`) |
| `AI_KNOT_EPISODIC_TTL` | `24` | episodic-memory time-to-live (hours) |

### LLM (only needed for `learn()` extraction — recall never calls an LLM)
| Variable | Default | Meaning |
|---|---|---|
| `AI_KNOT_PROVIDER` | `openai` | provider name |
| `AI_KNOT_MODEL` | _(provider default)_ | model override |
| `AI_KNOT_API_KEY` / `OPENAI_API_KEY` | — | provider credential |
| `AI_KNOT_LLM_RECALL` | `0` | enable optional LLM recall path |

### Embeddings (optional dense channel; degrades gracefully if unreachable)
| Variable | Default | Meaning |
|---|---|---|
| `AI_KNOT_EMBED_URL` | — | embedding endpoint (e.g. Ollama / OpenAI-compat) |
| `AI_KNOT_EMBED_MODEL` | — | embedding model |
| `AI_KNOT_EMBED_API_KEY` | — | embedding credential |

### Retrieval tuning (advanced — defaults are validated; change with a benchmark)
| Variable | Default | Meaning |
|---|---|---|
| `AI_KNOT_RRF_WEIGHTS` | _(tuned)_ | comma-separated RRF signal weights |
| `AI_KNOT_EXPANSION_WEIGHT` | _(tuned)_ | query-expansion weight |
| `AI_KNOT_DENSE_WEIGHT_MULT` | `1.0` | dense-signal multiplier in RRF |
| `AI_KNOT_RERANK` | `0` | enable cross-encoder rerank (**needs the `rerank` extra**) |
| `AI_KNOT_RERANK_MODEL` / `AI_KNOT_RERANK_N` | — | rerank model / pool size |

> **Cost note.** `learn()` calls an LLM per ingest batch. `recall()` is pure
> retrieval — no LLM, no network (unless the optional dense/rerank channels are
> configured). Keep recall on the hot path and reserve `learn()` for ingest.

---

## 4. Run the MCP server

ai-knot speaks the Model Context Protocol over stdio.

```bash
AI_KNOT_STORAGE=postgres AI_KNOT_DSN=postgresql://... ai-knot-mcp
```

**Claude Desktop / Claude Code** — register it in the MCP config:

```json
{
  "mcpServers": {
    "ai-knot": {
      "command": "ai-knot-mcp",
      "env": {
        "AI_KNOT_STORAGE": "postgres",
        "AI_KNOT_DSN": "postgresql://user:pass@host:5432/aiknot",
        "AI_KNOT_AGENT_ID": "assistant"
      }
    }
  }
}
```

The server exposes tools for `add`, `learn`, `recall` (incl. point-in-time
`recall(now=…)`), `add_resolved`, snapshots, stats, and (when wired) the
multi-agent pool. The TypeScript client (`npm i ai-knot`) is a thin wrapper over
the same tools.

---

## 5. Point-in-time recall (bi-temporal)

A fact records *when its knowledge held* (`event_time` → `valid_from`), and a
superseded fact closes at its successor's event time. So you can ask "what was
true on date D":

```python
from datetime import datetime, UTC
kb.add("User lives in Berlin", event_time=datetime(2023, 1, 1, tzinfo=UTC))
kb.recall("where does the user live", now=datetime(2022, 6, 1, tzinfo=UTC))  # excludes the not-yet-true fact
kb.recall("where does the user live")                                        # live query: current truth
```

Recall **without** `now` always sees the current state — production behaviour is
unchanged by the anchor.

---

## 6. Multi-agent pool (shared memory + governance)

When multiple agents share a memory layer, use the pool. It adds trust scoring,
evidence-gated belief, per-scope read ACLs, and an append-only audit ledger on
top of the same storage backend.

```python
from ai_knot.pool import SharedMemoryPool
pool = SharedMemoryPool(storage=create_storage("postgres", dsn="…"), persist_stats=True)
pool.grant_read("analyst", scope="finance")     # durable ACL grant
pool.publish("writer", fact_ids=[...], kb=kb)    # publish into the shared layer
ctx = pool.recall("analyst", "q3 revenue")       # scope-filtered, trust-weighted
```

- `persist_stats=True` persists trust state, ACL grants, and the event ledger so
  they survive a restart.
- Known-malicious agents (trust below threshold) are discounted even on wide
  recall; quick-invalidation of published slots penalises trust.

---

## 7. Observability

- **Recall trace** — `kb.recall_facts_with_trace(...)` returns the per-stage
  candidate sets (BM25 / rare-token / entity-hop / dense), RRF weights, and the
  selected ids. Log it to debug retrieval.
- **Pool stats** — trust scores, publish/use counts, quick-invalidation counts.
- **Audit ledger** — `trust_events` and `usage_events` tables answer
  "when / why / who-affected-whom" for every trust change and recall use.
- **Health** — assert the version with `ai_knot.__version__`; the three version
  files are CI-guarded to stay in sync.

---

## 8. Backup & snapshots

- **SQLite**: back up the DB file (or use `litestream`/WAL archiving).
- **PostgreSQL**: standard `pg_dump` of the `ai-knot_*` tables.
- **Snapshots**: `kb.snapshot("label")` versions the knowledge base; list and
  diff snapshots to audit state changes over time.

---

## 9. Security checklist

- [ ] Credentials (`AI_KNOT_*_API_KEY`, `AI_KNOT_DSN`) come from a secret manager,
      not the process args — they are read from env and never logged.
- [ ] PostgreSQL role for ai-knot is scoped to its own database/schema.
- [ ] Multi-agent deployments set per-scope ACLs (`grant_read`) and run with
      `persist_stats=True` so grants survive restarts.
- [ ] `AI_KNOT_RERANK` left OFF unless the `rerank` extra is installed and the
      added latency/dependency is acceptable.
- [ ] Network egress for the optional embedding endpoint is allow-listed.

---

## 10. Scaling notes

- `recall()` is CPU-bound and in-process (BM25 + RRF); it scales horizontally —
  run N stateless workers against one PostgreSQL backend.
- The dense and rerank channels add network / CPU; size them against your latency
  budget (see README "Performance").

## 11. HTTP sidecar

For a per-request HTTP surface instead of stdio MCP, run the optional sidecar:

```bash
pip install "ai-knot[server]"
AI_KNOT_SERVER_TOKEN=secret ai-knot --storage postgres --dsn "$AI_KNOT_DSN" serve svc --host 0.0.0.0 --port 8000
```

Routes: `GET /health` (open), `POST /v1/recall` (`{query, top_k, now}` → context +
facts), `POST /v1/facts`, `GET /v1/stats`. When `AI_KNOT_SERVER_TOKEN` is set, the
`/v1/*` routes require `Authorization: Bearer <token>`. Front it with a TLS-
terminating reverse proxy in production.
