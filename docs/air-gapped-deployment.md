# Air-gapped and regulated deployment

Updated: **July 5, 2026**

ai-knot is built for environments where data cannot leave the enclave — defense,
healthcare, national labs, and EU financial services under DORA / NIS2. The
deterministic write and read path makes **zero outbound network calls**, and that
is something you can verify rather than take on faith.

This page is written for the engineer who has to hand a security team a straight
answer, and for the auditor who will check it.

---

## The guarantee

The deterministic loop — `add`, `add_resolved`, `recall`, and the shared-agent
`SharedMemoryPool` publish/recall with its audit ledger — opens **no network
connection**. No telemetry, no phone-home, no version check, no license ping.

**Verify it yourself.** The guarantee is enforced by a test that blocks *all*
outbound sockets and then runs the full loop
([`tests/test_air_gapped.py`](../tests/test_air_gapped.py)):

```bash
pytest tests/test_air_gapped.py -q
```

If any code path tried to reach the network, that test fails instead of silently
phoning home. It is part of the suite, so the guarantee cannot regress unnoticed.

## What can touch the network (the complete list)

Only three components in ai-knot ever open an outbound connection, and none is on
the deterministic path:

| Component | When it connects | How to stay offline |
|---|---|---|
| **LLM providers** (`ai_knot.providers.*`) | only the opt-in `learn()` extraction and the optional semantic conflict resolver | use `add` / `add_resolved` (direct insertion, no model), or point `learn()` at a **local** model (Ollama / vLLM) |
| **Dense embedder** (`ai_knot.embedder`) | on recall, unless disabled — it targets `AI_KNOT_EMBED_URL` (default `localhost:11434`) | set `AI_KNOT_EMBED_URL=""` (the container's default) → deterministic BM25 recall, no connection; or point it at a **local** embeddings server |
| **PostgreSQL driver** | only if you choose the Postgres backend | it connects to **your** database; or use SQLite (a local file) or YAML |

SQLite and YAML are local file I/O and open no socket. There is no fourth thing.

## Install with no network

- **Core install:** `pip install ai-knot`. Runtime dependencies are `click`,
  `pyyaml`, and `httpx` — `httpx` is only exercised if you actually call a provider
  or the embedder; importing ai-knot makes no network call.
- **Fully offline (wheelhouse):** on a connected machine,
  `pip download ai-knot -d wheelhouse/`, transfer the folder into the enclave, then
  `pip install --no-index --find-links wheelhouse/ ai-knot`.
- **Container:** build the image from the repo [`Dockerfile`](../Dockerfile) on a
  connected box, then move it in with `docker save ai-knot | ...` / `docker load`.
  The image runs the HTTP sidecar locally so non-Python hosts (Node/TS, other
  services) can use the same deterministic core over `localhost` with no egress. It
  defaults to `AI_KNOT_EMBED_URL=""` — BM25-only recall, no outbound connection at
  all; set it to a reachable endpoint only if you want the dense channel.

## Storage under your control

- **SQLite** — a single local file; the default for a single server.
- **PostgreSQL** — your own cluster; tables are namespaced with an `ai-knot_`
  prefix so ai-knot coexists with your application schema.
- **YAML** — human-readable, for inspection and low-friction dev.

There is no managed service and no vendor data plane. The store is yours.

## The audit trail, produced offline

A `SharedMemoryPool` created with `persist_stats=True` writes a durable, append-only
audit ledger — trust changes, fact-usage events, and access-control grants — with no
network involved. Timestamps come from an injectable clock, so audited runs are
reproducible. Hand an auditor the whole ledger as JSON without writing code:

```bash
ai-knot --storage sqlite --data-dir ./data audit-export -o audit.json
```

See [multi-agent-governance.md](multi-agent-governance.md) for what each field means.

## Honest boundaries

The wedge is honesty, so the limits are stated plainly:

- **`learn()` (LLM fact extraction) needs a model.** Run a local one, or skip it and
  use `add` / `add_resolved` — the recall path never needs an LLM.
- **Dense recall is opt-in.** Deterministic BM25 recall needs no embeddings at all;
  turn on `embed_url` only if you run a local embeddings server and want the dense
  channel.
- **PostgreSQL connects to your database.** That is your infrastructure, not a third
  party — but it is a network connection, so SQLite is the strictest single-box path.
- **ACL enforcement is at the recall layer** (application-level), not database
  row-level security. For hard multi-tenant isolation, pair ai-knot scopes with your
  database's own RLS. See [multi-agent-governance.md](multi-agent-governance.md).

## When this matters

Anywhere "no data leaves the enclave" is a hard requirement rather than a
preference. The property that makes air-gapped *ingest* possible at all is **no LLM
on the write path** — most memory libraries must send your data to a model to
populate memory. See [positioning.md](positioning.md) and
[comparison.md](comparison.md) (the "No LLM required on write / ingest" row).
