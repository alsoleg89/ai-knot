# Returning to team memory after a pause

Use the optional always-on layer when agents sleep, restart or reconnect while teammates keep working. Ordinary `KnowledgeBase` and shared-pool recall remain unchanged. SQLite and PostgreSQL support the durable return protocol; no scheduler or model worker starts automatically.

The [executable example](../examples/shared_memory_return.py) uses a researcher and a planner. The researcher changes a deployment decision while the planner is away. The planner reloads saved evidence references from a host cache, rejects the old context, recovers pruned history and saves the new context and receipt together before ack:

```sh
PYTHONPATH=src python examples/shared_memory_return.py
```

## Drain changes before deciding to wake a model

```python
batch = memory.poll("deployment", limit=100)
# Apply invalidations/current revisions and persist a processing receipt first.
if batch.delivery_id is not None:
    memory.ack("deployment", delivery_id=batch.delivery_id)
# Keep draining while batch.has_more. Empty events can precede relevant updates.
```

`through_seq` is this delivery's scanned frontier; `head_seq` is the observed namespace journal head. `has_more` reports raw journal backlog beyond the frontier, including entries filtered out by your selector or access rules. A scan processes at most 1,000 raw entries and emits at most `limit` distinct facts. Pending delivery events/token remain stable across reopen; head/backlog metadata can grow. An empty batch with a token still needs a durable receipt and ack to advance. Debounce can instead return an empty batch without a token and with backlog; retry later using the host's timer.

Coalesce selected changes before invoking a model. Events carry IDs and revisions, not memory text. Treat them as invalidation hints: remove deleted entries, handle a fact moving out of a watched slot, and reload the selected current view. One agent's ack does not advance another agent's subscription. Notifications or scheduling are host concerns; this API performs no idle inference.

These fields describe a point-in-time journal read. Wall-clock expiry, future activation and verification deadlines can change admissibility without a new journal event. Use the existing waits/freshness APIs and host timers for time-based wakes; `has_more=False` is not proof that no timed work is due.

## Recover missing history without losing new writes

After `CursorExpired` or a `resync_required` control delivery:

```python
from ai_knot.always_on import EvidenceRef

snapshot = memory.resync("deployment", limit=100)
refs = tuple(
    EvidenceRef(event.namespace, event.fact_id, event.after_revision)
    for event in snapshot.events
)
view = memory.review_context(refs, max_chars=12000)
# Atomically replace the selected host cache AND save snapshot.delivery_id.
# Save refs together with accepted text, so a restarted process can review them.
memory.ack("deployment", delivery_id=snapshot.delivery_id)
# Poll again: writes after snapshot.through_seq are still pending.
```

The snapshot is a complete reference manifest for the subscription's currently active, supported, source-readable working set. `is_snapshot=True` means replace that selected cache, including with an empty set. Each synthetic `snapshot` event uses the captured journal head as its sequence; it is not a new journal write. Snapshot membership does not establish freshness or permission to act; `review_context` applies the stricter checks before returning text.

Cursor advancement happens only on ack. A crash before the host commit can replay; after that commit, an existing receipt makes processing idempotent. Explicit `resync` creates a new delivery and supersedes a pending unacknowledged token. Snapshot replay rechecks revisions, active time, ACL and lineage. If it no longer describes usable membership, replay returns no events, `resync_required=True`, `is_snapshot=False`; rebuild it. A snapshot is captured at one instant, not a live time-based membership subscription.

If a complete manifest exceeds `limit` (1..500), `SnapshotTooLarge` leaves cursor/pending state intact. Narrow the fact/slot selector or split the working set into subscriptions. There is no silent truncation or paginated corpus snapshot in this version. Recovery scans namespace metadata and selected candidates; indexed fact/slot lookup and corpus-export sessions may be added if measured workloads need them. Archived slot history is skipped before current-slot validation, avoiding repeated full-history checks.

## Admit saved context explicitly

```python
review = memory.review_context(saved_refs, max_chars=12000)
context = [entry.content for entry in review.accepted]
# Replace the old model context; do not append old rejected strings back to it.
```

At most 128 supplied references are checked in one read-only SQL snapshot. Accepted entries retain input priority and exact revision identity. Duplicate identical refs are counted once; conflicting revisions reject. Changed, missing, inactive, unsupported, conflicted, source-invalid or verification-due facts do not supply text. Revoked access issues redact their references. A whole fact that exceeds the remaining character budget appears in `omitted`; its text is never cut midway.

For a manifest larger than 128 references, select the task's working set before review or review separate batches; separate calls have separate read snapshots. Keep the full manifest/receipt distinct from the smaller model context.

`complete` means all supplied references passed and fit the budget. An empty input is complete too; this flag does not establish task-level evidence coverage. Character limits (1..65536, default 12000) exclude host prompt formatting and are not tokenizer limits. Review calls neither a model nor embeddings, changes no counters or intent state, and cannot erase text already delivered to a model. Rebuild that context in your host.

Keep existing `resume(intent_id)` for work state and unresolved effects, and `begin_action` for the final evidence/freshness/resource guard. Neither a snapshot receipt nor a context review transfers a lease, authorizes an external operation, or proves a statement follows from its sources. New undeclared semantic dependencies cannot be inferred by revision checks.

## Existing transports

| Python | HTTP POST | MCP | TypeScript |
|---|---|---|---|
| `resync(subscription_id, limit=500)` | `/v1/always-on/subscriptions/{id}/resync` | `memory_resync` | `alwaysOn.resync(id, limit)` |
| `review_context(refs, max_chars=12000)` | `/v1/always-on/evidence/review` | `memory_context_review` | `alwaysOn.reviewContext(refs, maxChars)` |

CLI supports `ai-knot changes resync AGENT SUBSCRIPTION --limit 100 --json` for its existing private-owner subscriptions. HTTP/MCP use the configured owner and existing authorization; requests cannot override identity. URL-encode subscription IDs, including `/`. Snapshot overflow returns HTTP 413 / domain code `SnapshotTooLarge`.

The return milestone added schema 7 on top of schema 6, preserving pending deliveries and receipts. Current source also includes [outcome-bound experience](team-experience.md) and initializes schema 8. Use compatible deployed clients; older binaries reject newer schemas. Package publication is separate from these source changes.

## Local load observations

`scripts/benchmark_memory_return.py` runs eight spawned readers with independent SQL connections, 2,500 unrelated writes, a target update, forced retention and return recovery. For each backend the observed invariants were: 16 target observations, eight recovered snapshots, eight rejected stale contexts, zero missed target updates, 80 idle checks and zero idle wake decisions. No model is connected; useful wakes are decisions of the scripted host.

| Backend | Return p50 | Return p95 |
|---|---:|---:|
| SQLite | 10.7 ms | 132.8 ms |
| PostgreSQL 16 | 158.2 ms | 213.5 ms |

These are local measurements on Python 3.12/macOS, not production capacity, an SLO or LLM answer accuracy. Raw reports: [SQLite](performance/shared-memory-return-sqlite.json), [PostgreSQL](performance/shared-memory-return-postgres.json). Reproduce with `PYTHONPATH=src python scripts/benchmark_memory_return.py --backend sqlite`; for PostgreSQL pass `--backend postgres` and set `AI_KNOT_TEST_PG_DSN`. The CLI creates and removes its own disposable database/schema.

Final local acceptance: 1,833 Python tests passed with PostgreSQL required (two expected environment-specific skips), coverage 88.93%; 57 TypeScript tests plus typecheck/build; lint/format/strict mypy passed. Existing process-recovery scenarios passed 12/12 on each backend; all 11 binding multi-agent thresholds passed. Five existing advisory misses remain unchanged. [Validation record](superpowers/validation/2026-10-27-memory-return.json) and [Astra/Ponytail review](superpowers/reviews/2026-10-27-memory-return-astra-ponytail.md) record scope and evidence.
