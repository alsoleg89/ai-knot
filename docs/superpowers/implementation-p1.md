# P1 production implementation

Implemented in the shared branch `codex/always-on-p1-p3`, 2026-10-27. No commit or publication performed by this worker.

## Delivered

- SQLite `BEGIN IMMEDIATE` and PostgreSQL namespace advisory transactions cover load, transformation, fact replacement, semantic diff, journal append, head and commit. The optional storage protocol remains separate from `StorageBackend`.
- Transaction-local SQL sessions enforce declared fact namespaces/read-only access and normalize PostgreSQL datetime rows. PostgreSQL uses sorted namespace locks and repeatable read snapshots; initialization serializes schema creation and all migrations with a schema-specific transaction advisory lock.
- `mutate_facts` returns callback results only after successful atomic commit. Managed add, learn, async semantic closing, recall telemetry, decay, publication, promotion and pool GC preserve concurrent foreground writes. Explicit clear/replace/restore retain intentional replacement behavior. P3 owns the final legacy episodic-consolidation refactor.
- Semantic revisions exclude access/ranking telemetry and include evidence/provenance/access/validity fields. Streams are explicitly enabled; existing facts create a baseline resync event. Events never contain fact text.
- Owner-scoped subscriptions persist immutable inflight batches, cursor, debounce and acknowledgment receipts. ACL replay redacts revoked/private tombstones without exposing closed IDs; old valid receipts remain idempotent until unsubscribe. Scan limit is 1000, page limit 500, and shared projection reads use indexed candidate IDs rather than materializing the namespace.
- Safe/manual pruning respects acknowledged retention frontiers and exposes forced gaps as `CursorExpired`.
- Durable ACL grant/revoke changes and addressed events share the transaction; shared recall refreshes current durable grants.
- PostgreSQL snapshots retain all fact columns; SQLite snapshot round-trip additionally preserves claim key and memory tier. Restoring facts does not restore intentions, actions or delivery history.

## Verification

Test-first checks reproduced missing SQL seam and guarded recall persistence, invalid nonfinite semantic writes and SQLite snapshot field loss. Regression coverage additionally reproduces delayed transaction commit ordering, entire SQL rollback, four concurrent PostgreSQL constructors in an empty schema, restart replay/ack/incarnation, tombstone redaction and retention gaps.

With the parent's isolated PostgreSQL 16 test service and `AI_KNOT_TEST_PG_DSN` set:

```
.venv/bin/python -m pytest tests/always_on/test_storage_core.py tests/always_on/test_subscriptions.py tests/always_on/test_managed_writes.py tests/test_sqlite_storage.py tests/test_storage_compat.py tests/test_concurrent.py tests/test_pool_field_parity.py tests/test_knowledge.py tests/test_knowledge_edge_cases.py tests/test_shared_pool.py tests/test_acl_ledger_storage.py tests/test_pool_governance_durable.py -q --no-cov
```

At the recorded combined run: **236 passed, 1 skipped** (the PostgreSQL-only constructor regression is intentionally inapplicable to the SQLite fixture). The additional bounded-projection regression was added afterward and its SQLite branch passed; the parent runs the final unified matrix.

`ruff check` and `ruff format --check` passed all 14 P1-owned modules/tests; strict mypy passed the 10 production modules again after the final bounded-projection refinement (parent performs final full-package check). The legacy simulated PostgreSQL driver fixture was updated for actual schema reads/column descriptions; its targeted regression passes. Live SQL tests, rather than that fake, establish transaction guarantees.

## Integration notes

- Upgrade every managed writer before enabling streams; old binaries and raw external SQL are outside journal guarantees.
- YAML keeps its legacy single-writer fallback and explicitly cannot enable always-on durability.
- The parent owns whole-suite coverage, subprocess crash recovery, benchmarks and Astra/Ponytail final review. This report does not claim those checks based on this worker's targeted tests.
