# Reliable return to shared memory

The product target is lightweight common memory for Muse-style agents that pursue goals across pauses, processes and teammates. The user explicitly authorized autonomous research, design and implementation. These design choices are the implementer's calls.

Research: [direction](../../research/2026-10-27-shared-memory-direction.md). [TRACE (2026-09-27)](https://arxiv.org/abs/2609.33517) independently motivates checking admissibility on return. This is deterministic checking of explicit revisions, not TRACE's semantic reconciliation or a claim of better LLM accuracy.

## Approaches

Adding graph/reflecting workers introduces infrastructure and recurring inference without solving delivery. Sending whole shared memory on each wake introduces unbounded context. Selected approach: extend existing SQL subscriptions, receipts and evidence validation; keep ordinary memory APIs unchanged.

## Delivery contract

Add backward-compatible `ChangeBatch` fields: `through_seq`, `head_seq`, `has_more`, `is_snapshot`. `through_seq` is the scanned/delivered frontier, `head_seq` the observed namespace head, `has_more` denotes remaining raw journal entries. Pending events/token remain stable across reopening; head/backlog can grow. Empty relevant events may still require ack and another page. Idle checks invoke no model.

`resync(subscription_id, *, limit=500) -> ChangeBatch` returns a complete, bounded, content-free manifest of selected active, supported, source-readable facts at a captured head, marked `is_snapshot=True`. The host replaces its selected cache, saves a durable receipt, then calls existing ack. Cursor advances only on ack; writes after capture remain for delta replay. Explicit resync replaces any pending delivery, invalidating its unacknowledged token. Overflow raises `SnapshotTooLarge` with pending state/cursor unchanged: narrow the watch, never silently truncate. The bounded working-set limit is 1..500; corpus export/paginated snapshot sessions are outside this milestone.

One additive subscription column stores snapshot mode. Upgrade preserves existing subscriptions and receipts; future schema versions fail. Snapshot replay rechecks exact revisions, active time and source readability. Invalid snapshots return an empty `resync_required` control delivery with `is_snapshot=False`. A pending delivery that still bridges retention can replay; a gap after its frontier expires. Manifests are references, not context text or action permits.

## Context contract

`review_context(refs: tuple[EvidenceRef,...], *, max_chars=12000) -> ContextReview` checks at most 128 supplied refs in one read-only SQL snapshot, using existing basis semantics: revisions, current slot, support, active time, ACL, freshness and transitive lineage. Immutable `ContextFact(ref, content, slot_key)` entries contain accepted exact revisions only. Result includes `issues`, budget-`omitted` refs, `checked_at`, and `complete`. Preserve input priority and deduplicate identical refs; conflicting versions reject. `max_chars` is an integer 1..65536; omit whole facts, never cut their text. Revoked issues redact refs per existing guards. Replace cached context with the accepted set. No model calls, embeddings, counters or intent mutations. Recheck action guards before effects: context review has point-in-time validity only.

## Surfaces and acceptance

Expose strict HTTP/MCP/TypeScript operations `memory_resync` and `memory_context_review`, retaining existing methods. Add executable two-agent sleep/reopen workflow and an 8-reader return workload. No scheduler, new dependency, graph service or background worker. Python >=3.11, SQLite and PostgreSQL.

Test >1000 irrelevant events before a target; page boundary; independent ack; forced retention; snapshot→write→ack→delta; overflow rollback; stale tokens; replay source/revision/ACL/expiry changes; context budget, priority, dedup and freshness; schema migration; HTTP/MCP/TS parity. Run full local Python required-PG suite, lint/format/strict mypy, TS typecheck/build/tests, existing recovery/MA acceptance and independent review. Separate deterministic safety from unmeasured real-model accuracy.

Work in `codex/always-on-memory-return` from main `3618bfb`. Preserve all original dirty release/research files using `/tmp/ai-knot-shared-memory/protected-before.json`. No version, changelog, publication, push or merge changes.
