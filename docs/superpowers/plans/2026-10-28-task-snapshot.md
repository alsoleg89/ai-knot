# Model-free task snapshot

Continue the active product goal after the semantic evaluation at `813daa5`. User authorization covers autonomous research, implementation, local PostgreSQL testing and independent review; release changes remain protected.

## Decision and alternatives

The previous case batch found a strong cheap scoped-bundle control, but editorial low-ID placement favors its truncation order. This is evidence to offer explicit scoped loading, not to make canonical truncation a relevance default. Public `select_context` needs lexical requirements, `review_context` needs known refs, and the semantic snapshot requires an embedding callback. Raw KB/pool listing does not provide the same source admission.

Add one opt-in synchronous read: `memory.task_snapshot(*, task_key: str, max_facts=8, max_chars=12000) -> tuple[ContextFact, ...]`. It returns **all** current admissible task facts as whole text and exact refs in deterministic namespace/ID order. If either output limit is exceeded, raise existing `SnapshotTooLarge` with no partial payload. Empty task scope is allowed explicitly and means generic/unlabelled records, as in existing selection. A fitting empty corpus returns an empty tuple. There is no new completeness/coverage flag, state table, schema, dependency, model call or scheduler.

Alternatives: a flagged canonical partial bundle would be smaller to use but invites ranking/completeness mistakes; `SemanticContext` with a dummy callback hides this model-free path behind inference setup; subscription resync changes durable delivery state and lacks task selection. The all-or-error read is the chosen lean contract. Snapshot size is never proof that task requirements are met, catch-up is complete, or an action is authorized.

## Implementation and acceptance

- [x] TDD source-checked queryless snapshots on SQLite/PostgreSQL: task/generic metadata, own/shared identity collisions, support/active time, ACL and inherited roots, freshness/outcome withdrawal, exact count/Unicode whole-text bounds, explicit overflow and empty results. No inference/stat/intent/subscription mutations; fresh refs after source changes verified.
- [x] Shared task applicability/scopes/source-admission iteration with semantic acquisition; retained pre/post-I/O checks, budget/cache behavior and ranking. Thirty-six SQL characterization cases compare exact prior/current results and callback inputs without paid reruns.
- [x] Added HTTP `POST /v1/always-on/evidence/snapshot`, MCP `memory_context_snapshot`, and TypeScript `alwaysOn.taskSnapshot({task_key, max_facts?, max_chars?}) -> ContextFact[]`, using existing strict owner/auth/codec paths. Tested auth, malformed fields/coercion, overflow, lifecycle and both transports; no owner override.
- [x] Documented a copyable host flow and extended the executable model-free task example with explicit overflow handling. Existing claim/review/action guards remain. Independent primary-source note distinguishes current/legacy mechanisms and adoption constraints.
- [x] Local standard CI/PostgreSQL: 2,335 passed, two expected skips, 89.92% coverage. Python 3.11/3.13 focused checks: 104 each. TypeScript: 61 tests, typecheck and both builds. Astra/Ponytail findings fixed. No new accuracy/default/capacity claim or paid rerun.
- [x] Original checkout HEAD/status/all 30 protected files verified unchanged; Python/npm versions remain 0.12.0. Local commit is the integration step, without push/release; broad product goal remains active.

Evidence: [API and host usage](../../task-snapshot.md), [acceptance metadata](../validation/2026-10-28-task-snapshot.json), [independent review](../reviews/2026-10-28-task-snapshot-astra-ponytail.md), and [primary-source adoption note](../../research/2026-10-28-task-bundle-adoption.md). The owner updated this progress checklist after reviewing the recorded source hashes and completed gates; implementation/test files remain the reviewed bytes.
