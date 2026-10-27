# P2 implementation report

Implemented on 2026-10-27 in `codex/always-on-p1-p3`; no commits created by this worker.

Production modules:

- `always_on/models.py`: public immutable state/lease/evidence/action/freshness objects, domain errors, canonical finite JSON validation, UTC and bound validation.
- `always_on/_state.py`: deterministic state transitions, inherited basis, lease expiry/fencing, action idempotency and recovery, freshness validity/backoff.
- `always_on/state_storage.py`: SQLite/PostgreSQL DDL and row codecs, integer revision CAS, additive migration marker to v5, PostgreSQL BIGINT and TIMESTAMPTZ declarations.
- `always_on/memory.py`: complete P2 facade on P1 SQL sessions. Namespace lock discovery/re-read/retry, current durable ACL, active/unambiguous slot projection, committed stale blockers, permit returned after commit, action journal, revision-bound verification.
- `always_on/__init__.py` and package root: public exports, lazy facade import to avoid storage/KnowledgeBase circular initialization.

The facade includes P1 subscriptions via `SubscriptionsMixin`. Constructor opt-in enables only its private stream. Shared capture requires matching persisted pool, enables the shared stream, then captures evidence inside one read snapshot. P2 DDL initialization is serialized under a schema lock independent of owner locks. It never runs external tools or models.

P2 tests: `tests/always_on/test_agent_state.py` and `test_state_contracts.py`.

Validation:

- Initial public lifecycle tests first failed with missing facade import, then passed after implementation.
- Additional tests reproduced UNKNOWN blocked-checkpoint rejection and missing model field validation; both corrected.
- Final P2 run with `AI_KNOT_REQUIRE_POSTGRES=1` and local live PostgreSQL DSN: **58 passed in 7.23s**, SQLite and PostgreSQL, no skips. Covers concurrent claims, exact expiry fencing, durable generation, CAS, owner isolation, deadlines, all STARTED checkpoint targets, UNKNOWN audit cancellation, inherited stale basis, waits/restart, durable shared ACL revocation, single permits, finish idempotency, retryable reconciliation stable keys, proven success with stale historical basis, freshness outages/backoff and slot replacement, schema types, malformed JSON and YAML incompatibility.
- Combined AO suite before the final schema-introspection additions: **118 passed, 1 skipped in 22.80s** (the skip belongs to another task's storage test); SQLite and PostgreSQL were enabled.
- Ruff check and format on owned files: passed.
- Strict mypy on `memory.py`, `state_storage.py`, `models.py`, `_state.py`: passed.

The parent owns the overall suite, coverage, end-to-end process crash matrix, eval gates, benchmarks and Astra/Ponytail final review. No external effect is performed by this memory layer. A permit fixes local guards at its commit point; conditional execution/fencing in an external service remains the host's responsibility.

## Final P2/P3 integration

Added lazy public maintenance exports (`MaintenanceBudget`, `MaintenanceReport`, `ConsolidationProposal`, both provider protocols and `run_maintenance`). Facade opt-in now creates the maintenance table and advances the complete schema to version 6 inside its existing schema-lock transaction. Regression first failed on the missing public export, then passed after implementation; it also asserts version 6 and an empty maintenance ledger before the first maintenance run.

Final combined targeted run after these integration edits: **84 passed in 11.27s** across SQLite and live PostgreSQL; strict mypy on the changed facade/package exports and Ruff passed. Parent/P3 subsequently own the independently reviewed maintenance-policy concurrency fix.

## Production review fixes: bounded due lookup and policy generation

Internal freshness rows now have a monotonic `revision` and indexed `dirty` marker, without changing the public FreshnessRecord/wire API. Every policy save increments its revision; semantic fact/slot mutations mark the affected policy in the same transaction and fence in-flight verification. Additive upgrades of existing schema 5/6 add these columns under the existing migration lock, including when opening a plain backend before creating the facade. Existing rows start dirty so no earlier mismatch is silently omitted.

`due_verifications(limit=...)` now selects eligible policies/current frontier IDs in bounded SQL pages, applies ACL/support/temporal/ambiguity filters before pagination, and decodes only selected facts or their slot peers via optional transaction-local fact filters. It continues pages after ineligible/reverted entries rather than permanently truncating eligible work. A scheduled slot frontier transition is detected even without any new fact write and while the old policy retry time is still in the future. Exact SQLite UTC scalar normalization and PostgreSQL TIMESTAMPTZ comparison preserve microsecond validity boundaries and offset timestamp ordering.

The old scaling regression reproduced **1001 full-namespace reads** for 1000 policies with limit 1, taking 12.09 seconds in the test run. The regression now requires at most 3 filtered reads and passes. Separate red/green cases cover a fact valid for one further microsecond, offset-created_at slot ordering, and a scheduled frontier switch with no intervening write.

The direct `record_verification` method documents its trusted-host contract: it attests the current policy for the exact fact; independently asynchronous hosts coordinate policy changes. Built-in `run_maintenance` additionally checks the saved internal policy revision before accepting provider responses.

Final focused SQL/backend regression run after code freeze: **249 passed, 1 expected skip in 40.35s**, with live PostgreSQL required. Strict mypy on memory/state storage/session/SQLite/PostgreSQL and Ruff on changed production/tests passed. The skip is the SQLite/PG-specific storage test counterpart, not a missing PostgreSQL execution.
