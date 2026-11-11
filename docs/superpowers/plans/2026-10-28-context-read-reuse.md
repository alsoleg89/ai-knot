# Transaction-local context read reuse

> **For agentic workers:** Execute this bounded task natively with tests first, then independent Astra/Ponytail source review. The user has already authorized autonomous research, implementation and local verification.

**Goal:** Reduce repeated SQL work when always-on agents read many lessons with shared roots, preserving the exact context and source/ACL/freshness boundaries.

**Architecture:** Reuse already scanned facts, identical point loads and identical read queries through an internal read-only adapter scoped to one existing read transaction. Only task snapshot, lexical selection, context review and semantic acquisition/revalidation use it. Keep write/action transactions and public interfaces unchanged. Do not reuse cached database lookups or admission decisions across requests, transactions or provider awaits; captured source objects may remain internal to semantic acquisition as before, followed by fresh revalidation. Do not add a persistent cache, database schema, dependency or model worker.

**Evidence:** The repository experiment delivered equal evidence sets to memory and the cheap control; its coding failures do not justify tuning retrieval on those three tasks. Source inspection instead finds repeated `_view` root/grant/freshness loads inside whole-corpus scans. Research and baseline measurements must confirm the narrower performance opportunity.

## Acceptance and risks

- Exact results, ordering, issue/overflow behavior and embedding inputs/costs must match the pre-change source. Preserve whole-fact budgets, own/shared ID distinction, applicability, root/intermediate/outcome currentness, slot ambiguity, freshness and ACL redaction.
- The adapter is internal and receives only the operation's read-only session; it cannot authorize writes. Returned SQL rows/lists must not mutate cached containers. Existing context code does not mutate loaded `Fact` objects; do not expose cached objects as a new public API.
- Point-only review must not acquire a full namespace. Whole scans may reuse their own rows by ID and slot, preserving original relative order. Never turn an output bound into a claimed bound on retained-corpus scanning.
- Both SQL backends must retain their current snapshot semantics when a writer commits mid-read, then observe changed evidence/permissions on the next operation. Semantic validation after provider I/O needs a fresh adapter/session.
- Predeclare performance checks: a 64-lesson/16-shared-root whole-corpus read should need fewer than 200 backing SELECT/load calls; point-only review must stay below 300 without scanning namespaces. The review allowance accounts for 64 independent slot lists plus 64 distinct fact reads and their policies; avoid replacing these with an unbounded scan merely to hit the scan budget. Report before/after SQL counts and five sequential latency samples with 0/2,000 unrelated retained facts. Seek at least 2x median improvement on this reuse-heavy workload, reporting any regression and all samples. This is a serial local profile, not fleet capacity or answer-quality evidence.

## Work

- [x] Add reproducible, model-free read workload and count existing SQL reads; freeze the pre-change base and record measurements before implementing reuse.
- [x] Write a failing backing-read budget regression and point-review/no-whole-scan test. Implement the smallest internal adapter and use it only in the four context read paths.
- [x] Exercise real concurrent SQL snapshots and fresh subsequent reads; compare exact public results and semantic callback inputs against pre-change code. Run focused tests including PostgreSQL, Python 3.11/3.13 and full local CI.
- [x] Independently review source and measured results with Astra/Ponytail. Record scope, limits, original-checkout protection and local performance data; make a reviewed local commit. Preserve deferred release files and keep the broad product goal active.

## Measurement outcome

The declared query budgets passed: shared-root snapshot 99 reads, lexical selection 83, point-only review 225, compared with 3,330/3,331/3,392 originally. All 24 public-result hashes matched. The sought 2x median latency gain was not met by most SQLite operations (1.34–2.04x on shared roots), while PostgreSQL shared-root profiles improved about 11.28–18.19x. Keep that limitation and all five samples; no retiming to meet the target. Root-free controls and retained-scan costs are reported separately. These are serial local performance results, not model accuracy, fleet capacity or completion of the product goal.
