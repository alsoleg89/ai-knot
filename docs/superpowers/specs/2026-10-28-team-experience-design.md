# Outcome-bound team experience

## Purpose and decision

Always-on peers need conditional lessons that retain the evidence and recorded result of the work that produced them. Existing `derive_fact` binds facts to source revisions, but ActionRecord is not an EvidenceRef: reconciliation of a failed outcome can leave a manually derived lesson untouched. Add an explicit host-authored experience operation. This is an autonomous implementation choice within the user's research/build goal.

Alternatives considered: a documented host-managed receipt plus ordinary derivation already works when the host updates every receipt; adding only an action ID qualifier fails to enforce retraction; a separate experience graph/model worker would duplicate existing lineage and increase idle cost. Choose existing procedural facts plus a small durable action-to-lesson binding table. Roots remain the supplied observations, not action IDs or copies.

## Public contract

`record_experience(action_id: str, *, expected_action_revision: int, experience_key: str, content: str, basis: tuple[EvidenceRef, ...], namespace: str | None = None, visibility_scope: str = "global", task_key: str = "", for_role: str = "") -> RecordedExperience`.

`RecordedExperience` is frozen: `lesson: ContextFact`, `outcome: Literal["succeeded", "failed"]`, `action_revision: int`.

- Require this configured owner's accessible action/workspace, exact current positive action revision, state succeeded/failed, nonempty recorded result and 1..128 unique currently admissible supplied evidence refs. A host supplies the text and observed evidence. Neither state nor arbitrary JSON independently proves remote success or semantic entailment.
- Create one procedural fact with an explicit visible `Recorded outcome: succeeded.` or `Recorded outcome: failed.` prefix, exact roots/intermediate dependencies, optional task/role labels and a reserved opaque action binding. No private request/result/action ID/intent text is copied to a shared lesson. A failure is a conditional warning; unknown/prepared/started cannot enter either result category.
- Idempotency key is scoped to owner, action and destination namespace. Replay of identical input returns the same lesson/ref and creates no fact/journal change. Reusing the key with different input raises IdempotencyConflict; removed/amended/retracted output is not silently recreated. The retained binding stores the digest and output evidence revision.
- Shared writes require existing same-store pool/workspace maintain authority. Reader permissions and source freshness/lineage remain mandatory. A shared projection cannot launder private evidence.
- Ordinary pool publication skips direct marked experience before shared-slot mutation; use the dedicated shared writer. Non-authoritative workspace terminal aliases cannot form current-outcome experience. Facts-only snapshot/import writes retain the durable binding decision and cannot resurrect outdated marked lessons or their source-dependent copies.
- Any later action revision invalidates bound lessons atomically with saving that action, including reconciliation to retryable/success, before any new permit. Derived copies then fail existing dependency gates. Reopening without a pool must still invalidate previously shared output. Discover destination locks from the binding table; ordinary actions must not acquire a global shared lock when no binding exists.
- Existing manual derivation remains available, with no change in default behavior or automatic reflection. Expose the opt-in operation through HTTP `/v1/always-on/actions/{action_id}/experience`, MCP `action_experience_record`, and TypeScript `recordExperience`.

## Storage and boundaries

Add one SQL table for owner/action/namespace/lesson ID, action revision, input digest and output revision; index by owner/action. Initialize idempotently on existing SQLite/PostgreSQL stores; schema version 8. Keep package versions 0.12.0. The central `save_action` hook retracts only registered lessons, inside declared destination locks. Fact mutation, journal changes and action transition roll back together.

No model call, graph dependency, new worker, success counter or automatic promotion. One observation copied by several agents remains one independent root. Selection/review remain point-in-time; action guards remain necessary. Already delivered model text cannot be erased. This milestone establishes recorded-outcome consistency, not useful generalization or better real-model answers.

## Acceptance

Both SQL backends: success/failure, unknown/empty result/stale revision rejection, strict transport validation, stable replay and conflicting inputs, original/intermediate correction/delete/freshness/ACL retraction, independent root counting across copies, shared authority and restart without pool, rollback on failed invalidation, concurrent recording versus reconciliation, and unrelated actions avoiding shared fact scans/locks. Include an executable two-role experience/retraction example and a reproducible bounded concurrent workload. Run full local required-PostgreSQL/static/TypeScript and existing recovery/MA gates; fresh Astra/Ponytail review. Preserve all protected original checkout/release files; local feature commit only.
