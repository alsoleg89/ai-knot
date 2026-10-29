# Guard a decision against changes to unselected task evidence

Selected evidence can stay current while another agent adds a separate prerequisite. `review_context` and an action's selected basis check those supplied refs; they cannot establish that no other applicable observation arrived. A `TaskContextGuard` adds an opt-in check of the entire currently admissible task set.

Capture the guard **before** retrieval/model I/O, preserve the original requirements and use the same guard when saving the proposed decision. This example assumes the host already owns its intent lease:

```python
guard = memory.task_guard(task_key="rollout", max_facts=128)
context = memory.select_context(
    requirements, task_key="rollout", max_facts=8, max_chars=2000,
)
# Your bounded model/domain policy assesses context; this read is not a verdict.
memory.checkpoint(
    intent_id, lease_token=lease.token, expected_revision=revision,
    state=IntentState.READY, progress=progress,
    basis=tuple(f.ref for f in context.facts), context_guard=guard,
)
```

Checkpoint revalidates selected basis and the guard while holding the existing source/intent namespace writer locks. A changed admitted set, evidence revision or new overflow raises `StaleEvidence` and rolls back the checkpoint. A selected source that remains valid cannot hide a newly added admissible generic/task condition. This does not establish that the selected context contains all prerequisites or that a proposed action follows from it.

The descriptor contains `task_key`, lowercase SHA256 `digest` and `max_facts` (integer 1..128, default 128). Its domain/version, logical owner, task scopes, task key, limit and sorted exact admissible namespaced refs enter the digest. It returns no fact text/IDs and requires no model or stored token registry. Empty sets are valid; overflow returns no partial guard. Fact content has no guard-specific character cap. Candidate scans, decoding, lineage and database/native buffers remain separate costs; the count bound does not make those costs constant.

## Recheck before an effect

The guard is not stored on an intent or action automatically. Preserve it in private host state and **explicitly pass it again** for a later action:

```python
permit = memory.begin_action(
    action.id, lease_token=execution_lease.token, context_guard=guard,
)
# Only now may the trusted connector use permit.idempotency_key.
```

The same guard detects an admissible unselected addition after checkpoint and before action admission. A mismatch follows existing stale-basis behavior: the owning intent is saved BLOCKED with reason `changed`, the action/reservation stays PREPARED, no permit is returned, and `StaleEvidence` is raised after committing the block. Normal lease, deadline, manual/unknown-action and workspace authority rules still apply. A guard supplies no authority or effect-success receipt. If it is omitted, the original selected-basis behavior is retained.

Do not recapture a guard after model output merely to make an old decision pass. Re-read/reassess changed evidence under the host's retry policy. A wrong task label with a freshly valid guard is still wrong host policy: intents have no implicit task-label binding. The guarded helper returns and privately persists the descriptor for deliberate reuse; see [bounded planning](bounded-planning.md).

## Exact boundary

The guard describes the **admissible source set** at the operation's captured time: existing task/generic applicability, support/active state, current slot, source/root ACL, lineage, freshness and recorded-outcome rules. Inaccessible, unsupported, expired or otherwise inadmissible new rows may leave it unchanged. ACL/freshness edits that preserve that exact set/ref revisions do too. Exact restored sets can match again (ABA); this is not a journal-history stamp.

Guard reads use one read-only SQL snapshot. Guarded writes lock all task scopes even if the chosen basis is private/empty, then recompute in their existing write session using the same `now` as basis/action validation. SQLite retains BEGIN IMMEDIATE for writes. PostgreSQL write transactions and schema initialization explicitly use Read Committed before their first lock/query; reads retain Repeatable Read Read Only. This avoids inheriting an RR snapshot that predates an advisory-lock wait. Cooperating library writers must use the existing namespace locking contract; arbitrary external SQL is not made safe by a checksum. See the [research and official isolation references](research/2026-10-28-task-context-guard.md).

No lock stops wall-clock expiry or the external world after validation/permit. No guarantee extends to arbitrary same-token concurrent controllers; keep the existing exclusive lease-controller contract. Guarding an admissible set is neither semantic completeness, lossless compaction, exactly-once inference nor a long-lived authorization grant.

## Transports

| Client | Operation |
|---|---|
| Python | `memory.task_guard(task_key="rollout")` |
| HTTP | `POST /v1/always-on/evidence/guard` with `{"task_key":"rollout"}` |
| MCP | `memory_context_guard` with the same fields |
| TypeScript | `memory.alwaysOn.taskGuard({task_key:"rollout"})` |

HTTP/MCP checkpoint/begin accept an optional nested `context_guard`. TypeScript checkpoint takes `context_guard`; `beginAction(actionId, leaseToken, guard)` appends an optional third argument, preserving old two-argument calls. Configured-owner authentication, strict scalar decoding/unknown-field rejection, `SnapshotTooLarge` capture and `StaleEvidence` conflict codes remain shared across transports. Release versions/schema/dependencies are unchanged.

The [growing-context study and verification](growing-context-evaluation.md) records the local SQL/transport tests, PostgreSQL lock-wait regression, paid decision results and exact PostgreSQL replay. Guard correctness and candidate quality are reported separately.
