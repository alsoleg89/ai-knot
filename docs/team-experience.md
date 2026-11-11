# Outcome-bound team experience

A useful team lesson needs both observations and the recorded outcome of the work that produced it. `record_experience` stores a host-authored conditional procedure bound to an existing action revision. Other agents can reuse it through [task working sets](task-context.md). Correcting that outcome withdraws the procedure and its exact-source descendants without a model job.

```python
experience = memory.record_experience(
    finished_action.id,
    expected_action_revision=finished_action.revision,
    experience_key="restore-checkpoint",
    content="For rollout42, verify the backup receipt before switching traffic.",
    basis=retained_receipt_refs,
    namespace="__shared__",
    visibility_scope="team:ops",
    task_key="rollout42",
    for_role="operator",
)
saved_refs = (experience.lesson.ref,)
```

The caller supplies observations and lesson text. The library verifies the recorded action/evidence relationship; it does not contact an external service to prove success or establish that the procedure follows from the receipt. A nonempty JSON result and current evidence can still be a mistaken host report. Action identity is a control dependency, not an additional independent observation.

## Recording and reuse

The action must belong to the configured owner and have its exact current revision, a `succeeded` or `failed` state and nonempty result. Prepared, started and unknown actions are rejected. Resolve unknown external effects through existing reconciliation before forming a current-outcome lesson. Workspace actions require their configured workspace and execute permission; terminal result aliases owned by another worker cannot create current-outcome experience. A historical alias is not authority over the current business operation.

Supply 1..128 evidence references. The writer checks current revisions, source/read permissions, intermediate dependencies, support and freshness in the same transaction. Roots remain the original observations; several copies, agents, lesson keys or action aliases do not turn one receipt into multiple confirmations. The text is stored as a procedural fact with a visible `Recorded outcome: succeeded.` or `Recorded outcome: failed.` prefix. Failure text should describe a conditional warning, not a universal rule that an approach never works.

`experience_key` is a nonempty exact label of at most 256 characters without surrounding whitespace. It is scoped to owner, action and destination namespace. Repeating identical input returns the same immutable `RecordedExperience` (`lesson`, `outcome`, `action_revision`) and emits no new fact/change. A different text, source set, action revision, scope or task/role under the same key conflicts. A removed, amended, stale or retracted lesson is not resurrected by replay. Keep a new key for a genuinely new lesson or corrected outcome. Content plus metadata uses the existing 65,536-byte canonical JSON request limit.

Shared recording requires a persistent same-store pool and the workspace maintain grant; readers must retain access to the source roots. Internal opaque action metadata excludes private request/result JSON, action IDs and intent text. The supplied lesson text is explicitly published by the host, so keep private details out of it. A public lesson over private roots remains unavailable to peers.

Create shared experiences directly with `namespace="__shared__"`. Ordinary `pool.publish` skips direct action-bound lessons before changing shared slots. This keeps publication on the guarded maintain/binding path. Ordinary manually authored facts keep their existing publication behavior.

Reuse `select_context` to get a bounded task/role working set; save exact refs and replace cached context with `review_context` on return. Neither coverage nor lesson delivery grants an action permit. Continue to use `prepare_action`/`begin_action` guards before effects, since sources may change after a read.

## Current outcome versus historical failure

Every later action revision withdraws registered current-outcome lessons atomically with the action change. This includes failed→succeeded reconciliation and failed→retryable recovery before another attempt. Copies keep exact intermediate dependencies and are rejected on future recall/select/review. Reopening the owning agent without a pool still discovers previously shared destinations and retracts them. Unbound actions do not acquire a shared fact-write lock or scan shared facts for this feature.

A fact snapshot does not roll back durable actions. SQLite/PostgreSQL apply the persisted action binding at their common fact-write boundary: restoring/importing an old marked lesson cannot make it supported after the action revision advanced. Imported marked lessons without a matching destination binding are unsupported. This is a trusted-host API guarantee, not protection against manually rewriting the underlying action/binding database. Already delivered model text cannot be erased; hosts must rebuild their context after invalidation.

An earlier failed attempt can remain useful even if a later attempt succeeds. Save its actual attempt receipt as a separate retained episodic fact and derive a conditional historical warning from that ref:

```python
warning = memory.derive_fact(
    "Under the old timeout setting, attempt one failed before retry; check its receipt.",
    basis=attempt_receipt_refs,
    namespace="__shared__",
    memory_type=MemoryType.PROCEDURAL,
    task_key="rollout42",
)
```

This historical warning survives a change to the current action outcome. Correcting, removing or revoking its own attempt receipt still retracts it. The mutable action table does not retain immutable versions of every past attempt; it cannot substitute for an actual retained receipt.

## Existing transports and storage

| Surface | Operation |
|---|---|
| Python | `memory.record_experience(action_id, expected_action_revision=..., experience_key=..., content=..., basis=...)` |
| HTTP | `POST /v1/always-on/actions/{action_id}/experience`, remaining named fields in JSON |
| MCP | `action_experience_record`, including `action_id` |
| TypeScript | `client.alwaysOn.recordExperience(actionId, input)` |

Configured-owner authentication and strict decoding reject actor spoofing, unknown fields and coercion. No new CLI flag, dependency, model worker or automatic reflection loop is required. One SQL binding table retains replay digests/output identities after lesson deletion, without retaining the lesson text there. Initialization upgrades existing stores to always-on schema 8; these operations are available since 0.13.0. Older clients that support only schema 7 reject the upgraded store. Keep database backups when upgrading clients.

Run the executable model-free two-role example:

```sh
PYTHONPATH=src python examples/team_experience.py
```

It records successful and failed lessons, checks peer reuse and copy-root collapse, corrects the failure from a new facade without a pool, and retains the distinct historical warning.

## Reproducible local load

```sh
PYTHONPATH=src python scripts/benchmark_team_experience.py --backend sqlite --output /tmp/team-experience-sqlite.json
# Supply AI_KNOT_TEST_PG_DSN for a disposable-schema PostgreSQL run:
PYTHONPATH=src python scripts/benchmark_team_experience.py --backend postgres --output /tmp/team-experience-postgres.json
```

The default workload uses four owners, eight overlapping processes with independent SQL connections, eight actions, eight keys per action, three replay rounds per worker and 500 unrelated facts. It expects 384 recording calls to produce 64 unique durable lessons. It checks peer ACL withdrawal, source correction, failed-outcome reconciliation from a fresh process without a pool, copy roots and a surviving historical attempt warning, under a three-fact/600-character selection budget. Inference/embedding HTTP is blocked in parent and worker processes.

Recording timings include first publication and successful idempotent replays. Return timing covers selection; the later explicit saved-ref review/assertions are outside that timer. Corrections occur between return phases; simultaneous recording versus reconciliation is tested separately by the core SQL race regression. This bounded local workload establishes consistency and descriptive timings, not production capacity, model-answer accuracy or an advantage over manual receipt maintenance.

Retained [SQLite](performance/team-experience-sqlite.json) and [PostgreSQL](performance/team-experience-postgres.json) runs both produced 384 calls → 64 unique lessons over eight independent source observations, with zero duplicate bindings, stale selections or model calls. ACL/source/outcome changes rejected 64/8/32 lessons respectively; four actions were reconciled after a process restart without a pool. A lesson and its transitive copies still contributed one root. Local p95 recording times were 67 ms/672 ms and selection times 24 ms/50 ms (SQLite/PostgreSQL); these include contention and are descriptive measurements of this fixed workload.

The final acceptance also compares the strong existing `retained observation → host-maintained current-outcome projection → derive_fact` control with `record_experience`, on both SQL backends. Both see the same retained observations, return the same one-fact text within 600 characters, make zero inference/embedding calls, and collapse to the same observation root. After reconciliation the binding retracts immediately; the control matches once the host explicitly updates its outcome projection. This establishes the saved bookkeeping/atomicity contract, not a better learning algorithm or an accuracy improvement over a correctly maintained receipt baseline. The public SQL regression is `test_equal_budget_managed_receipt_matches_final_view_with_explicit_host_update`.

See [primary-source research](research/2026-10-28-outcome-bound-experience.md), the [design contract](superpowers/specs/2026-10-28-team-experience-design.md) and [program roadmap](shared-memory-roadmap.md).

Local acceptance: 1,954 Python tests passed with PostgreSQL required, two expected skips and 89.37% coverage; 59 TypeScript tests and build/typecheck passed. Existing recovery passed 12/12 scenarios per SQL backend and all 11 binding multi-agent gates passed. The [Astra/Ponytail review](superpowers/reviews/2026-10-28-team-experience-astra-ponytail.md) records three P1 defects and the owner's verified repairs; the [validation record](superpowers/validation/2026-10-28-team-experience.json) retains source hashes and exact limitations.
