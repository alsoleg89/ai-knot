# Hosting an always-on team

Use AI Knot as durable shared memory inside your existing agent host. A queue message, webhook or timer invokes one bounded inspection; the host can then sleep again. Constructing a KB or always-on facade starts no scheduler, network service or model worker.

Run the complete model-free recipe from this source checkout:

```sh
PYTHONPATH=src python examples/always_on_team.py
```

One common corpus follows a planner and researcher through five quiet timer checks, 1,200 irrelevant changes before a relevant update, restart, bounded catch-up, retention recovery, stale-plan rejection, a recorded team lesson reused by a reviewer and later source/outcome withdrawal. Connector decisions/results are scripted and inference/embedding HTTP is blocked. Actual remote effects and model quality are not measured. The example uses a temporary SQLite database; integration tests run the same workflow with PostgreSQL and independent storage instances.

## Connect it to your host

The example's `Inbox`, `inspect_turn` and `TurnObservation` are application recipes, not another stable library API. Copy/adapt these host functions when wiring your runtime. Configure the existing [workspace authority and shared pool](multi-agent-always-on.md), stable per-agent subscriptions, task requirements and a durable local inbox path. Store the inbox somewhere the next invocation can read. Its key includes logical owner and subscription; it grants no authority over other agents.

```python
turn = inspect_turn(
    memory, inbox,
    intent_id=intent_id,
    subscription_id="rollout-watch",
    requirements=("deployment target", "backup receipt"),
    task_key="rollout42", role="operator",
)
logger.info("memory_turn", extra=turn.diagnostics())
if not turn.wake_model:
    return

# Two hosts may inspect READY; only one can claim it.
try:
    lease = memory.claim(intent_id, worker_id=worker_id, ttl_seconds=300)
except RevisionConflict:
    return

# Recheck after claiming, before sending context to your model.
checked = memory.review_context(tuple(item.ref for item in turn.context.facts))
if not checked.complete:
    current = memory.get_intent(intent_id)
    memory.checkpoint(intent_id, lease_token=lease.token, expected_revision=current.revision,
        state=IntentState.READY, progress={"next": "reselect context"}, basis=())
    return
# Your runtime now uses checked.accepted and persists its resulting decision.
```

Do not mark a work item handled just because its memory delivery was acknowledged. If the host crashes before inference/checkpoint, the intent remains available or becomes READY after lease expiry. A received context is not a completed decision. Paid inference may repeat after a crash; external effects still require their own idempotency/reconciliation contract.

For `replan`, the host claims the blocked intent, records its new plan with fresh basis through `checkpoint(..., state=READY, basis=...)`, then claims before execution. The example shows this sequence and a source change between preparation and `begin_action`. Keep the existing guarded action path even after successful inspection/context review: neither read is an action permit.

## Opt in to semantic candidates

The recipe also provides async `acquire_missing_context(turn, selector, ...)`. It calls the explicit semantic selector only when the lexical turn is `missing_context` and the host supplies `relevant_event=True`. The host must set that flag only for a new relevant event or a deliberate bounded retry. Unchanged idle checks use `False`; every other turn status returns `None` without acquisition. The default scripted example remains model-free.

Create one selector per owner and fixed embedding space, and reuse it within its event loop. `host_embed_once` is an explicit host callback that makes at most one embedding request per invocation, without retries; the host owns provider configuration and aggregate metering.

```python
from ai_knot.always_on import SemanticBudget, SemanticContext
from examples.always_on_team import acquire_missing_context

selector = SemanticContext(memory, embed=host_embed_once)

# Inside your async event handler, after inspect_turn:
candidates = await acquire_missing_context(
    turn, selector,
    relevant_event=new_relevant_event,
    budget=SemanticBudget(max_candidates=128, max_input_bytes=65536,
                          max_provider_calls=1, timeout=5.0),
    task_key="rollout42", role="operator", max_facts=3, max_chars=600,
)
if candidates is not None:
    logger.info("semantic_candidates", extra={
        "status": candidates.status,
        "provider_calls_reserved": candidates.provider_calls_reserved,
        "input_bytes_reserved": candidates.input_bytes_reserved,
    })
```

Keep `turn.context`, `turn.status` and the original requirements. The returned `review` contains source-checked candidates; its `complete` field describes that review, not whether the task requirements have been satisfied. Candidate availability never promotes the turn to `ready` or authorizes inference or effects. Budget exhaustion, unavailable embeddings and source changes remain acquisition statuses for the host to handle under its retry policy. A host decision to proceed still requires the existing successful claim, fresh `review_context` of the chosen refs, checkpointed basis and guarded action path.

## Act on the diagnostic state

| Status | Host response |
|---|---|
| `ready`, `replan` | Candidate for inference after a successful claim and fresh context review; then persist the resulting decision |
| `catching_up` | Schedule another bounded inspection; never infer that an empty page means no updates |
| `debounced` | Return to the host timer; backlog without a token is not an ackable delivery |
| `snapshot_too_large` | Narrow/split the subscription and recover a complete selected manifest; no partial snapshot was acknowledged |
| `waiting` | Sleep until the host's next relevant event or timer |
| `verification_due` | Run the existing trusted-host verifier/maintenance under its budget, then inspect again |
| `missing_context` | Supply the missing observations, adjust applicability, or increase a justified budget; incomplete lexical coverage does not authorize inference/actions in this recipe |
| `lease_busy` | Another invocation still owns the intent; do not replace its live lease |
| `reconcile` | Query the external connector/receipt; unknown does not prove success or safe retry |
| `blocked`, `deadline_exceeded` | Preserve the block for host intervention; source changes do not override a manual block |
| `completed`, `cancelled` | Stop scheduling this work item |

Diagnostics contain status/reason codes, counts and journal frontiers. They omit fact/goal content, IDs and lease tokens. The returned context/resume packet is richer private host state; avoid dumping it into shared logs.

## Durable delivery and time

Each inspected delivery replaces the selected host context and saves its receipt in one local SQLite transaction **before ack**. A crash before that commit leaves the delivery pending. A crash after the commit but before ack safely replays the same delivery; replacing a cache is idempotent. This handles memory consumption, not exactly-once model calls or connector effects. A failed cache write is surfaced and does not advance the memory cursor.

Inspection scans at most four deliveries by default (configurable 1..20). Each library poll scans up to 1,000 raw journal rows and emits at most 100 selected events. It may return an empty filtered page with a delivery token and more backlog. The recipe persists/acks it and continues, staying quiet if its page budget is exhausted. Debounce with backlog and no token returns after one attempt. Retention/ACL resync uses the existing complete snapshot bound of 500; overflow preserves the old cursor/cache and requires a narrower working set.

Task context defaults to three full facts and 600 Unicode characters. These are content limits, not model-token limits or semantic completeness. Selection still indexes the retained corpus; this is a small working-set recipe rather than a production capacity guarantee. Catch-up, resume and context reads are separate transactions, so writes can occur between them. Use action-time guards and rebuild cached text on invalidation.

Use timers even with event notifications. Future activation, expiry, freshness deadlines, wait conditions and lease expiry can change admissibility without a journal event. The recipe checks saved refs and reselects context on every invocation, then resumes work state. A notification is only a hint to read durable tables; it cannot replace subscriptions/cursors. PostgreSQL LISTEN/NOTIFY can be added by the host if measured polling cost justifies it; retain reconnect catch-up through SQL.

## Preserve experience and adoption boundaries

After a connector provides a known recorded result, use [record_experience](team-experience.md) for a current-outcome lesson with retained observation refs and a stable experience key. Unknown outcomes require reconciliation first. A separate historical attempt warning derives from its own retained receipt. Shared lessons require maintain authority and readable roots; they do not grant execute permission. Copies count the original observations, not agents or repeated summaries.

The library keeps ordinary KB use simple and these operations opt-in. There is no default reflection prompt, graph service, paid idle loop or inference-capable scheduler. Framework-specific integrations can call the same HTTP/MCP/TypeScript operations described in the [return](shared-memory-return.md), [task context](task-context.md) and [team experience](team-experience.md) guides. Published package versions remain unchanged; the new operations require the matching source engine/client until a separate release.

Final local acceptance:37recipe SQL/CLIcases and1993fullPython tests passed with PostgreSQL required (two expected skips,89.38%coverage);59TypeScript tests/build/typecheck and all static/recovery/binding-MA gates passed. See the [four-stage completion audit](shared-memory-completion.md) and [source/test validation](superpowers/validation/2026-10-28-shared-memory-program.json). The audit distinguishes the independent review from owner verification of its manual-block repair.
