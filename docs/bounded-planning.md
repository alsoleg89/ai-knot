# One planning turn over current shared memory

Optional [required_refs](required-context.md) preserve host-known conditions in selected context and checkpoint basis. They enter the input fingerprint when supplied; this is not automatic prerequisite discovery.

The opt-in `plan_once` host recipe connects current task observations to one persisted proposal. It is useful when evidence is available but the lexical requirements use different wording: a missing keyword match need not leave an always-on team silent indefinitely. Existing inspection and its coverage diagnostics stay unchanged.

Run the complete model-free workflow:

```sh
PYTHONPATH=src python -m examples.always_on_planning
```

A research connector publishes shared observations, the operator waits for sign-off, sleeps through five idle checks, reconstructs its facade after restart and plans from the updated evidence. A separate application gate checks the structured observations before a guarded, scripted effect. An unknown response requires receipt reconciliation before completion. The example's callback and connector are scripted; SQL integration tests block inference and embedding HTTP.

## Connect your callback

These are source-checkout application examples, rather than a new stable library or scheduler API. Copy/adapt them in your host. The callback must be async and own aggregate provider request/token metering across all its agents and invocations. It receives JSON-compatible `goal`, original `requirements`, application `progress` and `facts` with one-based `index` and full `content`. The helper adds no internal source refs, lease token or work identity. Host-supplied progress itself must be appropriate to send to the provider.

```python
from examples.always_on_team import plan_once

# Inside your relevant-event/timer handler, after inspect_turn/catch-up:
report = await plan_once(
    memory, turn, metered_async_planner,
    trigger_key=stable_host_event_key,  # None on idle checks
    requirements=("activation destination", "approval receipt"),
    worker_id="event-host", task_key="rollout",
    max_model_calls=1,  # default 0 makes no callback
)
```

The callback returns a JSON **string** with exactly these fields:

```json
{"action":"plan","next_step":"Activate Boreal.","basis":[1,2],"data":{"operation":"activate","target":"Boreal"}}
```

`action` is `plan` or `wait`; a plan requires at least one current citation. A wait may have no citations and saves a 60-second host timer. `data` is an opaque application-owned object. Neither valid JSON, current citations, nor READY proves the proposed domain decision is correct. Validate the data and business preconditions independently, then use the existing claim/prepare/begin/finish/reconcile path for effects. The helper never invokes a connector, publishes a shared fact, records a verified lesson or marks work completed.

## Scheduling, bounds and recovery

Only inspected `ready`, `replan` and `missing_context` turns are candidates. Fresh authoritative reads reject manual blocks, deadlines, waiting/terminal state, live foreign leases, pending actions including PREPARED and due verification. Source-only blocks can be replanned after current evidence checks. A successful claim precedes the callback. **One controller must exclusively use that lease token until the helper returns**; do not concurrently use it from another task/thread/process. The callback receives no token.

The small-task path uses `task_snapshot`: all admitted task/generic facts must fit the default 8-fact/12,000-character output budget. Empty snapshots make no callback. Overflow makes no callback by default; an explicit optional semantic selector can supply a checked subset as described below. The full guard universe remains capped at 128 admissible facts. These bounds limit output/candidates, not SQL scan/lineage cost. Serialized public input is capped at 65,536 UTF-8 bytes. Callback output is capped at 16,384 UTF-8 bytes before strict parsing; duplicate keys, nonfinite values, invalid/duplicate/bool/out-of-range citation indices and empty/oversized steps are rejected. The planner timeout defaults to 30 seconds and is configurable up to 60 seconds against a 300-second lease. Async cancellation is cooperative; a callback that suppresses it can return later. The helper rechecks lease and deadline before accepting that result.

`model_calls_reserved` counts local planner invocations, including failures, invalid output and timeouts. `acquisition_calls_reserved` is separate and reflects the semantic selector's reservation on normal return; cancellation/failure without a result retains a conservative upper reservation if an embedding call was authorized. Neither is a billable-request count or global budget. There are no automatic model retries. Configure a durable shared meter across both callbacks if cross-process accounting is required.

The latest attempted input marker is stored privately under `progress["_ai_knot_planning"]`. Identical immediate replay after reopening sees `unchanged` before paid acquisition or planning. Trigger, goal/original requirements, admitted-set guard, selection/bounds and application progress enter that fingerprint; the helper's own proposal/metadata do not. Prior proposals are not automatically fed back as evidence. A new trigger or changed progress/evidence can permit another explicitly budgeted attempt. Callback identity, model/embedding space, budget and timeout settings are not inferred from Python objects: after a paid attempt, deliberately changing those requires a fresh host trigger. A zero-call acquisition refusal or cancellation/failure with zero planner and acquisition reservations stores no suppression key, allowing a bounded retry of that event. This is one latest marker, not a history of every event or an exactly-once inference ledger; a crash before checkpoint or A→B→A can repeat a call. The guarded recipe uses a new fingerprint policy; old recipe markers are not a migration-safe inference ledger.

Before retrieval the helper captures a [task-context guard](task-context-guard.md). After provider I/O it resumes authority, compares application progress and reviews chosen refs; the final checkpoint atomically compares the entire admissible universe under its source locks as well as selected basis/intent revision. Admissible additions/deletion/revisions and admission changes reject the proposal, including changes after the last host review. The successful report/private marker includes `context_guard`; pass it explicitly to `begin_action` again to reject later unselected changes before an effect. A checksum does not certify selected-context sufficiency or the future world. PREPARED actions do not bump intent revision; concurrent reuse of the same token remains outside the exclusive-controller contract.

## Opt in when the whole task corpus does not fit

```python
from ai_knot.always_on import SemanticBudget, SemanticContext

selector = SemanticContext(memory, embed=metered_embed_once)
report = await plan_once(
    memory, turn, metered_async_planner,
    trigger_key=stable_host_event_key, requirements=requirements,
    worker_id="event-host", task_key="rollout", max_model_calls=1,
    max_facts=8, max_chars=2000,
    selector=selector, semantic_budget=SemanticBudget(max_provider_calls=1),
)
```

Reuse the selector in its fixed owner/embedding space. Only whole-snapshot overflow enters acquisition; fitting snapshots use no embeddings. The default semantic budget authorizes zero provider calls, including when a selector is supplied. The host rechecks authority before acquisition, rechecks its full guard after acquisition before planning, and accepts only fresh, nonempty reviewed candidates. Semantic omissions are retained as selection semantics; their presence is not a source-review failure or a task-readiness verdict. An omitted pre-existing condition can still make a model decision wrong. The original requirements/lexical observation stay intact. No SQL transaction spans either callback; acquisition timeout is also limited to 60 seconds in this recipe.

Safe failed attempts clear obsolete next-step/basis, persist a sanitized status and release their lease through checkpoint. A newly blocked intent or replacement lease is preserved; cleanup may leave the old lease to expire. If checkpoint/cleanup cannot persist, the returned failure reservation remains local and replay suppression is not durable. Cancellation is re-raised after best-effort cleanup. No SQL transaction spans callback I/O.

Useful statuses include `planned`, `waiting`, `unchanged`, `idle`, `skipped`, `budget_exhausted`, `blocked`, `lease_busy`, `no_context`, `snapshot_too_large`, `input_limit`, `context_changed`, `invalid_proposal`, `unavailable`, `failed` and acquisition-prefixed `budget_exhausted`, `input_limit` or `unavailable`. A host chooses its bounded retry/maintenance policy; polling a paid failure with a fabricated new trigger is still paid work.

See the [bounded-planning validation](superpowers/validation/2026-10-28-bounded-planning-turn.json) for SQL/compatibility checks and the frozen GPT‑4.1 mini smoke. The smoke compares the unchanged lexical gate, a simple current-task-bundle host and this helper with identical bundle inputs, prompt, schema and application/action gate. It evaluates adoption of this small workflow, not general memory accuracy or a real external effect.
