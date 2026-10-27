# Durable memory for always-on agents

This feature is implemented in the source checkout and is listed under
**Unreleased**. Install the matching source engine with `pip install -e ".[mcp,server]"`
and build the TypeScript client in `npm/` before using the new transport methods.
The published package version remains `0.12.0` until the release workflow runs.

AI Knot stores unfinished work, acknowledged memory changes, decision evidence,
external action outcomes and knowledge rechecks. The host runs tools and schedules
batches. Creating a normal `KnowledgeBase` starts no worker and enables no journal.

SQLite and PostgreSQL support these guarantees. YAML remains supported for legacy
memory operations and reports `UnsupportedAlwaysOnBackend` for always-on state.

For multiple logical agents, see [workspace coordination and shared memory](multi-agent-always-on.md).

```python
from ai_knot import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.always_on import AlwaysOnMemory, IntentState, WatchSelector

kb = KnowledgeBase("deploy-agent", storage=SQLiteStorage("memory.db"), embed_url="")
memory = AlwaysOnMemory(kb)  # explicit opt-in for this namespace
memory.subscribe(subscription_id="deploy-watch", selector=WatchSelector("deploy-agent"))
intent = memory.create_intent(client_key="deploy:42", goal="Deploy after CI")
lease = memory.claim(intent.id, worker_id="host-1")
current = memory.get_intent(intent.id)
memory.checkpoint(intent.id, lease_token=lease.token,
                  expected_revision=current.revision, state=IntentState.READY,
                  progress={"stage": "planned"})
```

A subscription keeps one durable delivery until `ack`. Store your inbox processing
receipt before acknowledgment. Reopening a facade or process returns the same
unacknowledged delivery. ACL revocation can redact its events while keeping its
identity and setting `resync_required`. Filtered pages may have an empty event list
and a nonempty delivery ID; acknowledge those pages too. Receipt history grows
until `unsubscribe`. A recreated subscription ID receives a new incarnation.

`capture_evidence` captures revisions of accessible, active facts. Save those refs
in intent checkpoints; actions inherit them. `begin_action` validates evidence,
ACL, freshness and the live lease in one transaction, commits `started`, then
returns an idempotency key. Pass it to the external API, then call `finish_action`.
This transaction does not include the external API. A remote service must itself
support idempotency or conditional fencing for those guarantees to extend remotely.

Expired leases fence old workers. A started action becomes `unknown`; a blocked
intent can receive a recovery lease for `reconcile_action`. A missing response is
not proof of failure: obtain independent external evidence. Until reconciliation,
AI Knot prevents another action from starting. Started actions also prevent a
checkpoint from releasing the lease. Unknown actions permit only blocked or
explicitly cancelled checkpoints, retaining their audit history.

Freshness is opt-in per slot or unslotted fact. Registration is not confirmation.
`confirmed` verifies the exact current revision for the configured interval;
`changed` and `unavailable` immediately block protected actions. `retry_at`
schedules verification; it never authorizes an action. Hosts inject typed verifier
objects; HTTP and MCP do not accept arbitrary fetch URLs, providers or credentials.

`record_verification(ref, result)` records the host's attestation against the current
policy. A host running its own asynchronous source checks must coordinate policy
changes before submitting that attestation. Inject a provider into `run_maintenance`
to use the built-in policy-generation guard: responses from an earlier registration
cannot confirm or overwrite a newer policy, even after an identical reset.

```python
from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance

report = await run_maintenance(memory, run_key="host-batch:42",
                               budget=MaintenanceBudget(max_model_calls=0))
```

A batch verifies due facts, consolidates supported episodes at least 24 hours old,
then applies existing pool TTL and safe journal pruning within the remaining item
budget. No private fact deletion TTL is added. Without providers it reports skips.
Providers declare a credential-free `config_key` and a conservative integer
`model_cost`; reservations and aggregate UTF-8 input bytes persist across crashes.
A timed out call is not refunded. Same-key terminal replay returns the report;
changed budget/provider configuration conflicts. Providers run outside SQL locks.
Consolidation checks input revisions and output-slot changes again before commit;
a concurrent correction causes a retryable conflict rather than an overwrite.
If pruning removed the history needed for that comparison, commit also conflicts.
Outstanding verifier results compare the captured full policy and its internal
monotonic revision before changing it. A reconfiguration, identical reset or newer
verdict invalidates the old response, including error/cancellation diagnostics.
Cancellation propagates even if best-effort diagnostics cannot be written.

`max_wall_seconds` is a cooperative scheduling timeout, not a hard realtime bound
for a connector that ignores cancellation. An optional workspace adds a persistent
daily maintenance budget; scheduling, credentials and actual external effects
remain responsibilities of the host.

## Transports

Inject `always_on=memory` into `server.create_app(kb, token=...)` or
`mcp_server._make_server(kb, always_on=memory)`. For CLI-served HTTP/MCP processes,
set `AI_KNOT_ALWAYS_ON=true`; it defaults to false and rejects YAML. HTTP routes
start at `/v1/always-on`; every route reuses the configured bearer authentication.
The configured KB owns every request; caller-supplied owner fields are rejected.
Shared-pool access requires explicit Python host injection with durable ACLs.

HTTP, MCP and CLI share strict JSON decoding: unknown typed fields, duplicate
keys, naive datetimes, nonfinite numbers and booleans in integer fields are
rejected. Arbitrary freeform JSON object keys are supported in progress, request,
result and evidence. Each logical JSON field is limited to 65536 UTF-8 bytes.
Pages default to 100 (maximum 500), raw change scans to 1000. Lease TTL is 1–3600
seconds and debounce is 0–3600 seconds. Dates are UTC ISO strings on the wire.

TypeScript exposes `client.alwaysOn` on `HttpKnowledgeBase` and MCP `KnowledgeBase`.
It creates no subprocess or polling until a method is called. All twenty Python
operations have typed equivalents. Errors retain `AlwaysOnError.code` rather
than relying on English text. Acknowledgment and unsubscribe resolve to void.

```sh
ai-knot --storage sqlite --data-dir .ai_knot work create deploy-agent \
  --client-key deploy:42 --goal 'Deploy after CI' --json
ai-knot --storage sqlite --data-dir .ai_knot work list deploy-agent --json
ai-knot --storage sqlite --data-dir .ai_knot changes subscribe deploy-agent watch --json
ai-knot --storage sqlite --data-dir .ai_knot changes poll deploy-agent watch --json
ai-knot --storage sqlite --data-dir .ai_knot maintenance run deploy-agent --run-key batch:1 --json
python examples/always_on_recovery.py --data-dir /tmp/knot-recovery
```

Checkpoint accepts `--input FILE` (stdin by default). Every command executes one
operation; no command starts a resident scheduler. The offline example uses three
real processes, deliberately exits after the external effect, reconciles from an
independent SQLite database and reports one effect even when run twice. A second
branch without external evidence remains blocked.

## Storage and operations

Back up the complete SQLite database with a consistent backup API, or the complete
PostgreSQL schema with a consistent database backup. Fact export and named fact
snapshots are not backups of agent state. The additive schema marker is
`ao_schema` version 6 (`ai-knot_ao_schema` on PostgreSQL). State tables are streams,
changes, subscriptions, delivery_receipts, intents, actions, freshness and
maintenance_runs, all prefixed `ao_` (PostgreSQL: `ai-knot_ao_`). A fact snapshot
restore emits new changes; it does not rewind actions, leases, receipts or jobs.

Do not run older writer binaries after enabling this journal: they do not append
its events. PostgreSQL writers lock all participating namespaces in sorted order
with stable advisory transaction locks. Write reads occur after locks under READ
COMMITTED; snapshot readers use REPEATABLE READ. SQLite uses BEGIN IMMEDIATE for
writes. Neither backend holds these locks across provider calls.

For PostgreSQL hosts, synchronize UTC clocks. Keep operational clock skew within
one second and choose leases with adequate scheduling/network margin; test the
host's clock monitoring separately. Local generation fencing still applies, but
AI Knot does not measure clock skew or fence a remote API that ignores its token.

Automatic pruning stops at the minimum acknowledged cursor. Forced manual pruning
can cause `CursorExpired` (HTTP 410). Recover by unsubscribing, subscribing at
`now` with a new incarnation, then reading current authorized memory. Reading a
snapshot before subscribing can lose intervening changes. Retain receipts until
unsubscribe; include their growth in capacity estimates.
