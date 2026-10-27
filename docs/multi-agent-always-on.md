# Memory for several always-on agents

This guide describes the Python host API in this checkout. It extends the
[single-agent always-on API](always-on.md) on SQLite and PostgreSQL without
starting a scheduler, model worker or network service.

Keep three identities separate: the workspace coordinates shared work, an agent
owns private memory and intents, and a worker is a replaceable process executing
an agent's work. A business `operation_key` survives retries and agent handoff;
it is not an attempt ID or a hash of an arbitrary request.

## Configure a team

```python
from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory, WorkspaceMemory, configure_workspace_budget,
)
from ai_knot.storage.sqlite_storage import SQLiteStorage

storage = SQLiteStorage("team.db")
workspace = WorkspaceMemory(storage, "deployment-team")
pool = SharedMemoryPool(storage, persist_stats=True)

# Trusted host configuration; neither call starts an agent.
workspace.grant("researcher", "maintain")
workspace.grant("operator", "execute")
pool.grant_read("researcher", "team:ops")
pool.grant_read("operator", "team:ops")
configure_workspace_budget(workspace, max_model_calls_per_day=100)

researcher = AlwaysOnMemory(
    KnowledgeBase("researcher", storage=storage, embed_url=""),
    pool=pool, workspace=workspace,
)
operator = AlwaysOnMemory(
    KnowledgeBase("operator", storage=storage, embed_url=""),
    pool=pool, workspace=workspace,
)
```

Each process can reopen these objects against the same storage. Grants and budget
configuration persist; setup need not be repeated on every poll. `grant`/`revoke`
are host administration calls. `maintain` permits shared freshness registration,
verification, derivation and cleanup; `execute` permits workspace operations.
Neither permission grants read access to a fact. Source ACLs still apply.

`WorkspaceMemory` is coordination inside a trusted host, not tenant
authentication. Shared facts still live in `__shared__` for the selected database
or schema. Different workspace IDs do not isolate those facts. Independent tenants
need separate databases or schemas with appropriate database-role privileges.
HTTP/MCP continue to use their configured owner; this API does not introduce a
multi-principal transport. Do not accept caller-supplied agent identities or hand
out direct database credentials as a substitute for authentication.

## Coordinate external operations

```python
intent = operator.create_intent(client_key="deploy:build-42", goal="Deploy build 42")
lease = operator.claim(intent.id, worker_id="host-1")
action = operator.prepare_action(
    intent.id, lease_token=lease.token, action_key="deploy",
    operation_key="release:build-42:production",
    resource_key="deployment:production",
    request={"build": "42", "environment": "production"},
)
permit = operator.begin_action(action.id, lease_token=lease.token)
# Call the external service OUTSIDE the memory transaction. Pass:
# permit.idempotency_key, permit.resource_key, permit.fencing_token.
# Then finish_action(...), or reconcile_action(...) after an uncertain outcome.
```

Both shared keys are required together. The workspace checks the current execute
grant and reserves the resource atomically with the action. Different intents or
agents cannot reserve the same resource for conflicting operations. Matching
replays use the same external idempotency key; a changed payload or resource raises
`IdempotencyConflict`. Matching terminal replays return the recorded terminal
result without issuing another permit.

A prepared operation can hand off by preparing it on another authorized agent's
intent. The resource generation increases and the former action loses authority.
`workspace.abandon(agent_id, operation_key)` releases an owner's prepared
reservation with a durable failed outcome. It cannot abandon started or unknown
effects. Reopening a bound action without its matching workspace fails closed.

Started/unknown operations retain resource exclusion after lease expiry. The
original logical owner must recover and reconcile with external evidence. Explicit
retryable reconciliation returns the operation to prepared and increases fencing
generation; it does not prove a remote failure by itself. Succeeded/failed outcomes
release the resource. A revoked execute grant blocks further transitions; the host
must explicitly restore authority for legitimate recovery.

The external service must enforce the stable idempotency key or fencing token to
extend these guarantees to real effects. A local SQL transaction cannot cancel an
already-sent request. Keep the external receipt and use reconciliation for unknown
outcomes. Legacy actions omitting the two shared keys retain per-intent semantics.

## Share rechecks and control cost

Shared freshness registration and verdicts require `maintain` and current source
readability. Ordinary readers can maintain their private state without authority
to modify shared policies. Private facts retain their existing ownership rules.

Maintenance claims a shared fact before invoking the verifier. Exclusion is keyed
by the actual shared subject in the database, including callers attached to
different workspace coordinators. The callback must still match its claim,
evidence revision, policy revision, lease deadline and current authority. A stale
callback cannot release a newer claim.

Timeouts, cancellations and unknown provider results retain the claim until its
deadline. Successful guarded writes release it. Timeout is cooperative: a connector
must honor cancellation; a remote request that ignores it may outlive the local
deadline. Claims fence local commits and bound retries, not remote execution.

The workspace budget charges every maintenance provider reservation in that
workspace: shared/private verification and private consolidation. It measures the
sum of integer provider `model_cost` units per UTC calendar day, from 00:00 inclusive
to next 00:00 exclusive. No configuration permits zero paid calls. Changing a limit
does not reset usage; failed, cancelled or crashed reservations are not refunded.
The per-run `MaintenanceBudget` remains an additional ceiling. Hosts must use
coherent UTC clocks across workers.

Durable subscriptions already provide independent cursors per logical agent. Save
a processing receipt before `ack`; repeated deliveries must be safe. Host scheduling
should subscribe to relevant facts/slots and avoid waking every model on every
change. The library adds no background polling and makes no idle model calls.

## Preserve independent evidence

```python
refs = researcher.capture_evidence((("__shared__", "source-fact-id"),))
summary = researcher.derive_fact(
    "A concise statement supported by the source",
    basis=refs, namespace="__shared__", visibility_scope="team:ops",
)
summary_refs = researcher.capture_evidence((("__shared__", summary.id),))
roots = researcher.independent_evidence((*refs, *summary_refs))
# Repeated derivation does not add a new root for the same observation.
```

Derived facts persist flattened namespace/fact/revision roots and input dependencies
in qualifiers, each bounded to 128 unique references. Copies share roots for evidence
counting; intermediate revisions and ACLs still constrain descendants. Source
correction, deletion or read revocation prevents future guarded use and recall of
the derived text. Consolidation records exact archived input revisions in the same
transaction that closes episodes and stores the summary. Historical inputs may be
closed; changing their recorded revision still invalidates descendants.
Archiving carries an exact matching freshness attestation to the closed revision
without renewing its original deadline. Actions also check the freshness policies
of roots and intermediate inputs.

Both agent sync APIs apply the same ACL/dependency checks as recall. Revoking
access or changing an input emits a blank invalidation for previously delivered
cached facts, even without a new pool version. Process those control invalidations
even when their version equals the cached version. Sync cursors and delivery
metadata remain instance-local; reopen clients with a full sync.

This records host-declared dependencies, not a mathematical proof that a generated
sentence follows from them. Arbitrary text imported without lineage is not analyzed
for hidden dependencies. Already-read model context cannot be erased by revoking
future access. Raw host administration/export APIs remain trusted operations.
New ordinary publications carry an opaque source-revision identity so the private
original and its accessible published copy count as one observation. Existing
unstamped publications are not guessed or retroactively relabeled.

## Durable statistics

Persistent SQL pools merge counter deltas and consumer-set unions atomically.
Repeated flush does not double increments, and existing instances refresh peer
trust totals. SQL publication commits facts, trust events and statistics together.
Legacy snapshots are retained without resetting counts. Recall usage is buffered
until `flush_stats()` or a later publication/recall; a crash can lose unflushed usage.
An ambiguous network failure at SQL commit is not automatically retried.

## Local correctness and performance checks

```sh
pytest tests/always_on --no-cov
AI_KNOT_REQUIRE_POSTGRES=1 pytest tests/always_on --no-cov
python scripts/benchmark_multi_agent.py --backend sqlite --agents 8 --operations 25
python scripts/benchmark_multi_agent.py --backend postgres --agents 8 --operations 25
```

Set `AI_KNOT_TEST_PG_DSN` for PostgreSQL. Tests create isolated schemas. The benchmark
CLI creates and drops its own disposable database/schema; its Python callable
expects an already isolated empty store. Each agent uses a separate process and
connections for mixed publication/recall and resource contention. Shared refresh
uses an asynchronous fleet with independent connections and a scripted verifier.
No external embedding/model endpoint is called.

Reports include workload/environment, throughput and monotonic p50/p95 latencies,
plus exact publication/usage/consumer totals, one resource winner, one shared
provider call and zero idle calls. Correctness assertions do not impose a
machine-specific latency threshold. See [local measurements](performance/multi-agent.md).
