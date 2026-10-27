# Always-on P3 — concrete maintenance, transport and recovery contracts

This is implementation reference for [P3](../plans/2026-10-26-always-on-03-maintenance-surfaces.md),
based on the current `0.12.0` sources. New routes, modules and SDK methods below
record the original implementation blueprint. The source implementation is complete; see [final verification](../implementation-final.md). The new API awaits a package release. The canonical
[specification](../specs/2026-10-26-always-on-memory-design.md) defines public Python
signatures. Code fragments specify changes to those files; they are not a second API.

## Current code and exact integration seam

| Current file / symbol | Concrete change | Regression coverage |
|---|---|---|
| `src/ai_knot/learning.py::_LearningMixin.consolidate_episodic` | Replace the call to mutating `learn_async` followed by whole-namespace `save`. Select bounded old episodes, prepare detached candidates, then guard and atomically insert/close only covered inputs. | maintenance races plus legacy learning suites |
| `learning.py::_extract_phase`, `_candidate_phase`, `_resolve_phase`, `_consolidate_phase`, `_commit_phase` | Reuse extraction/resolve logic after separating preparation from persistence. `_resolve_phase` and `_consolidate_phase` mutate their input Fact objects: copy inputs first; do not let provider preparation mutate live storage state. `_commit_phase` must not remain a whole-namespace overwrite. | unchanged retrieval/learning behavior; foreground insertion survives |
| `src/ai_knot/types.py::Fact.provenance`, `Provenance` | `Provenance.consolidation_ids` already exists but accessor does not reconstruct it. Decode the qualifier string into a tuple, validate all entries are strings; missing field means empty tuple. | SQL/YAML/snapshot round-trip |
| `storage/sqlite_storage.py::_build_rows`, `_fact_from_row`; `postgres_storage.py::_build_rows`, `_fact_from_row`; `yaml_storage.py::_fact_to_dict` and load | Existing `qualifiers: dict[str,str]` serialization already works; persist `qualifiers["consolidation_ids"] = json.dumps(ids)`. Preserve other qualifiers. No required Fact column for this list. | existing storage compatibility tests plus new lineage tests |
| `src/ai_knot/server/app.py::create_app`, nested `require_auth` | Add keyword-only `always_on=None`, include new router only on injection; reuse the existing bearer dependency for every route. Existing `/health` remains unchanged. | `tests/test_server.py`, new surfaces suite |
| `src/ai_knot/config.py::AIKnotConfig.from_env` | Add `always_on_enabled=False`; reject enabled YAML before creating any sidecar database. | `tests/test_config.py` |
| `src/ai_knot/cli.py::serve`, `serve_mcp`; `mcp_server.py::_build_kb`, `_make_server`, `main` | Construct scoped facade only when explicitly enabled, pass it to app/server; add keyword-only `always_on=None` to `_make_server`. `_make_kb` currently reads Click options; `serve` must explicitly propagate enable config rather than assuming config.from_env is already used there. | CLI and real MCP subprocess tests |
| `src/ai_knot/_mcp_tools.py::tool_capabilities` | Keep JSON array with `name` and `description`; append enabled AO entries only. Keep legacy tools and output shape intact. | `tests/test_mcp_tools.py`, `test_mcp_e2e.py` |
| `npm/src/index.ts::KnowledgeBase` | Public MCP facade lives here, not in `client.ts`. Add `readonly alwaysOn` using a callback bound to this facade's existing `McpClient`. Export types and `AlwaysOnError`. | new AO fixtures plus index/API/ESM/CJS tests |
| `npm/src/client.ts::McpClient.call`, `_handleLine` | Reuse lazy subprocess/JSON-RPC lifecycle. AO returns structured JSON tool text. Preserve domain error code; never infer it from an English exception. | crash/reconnect/pending-request tests |
| `npm/src/http.ts::HttpKnowledgeBase`, private `requestJson`, `requestVoid`, `buildHttpError` | Construct AO adapter with bound callbacks inside class; keep fetch/auth configuration centralized. `DELETE unsubscribe` uses requestVoid. Parse AO structured errors into `AlwaysOnError`; preserve legacy error text behavior. | HTTP SDK tests with injected fetch |
| `.github/workflows/ci.yml::test-postgres`, `test-npm-client`, `ma-gate` | Extend existing jobs; do not create a mock PostgreSQL proof or change S8–S26 thresholds. Existing live PG command discovers new tests. Required AO PG runner fails on missing DSN. | CI report and runner exit status |
| `.github/workflows/benchmark.yml::benchmark` | Existing workflow runs on push/PR, not nightly. Add a separate explicit `schedule`/`workflow_dispatch` large replay job for 100k events; keep bounded PR smoke in normal CI. | uploaded AO JSON report with backend and skipped/required flags |

## Bounded async maintenance: executable model and production contract

Run the stdlib reference:

```sh
.venv/bin/python docs/superpowers/reference/always_on_maintenance_reference.py
.venv/bin/ruff check docs/superpowers/reference/always_on_maintenance_reference.py
.venv/bin/ruff format --check docs/superpowers/reference/always_on_maintenance_reference.py
```

The reference uses real SQLite transactions and a persistent `runs` ledger. It
executes the reservation → outside-transaction provider → guarded-commit protocol.
The ledger keeps the original run digest, generation, reserved model calls and
input bytes across reopen and expired-lease takeover. A repeated finished run
returns its stored report. A concurrent live attempt gets `run_in_progress`;
it must not replace the in-progress row or perform another call. A different
budget/provider configuration under the same key gets `IdempotencyConflict`.
After expiry, a replacement generation inherits spent reservations. A timeout,
provider exception or crash after reservation does not refund model calls.

Production `run_maintenance(memory, *, run_key, budget, verifier=None,
consolidator=None)` uses P1's real `AlwaysOnSession`, not this model's tables.
Provider identity/configuration used in the run digest must be stable, explicit
and credential-free; include model cost classification and model/prompt version.
There is no durable daily quota subsystem. Hosts supply daily/money limits.

1. Select due requests and active supported old episodes with
   `created_at <= clock()-24 hours` (existing consolidation default), SQL `LIMIT`, deterministic
   ordering and a shared `max_items` allowance. Do not load an unbounded namespace
   and then slice. Query GC/prune work with the same remaining allowance.
2. Reserve each configured external operation before invoking it. A verifier
   without an LLM has model cost zero, a model-backed verifier declares its cost;
   consolidation's extraction/embedding/resolver subcalls each need accounting.
   A wrapper that reserves one call but internally makes three is invalid.
3. Count canonical UTF-8 request bytes cumulatively for the batch, before calls.
   Output logical JSON fields also obey the 64KiB field bound and Fact validation.
4. Track scheduling deadline with injectable monotonic time. Pass remaining
   timeout to the connector and wrap cancellable work with `asyncio.wait_for`.
   Timeout is cooperative: a connector that ignores cancellation is not a hard
   realtime bound. Recheck deadline before every new unit/call, not just once.
   Catch `asyncio.CancelledError` separately from operational failures and always
   re-raise it. On cancellation, unavailable/report persistence is best-effort:
   a failed guard or diagnostic write cannot replace cancellation with Conflict.
5. Outside write transactions, produce detached structured candidates and exact
   input refs. Empty facts/coverage, duplicate/unknown source IDs, output ID
   collisions and incomplete provenance mappings reject the entire proposal.
   The union of all output `consolidation_ids` equals covered episode IDs; every
   covered episode belongs to the selected refs. All selected refs remain current.
6. In one short transaction, acquire the P1 namespace locks, reread refs/ACL,
   verify active inputs and current unexpired maintenance generation, validate
   source mappings, resolve against current target slots, insert candidates and
   close only covered inputs. Fact writes, journal events and ledger counters
   commit together. Fresh foreground episodes are neither closed nor overwritten.
7. Store the terminal report/proposal diagnostics under the same generation CAS.
   Late workers can report local conflict but cannot overwrite a successor's
   status, counters or facts. Keep uncommitted proposals for bounded diagnostics;
   never apply them automatically after a restart without rechecking guards.

Report fields are `run_id`, `status`, `inspected`, `verified`, `consolidated`,
`expired`, `model_calls_reserved`, `skipped`, `reasons`. Counters describe committed
effects: `verified` counts accepted confirmed checks, `consolidated` counts closed
covered episodes, `expired` counts committed GC removals. Provider-disabled work
increments `skipped` and has explicit reasons; zero work can complete without calls.
Budget exhaustion is a terminal result for that run key; the host supplies a new
run key for another batch. Source unavailability is an outcome, not fabricated
verification success and not a reason to erase facts.

`VerificationResult(outcome, evidence, source_version)` must carry the connector's
evidence pointer. Do not expose an arbitrary URL fetch operation through HTTP/MCP.
For current revision confirmed, set `checked_at=now`, `verified_until=retry_at=
now+interval`, `failure_count=0`. For unavailable, preserve Fact, last successful
verification boundary and checked time, increase failures, set retry by
`min(3600, 30 * 2**min(failure_count-1, 7))` seconds, and immediately block protected
actions via the unavailable outcome. `retry_at` never authorizes an action.
For changed, record `checked_at=now`, `verified_until=min(previous,now)` and
`retry_at=now+interval`; host writes a real replacement via `add_resolved`.
A newly changed slot frontier is due immediately even if the old policy retry
time is in the future. Old-ref confirmation cannot verify a new revision.

Reference limits are deliberate: one namespace, simplified Fact rows, string
verification outcomes, no age fields/24h cutoff, no ACL/journal/full slot resolver, no GC and no PostgreSQL.
It proves control flow and the stated miniature transaction predicates, not the
production feature, live PostgreSQL correctness, all provider cancellation modes
or performance. Production adapters must use full `VerificationResult` and the
P1/P2 locks, revision function, permissions and error classes. Production SQL calls
use `AlwaysOnSession.load_facts/replace_facts` and parameterized
`fetch_one/fetch_all/execute(SQLStatement(sqlite,postgres),params)` on the same
transaction. No session.save API or nested raw connection is introduced. Migration
versions are v1 journal, v2 subscriptions/receipts, v3 intents, v4 actions, v5
freshness, v6 maintenance; each addition remains rerunnable and forward compatible.

## Codec: one canonical wire representation

Create `src/ai_knot/always_on/codec.py`. It encodes dataclasses recursively,
enums as their values, tuples as arrays, `None` as JSON null, and aware datetimes
as normalized UTC ISO-8601 strings (`2026-10-26T09:00:00+00:00`). Decoder accepts
aware offsets then normalizes UTC; naive timestamps are rejected. Use a single
canonical serialization for idempotency digests:

```python
json.dumps(value, sort_keys=True, separators=(",", ":"),
           ensure_ascii=False, allow_nan=False).encode("utf-8")
```

Validate the 65536 UTF-8 byte bound separately for each logical JSON field
(progress/request/result/evidence/selector), after strict JSON parsing and before
mutation. The bound is not a whole-response or multi-record-batch limit. Reject NaN/Infinity, duplicate JSON object keys, non-string
map keys, unknown fields at every typed layer, booleans in integer fields,
nonpositive/nonfinite budgets and invalid enum values. Freeform `progress`,
`request`, `result`, `evidence` remain validated JSONObject values with arbitrary
user keys. Do not apply an object-level `extra=forbid` to their user keys.

Wait-condition tags are fixed as follows (lower snake case); no executable text:

```json
{"kind":"until","at":"2026-10-26T09:00:00+00:00"}
{"kind":"slot","namespace":"deploy-agent","slot_key":"ci::status","value_text":"green"}
{"kind":"change","ref":{"namespace":"deploy-agent","fact_id":"ci-1","revision":"78474aa3f6a0e1d8cff389dc86aacd14e5dc75af81d52652e00c3d11fc2ed191"}}
```

`EvidenceRef` is `{namespace,fact_id,revision}`. `WatchSelector` is
`{namespace,fact_ids,slot_keys}`; omitted ID arrays default to empty. Selector
namespace is the sole scoped selector field, and facade validates access. Owner
identity never comes from body/query/header. `capture_evidence` input uses
`{"refs":[["deploy-agent","ci-1"]]}` matching Python tuple pairs; do not silently
substitute unchecked revisions from callers for a consistent capture.

HTTP success returns the encoded public object directly (list methods return
arrays); ack/unsubscribe return HTTP 204 with no body. MCP ack/unsubscribe return
JSON `null` as tool text. A batch without new events is
`{"delivery_id":null,"events":[],"resync_required":false}`. Fully filtered scanned
events may return an opaque nonnull delivery ID with empty events; persist/ack it
to advance scan position. Never interpret the token as a sequence.

Domain errors have a shared envelope and a safe message:

```json
{"error":{"code":"LeaseLost","message":"The lease is no longer current."}}
```

| HTTP status | Domain codes / condition | SDK behavior |
|---|---|---|
| 401 | existing missing/invalid bearer token | preserve existing auth semantics |
| 403 | `AccessDenied` | `AlwaysOnError.code` preserved; no hidden IDs/content |
| 409 | `RevisionConflict`, `LeaseLost`, `IdempotencyConflict`, `StaleEvidence`, `InvalidDelivery` | no automatic mutation retry |
| 410 | `CursorExpired` | host resync; never silently skip missing range |
| 422 | invalid typed request / unknown fields / invalid JSON | no durable writes; safe `InvalidInput` boundary envelope |
| 501 | `UnsupportedAlwaysOnBackend` | no fallback SQLite file |

Missing owner-bound intent/action/subscription IDs use `AccessDenied` uniformly
with foreign IDs, avoiding cross-owner existence leaks. Unknown internal failure
uses HTTP 500 with sanitized `InternalError`; logs may contain server diagnostics,
wire output never contains stacktrace, bearer token, DSN or connector credential.
Invalid JSON/oversize logical fields are rejected before facade mutation; choose the above
422 consistently across HTTP/MCP, not framework-specific permissive coercion.
Source `unavailable` returns a normal FreshnessRecord, not HTTP 503 or StaleEvidence.

## Complete method/route/tool map

All HTTP paths have prefix `/v1/always-on`. `I`/`A`/`S` are path IDs supplied by
host and excluded from HTTP body. MCP dictionaries include the corresponding
`intent_id`/`action_id`/`subscription_id`. All ordinary parameters retain the
Python snake_case names. SDK camelCase methods convert names only at the boundary.

| Python / TS method | HTTP | MCP tool | Remaining request keys |
|---|---|---|---|
| create_intent / createIntent | POST `/intents` | `intent_create` | client_key, goal, next_step?, deadline? |
| get_intent / getIntent | GET `/intents/I` | `intent_get` | none |
| list_intents / listIntents | GET `/intents?limit=100` | `intent_list` | limit? (query for HTTP) |
| claim / claim | POST `/intents/I/claim` | `intent_claim` | worker_id, ttl_seconds? |
| renew / renew | POST `/intents/I/renew` | `intent_renew` | lease_token, ttl_seconds? |
| checkpoint / checkpoint | POST `/intents/I/checkpoint` | `intent_checkpoint` | lease_token, expected_revision, state, progress, next_step?, waiting_for?, basis? |
| resume / resume | POST `/intents/I/resume` | `intent_resume` | none |
| capture_evidence / captureEvidence | POST `/evidence/capture` | `memory_evidence_capture` | refs |
| subscribe / subscribe | POST `/subscriptions` | `memory_subscribe` | subscription_id, selector, start?, debounce_seconds? |
| poll / poll | POST `/subscriptions/S/poll` | `memory_poll` | limit? |
| ack / ack | POST `/subscriptions/S/ack` | `memory_ack` | delivery_id |
| unsubscribe / unsubscribe | DELETE `/subscriptions/S` | `memory_unsubscribe` | none |
| prepare_action / prepareAction | POST `/actions` | `action_prepare` | intent_id, lease_token, action_key, request, basis? |
| begin_action / beginAction | POST `/actions/A/begin` | `action_begin` | lease_token |
| finish_action / finishAction | POST `/actions/A/finish` | `action_finish` | lease_token, outcome, result |
| reconcile_action / reconcileAction | POST `/actions/A/reconcile` | `action_reconcile` | lease_token, outcome, evidence |
| set_freshness / setFreshness | POST `/freshness` | `memory_freshness_set` | ref, source_ref, recheck_after_seconds |
| due_verifications / dueVerifications | GET `/freshness/due?limit=100` | `memory_freshness_due` | limit? |
| record_verification / recordVerification | POST `/freshness/record` | `memory_freshness_record` | ref, result |
| run_maintenance / runMaintenance | POST `/maintenance/run` | `memory_maintenance` | run_key, budget |

Providers are host-injected Python objects. Wire maintenance accepts no provider,
model, arbitrary URL, secret or callable payload. GET operations remain domain
read-only; resume is POST because it can reconcile wait/lease state.

## HTTP/MCP golden exchanges

With a configured SQLite app bound to `deploy-agent`, the create input is:

```http
POST /v1/always-on/intents
Authorization: Bearer example-token
Content-Type: application/json

{"client_key":"deploy:42","goal":"Deploy after CI is green","next_step":"wait for CI"}
```

The response is HTTP 200 and a canonical Intent (opaque IDs below are fixture values):

```json
{"id":"intent-42","owner_agent_id":"deploy-agent","client_key":"deploy:42","goal":"Deploy after CI is green","state":"ready","progress":{},"next_step":"wait for CI","waiting_for":null,"revision":1,"basis":[],"lease":null,"lease_generation":0,"blocked_reasons":[],"deadline":null,"last_transition_at":"2026-10-26T09:00:00+00:00"}
```

An identical create retry returns the same ID even if the original HTTP response
was lost; a different goal with that key returns 409 `IdempotencyConflict`.
Adding `owner_agent_id`, `namespace` or an unrecognized nested typed field returns
422 before creating an intent. Claim sends
`{"worker_id":"host-1","ttl_seconds":60}` and returns the Lease object
`{"token":"opaque-lease","generation":1,"worker_id":"host-1","expires_at":"2026-10-26T09:01:00+00:00"}`.

The corresponding MCP JSON-RPC exchange uses the same body and encoded result:

```json
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"intent_create","arguments":{"client_key":"deploy:42","goal":"Deploy after CI is green","next_step":"wait for CI"}}}
{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"intent_claim","arguments":{"intent_id":"intent-42","worker_id":"host-1","ttl_seconds":60}}}
```

MCP returns its standard `result.content` text block containing exactly the
encoded JSON object. Domain failure is the shared JSON error envelope in that
text block; protocol errors retain normal JSON-RPC errors. `McpClient.call`
already joins text blocks; `McpAlwaysOnMemory` parses once and checks the envelope.
Do not add a new subprocess/protocol client to interpret these tools.

Freshness record input, including evidence pointer (no replacement fact text):

```json
{"ref":{"namespace":"deploy-agent","fact_id":"ci-1","revision":"78474aa3f6a0e1d8cff389dc86aacd14e5dc75af81d52652e00c3d11fc2ed191"},"result":{"outcome":"unavailable","evidence":{"connector":"ci","request_id":"check-42","reason":"timeout"},"source_version":""}}
```

Fixture record response after the first failure at 09:00 UTC, with an old verified
boundary at 08:59 UTC:

```json
{"ref":{"namespace":"deploy-agent","fact_id":"ci-1","revision":"78474aa3f6a0e1d8cff389dc86aacd14e5dc75af81d52652e00c3d11fc2ed191"},"subject_kind":"slot","subject_key":"ci::status","source_ref":"ci:build:42","recheck_after_seconds":60,"checked_at":"2026-10-26T08:58:00+00:00","verified_until":"2026-10-26T08:59:00+00:00","retry_at":"2026-10-26T09:00:30+00:00","last_outcome":"unavailable","failure_count":1}
```

## Adapter implementation fragments

New router boundary inside `server/always_on.py`; typed `CheckpointRequest`
and nested conditions use the shared decoder/strict fields. Duplicate-key rejection executes before framework JSON coercion; field byte limits
execute before facade mutation, and validation
error serialization must not echo raw secret input. The pattern for each handler:

```python
@router.post("/intents/{intent_id}/checkpoint", dependencies=[Depends(require_auth)])
def checkpoint(intent_id: str, req: CheckpointRequest) -> JSONValue:
    value = memory.checkpoint(
        intent_id, lease_token=req.lease_token,
        expected_revision=req.expected_revision, state=req.state,
        progress=req.progress, next_step=req.next_step,
        waiting_for=req.waiting_for, basis=req.basis,
    )
    return encode_always_on(value)
```

`create_app` appends `app.include_router(build_always_on_router(always_on,
require_auth))` only for injected facade. Domain exception handling belongs to the
new router/boundary so old HTTP route error contracts stay unchanged. App creation,
imports and `client.alwaysOn` construction never call a model or start polling.

`npm/src/always_on.ts` receives closures, allowing access to existing private
HTTP methods without exposing them or cloning headers/fetch logic:

Current `npm/src/http.ts::requestVoid` accepts only DELETE and sends no body.
Before binding the new closure, replace that helper with this concrete body;
otherwise POST acknowledgement either fails typing or loses delivery_id:

```typescript
private async requestVoid(
  path: string,
  options: {
    method: "POST" | "DELETE";
    body?: Record<string, unknown>;
    auth?: boolean;
  },
): Promise<void> {
  const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
    method: options.method,
    headers: this.buildHeaders(options.body !== undefined, options.auth !== false),
    body: options.body !== undefined ? JSON.stringify(stripUndefined(options.body)) : undefined,
  });
  if (!response.ok) {
    throw await this.buildHttpError(response);
  }
}
```

Reuse existing `stripUndefined`, `buildHeaders` and `buildHttpError`. This helper
does not parse a 204 body. Tests assert POST body+Content-Type+auth for ack and
unchanged DELETE without body/JSON header, plus no response.json() on success.

```typescript
type JsonObject = { [key: string]: JsonValue };
type JsonValue = null | boolean | number | string | JsonValue[] | JsonObject;
type JsonRequest = <T>(path: string, options: {
  method: "GET" | "POST"; body?: Record<string, unknown>;
}) => Promise<T>;
type VoidRequest = (path: string, options: {
  method: "POST" | "DELETE"; body?: Record<string, unknown>;
}) => Promise<void>;

// In HttpKnowledgeBase constructor, after resolving existing transport options:
this.alwaysOn = new HttpAlwaysOnMemory(
  (path, options) => this.requestJson(path, options),
  (path, options) => this.requestVoid(path, options),
);

// In KnowledgeBase constructor in index.ts, after creating this.client:
this.alwaysOn = new McpAlwaysOnMemory((name, args) => this.client.call(name, args));

// In the new adapter class, return types come from exported AO types:
async ack(subscriptionId: string, deliveryId: string): Promise<void> {
  await this.requestVoid(
    `/v1/always-on/subscriptions/${encodeURIComponent(subscriptionId)}/ack`,
    { method: "POST", body: { delivery_id: deliveryId } },
  );
}

async unsubscribe(subscriptionId: string): Promise<void> {
  await this.requestVoid(
    `/v1/always-on/subscriptions/${encodeURIComponent(subscriptionId)}`,
    { method: "DELETE" },
  );
}
```

Keep full structural types for Intent/Lease/ResumePacket/FactChange/ChangeBatch/
ActionRecord/ActionPermit/FreshnessRecord/VerificationRequest/Result/Budget/Report,
including ActionRecord integer revision for storage CAS and UTC strings rather
than automatic JavaScript Date conversion. WaitCondition is a discriminated union.
Do not reuse legacy `normalizeFact`, which intentionally drops unknown fields and
defaults dates. AO errors must retain the code as data:

```typescript
export class AlwaysOnError extends Error {
  constructor(readonly code: string, message: string, readonly status?: number) {
    super(message);
    this.name = "AlwaysOnError";
  }
}

function parseMcpResult<T>(text: string): T {
  const value: unknown = JSON.parse(text);
  if (value && typeof value === "object" && "error" in value) {
    const error = (value as { error: { code: string; message: string } }).error;
    throw new AlwaysOnError(error.code, error.message);
  }
  return value as T;
}
```

HTTP `buildHttpError` first detects the same envelope for AO paths and constructs
`AlwaysOnError(code,message,status)`; existing routes retain their legacy error
format. Type assertions above do not replace runtime validation of the error
shape: reject malformed error envelopes with safe `InvalidResponse`.
Transport retry policy never invents a new client_key, action_key, subscription_id,
run_key or external idempotency key. Do not automatically repeat begin_action
when its response is uncertain; inspect action state through resume.

CLI adds exactly `work create/list/resume/claim/checkpoint`, `changes
subscribe/poll/ack/unsubscribe`, `maintenance run`, all supporting `--json`.
Use Click's JSON file/stdin input for checkpoint payloads, run the same strict
decoder before mutation, keep diagnostics on stderr and JSON output on stdout.
These commands execute one operation or one bounded batch, not a resident daemon.

## Host recovery with concrete public Python calls

The future offline `examples/always_on_recovery.py` owns scheduling and an
independent SQLite fake external API with `effects(idempotency_key PRIMARY KEY,
result_json)`. It stores inbox processing receipts durably before ack, and external
effects durably before returning success. Keep its database separate from AI Knot
to exercise the non-atomic boundary. The fake's `execute_once(key, request)` inserts
or returns the stored result for that key; `lookup(key)` returns proven success,
proven absence, or unknown according to the scenario.

Public calls for creating a waiting checkpoint, without private storage access:

```python
intent = ao.create_intent(client_key="deploy:42", goal="Deploy after CI")
lease = ao.claim(intent.id, worker_id="host-1")
intent = ao.get_intent(intent.id)  # claim changed state/revision
basis = ao.capture_evidence((("deploy-agent", ci_fact_id),))
ao.checkpoint(intent.id, lease_token=lease.token, expected_revision=intent.revision,
              state=IntentState.WAITING, progress={"stage": "ci"},
              waiting_for=WaitForSlot("deploy-agent", "ci::status", "green"),
              basis=basis)
```

After restart, poll preserves unacked delivery identity. The host updates a local
inbox transaction with a unique `(subscription incarnation, delivery_id)` receipt,
does its durable work, then acks. Every previously issued receipt for the same
subscription incarnation remains ack-idempotent until unsubscribe; unsubscribing
invalidates old tokens. Use a new incarnation if recreating an ID. Ack after
durable processing, never merely after receiving a packet.

```python
batch = ao.poll("deploy-watch")
if batch.delivery_id is not None:
    inbox.apply_once(batch.delivery_id, batch.events)  # local durable transaction
    ao.ack("deploy-watch", delivery_id=batch.delivery_id)
packet = ao.resume(intent.id)
if packet.stale_basis or packet.freshness_due:
    # Host replans/refreshes from accessible current facts, then captures fresh refs.
    recovery = ao.claim(intent.id, worker_id="host-2")
    current = ao.get_intent(intent.id)
    fresh = ao.capture_evidence((("deploy-agent", ci_fact_id),))
    ao.checkpoint(intent.id, lease_token=recovery.token,
                  expected_revision=current.revision, state=IntentState.READY,
                  progress={"stage": "replanned"}, basis=fresh)
```

Production host checks packet reasons explicitly: access revoked requires a new
accessible basis; due freshness requires accepted confirmation; deadline-exceeded
allows cancellation only. The snippet's replan branch applies only after those
conditions are resolved. A ready checkpoint releases the old lease, so claim a
new execution lease and reread revision before further work. Under a live lease
resume.ready is false; owning code can continue its current guarded operations.

```python
lease = ao.claim(intent.id, worker_id="host-2")
action = ao.prepare_action(intent.id, lease_token=lease.token,
                           action_key="deploy-build-42", request={"build": "42"})
permit = ao.begin_action(action.id, lease_token=lease.token)
result = external.execute_once(permit.idempotency_key, action.request)
# Test checkpoint: kill process here with os._exit(73), before finish_action.
ao.finish_action(action.id, lease_token=lease.token,
                  outcome="succeeded", result=result)
current = ao.get_intent(intent.id)
ao.checkpoint(intent.id, lease_token=lease.token, expected_revision=current.revision,
              state=IntentState.COMPLETED, progress={"external_result": result})
```

On crash after begin, do not call begin again. After lease expiry resume reports
blocked/unknown; claim provides a recovery lease while preserving blocked state.
Only independent external evidence permits reconciliation:

```python
packet = ao.resume(intent.id)
recovery = ao.claim(intent.id, worker_id="host-recovery")
for action in packet.open_actions:
    observed = external.lookup(action.idempotency_key)
    if observed.is_proven_success:
        ao.reconcile_action(action.id, lease_token=recovery.token,
                             outcome="succeeded", evidence=observed.evidence)
    elif observed.is_proven_absent_and_retry_safe:
        ao.reconcile_action(action.id, lease_token=recovery.token,
                             outcome="retryable", evidence=observed.evidence)
    else:
        return {"completed": False, "state": "blocked", "reason": "unknown_action"}
```

No response is not evidence of failure. With no external idempotency or queryable
proof, the expected result is blocked/unknown and no automatic repeat. A restored
facts snapshot does not restore actions/leases/cursors/maintenance history.
Backup/restore documentation must name all AO SQL tables and schema version;
legacy facts JSON/YAML export is not a full AO backup. On multi-host PostgreSQL,
document synchronized-clock assumptions and tolerated skew; do not imply local
lease fencing constrains a remote API that does not accept fencing tokens.

## Test and recovery matrix: binding assertions

Future test files below are required implementation work, not tests already run
by the documentation change. All SQLite concurrency assertions use independent
connections; process crashes use real child-process termination. PG tests use an
isolated live schema and the same parametrized invariants; fake psycopg is not proof.

| Required test location/name | Fixture / fault | Assertions and recovery |
|---|---|---|
| `tests/always_on/test_maintenance.py::test_zero_model_budget_never_calls_consolidator` | model budget 0 | calls=reserved=0; due episodes remain active; explicit reason |
| `test_empty_consolidation_never_closes_episodes` / `test_invalid_source_mapping_rejects_entire_proposal` | empty output / absent or unmapped source | zero output facts and zero closed inputs; proposal diagnostic retained |
| `test_foreground_episode_survives_background_commit` | barrier during provider await, other connection inserts | inserted episode survives active; only covered input closes |
| `test_late_provider_result_is_fenced` | provider await crosses expiry and replacement generation | old commit/ledger update rejected; successor untouched |
| `test_concurrent_run_key_reserves_once` | two connections start same key | one external invocation; changed config conflicts; restart replay returns report |
| `test_timeout_consumes_reserved_call` | connector timeout or process exit after reserve | no refund; same-key replay/takeover cannot exceed allowance |
| `test_input_utf8_budget_is_enforced_before_call` / `test_wall_budget_stops_new_calls` | Cyrillic input, aggregate bytes, fake monotonic | no unreserved call, deadline checks before each unit; bytes count UTF-8 |
| `test_consolidation_lineage_round_trips_sql_and_yaml` | save/reopen semantic facts | actual provenance.consolidation_ids equals source tuple; other qualifiers retained |
| `tests/always_on/test_codec.py::test_strict_nested_wire_values` | unknown condition/selector field; NaN; duplicate keys; bool integer; logical field >64KiB | reject before DB change; Unicode byte bound; canonical digests across key order |
| `tests/always_on/test_surfaces.py::test_actor_cannot_be_spoofed` | absent token; forged owner; foreign tokens/IDs | 401/422/403/409 as mapped; no IDs or hidden contents leaked |
| `test_get_routes_are_read_only` / `test_legacy_app_has_no_always_on_routes` | GET due/list/get; create_app without facade | no revisions/cursors/leases changed; old routes unchanged; AO 404 |
| `test_surface_golden_parity` | every row in method map via Python/HTTP/MCP | identical IDs, states, UTC strings, safe codes; ack null/204 normalization |
| `tests/always_on/test_cli.py::test_transport_retry_keeps_identity` | lost create/checkpoint/ack response | same-key identity, revision conflict visible, old receipt ack remains harmless |
| `tests/test_mcp_e2e.py::test_always_on_restart_replays_delivery` | kill subprocess after poll, before ack | same delivery, safe ACL redaction on replay, no process on SDK import |
| `npm/src/__tests__/always_on.test.ts` | constructed clients, injected fetch/MCP callback, 409, malformed envelope | no calls at construction; callbacks reuse auth; AlwaysOnError.code stable; all methods route correctly |
| `tests/always_on/test_example.py::test_offline_recovery_twice` | run offline subprocess twice with same directory | completed=true; effects=1; stale_plan_blocked=true; model_calls=0; no-idempotency branch blocked |
| `tests/always_on/test_end_to_end.py::test_all_crash_windows` | after_poll/checkpoint/begin/external/finish | missing changes=0; stale starts=0; duplicate effects=0; complete only with reconciliation evidence |
| `tests/eval/always_on/runner.py` required SQLite/PG mode | 10 deterministic scenarios and performance report | nonzero on any binding failure; requested PG with missing DSN fails, not skipped |

The ten scenario IDs are `waiting_restart`, `duplicate_event`, `read_before_ack`,
`stale_plan`, `deleted_or_expired_basis`, `lost_lease`, `unknown_external_outcome`,
`verifier_outage`, `consolidation_race`, `snapshot_restore`. Each result records
setup/seed/backend, crash checkpoint, completion expectation, counters and reason.
Compare completion only when host supplies sufficient reconciliation evidence;
blocked is success for the explicitly unknown non-idempotent branch.

The runner report `always_on_report.json` includes per-scenario counters, aggregate
binding invariants, p50/p95 latency, scanned-event counts and delivered batches.
All seven binding counters/ratios in the spec are explicit: zero missing committed
changes, zero stale action starts, zero stale lease mutations, zero unintended
duplicate effects for the idempotent fake, zero blind unknown retries, zero
cross-owner/scope leaks, completion 100% when reconciliation is supplied.
Performance matrices use 1k/10k facts × 1/10/100 subscriptions; report enabled and
disabled baseline on the same runner with at least five repetitions/median.
100k events is a slow scheduled/manual job. Track receipt rows/bytes and their growth until unsubscribe. Check real query plans and scan caps;
the <=20% disabled-AO p95 recall overhead is a measured regression budget, not an
assertion that the documentation/reference already meets it.
