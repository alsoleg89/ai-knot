# P3 implementation report — 2026-10-27

Implemented production P3.1–P3.4 in the shared `codex/always-on-p1-p3` checkout.
P3.5 acceptance/performance/CI ownership belongs to the parent coordinator.
No commits or external publication were performed by this worker.

## Delivered

- `always_on/maintenance.py`: typed budgets/reports/providers/proposals, durable
  per-key identity and generation leases, persistent nonrefundable model/input
  reservations, bounded SQL selection of supported episodes at least 24h old,
  guarded freshness commits, exact source lineage validation, atomic
  consolidation/journal/counters, cancellation preservation, existing shared-pool
  TTL and bounded journal pruning stopping at live subscription cursors.
- Consolidation guards both exact input revisions and output-slot changes since
  the pre-provider stream anchor. Concurrent foreground corrections survive.
  Empty/invalid proposals close nothing; duplicate source IDs and duplicate refs
  reject the complete proposal. Per-output field limits do not cap batch responses.
- Legacy `consolidate_episodic` prepares detached extraction results before any
  mutation and atomically rechecks sources/current targets before inserting and
  closing. `Fact.provenance.consolidation_ids` reconstructs persisted qualifier JSON
  on SQL/YAML round-trips.
- One strict codec and invocation boundary serve all twenty operations. HTTP
  registration is optional and reuses configured bearer authentication. Raw JSON
  rejects duplicate keys/nonfinite values; typed layers reject unknown fields,
  missing condition discriminators and coercion. Error envelopes contain safe
  messages without lease/delivery tokens.
- MCP registers all twenty tools only when explicitly enabled; capabilities keeps
  its original array shape. Its metadata adapter passes raw arguments to the shared
  decoder, preventing MCP's permissive coercion/unknown-field dropping.
- CLI `work`, `changes`, and `maintenance` groups support JSON, stdin checkpoints
  and SQL errors; disabled legacy serving keeps its original construction calls.
- TypeScript typed HTTP/MCP `alwaysOn` adapters reuse existing transports and cover
  all twenty methods, preserving structured errors. `requestVoid` supports POST
  acknowledgment bodies and bodyless DELETE without reading successful204 JSON.
- `examples/always_on_recovery.py` uses three actual child processes, an independent
  external SQLite DB and deliberate `os._exit(73)` after the effect. Repeating the
  example reports one effect; no-evidence remote outcomes remain blocked.
- `docs/always-on.md`, README, CHANGELOG, integration/production documentation cover
  public usage, backups, cursor retention, clock requirements and host duties.

## Verification actually run

- P3 tests with real SQLite **and live PostgreSQL** using isolated schemas:
  `pytest tests/always_on/test_codec.py tests/always_on/test_surfaces.py
  tests/always_on/test_maintenance.py tests/always_on/test_mcp_surface.py
  tests/always_on/test_cli.py tests/always_on/test_example.py -q --no-cov`:
  **43 passed**, no skips, on PostgreSQL DSN port55439.
- Relevant legacy server/config/MCP/CLI + P3 local suite: **170 passed,
  17 PostgreSQL skips** before the live gate (the live gate above executes those
  PostgreSQL cases).
- `npm run typecheck`: passed. Complete `npm test`: **56 passed**, including four
  new tests covering construction side effects, ack204/body/auth/custom headers,
  DELETE, HTTP/MCP error codes and every operation's routing.
- Ruff for all owned source/test/example files: clean; formatter applied.
- Strict mypy for codec/maintenance/HTTP boundary/MCP boundary: clean.
- Parent coordinator additionally runs complete coverage/eval/MA/recovery/CI gates,
  build/package tests and independent Astra/Ponytail reviews. Their evidence is
  reported separately; it is not inferred from this worker's targeted tests.

## Operational choices

Maintenance uses existing shared-pool TTL and safe journal pruning. It adds no
private-fact deletion TTL. Wall time is cooperative. There is no broker, permanent
worker, daily quota service, arbitrary URL fetcher or mandatory dependency.
External APIs remain outside local SQL transactions and require real evidence for
unknown-outcome reconciliation. Multi-host PostgreSQL deployments must synchronize
clocks (documented operational skew budget1s); clock monitoring and remote fencing
remain host responsibilities.

## Independent review fixes

Astra's production review reproduced two async races; Ponytail reproduced invalid
output temporal state closing valid episodes. Added explicit failing regressions
before fixing them:

- Both ordinary public pruning and another maintenance run could remove a slot
  conflict event after the provider anchor. Commit now verifies retained history
  still covers the anchor while holding the same transaction/namespace lock.
- Old-source verification could apply after source/interval reconfiguration,
  same-configuration reset or a newer changed verdict. Preparation now captures
  the full policy plus the internal revision supplied by P2; all result paths
  (confirmed, operational failure and cancellation diagnostics) compare both.
  Cancellation still propagates rather than becoming a conflict response.
- Expired/future/naive/malformed output timestamps now reject the complete proposal
  before created_at/valid_from normalization or any source closure. Temporal
  validity is rechecked inside the commit transaction.
- The recovery example explicitly closes its SQLite connections with closing().

Fresh follow-up gate with live PostgreSQL and SQLite:
`pytest tests/always_on/test_maintenance.py tests/always_on/test_example.py -q --no-cov`:
**63 passed, no skips**. Ruff and strict mypy for the affected production module
also passed. Parent coordinates reviewer rechecks and the final whole-project gate.

Final scoped Ponytail recheck caught legacy consolidation output expiry while
waiting for its writer lock. The callback now recomputes UTC time and revalidates
prepared outputs before any mutation; source/target activity and closure times use
that fresh time. Fake-clock regressions reproduce the lock gap without sleeps on
SQLite, live PostgreSQL and YAML. Final maintenance suite: **65 passed, no skips**;
Ruff, strict mypy for learning.py and diff whitespace checks passed. Core frozen
for the parent coordinator's final whole-project gate/reviewer confirmation.
