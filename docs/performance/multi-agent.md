# Local multi-agent load measurements

Measured on 2026-10-27 with Python 3.12.13, macOS arm64, 15 reported logical CPUs,
SQLite 3.51.3 and local PostgreSQL 16.15. These are reproducible local observations,
not production capacity estimates, latency SLOs or an accuracy benchmark.

Run [benchmark_multi_agent.py](../../scripts/benchmark_multi_agent.py) from the
repository root. The CLI uses a disposable SQLite database or PostgreSQL schema.
PostgreSQL uses `AI_KNOT_TEST_PG_DSN`; reports omit connection credentials.

```sh
python scripts/benchmark_multi_agent.py --backend sqlite --agents 8 --operations 25 --output sqlite-load.json
python scripts/benchmark_multi_agent.py --backend postgres --agents 8 --operations 25 --output postgres-load.json
```

Each agent runs in a separate Python process with independent connections. Every
iteration publishes a fact and recalls one result as a distinct observer. A
separate contention phase gives all agents different operations on one resource;
the winner retains its permit until all contenders have attempted admission.
Shared refresh uses an asynchronous fleet with independent connections and a
scripted verifier; the next idle pass must make zero provider calls. No external
embedding/model endpoint or real deployment is involved.

## Eight-process workload

200 publications and 200 recalls, with eight resource contenders and eight shared
refresh requesters. These measurements were taken while development checks ran on
the same host; use repeated isolated runs before comparing changes in speed.

| Observation | SQLite | PostgreSQL |
|---|---:|---:|
| Publication p50 / p95, ms | 7.02 / 69.29 | 75.11 / 108.00 |
| Recall p50 / p95, ms | 9.17 / 121.11 | 96.37 / 133.66 |
| Combined publication + recall operations/s | 299.31 | 89.47 |
| Resource admission p50 / p95, ms | 1.37 / 62.12 | 34.12 / 80.95 |
| Shared refresh fleet wall time, s | 0.068 | 0.731 |
| Publications / usage increments / distinct observers | 200 / 200 / 8 | 200 / 200 / 8 |
| Resource permits / shared provider calls / idle calls | 1 / 1 / 0 | 1 / 1 / 0 |

Raw reports: [SQLite](multi-agent-sqlite-load.json),
[PostgreSQL](multi-agent-postgres-load.json). The initial four-process,
40-publication checks are also retained: [SQLite](multi-agent-sqlite.json),
[PostgreSQL](multi-agent-postgres.json).

Throughput uses the common mixed-workload wall interval, excluding process/setup
time and including final explicit flushes. Per-category throughput is achieved rate
during this mixed load, not isolated capacity. PostgreSQL numbers include the
existing backend's short-lived connection and transaction overhead. Fact set
growth, remote database latency, dense retrieval and actual model execution can
substantially change these figures.

## Correctness under load

The harness asserts exact per-agent publication counts, aggregate usage and
consumer-set union after repeated flushes. It also requires exactly one resource
permit, exactly one successful shared refresh and zero idle provider calls. A
regression deliberately corrupts the stored usage total and verifies that the
harness rejects it. Timing thresholds do not decide correctness.

The functional SQL tests separately exercise rollback of publication/events/stats,
restart, stale workers, prepared handoff, terminal replay, unknown effects,
permission revocation, provider cancellation, claim replacement, UTC budget rollover,
ancestor revisions/ACLs and freshness deadlines. They use real PostgreSQL schemas
and SQLite databases, not SQL mocks.

The refresh load phase uses one process with asynchronous tasks; it does not model
a remote provider ignoring cancellation or a prolonged multi-process soak. Local
resource permits are not evidence of exactly-once external effects: the external
API must enforce idempotency/fencing. See the [host API guide](../multi-agent-always-on.md).

## Independent review

Astra and Ponytail reviews found and closed workspace-binding recovery bypass,
sync-path lineage disclosure, missing dependency lock scopes, duplicate source
identity, reader cleanup/starvation and archive/freshness consistency issues.
Final scoped Astra verification passed 68 real SQLite/PostgreSQL cases; Ponytail's
final scoped check passed 40. Both ended with no outstanding actionable findings.
Full local verification results are recorded in [multi-agent-validation.json](multi-agent-validation.json).

Final local CI: Python 3.12 with required PostgreSQL **1562 passed, 2 explained skips**,
coverage **88.38%**; Python 3.11 and 3.13 **1411 passed** each. TypeScript **56 passed**,
MCP E2E **6 passed**, eval tests **70 passed**, process recovery **12/12** on each backend.
The legacy mock-judge multi-agent gate passed **11/11** binding thresholds; five
advisory quality metrics remain below their existing targets. No accuracy gain is claimed.
