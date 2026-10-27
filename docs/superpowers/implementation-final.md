# P1–P3: production implementation and acceptance

Completed 2026-10-27 in branch `codex/always-on-p1-p3`, based on `69e2adf`.
Package version remains `0.12.0`. This report accompanies the implementation
commit in the feature branch; package publication uses the separate release workflow.

The three implementation agents delivered all 13 tasks. The production layer now
supports durable intentions and leases, acknowledged memory subscriptions,
decision evidence, external-action recovery, freshness verification and bounded
maintenance. Python, HTTP, MCP, CLI and TypeScript use the same configured-owner
contracts. SQLite and PostgreSQL provide durable storage; legacy YAML remains
supported and explicitly rejects always-on state.

## Implementation map

| Tasks | Production code and acceptance coverage |
|---|---|
| P1.1–P1.2: SQL transactions and safe writers | `storage/{sqlite_storage,postgres_storage,base}.py`, `knowledge.py`, `learning.py`, `pool.py`, `_pool_recall.py`; transaction, concurrent writer and recall-race tests |
| P1.3–P1.4: journal and durable delivery | `always_on/{storage,changes,subscriptions}.py`; commit-order, restart/ack/incarnation, retention and ACL-redaction tests |
| P2.1–P2.4: work state, evidence, actions, freshness | `always_on/{models,_state,state_storage,memory}.py`; lease fencing, unknown outcome, stale basis, exact validity and policy-generation tests |
| P3.1: maintenance | `always_on/maintenance.py`, guarded legacy consolidation; persisted budgets, cancellation, provider/configuration races and output-validity tests |
| P3.2–P3.3: interfaces | `always_on/codec.py`, server router, shared tool dispatch, MCP/CLI, `npm/src/always_on.ts`; wire validation, auth/owner binding, actual MCP restart and all twenty SDK methods |
| P3.4: product example | [API and operations guide](../always-on.md), [three-process recovery example](../../examples/always_on_recovery.py), README/integrations/Unreleased changelog |
| P3.5: acceptance and CI | `tests/eval/always_on/`, `tests/always_on/`, CI PostgreSQL gate and benchmark workflow; deterministic reports and manual 100k replay job |

Implementation details and individual red/green evidence:
[P1](implementation-p1.md), [P2](implementation-p2.md), [P3](implementation-p3.md).
The earlier standalone reference models are preserved; production code does not
import them.

## Final verification

| Check | Executed result |
|---|---|
| Full Python CI suite, live PostgreSQL required, coverage enabled | **1453 passed, 2 skipped**, **87.80%** coverage, 24.67s |
| New always-on PostgreSQL cases within that full run | **93 passed**, real PostgreSQL 16.15, isolated schemas |
| Existing slow performance suite | **18 passed**, 72.06s |
| Ruff / format / strict mypy | Passed; 246 Python files formatted, 71 source files type-checked |
| TypeScript | Typecheck, ESM/CJS build and **56 tests passed** |
| Retrieval acceptance | **2 passed** |
| Existing multi-agent acceptance | **11/11 binding thresholds PASS**; advisory values and thresholds unchanged |
| Python 3.11 syntax compatibility | 44 changed/new Python files parsed using Python 3.11 grammar |
| SQLite and PostgreSQL acceptance runner | **12/12 scenarios passed per backend**, all six aggregate failure counters zero |
| Large SQLite journal replay | **100,000 events delivered**, maximum raw page 1000, no missing events |

The full run includes the six existing real MCP subprocess tests and the new
always-on MCP restart test. The two skips are intentional: a PostgreSQL-specific
constructor test's SQLite parameter, and a missing-psycopg error-path test when
psycopg is installed. PostgreSQL cases were executed, not skipped. The one warning
is the existing OpenClaw multi-turn-history warning, also present in the baseline.

Local runtime: Python 3.14.3, PostgreSQL 16.15 in a separate temporary Docker
container. Runtime coverage for Python 3.11/3.12/3.13 remains in the existing CI
matrix; the local Python 3.11 check above is syntax validation, not another full
runtime test run.

Reproduce the main gates from the repository root after configuring a live DSN:

```sh
export PATH="$PWD/.venv/bin:$PATH"
export AI_KNOT_REQUIRE_POSTGRES=1
# Set AI_KNOT_TEST_PG_DSN to an isolated test PostgreSQL database.
.venv/bin/python -m pytest -n auto --timeout=30 -m 'not slow' \
  --cov=ai_knot --cov-report=term-missing
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
.venv/bin/mypy src/ai_knot/
npm --prefix npm run typecheck
npm --prefix npm run build
npm --prefix npm test
.venv/bin/python -m tests.eval.always_on.runner \
  --storage postgres --output-dir /tmp/always-on-postgres
```

CI fails the required PostgreSQL gate when its DSN is missing. The large replay
job is explicitly manual (`workflow_dispatch`); no nightly schedule was added.

## Independent reviews

[Astra](reviews/2026-10-27-production-astra.md) reproduced two concurrency defects,
then independently verified their fixes with **30 SQLite/PostgreSQL tests** and
an additive SQLite migration check. Its final scoped verdict has no blocker.

[Ponytail](reviews/2026-10-27-production-ponytail.md) reproduced four defect classes:
pruning erased consolidation conflict evidence; old verifier responses confirmed
new policies; expired semantic outputs closed their sources; due selection loaded
the entire namespace per policy. All are fixed. Recheck also covered identical
policy resets, time-only slot activation, microsecond/offset boundaries, additive
migration and legacy output expiry while awaiting the writer lock. Independent
recheck: **134 SQLite/PostgreSQL tests plus 3 SQLite/PostgreSQL/YAML tests**.
Final verdict: **Ship**.

The due-selection reproduction improved from 1,001,000 decoded facts / 11.68s
to **2 filtered fact reads / 0.00375s** for the same 1000-policy, limit-one case.
This is one controlled local measurement, not a general throughput guarantee.

## Measured reports

JSON evidence is stored beside this report:

- [SQLite matrix](validation/always-on-sqlite.json): 1k/10k facts × 1/10/100 subscriptions.
- [PostgreSQL matrix](validation/always-on-postgres.json): the same six combinations.
- [100k replay](validation/always-on-100k-replay.json): raw scan counts, page bound,
  journal bytes, 205 receipts and their logical byte growth.
- [Recall baseline](validation/recall-baseline.json) and
  [current recall](validation/recall-current.json): AO disabled, offline BM25,
  ten measured recalls per repetition, median of five repetition p95 values.

AO-disabled p95 comparison on this machine: 1k facts **30.163ms → 30.407ms
(+0.81%)**; 10k facts **304.811ms → 310.574ms (+1.89%)**. Both are within the
planned 20% regression budget. Matrix reports include query plans, actual raw
event counts, delivered batches, journal logical bytes, receipt rows and logical
bytes. Receipt history grows until unsubscribe; no constant-space claim is made.

## Operational boundary

This is a memory-state layer. The host schedules batches and executes external
tools. A committed action permit does not make a remote effect transactional with
the local database. Unknown outcomes require external reconciliation; a host using
the low-level verification-attestation API coordinates its own asynchronous policy
changes. The injected maintenance verifier has built-in policy-generation CAS.

GitHub delivery uses the `codex/always-on-p1-p3` feature branch. Package releases
and deployment are separate operations. The release version has not been changed.
