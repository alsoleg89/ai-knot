# Production Readiness

A feature-by-feature view of what ai-knot guarantees in production, organised by
readiness dimension.

**Legend** — ✅ shipped · ⬜ planned

For how to run it, see [deployment.md](deployment.md). For design rationale, see
[ARCHITECTURE.md](../ARCHITECTURE.md) and [DECISIONS.md](../DECISIONS.md).

---

## 1. Storage & durability

| Item | Status | Notes |
|---|---|---|
| SQLite backend (WAL) | ✅ | single-server / edge |
| PostgreSQL backend | ✅ | multi-process; `ai-knot_` table prefix; tables auto-created |
| YAML backend | ✅ | tests / human-inspectable state |
| Snapshots (version + diff) | ✅ | `kb.snapshot()`, list, diff |
| `event_time` persisted across backends | ✅ | bi-temporal field round-trips (sqlite/postgres/yaml) |
| Durable multi-agent ACL grants | ✅ | `ACLStoreCapable` + `acl_grants` table |
| Append-only audit ledger | ✅ | `trust_events` / `usage_events` tables |

## 2. Configuration

| Item | Status | Notes |
|---|---|---|
| Env-var configuration | ✅ | documented in [deployment.md](deployment.md) |
| Typed, validated config object | ✅ | `AIKnotConfig.from_env()` — range/enum validation, secrets never logged |
| Heavy-dependency guard | ✅ | `AI_KNOT_RERANK` requires the opt-in `rerank` extra |

## 3. Bi-temporal correctness

| Item | Status | Notes |
|---|---|---|
| `valid_from` / `valid_until` / `is_active(now)` | ✅ | core temporal model |
| Point-in-time recall (`recall(now=…)`) — core filter | ✅ | superseded facts excluded at query time |
| Event-time-anchored validity | ✅ | `valid_from = event_time`; supersession closes at successor's event time; `learn(event_time=…)` |
| Point-in-time recall exposed over MCP | ✅ | `recall(now=…)` + `top_k` bound |

## 4. Multi-agent governance

| Item | Status | Notes |
|---|---|---|
| Shared memory pool (MESI/CAS, no lost updates) | ✅ | protocol scenarios S10/S11/S13 green |
| Trust scoring + adversarial discount | ✅ | known-malicious agents discounted on wide recall |
| Evidence-before-belief gate | ✅ | unsupported claims withheld |
| Abstention signal | ✅ | recall surfaces "should abstain" risk |
| Conflict resolution (deterministic + optional semantic) | ✅ | `ClaimFamilyResolver`; opt-in `SemanticConflictResolver` |
| Per-scope read ACL writer (`grant_read`) | ✅ | persisted + restored on init |
| Pool wiring: ACL persist+restore, trust-event ledger, injectable clock | ✅ | |
| Fact lineage / provenance (`supersedes_id`) | ✅ | `lineage()` + `memory_lineage` MCP tool |
| Acceptance gate (S8–S26, binding vs advisory) | ✅ | `ma_gate.py`; `equivalence_recall_at_1000` bound |

## 5. Retrieval quality & determinism

| Item | Status | Notes |
|---|---|---|
| BM25 + intent-weighted RRF fusion | ✅ | no heavy ML deps on the hot path |
| Dense channel (optional, graceful degradation) | ✅ | skipped if embed endpoint unreachable |
| Deterministic, reproducible recall | ✅ | hash-seed-independent candidate ordering + id tiebreak |
| Cross-encoder rerank | ✅ (opt-in) | OFF by default; behind `rerank` extra |

## 6. Observability

| Item | Status | Notes |
|---|---|---|
| Per-stage recall trace | ✅ | `recall_facts_with_trace()` |
| Pool stats (trust / publish / use / quick-inv) | ✅ | persisted when `persist_stats=True` |
| Audit ledger query ("when / why / who") | ✅ | `load_trust_events` / `load_usage_events` |
| Version introspection | ✅ | `ai_knot.__version__`, CI-guarded sync |

## 7. Testing & CI

| Item | Status | Notes |
|---|---|---|
| Unit/integration suite (mypy --strict, ruff) | ✅ | `format → lint → types → tests` pre-commit order |
| Multi-agent acceptance gate (mock-judge) | ✅ | `runner --multi-agent --ma-gate` |
| Version-sync regression test | ✅ | pyproject == `__init__` == npm |
| Recall determinism regression guard | ✅ | `test_ddsa_output_stable_across_calls` |
| CI: live S8–S26 gate job | ✅ | `runner --multi-agent --ma-gate` on every PR |
| CI: PostgreSQL service for backend tests | ✅ | service container + `AI_KNOT_TEST_PG_DSN` |
| CI: npm client job | ✅ | build → typecheck → vitest |

## 8. Release & versioning

| Item | Status | Notes |
|---|---|---|
| Single version across 3 files | ✅ | pyproject / `__init__` / npm/package.json |
| Drift guard | ✅ | CI test fails on mismatch |
| Semver discipline | ✅ | MINOR for new subsystems; see [CHANGELOG](../CHANGELOG.md) |

## 9. Client & integration surface

| Item | Status | Notes |
|---|---|---|
| MCP server (stdio) | ✅ | Claude Desktop / Claude Code |
| TypeScript/npm client | ✅ | `learn` / `addResolved` / `recall(now)` / `tags` |
| OpenClaw memory adapter | ✅ | drop-in adapter |
| FastAPI HTTP sidecar | ✅ | `ai-knot serve`: `/health`, `/v1/recall`, `/v1/facts`, `/v1/stats` + optional bearer auth |
| CLI lifecycle/audit ops | ✅ | `recall --now` (point-in-time), `lineage` (supersession audit trail), `decay`, `export`/`import` |
| CLI pool-scoped gov ops | ⬜ | shared-pool operator commands |
| Framework integrations (LangGraph / OpenAI Agents / CrewAI / AutoGen) | ⬜ | thin adapters |

## 10. Benchmarks & evidence

| Item | Status | Notes |
|---|---|---|
| LoCoMo QA accuracy (LLM-judged) | ✅ | **78.0%** cat1–4 full-10, gpt-4.1/gpt-4o; per-conv 74–84% — see [benchmarks.md](benchmarks.md) |
| LongMemEval QA accuracy (LLM-judged) | ✅ | **59.6%** Oracle, gpt-4.1/gpt-4o; single-session 95–98%, abstention 90% |
| Reproducible deterministic suite | ✅ | zero-network, fixed seeds, one command; MRR 0.18→0.83, `evidence_recall@5` 0.15→0.26 |
| LongMemEval point-in-time adapter | ✅ | `recall(now=question_date)`; bi-temporal correctness regression-tested |
| Live competitor bench-pack (Mem0, …) | ⬜ | side-by-side scorecard |

---

## Roadmap (planned, in dependency order)

1. **Governance state machine** — `review_state` + `review_queue` +
   submit/ratify/reject on the pool. *Optional:* overlaps the existing
   evidence-gate + trust + ACL + abstention controls; warranted only where
   explicit human/agent ratification is required before a fact becomes visible.
2. **Lifecycle engine** — decay / archive / consolidate jobs + lifecycle ledger.
3. **Temporal/relation graph** — `GraphStorageCapable` edges from provenance.
4. **Self-repair probes** — generate + run consistency probes over the KB.
5. **Ecosystem** — CLI pool/gov ops, framework integrations,
   competitor bench-pack + live Mem0.
