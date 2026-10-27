# Task context implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Opt-in task/role working sets with truthful evidence coverage and whole-fact budgets.

**Architecture:** Extend existing pure assembler with optional budget/admission; add one small SQL snapshot selector and immutable result. Reuse lineage, freshness, metadata and strict transports.

**Tech Stack:** Python >=3.11, existing SQLite/PostgreSQL/HTTP/MCP/TypeScript; no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-28-task-context-design.md`

## Global Constraints

- Ordinary recall/default derive interfaces unchanged; shared assembler correctness fixes also apply to existing callers. No automatic reflection or inference calls.
- Preserve original protected checkout/release preparation; no version/changelog/publish/push/merge.
- Coverage uses admitted selected evidence, never role-only affinity or rank-only scores.

## Review Focus

- Character budget/admission applies to coverage and backfill, with deterministic ties.
- Wrong-task labels cannot be treated as soft preferences; role cannot grant evidence coverage or authority.
- Read-only selection retains exact original namespaced revisions and source permissions/freshness.
- Same publisher can supply multiple necessary facts; rare sources cannot be routed away.
- Equal-budget trace gains must distinguish applicability, explicit requirements and role preference.

### Task 1: Bounded assembly

Files: `multi_agent/assembly.py`, `models.py`, `tests/test_multi_agent_assembly.py`.
Interfaces: optional `max_chars: int | None`, `admit: Callable[[CandidateFact], bool] | None`; AssemblyResult adds `budget_limited=False`.

- [x] Add budget, admission, duplicate/multi-facet and deterministic tie regressions; run RED.
- [x] Reuse existing coverage/backfill with whole-fact budget and selection admission; run assembly/service tests GREEN.

### Task 2: Task selection and typed derivation

Files: new `always_on/context.py`; facade `memory.py`; value objects/label validation in `models.py`; exports; tests `tests/always_on/test_task_context.py`.
Interfaces: select_context and derive_fact options exactly as spec; immutable TaskContext.

- [x] Add public-path SQL tests for task scope, role bias without false relevance, budget/missing coverage, namespace collisions, shared lineage/freshness and persisted metadata; run RED.
- [x] Implement source-readable snapshot corpus, original relevance, cached basis admission and exact original refs; run SQL and existing grounding/default-call regressions GREEN.

### Task 3: Surfaces, traces and verification

Files: shared tool map, HTTP route, TS client/types, matching tests; new trace harness/fixtures, guide, validation and review evidence.
Interfaces: memory_context_select, HTTP evidence/select, TS selectContext.

- [x] Add strict transport tests RED; connect typed operations GREEN.
- [x] Freeze 16 traces × four checkpoints with literal labels; run six arms (ordinary/scoped recall, ordinary/scoped Integrate recall, task-only and task+role) at equal corpus/output budgets on both SQL backends.
- [x] Record matched controls and separate required recall, precision, stale/forbidden selection, coverage, context characters, model calls and latency.
- [x] Run full required-PG Python suite, static checks, TS build/typecheck/tests and existing recovery/MA gates.
- [x] Independent Astra/Ponytail review; fix material findings with regression evidence; verify original protected files and commit feature-only changes locally.
