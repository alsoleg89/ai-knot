# Outcome-bound team experience implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Reusable conditional lessons that are withdrawn when their recorded action outcome changes.

**Architecture:** Existing procedural facts and exact observation lineage; one small action binding table supplies idempotency, destination lock discovery and atomic retraction in the central action save path. Ordinary actions do not scan shared facts.

**Tech Stack:** Existing Python >=3.11, SQLite/PostgreSQL, HTTP/MCP/TypeScript. No dependencies or model worker.

**Spec:** `docs/superpowers/specs/2026-10-28-team-experience-design.md`

## Global Constraints

- No version/changelog/release workflow changes or publication; preserve the protected original checkout.
- Action revision and supplied evidence are different gates; action identity is not an independent root.
- No remote-success/semantic-entailment/answer-accuracy claims from recorded host outcomes.

## Review Focus

- Failed outcomes can be reconciled to another state; all lesson copies must retract atomically.
- A process reopened without a pool still retracts shared output; unrelated actions avoid a shared global lock.
- Idempotent retries cannot resurrect deleted/retracted/amended lessons or reuse a key with different input.
- Shared lessons keep readable source roots and maintain authority without copying private action payloads.
- A source may change between record/selection and action; existing guards must still apply.

### Task 1: Durable recording and retraction

Files: new `always_on/experience.py`; `memory.py`, `models.py`, exports, `state_storage.py`; new `tests/always_on/test_team_experience.py`.
Produces the exact public signature/result in the spec; helpers discover binding namespaces and invalidate on `save_action`.

- [x] Write integration tests for success/failure, revision/outcome/result/evidence validation, replay/conflicts, source/ACL/freshness retraction, copy roots, no-pool reopening, atomic rollback and concurrent reconciliation; run RED.
- [x] Add schema8 binding table, scoped transaction discovery, source-derived procedural output and central retraction; run focused SQL tests GREEN plus existing action/lineage tests.

### Task 2: Configured-owner transports

Files: `_always_on_tools.py`, server route, TS types/client and their existing tests.
Consumes `record_experience`; exposes `action_experience_record`, HTTP actions/{action_id}/experience and `recordExperience(actionId, input)` with all named fields.

- [x] Add real HTTP/MCP and TS mapping tests RED, including unknown fields/coercion/owner denial.
- [x] Connect strict existing decoder/transport; run surface tests and TS typecheck/build/tests GREEN.

### Task 3: Adoption and acceptance

Files: `examples/team_experience.py`, `scripts/benchmark_team_experience.py`, example/workload tests; guide, research, review and validation records.

- [x] Add executable shared success/failure example with peer reuse, root dedupe, outcome correction and retraction.
- [x] Exercise concurrent writers/replays/reconciliation under a fixed fact/character budget on SQLite/PostgreSQL; report counts, model calls and descriptive timings, not model quality.
- [x] Run full required-PG Python/static, TS and existing recovery/MA acceptance; fresh Astra/Ponytail review and material RED/GREEN fixes.
- [x] Verify protected bytes, documentation links and retained evidence; commit feature-only locally.
