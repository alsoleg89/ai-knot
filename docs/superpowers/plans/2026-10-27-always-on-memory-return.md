# Always-on memory return implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Reliable return to team memory with bounded recovery and checked context.

**Architecture:** Reuse the subscription journal/receipts and evidence checks; one snapshot-mode column; read-only context assembly.

**Tech Stack:** Python >=3.11, existing SQL/HTTP/MCP/TypeScript; no dependencies.

**Spec:** `docs/superpowers/specs/2026-10-27-always-on-memory-return-design.md`

## Global Constraints

- Preserve original release/research bytes; work in isolated branch `codex/always-on-memory-return`.
- No release/version/changelog edits, push or merge.
- No idle inference; retain namespace locks, ACL/lineage/freshness and durable ack.

## Review Focus

- Empty filtered pages before backlog cannot imply caught up.
- Snapshot capture/ack must not lose concurrent writes or accept replaced tokens.
- Replay source/ACL/expiry changes invalidate snapshots without exposing text.
- Partial context/overflow cannot appear complete.
- Old schema, ownership, legacy calls and action guards retain behavior.

### Task 1: Honest delivery progress

Files: `always_on/changes.py`, `subscriptions.py`, `tests/always_on/test_memory_return.py`.
Interfaces: ChangeBatch metadata; poll/ack signatures unchanged.

- [x] Add tests for hidden-page backlog, pending/idle heads, page limit and independent agents; run RED.
- [x] Implement scan frontiers and replay metadata; run subscriptions/state tests GREEN; commit.

### Task 2: Snapshot recovery

Files: `always_on/subscriptions.py`, `storage.py`, `memory.py`, `models.py`, `__init__.py`; return/schema tests.
Interfaces: `resync(subscription_id, *, limit=500) -> ChangeBatch`, `SnapshotTooLarge`; existing ack.

- [x] Test retention gap, write after snapshot, restart/stale tokens, overflow rollback, selectors, expiry/revision/source/ACL changes and migration; run RED.
- [x] Implement bounded complete manifest, migration and replay invalidation; run SQL subscriptions/lineage tests GREEN; commit.

### Task 3: Checked context

Files: `always_on/models.py`, `memory.py`, `__init__.py`; `tests/always_on/test_context_review.py`.
Interfaces: ContextFact/ContextReview; `review_context(refs, *, max_chars=12000)`.

- [x] Test changed/revoked/derived/freshness/foreign refs, budgets, input priority, duplicate/conflicting revisions and no intent mutation; run RED.
- [x] Reuse basis views under one read-only snapshot, select whole facts; run grounding/state regressions GREEN; commit.

### Task 4: Team workflow and acceptance

Files: `_always_on_tools.py`, `server/always_on.py`, `always_on/codec.py`, `npm/src/always_on.ts`, transport tests; new example, guide and load harness/report.
Interfaces: HTTP `/subscriptions/{id}/resync`, `/evidence/review`; MCP and TS equivalents.

- [x] Add strict wire-boundary tests RED; expose methods GREEN.
- [x] Execute two-agent example and 8-reader sleep/return trace; document observed results and limitations.
- [x] Run format/lint/mypy, full required-PG Python suite, MCP, TS checks, recovery and existing MA gate.
- [x] Independent whole-branch review; material findings RED→GREEN; rerun appropriate checks.
- [x] Verify protected original hashes; record evidence and commit only feature files.
