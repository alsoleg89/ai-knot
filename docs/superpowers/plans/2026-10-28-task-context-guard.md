# Task-context guard and selected planning implementation plan

> **For agentic workers:** Use native `superpowers:executing-plans` in this session; parent implements, independent source/method review follows. User authorized autonomous execution without confirmation loops.

**Goal:** Let an always-on host select a bounded context while detecting changes to the full admissible task universe when saving a decision and explicitly admitting its action.

**Architecture:** Add a small immutable guard over the owner/task/policy and sorted admissible evidence refs, using existing source admission. Recompute it inside existing namespace-locked checkpoint and optional pre-action transactions. Reuse semantic selection only as an opt-in overflow path in the host example; selected refs still pass their ordinary checks.

**Tech stack:** Existing Python/SQLite/PostgreSQL, strict HTTP/MCP codec and TypeScript adapters; no dependency/schema/version change.

**Spec/design evidence:** [task-context guard research](../../research/2026-10-28-task-context-guard.md), [growing context](../../research/2026-10-28-growing-task-memory.md).

## Global constraints and contract

- Worktree branch `codex/always-on-memory-return`; preserve original checkout's deferred release preparation and all 30 protected files. Local tests only; no publish/push/merge.
- `TaskContextGuard(task_key: str, digest: str, max_facts: int = 128)` validates exact labels, lowercase SHA256 text and integer count1..128. `memory.task_guard(*,task_key,max_facts=128)` returns this descriptor with no content/ref list/model call. Empty sets are valid; overflow returns no guard.
- Hash domain `task-context-v1`, configured owner, task scopes, task key, cap and sorted exact namespaced refs. This guards the **admissible set**, not every stored row, policy edit, ABA history, semantic sufficiency or host task association.
- `checkpoint(..., context_guard: TaskContextGuard|None=None)` validates shape, locks all `task_scopes` in addition to current intent/basis scopes, checks normal lease/revision/state/basis, then recomputes with one shared `now` before saving. Mismatch or new overflow raises `StaleEvidence` and rolls back. DefaultNone behavior is retained.
- `begin_action(..., context_guard: TaskContextGuard|None=None)` similarly locks all task scopes and checks lease first. Mismatch appends an existing `changed` basis issue with no ref, persists BLOCKED through the existing pre-action rejection path, leaves PREPARED action/workspace reservation unchanged and raises `StaleEvidence` after commit. Guard must be explicitly passed again; it is not stored automatically or a grant of authority.
- Pin PostgreSQL writer transaction isolation to READ COMMITTED before advisory lock SELECTs; read transactions remain REPEATABLE READ READ ONLY. This prevents a configured RR snapshot from preceding a writer lock wait. SQLite keeps BEGIN IMMEDIATE/BEGIN.
- Guard must be captured before retrieval/model I/O. Time is evaluated at one `now`; expiry/world changes after permit/validation remain possible. Same-token prepared-action races remain outside the exclusive-controller contract.

## Review focus

- A newly added unselected generic or task condition must reject guarded checkpoint even when chosen refs are unchanged; irrelevant/inaccessible/inadmissible rows do not falsely change the set.
- Full task namespaces, shared ACL/freshness/lineage/activation clocks and new overflow must participate; no selected-only or post-model guard capture.
- Wrong owner/changed descriptor, bool limits, malformed digest and transport unknown fields fail without mutation; correct task association stays host policy.
- PostgreSQL defaultRR plus a concurrent committed source addition before lock release must not pass a stale guard.
- A fresh matching guard cannot bypass lease/deadline/manual/unknown-action/workspace checks; pre-action rejection must not start or duplicate an effect.

### Task 1: Atomic guard primitive and transports

**Files:** models/context/memory/__init__, PostgreSQL transaction setup, existing tool/HTTP maps, TypeScript always-on types/adapters; new `tests/always_on/test_task_guard.py`/surface tests and adapter tests.

- [x] Write/run failing SQL tests for stable/empty guards, changed membership vs irrelevant changes, large content/no text budget, strict caps and identity, rollback, checkpoint late addition, shared changes and begin rejection.
- [x] Implement the descriptor, model-free hash and optional checkpoint/begin checks without schema changes. Reuse task_sources inside the existing session; do not open a nested guard transaction while locks are held.
- [x] Add PostgreSQL defaultRR/advisory-wait regression and explicit writer isolation. Use barriers/observed lock waiting rather than arbitrary sleeps.
- [x] Mirror through HTTP `/evidence/guard`, MCP `memory_context_guard`, TypeScript `taskGuard` and optional checkpoint/begin guard, preserving existing signatures/default calls. Run strict transport/adapter tests.

### Task 2: Bounded selected planning and independent evidence

**Files:** existing `examples/always_on_team.py` and planning tests; prospective independent cases/protocol/artifacts and one adoption guide.

- [x] Capture guard before acquisition; fingerprint trigger/goal/original requirements/application progress/guard/bounds before paying. Keep own metadata out of input identity. Same input suppresses paid acquisition as well as planning after reopen.
- [x] Add explicit optional existing SemanticContext/SemanticBudget overflow acquisition; no default embeddings/retries. Require current reviewed nonempty selected facts; preserve lexical diagnostics and original requirements. Report acquisition and planner reservations separately. Persist/return the guard privately for explicit action use.
- [x] Use guarded checkpoint instead of a separate final whole-snapshot comparison; reject observed/new unselected changes and preserve existing cleanup/manual/deadline/unknown/lease behavior. Do not claim selection establishes prerequisites.
- [x] Freeze independently authored growing cases, common decision/scoring policy, strong full-task reference and budget-matched lexical/semantic arms before live GPT‑4.1 mini/text-embedding-3-small calls. Retain every call/failure and separate source recall, model decisions, guard rejection and actual scripted effects. If this study is not finished in the current context, leave it explicitly unfinished rather than calling the overall goal complete.
- [x] Full local gate with PostgreSQL, Python compatibility/static checks, TypeScript checks, independent Astra/Ponytail final review, protected checkout audit and local commit. Broad goal remains active.

## Acceptance record

Completed as recorded in [growing-context evaluation](../../growing-context-evaluation.md) and [validation](../validation/2026-10-29-task-context-guard.json). Final Astra follow-up was interrupted at the user's request; no final Astra acceptance is claimed. Ponytail source review and owner-run local tests are retained separately. The broad product goal remains active; selection quality still trails the strong full-context reference.
