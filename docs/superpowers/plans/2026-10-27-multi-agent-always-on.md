# Multi-agent always-on implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development and test-first implementation. Continue through integration and local PostgreSQL verification without a confirmation checkpoint.

**Goal:** Make shared memory safe and economical for several logical always-on agents, preserving the existing per-agent API.

**Architecture:** Extend the existing SQLite/PostgreSQL transaction seam. An explicit trusted-host `WorkspaceMemory` groups coordination and authority; private memories retain distinct agent identities. Reuse subscriptions, action reconciliation and evidence guards. No broker, scheduler or new dependency.

**Tech Stack:** Python >=3.11, SQLite, PostgreSQL 16, existing pytest fixtures.

**Spec:** `docs/research/multi-always-on-memory.md`, especially reproduced findings, immediate scope and invariant matrix. The user authorized implementation following the research.

## Global constraints

- Preserve preexisting deferred-release edits; no publication, push or merge.
- Work in `codex/multi-agent-always-on`; agents share this checkout, with explicit file/section ownership and no subdelegation.
- Write regression tests and observe meaningful failure before production changes; run against both real SQL backends.
- `WorkspaceMemory` is a trusted-host Python facility, not a tenant-security boundary. Independent workspaces' facts require separate databases/schemas with appropriate DB privileges. Do not claim HTTP/MCP multi-principal authentication.
- Shared evidence access still requires the pool read grant; workspace permission cannot grant access to a fact.
- Unknown remote outcomes retain resource exclusion until reconciliation; lease expiry alone never authorizes blind replay of an external effect.
- No automatic selector training, speculative framework or external service.

## Review focus

- Multiple connections/processes, stale pool snapshots and repeat flush must not lose or double counters.
- Operation keys survive cross-agent handoff; payload/resource mismatches fail; resource generations never decrease.
- Revoked maintain/execute permission is checked at commit/start, not only at construction.
- Shared refresh is claimed before reserving/calling a provider; cancellation/recovery cannot permit overlapping old and new calls before the claim deadline.
- Derived evidence cannot gain independence or visibility through repeated copies, revision changes or ACL revocation.

### Task 1: Atomic durable pool statistics (agent 1)

**Files:** pool.py, _pool_recall.py where needed, storage/base.py, SQLite/PostgreSQL stats methods, focused storage/pool tests.

**Interfaces:** Preserve existing snapshot load/save APIs. Introduce the smallest atomic delta merge seam required by `flush_stats`; legacy snapshots migrate without reset. Preserve consumer-set union and counter increments. Recall must see durable trust state from other instances without replaying local increments twice.

- [x] Add failing regressions: two independently initialized pools publish/recall/flush, restart, repeated flush, concurrent separate connections.
- [x] Implement atomic deltas and refresh local baselines; ensure publish evidence survives and stats writes do not silently overwrite others.
- [x] Run focused SQLite tests; provide PostgreSQL command/results and remaining atomicity limits.

### Task 2: Workspace operations and resource coordination (agent 2)

**Files:** new always_on/workspace.py, new operation helpers as needed, memory.py constructor/transaction/action sections only, new workspace tests. Root owns public export changes.

**Interfaces:** `WorkspaceMemory(storage, workspace_id)`; trusted-host `grant(agent_id, permission)` / `revoke(agent_id, permission)` for `execute` and `maintain`; `require(session, agent_id, permission)`, `lock_namespace`. AlwaysOnMemory gets optional `workspace=WorkspaceMemory`, stored as `_workspace`. `prepare_action` optional `operation_key` and `resource_key` together; legacy calls retain behavior. Produce an exact API report for start/finish/reconcile, handoff and fencing token.

- [x] Tests first: same operation two agents, mismatched replay, same resource distinct operations/intents, restart, denied/revoked authority, unknown result, takeover, workspace isolation.
- [x] Reserve operation and serialize resource atomically with action state. Use stable cross-agent external idempotency keys and monotone resource fencing generations. Preserve unknown/reconciliation semantics; no network calls under SQL locks.
- [x] Ensure permission checks do not expose a foreign operation result to an unauthorized caller. New DB initialization is additive and locked.
- [x] Test SQLite, report API needed by task 3. Do not edit freshness/maintenance functions or public export file.

### Task 3: Shared freshness authority, refresh claims and budget (agent 3)

**Files:** maintenance.py, focused new shared-maintenance helper(s), memory.py freshness functions only, focused maintenance tests. Do not edit workspace.py, constructor/actions or storage backend files.

**Interfaces:** Consume `memory._workspace`, `.lock_namespace`, `.require(session, memory._owner, 'maintain')`. Private policies continue to work unchanged. Shared policies/verifications require maintain permission plus current fact readability. Workspace grant operations are trusted-host calls.

- [x] Add failures for a reader changing policy, permission revocation, simultaneous two-owner refresh of the same evidence/policy revision, different policy/source, crash/cancel recovery, shared cost accounting.
- [x] Acquire a persisted shared claim before provider calls; commit checks claim generation, evidence and policy revision and current authority. Deduplicate across owners; retain per-owner run recovery.
- [x] Implement an explicit host-configured persistent workspace call budget with a fixed documented UTC window. Use a small standalone configuration function on the workspace if adding methods would collide with task 2; report exact API. Bound actual reserved provider costs globally, independently of each MaintenanceBudget.
- [x] Reuse SQL transactions; SQLite tests first, then PostgreSQL integration. Report any legacy shared-policy tests needing explicit maintain grants.

### Task 4: Integration, lineage and verification (root)

**Files:** public exports, new lineage helper/tests if needed, deterministic time fixtures/examples, docs/multi-agent-always-on.md and scoped links in existing docs.

- [x] Integrate task APIs; update existing shared maintenance tests to explicit host authority without weakening read ACL.
- [x] Add minimal persisted lineage for derived evidence: root identity includes namespace/fact/revision, copies flatten roots, access/revision changes invalidate future use, copies do not multiply independent support. Reuse existing Fact source metadata if it supports this cleanly; do not pretend lineage ensures arbitrary text secrecy.
- [x] Fix already observed calendar-dependent test/demo clocks by using coherent explicit times, then run the local CI-equivalent checks and recovery/eval gates.
- [x] Start an isolated local PostgreSQL 16 service, require PostgreSQL execution (no skips), run both-backend concurrency and restart scenarios.
- [x] Review each task for spec and correctness; run independent final Astra and Ponytail reviews, fix actionable findings and rerun affected tests.
- [x] Save local validation evidence and user documentation. No release publication.
