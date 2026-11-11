# Task context guard — Ponytail source review

What this change does: The optional task guard detects changes to the whole admitted task source set while saving a proposal or admitting its action. The host recipe can explicitly select a small semantic context after whole-task overflow, with separate local acquisition/planner reservations and cancellation cleanup.

Looks good. Ship, subject to the owner's final execution gates. No concrete must-fix, should-fix or lean finding was identified in this source review. This is a source-review disposition, not a claim that the product goal or full acceptance is complete.

Checked the current diff and connected paths: task_sources/task_scopes and exact reference revision hashing; strict descriptor validation and shared codec; sorted namespace locks, intent dependency expansion and same-transaction recomputation; checkpoint rollback and pre-action BLOCKED behavior; unchanged default calls; PostgreSQL connection initialization and writer isolation before advisory SELECT; TypeScript HTTP/MCP mapping and old two-argument begin calls. Read the new SQL/surface/adapter tests and coordinated PostgreSQL lock-wait regression, including schema initialization. No test was executed by this reviewer.

For the host, traced guard capture before retrieval, guarded final checkpoint, original requirements, input fingerprint before acquisition, paid-failure replay suppression, zero-call cancellation retry, output validation, lease/controller checks and best-effort cleanup. Read SemanticContext lock and reservation handling and the new warm-selector cancellation regression. The scripted execution example explicitly passes the returned guard to begin_action. Existing host examples that do not opt in keep selected-basis checks.

Assumed load: multiple cooperating library writers sharing SQLite or PostgreSQL, and one controller per leased intent token. The explicit 128-admitted-fact guard cap can reject a growing task. It does not bound raw SQL row/native-buffer scans; writes can serialize on shared namespace locks. Those limits are documented and are not throughput claims. A guard is an optimistic set/revision check, not an authority grant or a proof that selection includes all task conditions. It must be explicitly reused before action admission; correct task binding remains host policy.

No new schema, dependency, background service, persistent token registry or inference retry was introduced. Existing source admission, transactions, transport codec and optional selector are reused. No speculative simplification is proposed merely to reduce line count at the expense of validation or accounting.

Verdict: Ship, after final local gates.
Not checked: no tests, SQL, Docker, provider, Keychain, live-model quality, throughput measurement or outgoing Git/secret audit performed. The owner must inspect actual final execution evidence; this review does not stand in for it.

## Exact inspected file hashes

```json
{
  "examples/always_on_team.py": "7ce5ba335d9031843a34861497eb18aea6b030603d7676dd76e2b38f9b5031b3",
  "examples/always_on_planning.py": "b2e3d546130d50e141ad802cc4e87e5b37ca6b7a5e4a3ba7d676908ae0029857",
  "src/ai_knot/always_on/context.py": "aa171e3cc7075ede711211e764c32cbbfc9ffe4a1a9c3808665b240106fe05a4",
  "src/ai_knot/always_on/memory.py": "7f431525e9d20d7f7b5c5b48375661577c861971dc4e324edcee7d9894fe573a",
  "src/ai_knot/always_on/models.py": "999b26910055defe9c3cc68c8a6461f269c7e2052df0b97908f19a8e6599ee73",
  "src/ai_knot/storage/postgres_storage.py": "b0496ce8aaa59db51059af561e5d1b9ed0c907a540b0d2c76ebd5050e07fdde6",
  "src/ai_knot/_always_on_tools.py": "84f333dbae3c14db66614a39d6a9370028147c89bb4f6abf3297226cf29ebf21",
  "src/ai_knot/server/always_on.py": "cdf3e9e367ad5e908b046bad42d9f4104e237a6ca2851990beec62bb84adbe65",
  "npm/src/always_on.ts": "95eefc49e84fcf23ef4de0d2e62d5ffb779f5872aac5497e7180b9a493400331",
  "tests/always_on/test_selected_planning.py": "c5d92013ff18551a4e2933986342a1367351b2ef25ad90dd4fed5ae06afcf319",
  "tests/always_on/test_task_guard.py": "d854c0f6fbe18a4cdfcdecd19475291c468b0cec83fdfa442c6464a07af248d7",
  "tests/always_on/test_task_guard_postgres.py": "6d7fae9dfc337e2820f3049edc4c7606b19566466534fc16ec91ac7bd771ceb5",
  "tests/always_on/test_task_guard_surface.py": "e78ab627271f969ba5a49660bd96d608c36f7527884ac03928f62ca205e95a85",
  "docs/task-context-guard.md": "1f7aafea53cd6394ad75c0e4392a198edc4cd84b25e8fa0079192def5389d459",
  "docs/bounded-planning.md": "0a7304b6da7703087fd294ba67bb2a30d71c10c9eaf6461859d18e55c9d20c64"
}
```
