What this change does: Adds an explicit scanned frontier and backlog state to durable memory delivery, plus a complete bounded recovery manifest whose cursor advances only after acknowledgement. Adds read-only context admission for exact saved evidence revisions and exposes both operations through HTTP, MCP, TypeScript and CLI recovery.

Review scope: whole uncommitted branch from `3618bfb` in `/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot`, including new tests, design/plan, storage transactions, evidence and lineage validation, current HTTP/MCP/TypeScript call paths, executable example, guide and process workload. Assumed load: eight independently connected agents, selected working sets of at most 500 facts, a much larger retained namespace and ordinary archived slot history. No source code edited by reviewer.

No unresolved correctness, security, load or unnecessary-complexity findings remain in the reviewed implementation. The following findings were corrected during review.

**Must fix — resolved**

1. **Recovery repeatedly decoded all archived versions** (`src/ai_knot/always_on/memory.py:640-652`, connected `_view` at `memory.py:300-317`).
   - **What this is:** Snapshot recovery checks candidate facts against current slots and source evidence while holding the namespace writer lock.
   - **Problem:** Each expired candidate loaded the whole slot before being rejected. One current fact and 1,000 archived versions decoded 1,004,003 facts and took 12.365 seconds locally; eight returning agents would contend for this same writer lock, whose SQLite wait timeout is 30 seconds.
   - **Fix:** The author now rejects unsupported or inactive candidates before the expensive slot/evidence check. The same independent probe now decodes 2,003 facts in 0.034 seconds; a deterministic count regression protects this path.
   - **If we skip it:** Ordinary long-lived slot history turns a one-fact recovery into a quadratic scan and blocks other memory work. Resolved and independently verified.

**Should fix — resolved**

2. **The restart example did not persist its evidence references** (`examples/shared_memory_return.py`, `save_cache` and the reader reopen).
   - **What this is:** The example commits replacement context and a delivery receipt before acknowledging a snapshot, then reopens the memory reader.
   - **Problem:** It originally persisted only text and delivery ID. Its saved evidence refs survived in a Python local; a real process restart would have no exact refs to pass to `review_context`.
   - **Fix:** The host cache now saves refs with context and receipt. The example deletes its original local refs and loads them from `cache.db` after reopening.
   - **If we skip it:** Users copying the example can persist text that cannot be safely checked on their next process run. Resolved; the updated executable example passed independently.

3. **The retained pending frontier needed a positive regression** (`src/ai_knot/always_on/subscriptions.py:244-248`).
   - **What this is:** An unacknowledged delivery can safely bridge pruned journal entries through its saved frontier.
   - **Problem:** Initial new tests covered a gap beyond that frontier, but not a successful replay when pruning stops exactly at it. Reverting to the old cursor check would break intended recovery without failing that negative case.
   - **Fix:** `test_pending_delivery_bridges_pruned_covered_history` now polls a delivery, prunes through its saved frontier, verifies identical replay and successful ack, then verifies a later write is delivered. The beyond-frontier expiry case remains.
   - **If we skip it:** The new retention-recovery guarantee lacks a focused test for the allowed path. Resolved; the new test passed independently.

Additional verified boundary correction: the new TypeScript test used `watch/id`, but the actual HTTP route originally returned 404 for its encoded slash. Existing subscription poll/ack routes shared this limitation. The author changed the four subscription resource routes to a path converter and added real HTTP coverage; encoded slash IDs now pass resync, poll, ack and delete.

Evidence:

- Final independent SQLite focused suite: **27 passed** across memory return, context review, HTTP surfaces and executable examples, including the newly corrected paths.
- Independent MCP/CLI/example suite: **7 passed**. The standard subprocess MCP restart/replay test and new CLI recovery operation passed.
- Independent eight-process workload with **10,000 unrelated updates**: 16 target observations, 8 snapshots recovered, 8 stale contexts rejected, 80 filtered empty pages, 80 idle checks, zero missed targets and zero model calls. Local return p50 0.226 seconds, p95 0.477 seconds. Output: `/tmp/ai-knot-shared-memory/reviewer-load-sqlite.json`.
- Independent scale probe `/tmp/ai-knot-shared-memory/reviewer-scale.py`: 100/500/1,000 archived versions changed from 10,403/252,003/1,004,003 decoded facts and 0.122/3.106/12.365 seconds to 203/1,003/2,003 decoded facts and 0.006/0.022/0.034 seconds.
- Read-through found captured head and selected reads serialized with writers; cursor and pending delivery change atomically, overflow rolls back, unacknowledged replaced tokens fail, and independent owners do not share ack state. Read-only context checks use SQLite snapshots or PostgreSQL REPEATABLE READ and do not touch telemetry or intent state.
- Exact revisions, current slots, source ACLs, dependency changes and freshness checks flow through the existing basis/lineage gate. Context redacts revoked refs, preserves input priority, deduplicates identical refs and omits whole facts that exceed the character budget.
- Schema 6 to 7 migration adds one column and preserves pending events and receipts. Existing future-version rejection remains. HTTP and MCP reuse the strict argument decoder; TypeScript operations map to the corresponding routes/tools and preserve metadata.
- No new dependency, idle inference, worker framework or unnecessary abstraction found. Existing basis semantics intentionally allow unchanged archived source revisions after consolidation, and context admission remains point-in-time rather than action permission.
- `docs/shared-memory-return.md` now explicitly states that `has_more` means raw journal backlog, not time-based membership. A future activation alone need not produce a journal event; host timers handle this. The guide distinguishes bounded supplied-reference completeness from task coverage and makes no measured LLM accuracy claim.

Verdict: Ship.
Not checked: PostgreSQL runtime, the full project check suite and final documentation-link synchronization by this reviewer; these remain in the author's acceptance run. The implementation and guide reviewed here have no outstanding finding.
