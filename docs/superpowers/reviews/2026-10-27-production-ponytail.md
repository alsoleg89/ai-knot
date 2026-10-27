What this change does: It adds durable work state, acknowledged change delivery, evidence checks, action recovery and bounded memory maintenance to the existing SQLite and PostgreSQL stores. It also exposes those operations through Python, HTTP, MCP, CLI and TypeScript.

Reviewed branch `codex/always-on-p1-p3`, base `69e2adf`, including the supplied implementation patch and connected production code. Expected load is 1,000–10,000 facts, 1–100 subscriptions, and several foreground/background processes writing the same store. This report retains the defects reproduced in the submitted implementation and records their verified fixes below. All four findings are closed on the final targeted recheck. Line references identify the originally reviewed code and can shift during fixes.

**Original must-fix findings — all resolved**

1. **[P1] Pruning hides a foreground correction from consolidation** (`src/ai_knot/always_on/maintenance.py:324–345`, `commit_consolidation`).
   - **What this is:** Consolidation prepares a summary outside the database transaction. Before replacing a slot, it checks the change journal for writes after its preparation sequence.
   - **Problem:** If a foreground writer corrects that slot and normal `prune_changes()` or another maintenance batch removes those events before the summary commits, the query finds no conflict. A real SQLite reproduction returned `status='completed', consolidated=1`, closed the foreground correction and left the stale summary active.
   - **Fix:** In the same commit transaction, reject a slotted proposal when the journal's `min_retained_seq` is greater than `anchor_seq + 1`, before checking the remaining events. Add a regression that corrects and prunes the output slot while the provider is running.
   - **If we skip it:** Ordinary maintenance can silently undo newer foreground knowledge. The namespace lock does not restore conflict evidence already pruned before that lock was taken.

2. **[P1] An old verifier response confirms a newly configured source** (`src/ai_knot/always_on/maintenance.py:357–375`, `_verify`; `src/ai_knot/always_on/memory.py:620–639`, `set_freshness`).
   - **What this is:** A verification request captures a fact revision and source name, then the provider runs outside the transaction. The result is later committed to the current freshness policy for that fact.
   - **Problem:** Change the policy from source A with a 60-second interval to source B with a one-day interval while A is being checked: the response from A confirms B for one day. The SQLite reproduction stored `source_ref='source-B'`, evidence `{'checked':'source-A'}`, `verified_until=now+1 day`, and returned no due verification.
   - **Fix:** Capture a monotonically increasing policy revision before calling the provider, and compare it under the commit lock for confirmation, errors and cancellation diagnostics. Increment that revision on every registration and recorded outcome; comparing policy values alone misses an identical reset at the same clock time.
   - **If we skip it:** Actions can pass the freshness guard using evidence from the wrong source or an earlier registration. An older unavailable response can also overwrite a newer policy outcome.

3. **[P2] An already expired summary closes its only active source** (`src/ai_knot/always_on/maintenance.py:271–284`, `validate_proposal`; `:337–346`, `commit_consolidation`; the same validator is used by `learning.consolidate_episodic`).
   - **What this is:** Proposal validation checks semantic type, support, size and source lineage. Committing resets the summary's creation and start times and closes its covered episodes.
   - **Problem:** A supported semantic `Fact` with valid lineage and `valid_until=now-1 second` passes validation; the commit preserves that expiry. The real SQLite run reported successful consolidation while both the input episode and its replacement summary were inactive.
   - **Fix:** Validate the output's effective validity at commit time before changing any input or target: reject an output whose preserved `valid_until` is at or before the new `valid_from`. Apply that check to both maintenance and legacy consolidation and test that the complete proposal rolls back.
   - **If we skip it:** A malformed provider result removes information from active recall while reporting success. The stored rows remain available historically, but the intended replacement is unusable.

4. **[P2] Fetching one due verification decodes the entire store once per policy** (`src/ai_knot/always_on/memory.py:653–693`, `due_verifications`; `:225–227`, `_view`; `src/ai_knot/always_on/maintenance.py`, the call to `due_verifications` before the first `remaining()` check).
   - **What this is:** `due_verifications(limit=1)` scans all policies, scans all facts for each policy and calls basis validation for each candidate. Basis validation reloads the full namespace each time.
   - **Problem:** With 1,000 facts and 1,000 policies, a real SQLite call returning one request made 1,001 full loads, decoded 1,001,000 fact objects and took 11.68 seconds. The limit is applied after this work, so a one-second maintenance batch can spend many seconds in synchronous selection before checking its deadline; 10,000 facts and policies imply about 100 million decodes by the same loop.
   - **Fix:** Select bounded candidate rows and validate them using one transaction-local fact/slot projection, rather than reloading the namespace for every policy. Preserve immediate detection of a changed binding, and add a query/decode-count regression with a small requested page and a large store.
   - **If we skip it:** Freshness selection becomes impractical at the documented 10,000-fact load, consumes the maintenance budget before useful work starts, and blocks the async HTTP/MCP handler while it runs.

Reproduction evidence: `/tmp/ponytail_prune_repro.py`, `/tmp/ponytail_freshness_repro.py`, and `/tmp/ponytail_repro.py` use real SQLite, public facade operations, a fixed UTC clock and injected providers. They reproduced all four findings independently of the reported aggregate test results. Inspection also covered the SQL session/locking seam, journal projection and replay, lease/action transitions, managed write callers, the shared codec, HTTP/MCP/CLI adapters, and TypeScript operation mappings.

A targeted run during concurrent remediation produced 34 passed, 35 skipped and 3 failed tests across maintenance, agent state and subscriptions. The failures were the newly added identical-policy-reset cases for confirmed, error and cancelled outcomes; they establish why a policy revision is necessary instead of value comparison. PostgreSQL cases were skipped in that run because no test DSN was supplied to this reviewer process.

**Targeted remediation recheck**

The four original standalone SQLite cases now behave correctly. Pruning returns conflict and preserves the foreground correction; a response from source A cannot confirm the replacement source B; an expired summary leaves its episode active; selecting one verification from 1,000 policies uses two filtered reads and decodes two facts in 0.00375 seconds, instead of 1,001,000 fact decodes in 11.68 seconds.

Code inspection confirmed that the conflict-history frontier check runs under the commit lock and the maintenance output is checked again after locking. Verification commits compare both the complete policy snapshot and a monotonically increasing internal revision, including the error and cancellation paths. Dirty-policy updates commit with logical fact writes, SQL selection loads targeted facts/slots, and both backends compare temporal values as UTC instants.

The first SQL rewrite missed a frontier changing solely because a pre-stored future fact became active. That case was reproduced, fixed with current/bound fact-ID mismatch eligibility, and rechecked: the new active ref is immediately due without another fact write. Additive migration also upgrades existing version-5/6 policy columns before a plain fact writer uses them.

Independent targeted SQLite and live PostgreSQL verification: `tests/always_on/test_maintenance.py` plus `tests/always_on/test_state_contracts.py`, **134 passed in 30.58 seconds**. This covers public/maintenance pruning, all 12 policy configuration/outcome combinations, malformed/inactive outputs, bounded 1,000-policy reads, same-ID changes, microsecond expiry, offset timestamp ordering, scheduled frontier changes and old-schema reopening.

The final recheck also reproduced a legacy variant of finding 3: a valid summary expired while waiting for the writer lock, then both source and summary became inactive. The final fix recomputes UTC time and repeats shared proposal validation inside the legacy mutation callback before any source/target mutation. Independent final regressions for SQLite, live PostgreSQL and YAML all passed: **3 passed in 0.37 seconds**, using a controlled clock without sleeps.

No actionable finding remains in the scoped remediation recheck.

Verdict: Ship.

Not checked: the full regression/retrieval gates were not independently rerun by this reviewer; the parent runs those broader gates.
