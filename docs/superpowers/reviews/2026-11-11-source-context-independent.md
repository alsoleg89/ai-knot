# Independent review: source context and study corrections

Date: 2026-11-11. Base: `fd6a2e2`. Two read-only reviewer contexts were dispatched
using the requesting-code-review procedure. Neither invoked Astra, paid inference,
Keychain access or further agents. No tracked file was edited by a reviewer.

## Review of the completed study runner

Reviewer `independent_team_cycle_review` checked the earlier major range
`a56a848..fd6a2e2`, connected code, tests and retained SQLite/PostgreSQL evidence.
It reproduced two P2 issues:

1. A zero-charge embedding `BudgetExceeded` became semantic `unavailable`, then
   `ValueError`, bypassing the coordinator's arm-budget isolation. The study's
   embedding adapter now captures/rethrows that exact exception after production
   semantic acquisition returns. The public semantic API is unchanged. Reported
   overspend retains the global-stop path.
2. Local invalid handoff parsing could bypass `ReplayClient.assert_consumed`.
   A valid trailing recorded request was silently omitted from the replay ledger.
   The consumption assertion now runs in formation cleanup after recording the
   consumed snapshot, including failed formation.

Owner regressions observed both failures before the fixes. The real SQL/semantic
budget test now reports `BudgetExceeded`, retains zero charge and saved code; the
trailing-request test now rejects the replay without a quality claim. The original
reviewer accepted both fixes on follow-up and independently ran 95 focused tests
(8 PostgreSQL cases skipped in that reviewer environment). Parent PostgreSQL
verification is recorded separately in the validation artifact.

Neither trigger occurs in the historical measured trace: its embedding calls
succeeded and its invalid formation has exactly one request. The reviewer checked
that retained live/PostgreSQL semantic projections and source hashes match. The
old raw archive and scores remain unchanged, with reproduction pinned to `fd6a2e2`.

The initial reviewer test command inadvertently wrote an ignored coverage file;
the focused coverage threshold failed because the whole-suite default was applied.
Subsequent reviewer commands disabled coverage and cache writes. This is not a
tracked source change or a full-suite failure.

## Fresh review of the source-view feature and corrections

Reviewer `source_context_final_review` checked the complete current diff against
`fd6a2e2`: strict optional boolean, existing derivation classifier, filtering before
bounds/lineage, default compatibility, source authority/currentness, complete task
guards, HTTP/MCP decoding, TypeScript forwarding, example and the two fixes above.
It found no actionable issues and judged the change merge-ready conditional on
parent local/PostgreSQL gates passing. No independent performance or accuracy
advantage was claimed.

## Verification limits

Reviewers did not rerun paid evaluation, historical replay, a competing PostgreSQL
suite, or release checks. Parent tests cover SQLite and PostgreSQL, public surfaces,
the runnable example, source typing/formatting and the TypeScript build. Neither
review proves semantic sufficiency, unmarked generated prose detection, truth,
longitudinal utility or concurrent fleet performance. These remain outside the
source-view contract and the broader product goal remains active.
