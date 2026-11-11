# Source context and review corrections

> Native execution under the user's explicit autonomy instruction. Apply
> `superpowers:executing-plans`; independent review without Astra. No approval
> handoff, paid model calls, release preparation, schema or dependency changes.

**Goal:** Preserve exact experiment accounting, then let always-on hosts read
admitted non-derived task facts without depending on a generated handoff.

**Architecture:** Keep existing snapshot/authority/lineage paths. Add the optional
strict boolean `include_derived=True` to `task_snapshot`; false excludes annotated
derivations before output bounds and lineage work. Guards retain their full
admissible universe. HTTP/MCP derive arguments from the Python signature;
TypeScript mirrors the optional flag. There is no new scheduler or summarizer.

**Tech stack:** Existing Python, SQLite/PostgreSQL, HTTP/MCP and TypeScript client.

**Spec:** The current goal, the independent review of `a56a848..fd6a2e2`, and the
measured failures documented in `docs/team-work-evaluation.md`. Primary-source
comparison is recorded separately; it does not prove task-quality improvement.

## Global constraints and review focus

- Defaults and ordinary KB/selection behavior remain unchanged.
- Source-only means non-derived by existing provenance metadata, not certified truth.
- Hidden/expired/stale/unsupported/ambiguous/wrong-task facts remain excluded.
- Output bounds remain all-or-error. No clipping, implicit provider calls or writes.
- Filtered reads never weaken a later task guard or action authorization.
- Original release checkout hashes/status stay unchanged; no Astra.

## Task 1: Correct independently reproduced experiment failures

- [x] Add a real semantic/SQL regression where embedding's zero-charge
  `BudgetExceeded` must reach the runner's existing arm-isolation path; watch RED.
- [x] Add invalid-formation replay with one trailing recorded request; it must
  become incomplete with no quality claim rather than discard the request; watch RED.
- [x] In the study embedding adapter, retain/rethrow only budget failures after the
  production selector returns. Assert all formation replay requests are consumed
  on invalid output too. Keep existing provider overspend protection.
- [x] Run focused tests, request the original independent reviewer's follow-up,
  and record findings/resolutions. Existing measured trace stays immutable.

## Task 2: Add an opt-in non-derived snapshot

- [x] Add SQL-backed tests for default inclusion, false filtering before bounds,
  raw-source overflow, read-only state, ACL/currentness and strict boolean validation;
  watch RED before implementing.
- [x] Extend `AlwaysOnMemory.task_snapshot(..., include_derived: bool = True)` and
  internal task-source enumeration using existing `lineage.is_derived`.
- [x] Mirror `include_derived?: boolean` in `npm/src/always_on.ts`; test actual HTTP
  and MCP request/response behavior and invalid booleans through the shared codec.
- [x] Document a bounded source fallback using the existing facade, with no default
  retry, synthetic accuracy claim, guard weakening or new product dependency.
- [x] Run local PostgreSQL/Python/static and TypeScript gates, obtain fresh
  independent/Ponytail review, verify protected release hashes, and commit locally.

## Owner execution record

The two review regressions and source-view regressions were observed failing before
their fixes. Initial full validation exposed one additional exact-output consumer
in `tests/always_on/test_example.py`; its expected diagnostic now includes
`source_only_facts=1`. The repeat gate passed 2889 tests with PostgreSQL required,
9 skips and 90.16% coverage, plus 106 focused SQL/surface tests and all63 client
tests/typecheck/build. Independent follow-up accepted both review corrections;
a fresh reviewer found no source-view issue. The historical paid trace remains
unchanged and pinned to `fd6a2e2`. This addendum makes no accuracy claim.
