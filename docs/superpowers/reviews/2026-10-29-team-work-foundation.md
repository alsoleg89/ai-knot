What this change does: It adds a frozen corpus loader, a study-only source publisher, canonical evidence ordering, and offline reference controls. It prepares a future continuation study; it does not implement or complete that study.

Independent non-Astra Ponytail correctness and fixture-adequacy review, 2026-10-29. No code or corpus changes made. Reviewed the three team_work modules, their tests, the foundation plan and method, the reused prompt/record/runner/sandbox boundaries, and all nine public snapshots, private contracts and reference bodies. Assumed the specified load: three frozen synthetic cases, chronological stage application, and replay of the current stage only.

No must-fix generic implementation bug found within that scope. The following are concrete fixture gaps to resolve or explicitly retain as limitations before a paid quality run; neither changes the public requirements nor invalidates the foundation code.

## Should fix

1. **The stage-three casefold rule is only checked with ASCII** (`cases-original.json:769,828`, artifact-audit-maintenance/stage-3, `ManifestChecks.test_03`).
   - **What this is:** Stage three explicitly changes duplicate identity to Python `casefold()` while retaining exact spelling for required-file presence.
   - **Problem:** The sole changed-case example is `readme`/`Readme`. Replacing the reference's `path.casefold()` with `path.lower()` would satisfy every existing assertion, although valid paths `Straße` and `STRASSE` must be duplicates and are not equal under `lower()`.
   - **Fix:** In a separately versioned, transparently recorded fixture repair, add that Unicode pair with valid metadata and both required spellings. With first size 2, second size 3, and maximum 20, the required result is total 2 and problems `["duplicate:STRASSE", "missing:STRASSE"]`. Preserve the frozen original bytes and rerun controls for the repaired version.
   - **If we skip it:** An implementation that violates the newly introduced duplicate policy can receive full stage-three credit. This is a static counterexample; no mutant or reference was executed by this reviewer.

2. **The newly explicit zero-budget boundary is untested** (`cases-original.json:745,828`, artifact-audit-maintenance/stages-2-and-3, `ManifestChecks.test_05`; public `manifest/budget.md`).
   - **What this is:** The changed public rule explicitly says a zero-byte total with a zero limit produces `budget`.
   - **Problem:** Both suites check equality only with a positive total. An implementation using `if total and total >= max_bytes` instead of `if total >= max_bytes` would satisfy all current checks but return `ok=True` for `audit_manifest([], [], 0)`.
   - **Fix:** Preserve the original and add one independently recorded fixture assertion in the repaired stage-two and stage-three suites: empty entries and requirements, maximum 0, total 0, problems `["budget"]`.
   - **If we skip it:** Full credit includes a plausible implementation that misses an explicitly documented changed-boundary case.

## Scope and method observations

- Public requirements are sufficient for the checked policies. I found no private expected behavior contradicting the public snapshots and no reference/public mismatch by inspection. Every stage retains checks for earlier obligations, and each changed rule or withdrawn grant has a distinguishing assertion, subject to the two gaps above.
- The reused prompt functions explicitly project public fields and public entry metadata. They do not serialize `Stage.test_source`, `Stage.reference_solution`, corpus provenance, or reference identities. Receipt names/counts remain the intentionally bounded feedback channel described by the method.
- `publish_stage` replaces only matching `repo-observer` facts under the case's `id + "::"` prefix. It does not touch executor intents; unrelated actors and derived facts have different provenance and are preserved. The resume test appropriately expects ordinary resume to persist BLOCKED after a stale basis is discovered.
- Canonical ordering sorts namespace, fact ID and revision, then assigns fresh citation indices. This makes equal valid entry sets render equally even if storage iteration differs. There is no paid runner integration in this foundation, so its future acquisition branches must actually call this helper before claims about model-input equality.
- The foundation plan line 17 says “Replay of a stage is idempotent.” Read that as replay of the current stage within chronological application, as instructed for this review. A delayed `publish_stage(case, 0)` after stage 1 would restore the earlier source snapshot/version; the function has no monotonic delivery guard. Clarify that wording to say “Replay of the current stage” so it does not imply out-of-order safety. No additional state machine is needed for the promised scope.
- Reference controls count the expected static unittest methods and require complete matching receipts. Initial stubs legitimately fail through NotImplementedError. Aggregate negative-control success establishes rejection of those candidates, not universal acceptance of valid alternatives or comprehensive contract coverage.
- The method clearly leaves paid execution, independent repetitions, exact PostgreSQL response replay, cost comparison, and product-quality conclusions prospective. The generic source hash freeze is unchanged for every reviewed new module/test, and the corpus digest matches `8557254849082076acc9f36948804718f8b2df0d76403a2738f74aacc2f36265`.

Verdict: Ship the generic foundation after the parent's required validation. Fix 1 and 2, or prominently retain their limitations, before treating these fixtures as a complete contract for paid quality outcomes.

Not checked: No tests, Docker, SQL, provider calls, Keychain access, candidate execution, reference execution, or external-source citation verification performed. Docker and backend results belong to the parent's separate validation. Fixture adequacy is a static review, not exhaustive proof.

## Owner resolutions

Both fixture findings fixed in v3 private contracts only. All public inputs, initial code and reference implementations are unchanged by structural JSON comparison. Exact same three mutation hashes pass original contracts and fail revised contracts by assertion with zero execution errors. All24v3positive/negative controls match. No repeated independent review or paid quality acceptance is claimed. Chronological snapshot/current-stage replay scope is stated in the report; arbitrary out-of-order source delivery remains unimplemented.
