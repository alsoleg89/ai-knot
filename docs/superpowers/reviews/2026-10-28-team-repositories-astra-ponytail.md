# Independent pre-live repository-task protocol review

Reviewer scope: design/method review, read-only repository access, no provider calls or credential access. The sealed `tests/eval/team_repos/cases.py` and `tests/eval/test_team_repo_cases.py` have not been opened. Reviewed existing `docs/team-decision-evaluation.md`, `docs/semantic-context-evaluation.md`, `docs/task-snapshot.md`, the existing metered provider adapter, and the production semantic selector. This note does not approve an implementation that has not yet been inspected.

## Current agreed design

Three independently authored small Python repository tasks, with three stages each: initial task and two exogenous source updates. The same stateless reader model emits a complete `solution.py` replacement, with at most two attempts at each stage. Hidden tests run only in a hardened networkless Docker sandbox. Each arm carries its own candidate code between stages, and the SQL worker really reopens between stages. Public task requirements, current/initial code, source evidence, and sanitized test receipts may reach the reader; reference implementations, expected literal values, and hidden test bodies may not.

Four arms are now planned:

1. `direct_sources`: all current public raw source documents, no researcher formation.
2. `current_docs_handoff`: all current public raw source documents plus the exact latest common researcher handoff.
3. `raw_team_history`: all retained raw source revisions and researcher handoffs, subject to the declared common context ceiling.
4. `task_memory`: current source-checked task snapshot, or the existing explicit `SemanticContext` acquisition on overflow.

One common researcher produces each frozen stage handoff and its source basis before any coder arm runs. Each of the three formation-using arms is charged the FULL actual formation calls/tokens in its standalone logical ledger. The actual account-spend ledger counts each shared call only once. No division by the number of consuming arms.

The agreed per-case/per-arm ceiling is 12 provider requests and 60,000 tokens, inclusive of allocated formation, embeddings and reader work. A 24,000-character source-context ceiling applies to every arm. Reader output ceiling is 2,048 tokens; researcher output is 768. The provider's current 128-token default remains unchanged, with a constructor-level explicit opt-in ceiling for this evaluation. Task snapshot/semantic selection uses at most eight facts. Semantic acquisition permits at most one combined query/document embedding batch per stage and no hidden fallback.

## Findings and smallest fixes

**1. Formation accounting must represent an independently deployable arm. Resolved in agreed design.** Splitting a common researcher call across the three using arms would subsidize their standalone budgets. Debit full common formation to every using arm, and separately report deduplicated actual account spend. A shared call with unknown provider usage keeps its full conservative reservation in both relevant views; failed calls and cleanup failures must not erase incurred charges. Charge formation even if a later acquisition omits the handoff or a reader cannot run.

**2. A current-documents-plus-handoff control is needed to isolate memory acquisition from research summarization. Resolved in agreed design.** Comparing memory against direct raw sources combines formation, summarization and acquisition. Comparing memory against retained history combines acquisition and source-currentness filtering. The added `current_docs_handoff` arm is the smallest useful control: it receives exactly the same researcher output without the knowledge layer. Do not claim a library benefit if it ties or loses to that arm.

**3. Twelve requests leave no spare retries. Confirm in implementation.** At three stages, three formation calls plus three single-batch semantic calls plus six patch calls is exactly twelve. The existing `SemanticContext` combines missing documents and the query into one callback, so this is feasible. Select once per stage and reuse/re-review that selected context for the second patch attempt; selecting again through a fresh cold helper would spend extra calls. Preserve failed/malformed attempts in the two-attempt allowance. Source updates happen only at the declared stage boundary. Equal ceilings with this fixed attempt schedule are not a comparison of optimized autonomous budget policies.

**4. Keep code state endogenous and research state common. Confirm in implementation.** Every arm starts from the identical initial module, then continues from its own latest candidate, including a failing candidate. Never reset to a passing/reference implementation between stages. Freeze researcher inputs to common public task/source inputs; do not let the common researcher inspect one arm's generated code, test receipts, or results. Otherwise a shared note transfers an arm's work to competitors. A small per-stage hash ledger for inputs, handoff and starting candidate is sufficient.

**5. Match the cheap control's note universe. Confirm before freeze.** Use one stable handoff path/slot per repository with successive revisions; retain older revisions only in raw history. Alternatively, expose every still-current note to `current_docs_handoff` as well. If task memory retains several older notes whose cited sources remain current while the cheap control receives only the latest note, their evidence differs beyond acquisition. Source lineage should be actual researcher-selected refs, not hidden expected evidence groups. A handoff is a model-produced interpretation with provenance, not a new authoritative requirement.

**6. Raw history must actually fit, or its truncation must be explicit. Confirm with sealed author/frozen manifest.** Natural SHA(path) IDs remove hand-assigned priority but do not make a prefix unbiased. Preflight the fully rendered source context, including paths, revision/authority labels, citation handles and handoff wrappers. If any history exceeds 24,000 characters, freeze a transparent whole-entry truncation policy and rename/report that arm accordingly; never silently clip bodies or choose a gold-relevant prefix. Prefer author-confirmed fitting histories for this small study. Current-doc baselines must likewise genuinely include all current raw docs within the common ceiling. Code, system prompt, response schema and receipts still count toward the aggregate provider token budget.

**7. Currentness guards must not become hidden answer oracles. Confirm in implementation.** A common action path may require a valid lease and fresh readable cited refs. It may not require hidden expected source groups or paths to authorize a patch. Hidden tests decide functional correctness; current basis validity is a separate provenance outcome. A citation to a source-current researcher note does not prove its interpretation true or its patch justified. Enumerate all visible citation handles, including stale entries in raw history, rather than only admissible/correct ones. Reject stale basis consistently and record the rejected attempt; do not silently replace it with current refs.

**8. A safe sandbox is not by itself a trustworthy test oracle. Confirm the small runner contract.** A candidate's print output or process exit code must not be accepted as a passing test receipt. Use a sealed fixed harness and require a completed structured test run with the expected test count, rejecting premature exits or malformed receipts. Only fixed neutral test names/count/pass-fail may reach the model: assertion text, traceback source lines, exception values and test names containing expected literal values must not. Keep the candidate's generated file out of host execution. Do not add an elaborate anti-cheating framework for this narrow non-adversarial study; document that Python code co-resident with its tests is not adversarially hidden from the executed program.

**9. Freeze and report the actual route, without forcing a semantic win. Confirm before live.** Freeze case data, source schedule, reader/researcher prompts, acquisition parameters, runtime hashes and test contract before the parent opens the sealed cases or runs paid models. Report snapshot/semantic/input-limit/unavailable outcomes and actual embedding calls. If all corpora fit, call the result a snapshot/bundling study; do not enlarge selected cases after inspecting outputs to manufacture semantic use. A failure of budget/provider/acquisition remains a failed or incomplete case with charges preserved, not a dropped row.

## Claims this study can support

Primary outcomes should be complete passing stages and complete passing repositories, with patch attempts, invalid/stale-basis rejections, time and actual logical cost alongside them. A later successful stage does not erase an earlier failure. Use tests for the current complete contract, so an update cannot score well by dropping previously required behavior.

The most useful contrast is `task_memory` versus `current_docs_handoff`: source-checked acquisition against a cheap current-document bundle supplied with identical research. `current_docs_handoff` versus `direct_sources` estimates whether this researcher handoff helps under the specified costs. `task_memory` versus `raw_team_history` shows the behavior of selected current evidence versus accumulated team history under the fixed prompt/attempt policy.

Three repositories are three independent authored task definitions, not nine independent tasks because they have three stages, and not twelve independent tasks because they have four arms. Actual model-generated code and hidden tests improve realism over closed-choice fixtures, but these remain small synthetic repositories unless drawn from genuinely existing repositories. Sequential researcher/coder roles and SQL reopen establish a bounded handoff/durability experiment, not a concurrent fleet or sustained always-on production workload. Natural hashed source order does not establish representativeness. No generalized quality or statistical-superiority claim follows from one live pass.

## Minimality review

The smallest adequate implementation is one evaluation runner, one isolated candidate-test harness, the existing provider ledger with an explicit ceiling extension, and the independently authored cases. Reuse the production snapshot, semantic acquisition and action guards. Do not introduce a plugin system, general agent scheduler, vector cache service, custom retry framework, or a second provider implementation for this study. No line-level complexity finding is asserted before the new runner exists.

Status: design is suitable for implementation after the confirmed accounting/control repairs. Live execution still requires implementation review and an immutable freeze. No providers, credentials, sealed cases, PostgreSQL instance, or production state were accessed by this reviewer.

## Generic sandbox inspection before freeze

Read `tests/eval/team_repos/sandbox.py` and `tests/eval/test_team_repo_sandbox.py` when they appeared; neither imports nor reveals the sealed case contents in the inspected text. No tests or candidate code were executed by this reviewer.

**Open correctness finding — skipped/partial test suites can pass.** In the inspected `_RUNNER`, the `Result` overrides failures, errors, unexpected successes and subtests, but inherits `addSkip` and `addExpectedFailure`. The pass predicate is only `testsRun > 0` plus zero counted failures/errors. Thus a candidate function raising `unittest.SkipTest` can cause every test calling it to be skipped while receiving a pass. Expected failures similarly do not count against success. Separately, no expected suite count is compared with the completed test count, so an early stop after some successful tests can pass. Smallest repair: count skipped/expected-failure cases as failures or incomplete execution, capture the trusted suite's `countTestCases()` before running, and require completed `testsRun` to equal that count. Add synthetic sandbox regressions; no changes to sealed tasks are needed. This finding concerns the validity of a success receipt, not a host-isolation escape or an observed model attempt.

The Docker command already uses a fresh non-root container, network none, read-only filesystem/mount, dropped capabilities, no new privileges, bounded memory/CPU/PIDs/tmpfs, no pull, a pinned default image, sanitized environment, and timeout cleanup scoped by an ownership token. Ordinary stdout/stderr are redirected away from the receipt descriptor. The module correctly acknowledges that same-interpreter unittest is not an adversarial attestation mechanism; this review does not ask for a new anti-cheating framework.

## Generic implementation review before the live freeze

The reviewer subsequently inspected `runner.py`, `protocol.py`, `experiment.py`, the revised sandbox, the provider-adapter diff, and their generic tests. Sealed case files remain unopened. These source inspections were performed while the owners were still finalizing files; final hashes and closure checks are required before declaring review complete.

The earlier sandbox finding is repaired in the inspected source: `addSkip` counts an error, `addExpectedFailure` counts a failure, and a completed count differing from `suite.countTestCases()` counts an error. Synthetic opt-in Docker regressions exercise all three paths. The parent reports those tests passed; this reviewer has not independently executed them. Exact run-label ownership checks now support cleanup after the outer worker is interrupted.

**Resolved during inspection — CLI workers launched the wrong module.** The original `run_worker` used `python -m __name__`. With the CLI itself invoked using `-m`, `__name__` is `__main__`, which cannot launch the intended worker. The owner replaced it with the literal `MODULE = "tests.eval.team_repos.experiment"`; the inspected command now uses that constant. A contract regression should explicitly cover the module argument.

**Open — outer CLI failures can erase paid-work evidence.** The inspected `main` puts paid execution and `TemporaryDirectory` cleanup inside one `try`; its `except` unconditionally writes a new two-field failure report. A cleanup failure after successful paid work therefore overwrites the full results/cost ledger already written. A final report-write failure escaping `run_experiment` can also reach this path. Keep the in-memory report outside the `try`, update its failure status, and preserve existing partial evidence if no full result is available. An injected outer cleanup failure is the smallest regression.

**Open — complete tested candidates are not retained.** Applied attempts store a SHA256 and the first 4,096 characters of sanitized model output, while the default CLI deletes the temporary SQLite databases. A valid 2,048-token code replacement can exceed 4,096 characters, leaving no recoverable tested `solution.py` after the run. Store the full already-parsed/validated candidate per applied attempt or in a retained stage artifact, with its hash. Do not retain rejected credential-shaped raw text.

**Open — unknown reservations are reported as unique spend.** On an unreadable/lost worker result, the parent correctly reserves all remaining requests/tokens and stops the arm. Those synthetic `unknown_worker` entries are then summed into `unique_calls` and `unique_spent_tokens`. Keep the conservative budget charge, but report known actual usage separately from unknown reservations; the combined value is a bound, not observed account spending. Unknown provider billing within an otherwise retained call needs the same distinction.

**Open — declared source-context ceiling is currently content-only.** The runner checks only the sum of content string lengths; the public context also includes citation indices, versions and observation timestamps. Either enforce the rendered public-entry ceiling, or explicitly freeze and document the 24,000-character cap as content-only. Actual provider accounting already charges prompt/schema overhead, but that does not make the content check a full source-context bound.

**Freeze check — case data is outside the current runtime hash fence.** `source_hashes()` deliberately omits sealed cases during generic development. Before live execution, include an author-supplied immutable case manifest/digest and assert it throughout the run, or add the sealed source file hash without presenting its content to the parent. Runtime-only hashes cannot establish that tasks/tests stayed fixed.

Reviewed positive properties: both model prompts explicitly project only public fields; current code is supplied only to coders; hidden test source is handed only to the sandbox; the common researcher completes all common public handoffs before any coder arm; formation is charged in full to each using arm and deduplicated in the experiment ledger; every stage carries its own previous candidate via SQL; acquisition runs once per stage; no correct-answer source group is used as an action admission oracle; every visible history entry remains in the citation schema; `review_context`, `prepare_action` and `begin_action` all participate in the common guarded path; optional visible version/time metadata is now included in both prompts. The provider adapter preserves its old default while requiring explicit authority for larger outputs.

Additional interpretation limit: `seed_stage` replaces the shared source corpus with the current documents and current handoff, while `_prepare_stage` clears checkpointed intent basis before the worker runs. This measures current source acquisition, rejection of stale history citations, persisted own-code handoffs, and real patch/test outcomes. It does not independently demonstrate autonomous invalidation of retained source knowledge or stale checkpoint basis; stage loading supplies the current corpus explicitly.

Blinding disclosure received from the parent: one orchestration contributor accidentally matched two sealed lines containing `checkpoint` in a search, stopped and disclosed it; no gold was sent to the parent and no case-dependent algorithm branch was introduced. This is partial implementer blinding, not a double-blind experiment. This reviewer and the parent remained blinded to the sealed case bodies during the generic review.

## Closure checks on the owners' follow-up changes

Source inspection confirms the following repairs:

- The runner checks `protocol.context_chars(selected)`, which uses the same projected entry JSON as the prompt, and records `rendered_context_chars`. A metadata-overflow regression was added.
- The full parsed, credential-checked `candidate_code` is retained before provenance admission; tested attempts retain the code hash and receipt. This closes the truncated-candidate artifact gap.
- The subprocess command uses a fixed module constant; the generic launch test now patches the imported module's `__name__` to `__main__` and still expects the proper module path.
- Experiment output now distinguishes `unique_tokens_plus_reservations`, `known_unique_spent_tokens`, unknown worker reservations, and unknown provider billing. The known-usage total is computed from valid reported provider usage, not placeholder charges.
- The source fence now includes the sealed case file's bytes as a digest without rendering/importing them during generic development; the report also records an author-supplied manifest digest and the partial-blinding disclosure.
- The CLI initializes the report outside its `try` and preserves it across temporary-directory cleanup errors. The added outer cleanup regression verifies retained costs.

**Remaining persistence edge:** the last `update()` inside `run_experiment` is outside its error-catching block. If report persistence fails just after newly paid formation or a stage, and fails again in this final update, the function raises without returning its in-memory report. The CLI's fallback reads only the prior successfully persisted report, losing the latest charges. Preserve/return the in-memory failed report on write failure, or attach it to the raised error, and stop further paid work. A transient writer failure after one metered call is the smallest regression. Persistent storage failure cannot be made durable by software alone, but the authoritative in-memory ledger need not be discarded.

## Pre-live review closure

The persistence edge is repaired in the final inspected source. `update()` catches report-write failures and preserves the full in-memory report; its false return stops the arm before another stage, and the experiment stops further paid work. The CLI receives that object and retries persisting it. A generic test injects failure after the first paid worker result and asserts that the latest 190 known tokens remain in the returned report, only one worker ran, and status is failed. With a persistent filesystem outage, the last successfully saved checkpoint remains the durable limit; the code no longer silently replaces its in-memory ledger with an older file during ordinary/transient failures.

No unresolved material source/design finding remains in the reviewed generic implementation. The accounting/control repairs, concrete sandbox/launch/reporting fixes, full candidate retention, runtime/case hash fence, and explicit interpretation limits make this suitable for the planned frozen exploratory run. This approval covers the generic implementation and stated method, not the unseen case validity or future live outcome. The parent owns tests, the immutable freeze, sealed-case author checks, and any paid execution. This reviewer executed no providers, credentials, hidden tests, generated candidates, PostgreSQL checks, or test suite and modified only this review note.

Minimality: the implementation reuses the production selectors, action guards, provider meter and existing isolated storage helper; no new scheduler/provider/cache framework was added. No concrete complexity deletion is requested. Lean already. Ship.

SHA256 values at the final generic source inspection:

| File | SHA256 |
| --- | --- |
| `tests/eval/team_repos/runner.py` | `74f6201f491fe28d60fe059173f2d27e362266f8623e90c8edb4d642e2d019ac` |
| `tests/eval/team_repos/protocol.py` | `edfc765d2c96e8b31feba46bf71ca17fb0e6376c70b169709b00596b277de824` |
| `tests/eval/team_repos/sandbox.py` | `d0fc68ca31f004e4252cf4d011b612ae29fdd40bd6aa45f327498d38e66a741b` |
| `tests/eval/team_repos/experiment.py` | `43ddb0cde89f5def4886af18d44c1cd71ab35a0b79c534815b48257a0a2e389a` |
| `tests/eval/team_decisions/openai.py` | `58ab9cb4e31f6f28e18057ad1582fdb8c254b1f49df11f9c54d8362965628ba4` |
| `tests/eval/test_team_repo_runner.py` | `185a96e92876780ad0be330053ef1850368bee4df569020368d2cb593383dffc` |
| `tests/eval/test_team_repo_protocol.py` | `7ddfd728a676219d5d97d94b96f8bba5a7497716ec14dff4a42e0f947931f630` |
| `tests/eval/test_team_repo_sandbox.py` | `f604113ca711bde477d5bfb32374466782bbc9361cff7c60cf92b5660f0a5ddc` |
| `tests/eval/test_team_repo_experiment.py` | `c376757bd34e63e61d7482d4fe3545975859b08d60b73b5565429fda38559b79` |

## Pre-live interface correction after the first freeze

After the first runtime freeze, the primary implementer accessed only public case IDs, goals, stage briefs and counts. This exposed a cross-component contract error: a repository's `Case.id` was different from the business task label in `Doc.task_key`, while the generic runtime assumed they were equal. The earlier review did not catch this because its generic fixtures used matching values. No provider request or model output had occurred. The initial freeze and original authored manifest are retained rather than relabeled as the corrected protocol.

The author added one explicit public `Case.task_key` metadata field. The generic runtime now uses it at every applicability boundary: researcher source filtering, retained-history admission, handoff qualification, current source filtering for direct/cheap controls, `task_snapshot`, and `SemanticContext`. `Case.id` remains the repository/report/work identity and the input to source/path hashes and slot keys. The reviewer inspected every generic consumer and found no case-specific branch, automatic relevant-source inference, hidden expected-source group, changed ranking parameter, or answer repair. A new synthetic regression deliberately sets `case.id != case.task_key`, includes an unrelated task, and checks common formation plus all four arms over successive stages.

The author-produced metadata-only receipt at `/tmp/ai-knot-product-quality/repo-cases-metadata-repair-proof.json` reports:

- Original canonical manifest: `db505790d96c27c3484a3ab77f552bc05010396f552b997bc9b14d0e970b5f03`.
- Corrected manifest: `148d2368d1c841fd5005879118b041d6d4ce7b9cfe574c8038a4c65af1ffc656`, 55,554 UTF-8 bytes.
- Removing only the new case-level `task_key` exactly reproduces the original canonical manifest bytes; zero existing case fields changed.
- Three cases, nine stages, 60 stage-document entries and all 110 authored assertions remain unchanged. The author reports nine reference passes, nine initial implementations rejected by assertions, six prior-stage references rejected by updated assertions, and six passing integrity tests.
- Zero provider calls occurred before this adaptation.

This reviewer read that metadata-only proof and the generic repair, but did not open the current case file, the pre-adapter case archive, canonical case-content manifest, hidden tests or reference implementations. The equality assertion and sealed-task execution results are author-produced evidence, not independently executed by this reviewer.

The repair is acceptable as a disclosed pre-live interface adaptation. It does not support a fully unseen or double-blind claim: public case metadata was accessed before the correction, and the earlier contributor keyword exposure remains disclosed. No live outcome was available to tune. Preserve both freezes, label the forthcoming live run with freeze v2 and the corrected manifest, and state that original source documents, task text, tests and references were preserved by the author comparison. The report now includes an explicit `interface_correction_note`, and its manifest constant matches the corrected digest.

No unresolved material finding remains in this metadata/interface repair. It is suitable for the second immutable freeze and planned exploratory live run, subject to parent-owned final tests and freeze verification. Relevant generic hashes at this follow-up inspection (superseding the earlier table for these files):

| File | SHA256 |
| --- | --- |
| `tests/eval/team_repos/runner.py` | `e381a23d21ee8c60375a771a025f04b2e99da4dcc72bec715f73b5ad3197a774` |
| `tests/eval/team_repos/experiment.py` | `889355aa6c1454f51bb9f8538a0272611d514c4a9424c54db412a97cce53b581` |
| `tests/eval/test_team_repo_runner.py` | `df03023081f7c3650a35e2130fe5f14dd6c78897825172eddbe997a4192cf37a` |
| `tests/eval/test_team_repo_experiment.py` | `e43c267075647662dbcc3fd8f8b3093da995a2e4c2b40af2d1d8b45ab785ad65` |

## Independent audit of the retained live reports

The reviewer inspected the completed SQLite report and, after its report became complete, the PostgreSQL report. The audit parsed retained public evidence, attempts, receipts and metering records; it did not open hidden test bodies or reference implementations, execute generated candidates, call providers, access credentials, run Docker/full CI, or change runtime/data. A pure `union_basis` call using the recorded failing refs was executed locally to identify the SQLite exception. Only this review note was edited.

Artifact identities:

| Report | SHA256 | Run status | Actual known calls | Actual known tokens |
| --- | --- | --- | ---: | ---: |
| `/tmp/ai-knot-product-quality/repo-live-sqlite.json` | `34d1d80cd1e4ce8b300bc4abac397ffbb4710482c52a01101a97a4fc192f593a` | failed | 56 | 121,994 |
| `/tmp/ai-knot-product-quality/repo-live-postgres.json` | `b82542d7a7695005956837244f9f9d46e1954b36b4f463986c330dcc98ed74d4` | complete | 57 | 129,682 |

All 113 retained provider calls have successful provider status, valid usage, kind `complete`, and returned model `gpt-4.1-mini-2025-04-14`. There are no embeddings, unknown worker reservations, or unknown provider bills. Combined known actual account usage is **113 calls and 251,676 tokens**. Both reports' before/after runtime source hashes are equal, and the complete runtime hash mappings match across the two runs. Both use the corrected manifest `148d2368d1c841fd5005879118b041d6d4ce7b9cfe574c8038a4c65af1ffc656`.

The audit independently recomputed each stage's charges, each arm's formation allocation plus worker charges, and each experiment's deduplicated account ledger. Every formation-using arm contains the exact full common formation ledger; direct sources contains none. All 12 case/arm records and all three scheduled stages are retained in each backend. Every retained applied candidate's code hashes to its recorded candidate SHA256. Per-stage attempt counts and rendered context bounds are respected, and every individual arm remains within 12 requests/60,000 tokens. No paid row was dropped from accounting.

SQLite common formation cost is 9 calls/12,248 tokens. Its standalone logical arm ledgers sum to 74 calls/146,490 tokens, with the extra 18 calls/24,496 tokens representing two additional allocations of that same formation. PostgreSQL formation is 9 calls/12,256 tokens; its logical arm ledgers sum to 75 calls/154,194 tokens, with the extra 18 calls/24,512 tokens again arising solely from the full formation allocations. These logical totals are not additional actual account charges.

### Outcomes and cost

Numbers separated by `/` below mean **SQLite / PostgreSQL**, not a pooled sample. A repository passes all stages only if each of its three stage contracts passed. Final-contract success is shown separately and does not erase earlier failures.

| Arm | Passing stages, out of 9 | All-stage passing repositories, out of 3 | Final contracts passed, out of 3 | Logical calls | Logical tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct current sources | 7 / 7 | 2 / 2 | 3 / 3 | 12 / 11 | 22,971 / 20,532 |
| Raw team history | 6 / 5 | 1 / 1 | 2 / 2 | 20 / 22 | 44,140 / 53,594 |
| Current docs + handoff | 8 / 7 | 2 / 2 | 3 / 3 | 19 / 20 | 34,998 / 36,651 |
| Task memory | 5 / 5 | 1 / 1 | 2 / 2 | 23 / 22 | 44,381 / 43,417 |

SQLite has 47 reader attempts: 46 reached tests, one aborted on conflicting citations. PostgreSQL has 48 reader attempts, all reaching tests. `status="complete"` denotes completion of the evaluation loop, not passing all tests; several complete stages have `tests_passed=false`. The SQLite run remains failed because one stage aborted. The PostgreSQL run remains complete despite failed task contracts. No denominator or status should be rewritten to make either report look more successful.

All nine task-memory acquisitions in each backend are `snapshot`: each loads six current evidence entries. **There is no measured semantic-acquisition or embedding benefit in this batch.** Within each backend, task memory and current-docs-plus-handoff have identical full evidence/metadata sets at every matching stage; their order differs. The current-source baseline is materially cheaper, and task memory does not beat either current-document control on these outcomes. The data do not support general memory superiority or a default change.

### SQLite history abort: observed reader error and harness limitation

At stage index 2 of `notes-digest`/`raw_team_history`, the reader emitted a syntactically valid patch with basis indices `[7,8,9,10,11,12,13,14]`. That basis cites versions 2 and 3 of four logical sources: `handoff.md`, `notes/adapter.md`, `notes/export.md`, and `notes/formatting.md`. The local reproduction with those eight recorded refs raises `StaleEvidence("conflicting revisions of one fact")`; `StaleEvidence` is not a `ValueError`.

`review_context` calls `transitions.union_basis(refs)` before opening its read transaction. The conflicting revisions therefore raise before a `ContextReview` exists. The evaluator's per-attempt handler catches only `ValueError`, so the domain exception reaches the outer stage handler. The report retains the paid response and full parsed candidate, but leaves `basis_valid=false`, `applied=false`, no test receipt, and no per-attempt status. Only one of the two allowed attempts was used; the stage aborted and no candidate was applied or tested.

This combines a model's invalid citation basis with an evaluator recovery defect. It is not a hidden-test failure, a provider/budget failure, or evidence that the candidate would have passed if executed. The action/source boundary rejected the contradictory basis correctly. The missing per-attempt domain-error handler removed the intended second correction opportunity, so the measured raw-history result is qualified by that benchmark limitation. Keep its failed stage and actual charges in the retained result; do not impute success, delete the case, or retrospectively substitute a repaired-policy score.

The failed stage's `source_unchanged=false` is the initialized placeholder: the final comparison was never reached. It is **not an observed source-mutation finding**. The identified exception occurs before the review transaction; the separate runtime-source hash fence passed. A future reporting repair should distinguish an unperformed check from a performed unequal comparison.

The minimal post-run repair is to record `StaleEvidence` alongside `ValueError` as a rejected attempt, retaining its charge and allowing the already-budgeted second attempt. Add a synthetic mixed-revision regression that rejects the first attempt, executes a valid second candidate, records both calls, and proves no action occurred on the bad basis. Do not change the frozen live runtime until both processes have terminated; any future measurement after repair is a separately versioned protocol on already inspected cases.

### Database comparison and ordering confounds

The reviewer compared every matching case/arm/stage across the two reports, excluding only the researcher handoff identity when comparing raw source records. All **36/36** raw-source evidence/metadata multisets match exactly, including path/content, namespace/ID/revision, version and observed time. Raw-source order matches at only **13/36** positions: all nine snapshot positions and four direct-source positions. The controls and history use unspecified `storage.load` order, so hashed path IDs did not establish canonical or natural presentation order for those arms.

All nine common researcher handoff records differ between the independent SQL runs. Consequently full context multisets match at only the nine direct-source positions; full ordered contexts match at only four of those. Within each backend, the cheap handoff control and task-memory evidence sets match at all nine points, but their ordering differs. The measured differences therefore combine source ordering, independently generated summaries, model variation, and subsequent arm-specific code/receipts. They are **not evidence of a database-induced quality difference**, nor a controlled estimate of a pure ordering effect.

For a future parity-specific experiment, declare one source serialization order before measurement and reuse the same frozen formation data across backends. That would be a new protocol; do not reorder or regenerate this run after seeing its outcomes. The current two backends still involve only three authored task definitions, not six independent repositories or eighteen independent task samples.

### Read-API alternative considered, not adopted

The reviewer separately assessed deleting only `review_context`'s early `union_basis(refs)` call. That is technically coherent for a read-only candidate batch: existing per-ref checks could admit the exact current ref, reject stale refs as issues, return `complete=false`, preserve input priority/deduplication/budgets and ACL redaction, and leave all checkpoint/derive/action unions strict. It would not authorize actions from contradictory refs.

However, it is an intentional public behavior change, not required to fix this evaluator. `test_context_review.py` explicitly expects the conflict exception; the original design and public return documentation state that conflicting revisions reject; the HTTP response currently changes from domain error 409 to a successful incomplete review. The earlier team-decision regression also explicitly expects `basis_issue="StaleEvidence"`; changing the read API would change that diagnostic unless its caller is updated. A production change would need its own documented contract, mixed-revision/denied/budget/no-mutation tests, still-strict action tests, and transport checks. It would not by itself handle later domain exceptions from action guards.

The parent and reviewer therefore agreed to retain the documented strict read/action API and make the minimal evaluator caller repair after the frozen runs. No model-quality gain is inferred from this alternative or from the future recovery fix.

Post-run verdict: accounting and retained evidence are internally consistent, with the failure and ordering limitations disclosed. These exploratory results show no benefit over the cheap current-document baselines and no semantic benefit. They do not establish sustained always-on fleet usefulness, generalized repository performance, or statistical superiority. Parent-owned full CI results are separate engineering acceptance evidence, not a model-quality score.

## Post-measurement caller fix and measured-runtime provenance

The parent confirmed both live processes had terminated before changing the evaluator: PostgreSQL exited 0 and SQLite exited 1. The reviewer then independently compared the archived measured runner against the current runner. Exactly three changes exist: import `StaleEvidence`; catch `(ValueError, StaleEvidence)` inside the per-attempt rejection handler; initialize `source_unchanged` to `None` so an unperformed check is distinct from an observed mismatch. No production read/action API or guard was relaxed. Current `memory.py` and `_state.py` remain byte-identical to their measured versions.

The new generic regression constructs old/current refs for the same source, submits both on the first paid attempt and only the current ref on the second, and asserts two charged attempts, a recorded `invalid_proposal`/`StaleEvidence` first result, no application of that first candidate, and exactly one sandbox-port invocation. A separate regression retains a provider failure's charge while reporting the source check as `None`. The source behavior and regression are appropriate for the observed evaluator defect. The parent reports 37 passing focused tests across the two SQL backends; the reviewer inspected those tests but did not independently run them or the final full CI.

**The caller finding is closed.** This is an unmeasured recovery/reporting repair after observing the original run. Neither live report was rerun or rescored. SQLite's failed status and its missing second attempt remain in the retained evidence. No improved model-quality outcome is attributed to the repaired caller, and the strict read API considered above remains unchanged.

The reviewer decompressed `docs/performance/team-repos-measured-runtime.json.gz` and checked archive metadata and generic runner text. Sealed fixture blobs were hashed/compared only; their contents were not displayed or inspected. The archive records base commit `0dc11976dc563182e4185697fa43a0ced5880485`, seven source overrides, 86 measured source hashes, both freezes, and the original pre-adapter source/manifest. Reconstructing each of the 86 source hashes from its override or `git show` of the base commit produced **zero mismatches**. The complete archived hash mapping equals both live reports' before/after mappings and freeze v2's mapping.

The archived original manifest bytes hash to `db505790d96c27c3484a3ab77f552bc05010396f552b997bc9b14d0e970b5f03` and exactly equal the preserved pre-adapter manifest file; the archived original source bytes also equal the preserved pre-adapter source. The initial/final freezes retain the original `db505…` and corrected `148d…` manifest identities respectively. Thus the later interface and caller changes do not overwrite the provenance of the measured or originally authored states.

| Artifact/source | SHA256 |
| --- | --- |
| Measured runtime archive | `f1e2ebfde3fe0ee87870a79941be3f6b6df5ebe5bc3829d50b040f2eac50e0e4` |
| Archived measured runner | `70d3d371901dcd917beea2c0ab88767e29c2c8d31c69765dd4c1b92fe7e9f903` |
| Current repaired runner | `1126628e7adbbd2fe8e1f7b7371d8085400c12377ea4f01a2bf2e977a27715cd` |
| Current runner regression tests | `974e8ddf0948f7d28319edc0175691fdd3daeb9d00bf796af6cb02b78d9bcf6c` |
| Current experiment regression tests | `e43c267075647662dbcc3fd8f8b3093da995a2e4c2b40af2d1d8b45ab785ad65` |
| Unchanged production `memory.py` | `e2958caa58ddea30bc566e97e32d263b69b6641535d32cbd05e3f72c24170194` |
| Unchanged production `_state.py` | `c238d8a2ccf7f867f39c9e1bb6970fb6be2602cafd79c6176e43093e0b12fb5c` |

Final review verdict for this follow-up: no unresolved material finding in the minimal caller repair or archived-runtime provenance. Keep the measured runtime, unchanged raw outcomes, partial-blinding disclosure, source-order/formation confounds, and negative comparison with cheap controls visible in the final artifact summary. The broader product goal remains unproven by this small experiment.

## Python 3.11 sandbox receipt compatibility follow-up

The parent reproduced the existing `deep_json` regression failing on Python 3.11 because `json.loads(payload)` raises `RecursionError`, which is outside the previous `(OSError, ValueError, TypeError)` handler. The reviewer independently inspected the source, existing regression and archived/current diff: the sandbox's sole change is adding `RecursionError` to that fail-closed handler. The test supplies 8,000 nested JSON arrays, a 16,000-byte receipt below the 16,384-byte outer size limit, and expects sanitized `status="error"`. The handler now returns that existing error receipt and still records elapsed time; it does not accept the malformed payload or weaken receipt validation or container isolation.

No unresolved finding: this is the smallest appropriate compatibility repair. The parent reports red then green on the same Python 3.11 test and owns the rerun of full CI and Python 3.11/3.13 suites. The reviewer performed read-only source/artifact checks, not test execution, and invoked no API or credential tooling. No new paid run or rescoring occurred, and no model-quality gain follows from this repair.

The measured sandbox remains archived as `d0fc68ca31f004e4252cf4d011b612ae29fdd40bd6aa45f327498d38e66a741b`; the repaired sandbox is `91df40df0bfb7769037024c1bc8ea0f7e09e3e98095132321790464109ba8922`. The existing sandbox test file is unchanged at `f604113ca711bde477d5bfb32374466782bbc9361cff7c60cf92b5660f0a5ddc`. The runtime archive and both raw live reports retain the exact hashes recorded above (`f1e2ebfd…`, `34d1d80c…`, and `b82542d7…` respectively).
