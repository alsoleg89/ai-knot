What this change does: The shared pool now keeps dense vectors only for the exact current text and requested validity time. A separate test harness compares task decisions over frozen source revisions, records simulated effects separately from rejected proposals, and meters reader plus embedding calls under equal ceilings.

Reviewed independently against HEAD `6f23a03c9cb726e72895c5102ec93d012e4d1ffe`, using the Ponytail review skill. Expected load: long-lived shared-pool readers with concurrent source revisions; the evaluation itself runs one sequential scenario per arm and parallel independent arms in one event loop. This review made no source edits, accessed no user credentials, made no provider calls, and did not touch the original checkout or release files.

**Must fix found during review — repaired and rechecked**

1. **The original hybrid arm paid for vectors without using them** (`tests/eval/team_decisions/runner.py`, `_context`; connected path `src/ai_knot/_pool_recall.py`, `_recall_locked`).
   - **What this is:** The original hybrid arm called async pool recall with the same “Integrate ...” query as the strong lexical control. The existing multi-source path handles that query before hybrid ranking.
   - **Problem:** At runner SHA256 `95e8bdd3738c11a4bd40a26586ada72b2b3138d291d179aa390355cd81ebbd67`, I reproduced all three scenarios making document/query embedding batches `[4, 1]` while `DenseRetriever.search` was called zero times. Checking that an embedding request succeeded would not detect this.
   - **Fix:** The parent preserved the strong Integrate control, gave the hybrid arm a flat requirements query, added a matching `scoped_flat_pool` lexical control, and rejects a hybrid run without an actual dense search. The final protocol has five arms; this was changed before live measurement.
   - **If we skip it:** A comparison labeled hybrid would measure the same lexical route with extra provider cost. Recheck: all 15 hybrid steps in the three-scenario offline matrix now call dense search once.

No unresolved material finding remains in the reviewed source state. The parent also repaired the independently corroborated as-of-time mismatch: both document-embedding snapshots now use the explicit recall time. Past and future validity-window tests pass.

**Evidence checked**

- Focused offline suite: **121 passed, 8 PostgreSQL variants deselected**, in 2.17 seconds. Command: `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /Users/alsoleg/Documents/github/ai-knot/.venv/bin/python -m pytest --no-cov -o addopts='' -o cache_dir=/tmp/ai-knot-astra-pytest-cache tests/test_pool_embedding_revisions.py tests/test_retriever.py tests/test_retriever_relevance.py tests/eval/test_team_decision_openai.py tests/eval/test_team_decision_world.py tests/always_on/test_team_decisions.py -k 'not postgres'`.
- A separate stable-source SQLite matrix completed all **15 harness runs**. English and Russian scripted proposals complete both waves in all five arms. In the paraphrase stress case, raw history and the hybrid fixture complete; the three lexical selectors omit required evidence and correctly record no completed task. These are scripted contract results, not model-quality evidence. Artifact: `/tmp/ai-knot-product-quality/team-decision-astra-offline-matrix.json`.
- Raw history retained all authored revisions: entry counts `[5, 6, 6, 8, 9]`; maximum context size across those arms was 3,890 characters, below the common 12,000-character ceiling. No history was quietly truncated in these fixtures.
- Cache tests exercise same-ID changes, changes during document and query I/O, a late old response after a newer embedding, failed replacement requests, deletion/expiry/future activation, and private-vector-only fallback. Exact text comparison occurs after provider I/O and again against the ranking snapshot. I also traced the changed `DenseRetriever` behavior into `KnowledgeBase`; missing vectors no longer receive an artificial zero-score dense rank.
- Meter tests cover actual usage settlement, unknown billing reservations, concurrent reservations, sticky budget exhaustion, model substitution, truncation, malformed usage/vectors, redirects, cancellation, and sanitized public errors. The runner uses the same per-scenario ledger for document formation, query embeddings and reader calls; lexical selectors have no hidden model calls.
- The source schedule is independent of executed actions. Hidden physical state and the contingent oracle stay out of model inputs. The explicit execution citation contract now matches the guard's requirement for the three current source entries. Source changes during reader I/O reject stale evidence.
- Unsafe executions remain unsafe proposals even when the host prevents harm. A malformed completion stopped after one reader call, with failed status, zero steps and zero effects. Five empty-target holds completed only the evaluation loop; task completion stayed false and physical effects stayed zero.
- Fresh subprocesses reopen SQL and compare sorted source/progress snapshots. The continuing parent then constructs fresh memory/pool objects over reopened storage. This establishes SQL reload and stateless reader handoff for the fixture; it is not an external worker crash/recovery experiment.
- `git diff --check` passed. The tracked product diff is limited to the three pool/retriever files; this review does not replace the parent's protected-file audit.

**Interpretation limits**

`decision_matches_oracle` is action/target agreement only. An execution with a missing basis can match those two fields and still be rejected; report that metric beside invalid/unsafe proposals and confirmed destinations, not as complete decision correctness. A completed evaluation loop also does not mean the simulated task completed.

These five-step authored scenarios have only three supported task facts, idealized source labels, fixed peer confirmations, and a common simulator admission policy. They can expose a concrete failure and compare these selected routes under the stated ceilings; they cannot establish production usefulness, safe external effects, statistical superiority, or general multilingual quality. Uniform fake vectors in the scripted matrix prove the route is exercised, not embedding quality. Exact vector input text reuse is the only product cache claim reviewed here; this change does not fix unrelated private-KnowledgeBase cache behavior.

World manifest: `e4453f9db922bcb521cf6106e0d25a221676e919f0e67c850c82047da3eb300d` (recomputed and matched).

Protocol: `1ca3faeb5654fb0f0740da13ba035d24550e143b592ad876f369e8696c48a280`.

**Reviewed SHA256 source state**

| File | SHA256 |
| --- | --- |
| `src/ai_knot/_pool_recall.py` | `284134a95573f5b1d90eb0a2778078c9b0a6a1b8ced08b2325bde8436f37877c` |
| `src/ai_knot/pool.py` | `3a29bb621983deb370a66ee44630670564dd3b14b34daa5900371b7c8018cdc3` |
| `src/ai_knot/retriever.py` | `3dee843db0fa0d9ab198a3d29527406317f342f4b0f3243e2b0a10ea0e9e6d31` |
| `tests/eval/team_decisions/__init__.py` | `35ae31bf5b6bc9ffb13f31a4136cb0ba9c70a1c03a10a5e54ccb46b2c7e2d2cc` |
| `tests/eval/team_decisions/openai.py` | `8c7f51964d4576f4425da77ceed35677455a43478cf139654749d198f6cbfa5b` |
| `tests/eval/team_decisions/world.py` | `adbff820a609eab34dba758ac9fbf314157181f10e58f81ce20560d89d18b6ed` |
| `tests/eval/team_decisions/runner.py` | `1e697059993f71ff5641e13403677e2fd031743b77ae0244333bcbb669841fc7` |
| `tests/test_pool_embedding_revisions.py` | `a1cf3922019348e26a73da06052128eee4e3f1f0b6a51122c65a4ba08f707f57` |
| `tests/eval/test_team_decision_openai.py` | `58da0b52da10f2363d16e086fc06715281827cc075587e71887399140504fe95` |
| `tests/eval/test_team_decision_world.py` | `7ac43864a34e4aa066dac9ac063c11c40fa8b429b122d669eed0edf4ab9d1bda` |
| `tests/always_on/test_team_decisions.py` | `91a66e72ef678c92b48f152e2d555eefd3e4dc836558487b82e3bc524fc67352` |

Verdict: Ship. The repaired code supports the bounded evaluation claim; live results still need their own interpretation.

Not checked: real OpenAI responses, PostgreSQL execution, full repository CI, or final release/protected-file acceptance; those remain parent-owned checks.


**Protocol 2 follow-up — post-pilot harness repair**

What this change does: Protocol 2 specifies that `target` is a destination name for execute, verify and reconcile, and explicitly defines the empty-target hold. Malformed proposals now consume a rejected step, conflicting revisions mark execution evidence invalid, and bounded reader diagnostics are retained; provider and budget failures still stop the case.

The original review did not anticipate the live reader's fact-ID/operation-ID interpretation. Scripted responses supplied destination names, so those tests could not expose the incomplete argument contract. The preserved protocol 1 run is a pilot failure, not evidence that one memory arm improves decisions. Its artifact `/tmp/ai-knot-product-quality/team-decisions-sqlite.json` has SHA256 `d15bc962a9918a4f8b5f2ac1ece24001450eff29fd842873bbea5384e2feb412`: 91 calls, 55,703 charged tokens, seven aborted cases and zero completed tasks. Per-call charges equal the sum of per-case spending; the source fence passed. The original responses that caused some ValueError aborts were not retained, so their exact cause remains unknown.

The change is methodologically acceptable as a disclosed post-pilot repair on the same authored fixtures. It gives every arm the same argument specification, does not select the correct target for the model, does not repair a model's proposal into a successful action, and does not change world facts, expected decisions, selectors, product code or budget ceilings. Malformed proposals receive `invalid_proposal`, then the unchanged world records a rejection; there is no free retry or skipped denominator. A later completed task can coexist with earlier reader errors and must be reported that way. Results from the two protocols must not be pooled or interpreted as a memory-quality improvement.

**Should fix**

2. **Redacting only the displayed response leaves its parsed target in receipts** (`tests/eval/team_decisions/runner.py`, reader-output recording, SHA256 `7a8a4323b9df75109a1f7dd4b978f5dba7077d2db82dd9667f9e84b425effe78`).
   - **What this is:** The new diagnostic field masks key-shaped strings in `reader_output`. The same response is parsed before that masking and copied into proposals and host progress.
   - **Problem:** A fake key-shaped verify target was masked in `reader_output`, but remained in `proposal.target`, `outcome.target` and the next reader's progress. This reproduction used no real credential; the fixed scenario corpus contains none, so it does not show that a historical run leaked a credential.
   - **Fix:** Reject or sanitize key-shaped reader text before parsing and persisting it, and add a fake-token regression covering the whole report and subsequent model input. The parent has accepted this fix and is preserving the active live run's source state until it finishes.
   - **If we skip it:** The new redaction guarantee covers one display field while other copies of that same text remain durable and can be sent back to the model.

Independent protocol-2 checks at this point: **68 passed, 10 PostgreSQL variants deselected**, in 2.82 seconds, for `tests/always_on/test_team_decisions.py`, `tests/eval/test_team_decision_openai.py`, and `tests/eval/test_team_decision_world.py`, using the same offline pytest options above. A separate five-malformed-response probe produced five rejected/wasted/unsafe proposals, zero oracle matches, zero effects and incomplete task status while finishing the evaluation loop. A separate `BudgetExceeded` probe stopped after one reader attempt with failed status and zero steps. Conflicting historical refs are covered by the new SQL regression and never authorize execution.

Runtime protocol-2 runner hash reviewed: `7a8a4323b9df75109a1f7dd4b978f5dba7077d2db82dd9667f9e84b425effe78`. Initial protocol-2 test hash: `52f2799d0564e6f7787370641b22565be2d407a96fce8a7ae95629fd9ee8dd2c`. The world, provider and three product-file hashes still match the preceding review. Protocol hash: `30ae75abf85cca803e054e68a7cdb70f37ee2b39680f5b7bbfb0def22f570d62`.

Interim verdict: fix 2 before calling the diagnostic path fully redacted. No quality-scoring blocker remains in the protocol repair. The closed-corpus historical results are not invalidated solely by the fake-token reproduction.

Not checked in this follow-up: real provider calls or PostgreSQL execution; this reviewer only read the preserved pilot artifact and ran offline checks.


**Protocol 3 follow-up — structured response contract**

What this change does: The reader receives explicit zero-based indices on every visible context entry and a strict response schema. The API adapter reserves schema bytes before transport and records a hash of the complete request payload. The proposed action, destination, supporting evidence and simulated effects remain model choices checked by the same host policy.

No new quality-scoring defect was found in the schema change. The enum includes every displayed index, including superseded or unsupported entries in raw history; it does not enumerate only correct evidence. The target field is an unrestricted string, so the schema supplies no current target or oracle answer. Empty context permits only an empty basis. Both displayed indices and the formatting they add are included before the whole-entry context ceiling is checked. All arms use the same response contract and still have one proposal per source step.

The provider adapter serializes the schema into the request, includes its UTF-8 bytes in the preflight reservation, and retains that reservation on unknown billing. Actual usage settles the whole provider call as before. `request_sha256` covers the schema and request settings; `input_sha256` retains its original observation-only meaning. The offline tests verify schema forwarding, payload hashing, rejection before transport when schema bytes exceed the remaining budget, and sanitized serialization failures.

The plain-text form of finding 2 is fixed: credential-shaped raw text is rejected before parsing or host storage, and a regression checks the whole report and next prompt. **Finding 2 remains open for JSON escapes.** A valid response whose fake key-shaped target starts with `\u0073k-` passes the raw-string regex, then decodes to the same `sk-...` target. Independent reproduction returned `reader_output_redacted=false`; the decoded fake token appeared in the report and the next prompt. No real credential was used. Check the decoded proposal, or a canonical serialization of the parsed response, before applying it; reject and redact that case too. This is a diagnostic-boundary follow-up, not evidence that the credential-free historical corpus leaked a secret or that its model results should be invalidated.

Independent verification: **74 passed, 11 PostgreSQL variants deselected**, in 2.77 seconds, across the same three focused files used for the protocol-2 follow-up. Additional direct schema checks covered context counts 0, 1, 3, 9 and 128 and confirmed that the target never acquires an enum. The world manifest was recomputed and still matches its original frozen value. Product and world source hashes are unchanged.

The preserved protocol-2 artifact `/tmp/ai-knot-product-quality/team-decisions-v2-sqlite.json` has SHA256 `c25536f4187b0515ccb7d8091c1993d09462285784106047473ec2bf5658da11`. Its source fence passed; all 75 steps ran, but there were 20 recorded reader errors and zero completed tasks. It retains 105 provider calls and 79,114 charged tokens, equal to summed per-case spending. Protocols 1 and 2 together therefore retain 196 calls and 134,817 tokens before protocol 3. These costs remain part of the investigation even though the runs are not evidence of a favorable memory ranking.

Protocol 3 is another exploratory repair made after inspecting pilot outputs on these same authored fixtures. A lower format-error rate under it would establish a clearer response interface, not a memory gain. Any eventual task comparison describes this fixed five-proposal policy: one proposal at each of five source steps, with equal 20-call/20,000-token upper bounds. It does not measure unrestricted autonomous agents that can spend spare budget on extra reasoning, self-review, retries or additional actions. Keep pilot and follow-up results separate and report all spending.

Protocol hash: `f3ddea2a84a999d1e850f3f7de5c9918ea0f920430f48b169e736cff80f15d80`. World manifest: `e4453f9db922bcb521cf6106e0d25a221676e919f0e67c850c82047da3eb300d`.

| Reviewed file | SHA256 |
| --- | --- |
| `tests/eval/team_decisions/runner.py` | `de0b6d2e156616d78e974de8c1e04da6e168f941c72d48f410c9cd1659f52af4` |
| `tests/eval/team_decisions/openai.py` | `0d16e646eb5d37d50dd28c0462801695c80fd4418b67fa3af7131671982533b9` |
| `tests/always_on/test_team_decisions.py` | `d9a6f37f864e2ce4fe200ae3f6538fe76a67ef0f42281f752ff1f6ae3c69af5a` |
| `tests/eval/test_team_decision_openai.py` | `7757214568635f22d25cf7189fef8f2c2ed6527fa4b9cb6487d0262f2a05c8af` |
| `tests/eval/team_decisions/world.py` | `adbff820a609eab34dba758ac9fbf314157181f10e58f81ce20560d89d18b6ed` |
| `src/ai_knot/_pool_recall.py` | `284134a95573f5b1d90eb0a2778078c9b0a6a1b8ced08b2325bde8436f37877c` |
| `src/ai_knot/retriever.py` | `3dee843db0fa0d9ab198a3d29527406317f342f4b0f3243e2b0a10ea0e9e6d31` |
| `src/ai_knot/pool.py` | `3a29bb621983deb370a66ee44630670564dd3b14b34daa5900371b7c8018cdc3` |

Interim verdict: fix the JSON-escape case in finding 2 before calling diagnostic redaction complete. The structured-response comparison itself has no unresolved material integrity finding.

Not checked: real provider calls, PostgreSQL execution or the active protocol-3 live outcome; this follow-up used only local code, offline tests and the preserved protocol-2 report.


**Final recheck — findings closed and measured runtime preserved**

What this change does: The diagnostic check now decodes valid JSON before looking for credential-shaped text, then rejects flagged output before proposal parsing or host storage. Normal JSON formatting changes do not trigger rejection because the redaction flag is compared with the normalized diagnostic text.

**Finding 2 is closed.** Both literal and Unicode-escaped fake-token regressions pass on the reviewer's SQLite run, and the full focused offline set passes: **75 passed, 12 PostgreSQL variants deselected**, in 3.16 seconds. `git diff --check` also passes. The parent separately owns its reported PostgreSQL regression run. No unresolved material code finding remains in this review.

Final reviewed runner SHA256: `24b98de28211f5c9c1e1bd6d75c8592b4668fe827deb1f89ba52ea697bcdb591`.
Final reviewed SQL/protocol test SHA256: `0fd7771d91ab3a1db9187a790cb9523eda44bf6d30eaca2ef7166a3f43fc06a8`.
The provider, world and three product-file hashes remain the same as in the protocol-3 table above. The response protocol hash remains `f3ddea2a84a999d1e850f3f7de5c9918ea0f920430f48b169e736cff80f15d80`.

The measured protocol-3 runner remains the earlier `de0b6d2e156616d78e974de8c1e04da6e168f941c72d48f410c9cd1659f52af4`; the final diagnostic patch is not retroactively described as measured code. I independently decompressed `/tmp/ai-knot-product-quality/team-decisions-v3-runtime.json.gz`, hashed its six source texts, and verified that they match both its own manifest and the source hashes in both live reports. Only the runner differs between that archive and the final six-file runtime. Archive SHA256: `6a8d2e153a6273db87eb032917ab7ee202f1ee1db878585745a77eb30406ef05`.

| Preserved protocol-3 report | Artifact SHA256 | Calls | Charged tokens |
| --- | --- | ---: | ---: |
| `team-decisions-v3-sqlite.json` | `258d714f5bd85e9e86b064fe471689461c9b5907a2f06b6f1f5f7c7fe686e99e` | 105 | 87,249 |
| `team-decisions-v3-postgres.json` | `8a9ff2834e7e275f7e5a0ad2ee504522251fc810c2ff5ee7770ce5adeb3d9fa5` | 105 | 87,275 |

Both reports have complete evaluation status and a passing source-change fence. Per-call token charges equal summed per-case spending. Across both reports, all 150 preserved reader outputs remain unflagged by the final canonical-JSON redaction check; the patch therefore changes no observed admission decision on this closed dataset. All 75 corresponding SQLite/PostgreSQL context selections compare equal as JSON structures. The differing task outcomes do not establish a retrieval advantage for one SQL backend.

The raw-history control completed 2/3 tasks in the SQLite run and 3/3 in PostgreSQL. That strong control prevents a general memory-dominance claim. These remain exploratory results on three authored scenarios per arm, with exactly five proposal opportunities per scenario. The two SQL executions do not create six independent task definitions. They do not establish production readiness, sustained always-on costs, or an optimal autonomous policy under the 20-call ceiling. The earlier qualifications about action/target agreement versus valid evidence and actual task completion still apply.

The four preserved live studies (protocol 1, protocol 2 and protocol 3 on both SQL backends) account for **406 calls and 309,341 charged tokens** in total. This is a ledger reconciliation of the supplied artifacts, not a claim that the reviewer initiated or independently reran those provider calls.

Verdict: Ship. Both concrete review findings are repaired; retain the pilot failures, exact measured runtime and narrow empirical scope in the final report. The broad product goal remains open.

Not checked by this reviewer: direct OpenAI execution, direct PostgreSQL execution, full repository CI or protected-release acceptance. The reviewer performed offline tests and inspected the parent's preserved live evidence; no credentials were accessed and no source files were edited.
