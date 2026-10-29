# Continued repository work with team memory

The retained GPT-4.1 mini experiment does **not** demonstrate a task-quality advantage for task memory. On both SQLite and PostgreSQL, the memory arm passed the final contract for two of three repositories; current documents plus the same researcher handoff passed all three. No default or product accuracy claim changes on this evidence.

This extends the [closed-choice study](semantic-context-evaluation.md) to generated Python code executed against private tests. It remains a small exploratory study of three authored single-file repositories, not an evaluation on existing production repositories or a concurrent always-on fleet.

## Tasks and execution

Three independently authored tasks cover parcel routing, a Russian-language notes export, and a paginated catalog client. Each has an initial stage and two public source updates. There are 60 document occurrences and 110 assertions in 18 test methods across the nine stage contracts. Every arm starts from the same initial module, then carries its own latest candidate code, including failures, into the next stage. No reference solution replaces an arm's work.

A researcher forms one common public-source handoff per stage before any coder arm executes. It sees no arm's code, test receipts or results. The coder receives the public goal, requirements, brief, its own current code, displayed source entries and bounded feedback from its preceding attempt. It emits a complete replacement `solution.py` and one-based evidence citations. Private test bodies, expected values, reference code and internal source identities do not enter model prompts.

Each stage uses a fresh executor Python process that reopens SQL state: 36 worker processes per backend, 72 total. This verifies bounded process restart and handoff, not simultaneous agents or long-running production operation. All arms use the same SQL intent, saved-code and action-guard mechanisms. The direct-source arm omits the knowledge-acquisition layer; it is not a completely memoryless agent. Stage events explicitly project current source revisions and clear the intent basis, so this experiment does not establish automatic invalidation of an untouched accumulating memory.

## Four acquisition policies

| Arm | Knowledge supplied to the coder |
| --- | --- |
| `direct_sources` | All applicable current raw source documents; no researcher formation cost |
| `raw_team_history` | All retained applicable source and handoff revisions, without hidden truncation |
| `current_docs_handoff` | Current source documents and the exact latest common researcher handoff |
| `task_memory` | Production `task_snapshot`, with explicit semantic acquisition available on overflow |

The cheap current-documents-plus-handoff control is necessary: summarization must not be mistaken for a benefit of memory acquisition. All 18 memory acquisitions used a six-fact snapshot. **No embedding calls or semantic fallback occurred.** The configured embedding model was `text-embedding-3-small`, but this experiment measures snapshot/bundling, not semantic retrieval.

The same evidence and observation-metadata sets reached the cheap control and memory in all nine positions on each backend. Their ordered contexts differed in every position. Across backend runs, raw source sets matched in all 36 positions, raw source order matched in 13, and complete contexts matched in four. Storage iteration for controls/history was unspecified; the nine independently generated handoffs also differed between backend runs. Own carried code and stochastic model outputs add further differences. Results cannot identify an ordering effect or a database-caused quality difference.

## Frozen budgets and accounting

The requested reader/researcher model is `gpt-4.1-mini`; responses reported `gpt-4.1-mini-2025-04-14`. Each case/arm has the same ceiling of 12 requests and 60,000 reported or conservatively reserved tokens, including formation. Each stage permits at most two coder proposals/tests, with 2,048 output tokens per proposal; researcher output is capped at 768. Source context is capped at 24,000 Unicode characters in its rendered public JSON representation, including citation indices and observation metadata. Code, prompts and schemas additionally count toward total provider usage.

Each consuming arm is allocated the **full** common formation cost, including when a later attempt fails. The direct-source arm needs no formation. Actual account spend counts shared calls once; standalone logical arm budgets count them in each using arm. Equal ceilings do not imply equal spending or optimized use of the available budget.

| Backend | Arm | Passing stages / 9 | All stages passed / 3 repos | Final contract passed / 3 repos | Logical calls | Logical tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SQLite | Direct sources | 7 | 2 | 3 | 12 | 22,971 |
| SQLite | Raw team history | 6 | 1 | 2 | 20 | 44,140 |
| SQLite | Current docs + handoff | 8 | 2 | 3 | 19 | 34,998 |
| SQLite | Task memory | 5 | 1 | 2 | 23 | 44,381 |
| PostgreSQL | Direct sources | 7 | 2 | 3 | 11 | 20,532 |
| PostgreSQL | Raw team history | 5 | 1 | 2 | 22 | 53,594 |
| PostgreSQL | Current docs + handoff | 7 | 2 | 3 | 20 | 36,651 |
| PostgreSQL | Task memory | 5 | 1 | 2 | 22 | 43,417 |

Logical columns sum the three standalone case budgets for that arm. Passing a final contract does not erase an earlier failed stage. Account totals were **113 actual completion calls and 251,676 tokens**: SQLite 56/121,994 and PostgreSQL 57/129,682. Formation accounted for nine calls per backend and 12,248/12,256 tokens respectively. Every provider response had known usage; there were no lost workers, unknown-billing reservations or embedding charges. These totals are reported tokens, not a dollar-price estimate.

## Retained failure and later repairs

The original SQLite run exited 1. In the final notes/history stage, the first proposal cited both old and current revisions of four sources. Production `review_context` correctly raised `StaleEvidence`. The evaluation caller caught `ValueError` only, so the stage aborted before its second allowed proposal and before executing that candidate. Its paid request remains in the results. The original `source_unchanged=false` field in that aborted stage was an unperformed-check placeholder, not an observed source mutation; the separate runtime hash fence passed.

Only after both live processes terminated, the evaluator was changed to reject `StaleEvidence` per attempt and preserve the remaining attempt. An unperformed source check now reports `null`. A regression submits contradictory revisions followed by current evidence: both attempts remain charged, only the valid candidate reaches the sandbox. The strict production read/action API is unchanged.

A subsequent Python 3.11 compatibility check exposed `RecursionError` when parsing a deeply nested malformed sandbox receipt. The evaluator now returns its existing sanitized error for that exception, covered by the existing malformed-receipt test. Neither repair was used to rerun or rescore the model results. The archived measured runtime retains both original implementations.

PostgreSQL's original CLI exit was 0. Its report's `status=complete` means the stage loop completed, not that all private tests passed. The outcome table, not CLI success, is the task-quality measure.

## Execution and fixture validity

Generated code ran only in fresh Docker containers using cached image `sha256:b64631e04e4920160c50fbe8d8df828f7f35f06f425cb44aa09bca53e708a35a` on arm64 Linux: network disabled, read-only root, non-root user, dropped capabilities, no-new-privileges, CPU/memory/PID/time limits and a read-only mount containing only that run's three temporary files. No credentials, user files or daemon socket were mounted. Host execution of model code was not used.

Receipts contain bounded test counts/names, never candidate stdout, tracebacks or assertion values. Zero/partial tests, skips, expected failures, abnormal exits, timeouts and malformed receipts cannot pass. After the live runs, 24 reference/negative controls used the same Docker grader: nine stage references passed, nine initial modules and six previous-stage references failed assertions as expected. The grader shares a Python interpreter with candidate code and is not adversarial tamper-proof attestation. Minimal feedback also differs from ordinary coding tools' richer diagnostics.

The generic runtime was frozen before the primary implementer opened fixture content. Public case metadata then exposed a mismatch between repository IDs and business task scopes. Before any API call, the author added only an explicit `Case.task_key`; no existing field, document, assertion or reference changed. Removing that field reproduces the original canonical manifest byte for byte. An orchestration contributor had earlier seen two fixture lines in a keyword search. This is partial blinding, **not** a fully unseen or double-blind benchmark. The metadata repair and exposure are retained in the provenance records.

The final case manifest is `148d2368d1c841fd5005879118b041d6d4ce7b9cfe574c8038a4c65af1ffc656`; the pre-adapter manifest is `db505790d96c27c3484a3ab77f552bc05010396f552b997bc9b14d0e970b5f03`. Both live reports match the same 86-file runtime hash mapping before and after execution. There were no after-result fixture changes or paid repetitions.

## Reproduction and records

The later [team-work foundation](team-work-foundation.md) prepares new independent tasks, incremental publication without producer-side intent reset and equal-order context controls. Its Docker/SQL results are preflight evidence only; no new paid quality result replaces the experiment above.

[Summary and artifact hashes](performance/team-repos-summary.json), original [SQLite](performance/team-repos-live-sqlite.json.gz) and [PostgreSQL](performance/team-repos-live-postgres.json.gz) reports retain complete validated candidate code, costs, context and receipts. [Docker controls](performance/team-repos-docker-feasibility.json), [metadata repair proof](performance/team-repos-metadata-repair.json) and [contradictory-ref reproduction](performance/team-repos-conflicting-ref-repro.json) are separate records.

The [measured-runtime archive](performance/team-repos-measured-runtime.json.gz) contains base commit `0dc11976dc563182e4185697fa43a0ced5880485`, seven source overrides, all 86 source hashes, both freezes and the original pre-adapter fixture source/manifest. Reconstruct measured sources from that base plus overrides; the current caller and sandbox include the later repairs above. The [independent Astra/Ponytail review](superpowers/reviews/2026-10-28-team-repositories-astra-ponytail.md) and [local validation record](superpowers/validation/2026-10-28-team-repositories.json) distinguish those states.

Ordinary evaluation tests use fake providers. The following command starts **new paid calls**, requires macOS Keychain access and the cached Docker image, and is not needed to inspect the retained results:

```bash
PYTHONPATH=src python -m tests.eval.team_repos.experiment \
  --backend sqlite --keychain-service ai-knot.openai-api-key \
  --output /tmp/team-repositories-new-run.json
```

For PostgreSQL, select `--backend postgres` and supply `AI_KNOT_TEST_PG_DSN` for a local test server. Each worker reads the key from Keychain separately; it is not written to jobs, reports, container mounts or Git. No automatic paid retry follows a failed run.

This batch establishes an executable task-outcome evaluation and repairs its observed recovery/receipt handling. The next useful quality work must control source ordering and handoff variation, then study genuinely accumulating context or existing repository tasks. The broad product goal remains active; this small negative result does not justify another default, a claimed accuracy gain or a fleet-capacity claim.
