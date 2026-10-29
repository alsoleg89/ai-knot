# Decisions with accumulating task evidence

Ten independently authored synthetic logistics tasks have two consecutive stages each. The source histories contain additions, updates, expiry, deletion, dated rules, exceptions and conflicts. This study measures proposed next decisions and required evidence; it does not execute external work or demonstrate real-team productivity.

The case data, common policy/schema/prompt, harness and 81 runtime/meter source files were frozen before live calls. The case SHA256 is `02b9ad58bd4d1525d6224a7385b750952df68658d4717955f74c3ef878b3659f`. No score-based tuning or paid rerun occurred. All arms use the same observations, task rules and request; native selectors determine evidence and ordering. The full-context reference is deliberately stronger: up to 128 facts and 8,000 total message characters. Lexical and semantic selection each receive eight facts and 2,000 rendered evidence characters. Every arm has the same offered six-request/20,000-token ceiling per two-stage task, including embeddings and the one scheduled late-update retry; consumption differs.

## Live SQLite result

GPT-4.1 mini supplied decisions; text-embedding-3-small supplied semantic vectors. JSON-object mode and identical local schema validation were used. All 30 case/arm runs and 60 final stages completed without provider or format failure.

| Arm | Correct decisions | All required sources supplied | Correct decision AND all sources AND context limits | Mean required-source recall | Both stages successful | Requests | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full task reference | 17/20 | 20/20 | 17/20 | 100% | 7/10 | 21 | 25,202 |
| Lexical selection | 15/20 | 5/20 | 4/20 | 82.85% | 1/10 | 21 | 19,741 |
| Semantic selection | 15/20 | 5/20 | 4/20 | 78.89% | 1/10 | 40 | 27,718 |

Total: 82 requests, 72,661 actual reported tokens. Request/token totals include three rejected provisional answers. Citation availability is reported separately and is not part of the combined success column. Required-source sets are frozen coverage obligations, not independently proven minimal explanations.

The semantic arm did not improve decisions over lexical selection and had lower mean required-source recall at greater inference cost. Neither selected arm matched the full-task reference. Some answers were correct despite missing required evidence; decision accuracy alone hides that omission. Three full-reference decision errors also show that supplied evidence does not guarantee correct reasoning. This small synthetic sample does not establish statistical significance, production accuracy or a general embedding comparison.

Each arm rejected the known late-update provisional proposal through its captured task guard, then used its one scheduled fresh-context retry. No effects were started and no tasks were claimed completed; final intents contain READY proposals. Guard rejection establishes source-set freshness at checkpoint, not semantic correctness or safe action authority.

## PostgreSQL and verification

PostgreSQL replayed the exact recorded embedding inputs/vectors and ordered completion requests/responses, with no provider credentials, network fallback or additional paid calls. All 30 pairs/60 stages matched SQLite's final metrics, intent states and guard outcomes. This is backend parity evidence, not a second independent model-quality sample.

The full local Python suite passed 2,737 tests with PostgreSQL required, eight expected skips and 90.09% coverage. Focused compatibility checks passed 222 tests each on Python 3.11 and 3.13, with three PostgreSQL-only regression skips for SQLite parameters. TypeScript passed 62 tests, type checking and its ESM/CJS build. Source/tests and changed host examples pass Ruff; strict mypy passes 80 source files. Focused compatibility runs disable coverage because the full suite separately verifies its global threshold.

Retained failures include the initial missing-feature tests, PostgreSQL lock-wait regressions without the isolation fix, an old route-count expectation, a full run affected by temporary PostgreSQL connection closures, a focused run that tripped the global coverage threshold despite passing tests, and whole-repo formatting failures in eight unchanged files outside CI's format targets. The unchanged full-suite retry passed. No failures were rescored as model success.

Ponytail's final source review found no concrete defect. Prior Astra source/method reviews are retained; its final follow-up was interrupted at the user's request. No final Astra acceptance is claimed, and further Astra use requires an explicit user request.

## Reproduction and next work

The [compressed evidence](performance/task-context-guard-raw.json.gz) retains frozen cases, protocol, harness, source bytes, offline cancellation/failure checks, exact live/replay reports, vectors, outputs and test logs. The [validation record](superpowers/validation/2026-10-29-task-context-guard.json) gives hashes and summaries; the [Ponytail review](superpowers/reviews/2026-10-29-context-guard-ponytail.md) states its scope. Extract text entries by their recorded relative paths. The harness has an offline-only CLI; live execution requires explicitly supplied credentials and the reviewed source digest. PostgreSQL replay requires the retained live recording and a local DSN, never credentials for a model.

The new [task-context guard](task-context-guard.md) and [selected planning recipe](bounded-planning.md) address stale decisions during concurrent updates. They remain opt-in. The next quality problem is selecting complete sets of applicable prerequisites under a bounded context. Investigate that against new independently authored cases and strong full-context controls, preserving the frozen results above. Release versions and deferred release preparation remain unchanged. The broader product goal remains active.

The subsequent [recorded-vector diagnosis](research/2026-10-29-selection-loss-diagnosis.md) exactly reproduces all 20 semantic selections without provider calls. Every source has positive similarity to both query facets, so reported facet coverage does not distinguish complete prerequisite sets. The frozen required sets fit the budget in all 20 cases; discovering those sets without gold remains the separate product problem. This diagnosis does not rescore or extend the quality sample above.
