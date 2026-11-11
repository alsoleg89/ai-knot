# Decisions across source changes and uncertain operations

Shared async retrieval now refreshes vectors when a fact's text changes under the same ID. It also discards responses made obsolete during embedding I/O and checks vector/text agreement again before ranking. Deleted or inactive entries leave the cache; unchanged text keeps its vector. A deterministic regression reproduced different rankings in a long-lived versus reopened pool before this repair. Updating one document after an initial ten embeds only that document.

Dense retrieval no longer ranks facts that have no vector, and lexical fallback works when cached vectors exist only for inaccessible facts. An explicit async `now` applies to both embedding snapshots and final retrieval; previously a historical/future query could rank facts that its embedding stage considered inactive. Provider defaults and opt-in configuration remain unchanged.

## What was measured

The [runner](../tests/eval/team_decisions/runner.py) compares five memory inputs with real `gpt-4.1-mini-2025-04-14` reader responses and `text-embedding-3-small`. Three authored traces cover English, Russian and paraphrased requirements. Scripted peers publish source observations; one stateless reader per arm proposes actions. A separate simulator applies a common admission policy and records actual simulated effects and receipts.

Each trace starts with a pending canary receipt. The peer confirms it, the first execution happens but returns an unknown outcome, and a SQL reopen checks durable source/host state. The approved destination then changes and its independent receipt arrives. Execution needs current references to the destination, rule and receipt. Reconciliation must confirm the earlier effect rather than repeat it.

The source manifest was frozen before retrieval/model runs. The policy has exactly five proposals, one per scheduled source/work step, within the same per-trace ceiling of 20 provider requests, 20,000 total reader/embedding tokens, 12,000 context characters and 128 output tokens per proposal. This is a tight decision trace, not an unrestricted autonomous agent budget: an extra cautious verification can consume a useful step.

All arms receive the same exogenous observations and authority. Their subsequent tool receipts can differ because their actions differ. A fresh subprocess verifies SQL state at the handoff, then the parent reopens storage; the model call itself stays in the metered parent. This does not demonstrate a fleet of independently running LLM workers.

| Arm | Memory input |
|---|---|
| Full history | All authored observations/revisions and their labels, within the same context ceiling |
| Scoped pool | Strong existing task-scoped `Integrate` query and facet assembly |
| Scoped flat pool | Existing lexical retrieval with space-joined requirements |
| Task context | Existing deterministic requirement/role working set |
| Hybrid scoped pool | Same flat query as its lexical control, plus explicitly configured OpenAI vectors |

The `Integrate` path uses lexical facet assembly, so it is retained as a strong control rather than mislabeled as hybrid. Every measured hybrid step asserts an actual dense search and accounts for document/query embedding calls. None of the other arms invokes embeddings. All source contexts at the 75 corresponding SQLite/PostgreSQL positions match exactly.

## Exploratory result

Each row below covers three traces and at most six confirmed destinations. "Unsafe proposals" counts attempts rejected by the common simulator policy; zero resulting unsafe effects is a guard property, not proof that a model made safe choices.

| Arm | SQLite tasks / 3 | PostgreSQL tasks / 3 | Confirmed destinations, SQLite / PostgreSQL | Unsafe proposals, SQLite / PostgreSQL | Total tokens, SQLite / PostgreSQL |
|---|---:|---:|---:|---:|---:|
| Full history | 2 | 3 | 5 / 6 | 2 / 1 | 24,076 / 24,073 |
| Scoped pool | 1 | 1 | 3 / 3 | 1 / 1 | 14,998 / 14,998 |
| Scoped flat pool | 1 | 0 | 3 / 2 | 1 / 1 | 14,958 / 14,975 |
| Task context | 2 | 1 | 4 / 3 | 1 / 1 | 14,961 / 14,963 |
| Hybrid scoped pool | 2 | 2 | 5 / 5 | 0 / 0 | 18,256 / 18,266 |

Hybrid used 45 calls per backend: 15 reader and 30 document/query embedding calls, including rebuilding after reopen. Embeddings contributed 1,999 tokens to each backend total. Other arms used 15 reader calls each. All individual trace/arm runs stayed within their ceilings; totals in the table combine three separately bounded traces.

The paraphrase trace exposes a concrete selection gap: lexical paths omit the current target/rule while the same source observations are available. Hybrid retrieval completes both destinations on both backends. Full history remains a strong control and also completes that trace in its PostgreSQL run. With only three inspected fixtures and a single live run per backend, this does not establish general superiority, statistical accuracy, a production latency/capacity target, or a gain over full history.

The two SQL runs have identical selected contexts but some different model decisions. Temperature zero does not make these repetitions evidence of a database-induced quality difference. `decision_matches_oracle` in raw records measures action/target agreement only; a rejected basis can still accompany that agreement. Read it alongside task completion, rejected proposals and receipts.

This study compares retrieval inputs under a common reader/guard policy. The hosting recipe's earlier missing-context suppression is not applied; it would avoid some model wakes but would not acquire the omitted evidence. Its full operational behavior remains covered separately by the SQL host tests.

## Protocol repairs are retained

Two preliminary runs exposed test-interface defects. Version 1 left the target argument's meaning implicit and aborted some traces on response/basis errors. It spent 91 calls and 55,703 tokens. Version 2 clarified destination arguments and continued after rejected proposals, but implicit numeric citation positions still caused 20 reader-format errors; it spent 105 calls and 79,114 tokens.

Version 3 displays zero-based citation handles and uses the same strict JSON Schema for every arm. The schema enumerates only visible indices, including old/unsupported history entries; it never enumerates hidden correct destinations or gold facts. Schema formation is included in budget reservation. The [official Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) describes this API mechanism. The world, oracle and selection algorithms were not tuned to make the reader pass.

The version changes are exploratory repairs on already inspected fixtures, not an accuracy improvement attributable to memory. All four multi-step runs remain in the [accounting summary](performance/team-decisions-summary.json): **406 calls and 309,341 tokens**, including failed preliminaries. The earlier [three-question pilot](performance/openai-reader-pilot.json) is separate.

## Reproduce and inspect

With an authorized `OPENAI_API_KEY` in the process:

```sh
PYTHONPATH=src python -m tests.eval.team_decisions.runner \
  --backend sqlite --output /tmp/team-decisions.json
```

For PostgreSQL, supply `AI_KNOT_TEST_PG_DSN` and use `--backend postgres`. The runner creates and removes only its UUID-named schema. On macOS, an existing Keychain record can instead be selected with `--keychain-service SERVICE`; no credential is written to reports or model inputs.

The summary links compressed raw reports for both preliminary versions and both measured version-3 backends. The [runtime archive](performance/team-decisions-v3-runtime.json.gz) contains the exact six source overrides at their measured hashes over its recorded base commit. Final runner code additionally rejects literal/JSON-escaped credential-shaped output before host storage; none of the 150 measured version-3 outputs matches that guard, so it changes no observed decisions.

The [validation](superpowers/validation/2026-10-28-team-decisions.json) and [independent Astra/Ponytail review](superpowers/reviews/2026-10-28-team-decisions-astra-ponytail.md) distinguish offline contracts, live measurements and engineering acceptance.

The broader [product goal](shared-memory-roadmap.md) remains active. Next work needs independent held-out tasks and bounded semantic candidate acquisition for missing always-on context, with current source/ACL/revision checks. Measure sustained formation/reopen costs before adding a shared vector cache or changing defaults. This batch fixes a demonstrated runtime defect and creates usable evidence; it does not finish the product.
