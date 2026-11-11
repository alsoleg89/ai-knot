# Semantic candidates: first independent case batch

The new [semantic helper](semantic-context.md) retrieves useful paraphrases that the lexical task selector misses in this batch. It does **not** beat the simple scoped task-bundle control. Keep it opt-in; these results do not justify changing the default or claiming general accuracy gains.

## Protocol and sealing

An independent agent authored 16 closed-choice cases across maintenance, parcel handling, database incidents and releases, with English/Russian queries. Twelve need current readable evidence; four require abstention. Eight admitted task corpora fit three facts/600 Unicode characters; eight contain readable same-task distractors and require selection. The 54 distinct records include denied scopes, unsupported observations, other tasks, expired approvals and stale derived copies.

The implementer did not open these cases until the semantic implementation, exports and tests were frozen. Those three files stayed byte-identical through both live runs. Fixture whitespace was formatted afterward; the canonical data/gold manifest stayed `f1cdd8136013d62f4f83e94492d6faa2ee589c942871ce4481210430331907e4`. Independent review repaired reporting/verification defects before live execution. No case, label, query, reader schema or runtime was tuned after observing results.

Protocol `c6cdbb9b97ce885f006f336274b27700c47aba2989124a2b2f79235b6318c0ce` compares:

- `lexical_task_context`: existing `select_context`, unchanged full query as one requirement.
- `scoped_bundle`: the same source/task/access/currentness admission, then canonical first whole facts that fit.
- `semantic_context`: the frozen helper, fresh cold cache per case; unchanged query and at most one embedding batch. A complete fitting task corpus returns `snapshot` without embeddings.

All arms use identical observations, task/access labels, context ceiling (three facts/600 content characters), strict one-based visible citation indices and GPT-4.1 mini prompt/output ceiling (128 tokens). A fresh common ledger allows at most two provider requests and 12,000 total tokens per case/arm, counting embeddings plus reader work. These are equal ceilings, not forced equal expenditure. The model receives the public question/choices/content, never expected answers or gold groups. Requested embeddings are `text-embedding-3-small`; the served reader was `gpt-4.1-mini-2025-04-14`. No retries were made.

## Retained result

Both SQLite and PostgreSQL produced this same table, with all 48 paired contexts and reader outputs equal:

| Arm | Grounded correct /16 | Required positive evidence visible /12 | Correct abstention /4 | Requests | Total tokens |
|---|---:|---:|---:|---:|---:|
| Lexical task context | 4 | 1 | 3 | 16 | 4,529 |
| Scoped task bundle | 15 | 12 | 3 | 16 | 5,068 |
| Semantic candidates | 14 | 12 | 3 | 24 | 6,111 |

The semantic arm made eight embedding calls, consuming 1,023 tokens per backend. Its other eight selections used the zero-call snapshot. Both complete task bundles and semantic retrieval recovered every required positive evidence group; a correct candidate set did not ensure a correct reader decision. On the two-condition release case, semantic selection included an extra probe-description distractor and the reader chose the wrong action. The task-bundle reader answered correctly on that case. This observation does not isolate which text or ordering caused the error.

Every arm answered the expired-attestation case incorrectly. Both lexical runs also returned `ABSTAIN` with `basis=[1]` in the maintenance-prerequisite case. The parser rejected that response because abstention requires an empty basis. Each request's 309 charged tokens remain in the ledger, and its success metrics stay false. Consequently **both raw run statuses remain `failed` and both CLI invocations exited 1**. All 48 records per backend and all actual provider charges are retained; there was no retry to make the report green. All 112 HTTP/provider calls succeeded structurally; the retained failures concern reader output/decisions. Total reported usage across both SQL backends is 31,416 tokens.

These are editorial closed-choice cases, not a population estimate or open-ended always-on task study. Repeating the same cases on two SQL backends checks parity, not a doubled independent sample. Canonical low-ID placement favors relevant records in the task-bundle control. `grounded_correct` checks the expected choice and citation of every required group; it does not penalize extra irrelevant positive citations or prove causal faithfulness. The cheap control and all negatives remain visible in the [machine-readable summary](performance/semantic-context-summary.json).

## SQL and cost profile

Five sequential cold/warm repetitions used deterministic two-dimensional vectors with all HTTP blocked. These measurements include source admission and post-selection SQL checks; they measure no embedding service latency, semantic quality, concurrent fleet or sustained capacity.

| Admitted facts | SQLite cold/warm median ms | PostgreSQL cold/warm median ms | Callback behavior |
|---|---:|---:|---|
| 3 | 0.82 / 0.72 | 10.25 / 10.07 | Whole snapshot; zero calls |
| 32 | 4.47 / 4.56 | 93.41 / 88.32 | Cold one batch, warm zero calls |
| 128 | 12.80 / 14.18 | 259.91 / 278.93 | Cold one batch, warm zero calls |
| 129 | 5.75 / 5.56 | 114.40 / 117.43 | `input_limit`; zero calls |

The warm cache saves paid embedding work, not source-checking SQL. The candidate cap does not bound namespace scanning, so these results do not establish large-corpus performance. [Raw repetitions](performance/semantic-context-profile.json) retain maxima and all samples.

## Acceptance and reproduction

Local Python CI with PostgreSQL required: **2,252 passed, two expected skips, 89.90% coverage**. Focused Python 3.11 and 3.13 suites each passed 137 cases. Ruff and strict mypy passed. Existing structural MA validation returned all 19 selected scenarios and 11/11 binding checks per backend. Five SQLite and six PostgreSQL advisory misses remain; the extra PostgreSQL miss is 173.014 ms p95 against the unchanged 150 ms advisory target. No threshold was weakened. The hardened gate rejects missing selected results, empty/non-finite measurements, execution errors and comparison-only verification.

[Astra/Ponytail review](superpowers/reviews/2026-10-28-semantic-context-astra-ponytail.md) records repaired defects and independent limits. [Acceptance metadata](superpowers/validation/2026-10-28-semantic-context.json) links source/test hashes and local checks. All 30 protected original-checkout files, its HEAD/status and deferred release changes remain unchanged. No package release, push or merge was performed.

From the matching source checkout, the following commands create a **new paid run**. They do not reproduce identical model outputs by guarantee:

```sh
PYTHONPATH=src python -m tests.eval.semantic_context_runner \
  --backend sqlite --output /tmp/semantic-sqlite.json \
  --keychain-service ai-knot.openai-api-key

# Set AI_KNOT_TEST_PG_DSN to a dedicated local test database first.
PYTHONPATH=src python -m tests.eval.semantic_context_runner \
  --backend postgres --output /tmp/semantic-postgres.json \
  --keychain-service ai-knot.openai-api-key
```

The [summary](performance/semantic-context-summary.json) links the exact compressed raw reports, runtime sources over base commit `cf5c210`, pre-fixture/pre-live freezes and artifact hashes. PostgreSQL runs create/drop only their own UUID-named schemas. The microprofile is an intentionally small owner measurement; it does not add a new benchmark framework.

The broad product goal remains active. Before another default or architecture change, test task bundling and semantic selection on a corpus with natural source order and longitudinal accumulation, with predeclared cases and the same aggregate inference budgets. Preserve the source/action guarantees and measure reader distraction as well as retrieval recall.
