# Cheaper source checks within a context read

Task snapshot, lexical task selection, saved-context review and semantic acquisition now reuse repeated database reads within their existing read-only SQL transaction. Public APIs, ranking, output budgets, source guards and provider spending are unchanged. This reduces repeated work when many team lessons depend on the same source observations.

The motivating workload has 64 task facts and 16 common shared roots. Previously, loading a fact for the task corpus did not prevent `_view` from loading it again; each descendant also reread shared roots, grants and freshness rows. A complete snapshot needed 3,330 backing reads even though its returned context was small.

## Lifetime and admission

The internal `ReadCache` reuses complete namespace rows by ID/slot, identical point loads and identical SQL read statements. It preserves the backing row order and keeps private/shared IDs separate. Empty results are reusable too. Returned containers and SQL row dictionaries are copied. `Fact` instances remain internal and are not mutated. Semantic acquisition can retain captured source objects across provider I/O, as before; no cached lookup crosses that await, and revalidation uses a new SQL snapshot. No public object cache or mutation interface is introduced.

Each operation constructs its own adapter inside the existing transaction and discards it afterward. A semantic provider call happens outside SQL; revalidation creates a new adapter over a new transaction. Write transactions and action admission keep their original sessions. No permission TTL, persistent fact cache, cache invalidation worker or new model call is added.

A writer can commit while a read retains its earlier snapshot. The next operation must observe the changed source or grants; already delivered text cannot be revoked. This follows the existing backend isolation rather than providing a new freshness guarantee. See the [primary-source note](research/2026-10-28-memory-read-scaling.md) and the real concurrent-commit regression tests.

Point-only `review_context` still uses targeted ID/slot queries; it does not scan a retained namespace to satisfy a query-count target. Full task reads still scan and deserialize retained namespaces, and revision/lineage checks still consume CPU. Reuse adds transient maps proportional to scanned facts and distinct queried rows. Output limits do not bound this work or memory.

## Local before/after profile

Baseline product source is commit `48b99d549bf8610dba8ecb047789033b1dc3a88c`. Both backends used 64 task facts, either 0 or 16 shared roots, and either 0 or 2,000 unrelated retained shared facts. Every operation has five sequential timing samples. PostgreSQL 16 ran in a local loopback Docker container; SQLite ran on the host. No CI or concurrent workload ran during profiling, and no model/embedding request occurred.

Counts cover backing fact-loader SELECTs plus read statements through `SQLSession`; they exclude transaction setup/connection negotiation. Timing covers the context operation, connection/setup and counting wrappers. The original shared-root baseline ran before implementing the adapter; the additional root-free control ran against the exported same pre-change product source. All sample bytes and source metadata are retained.

| 64 facts, 16 common roots | Reads before → after | SQLite median, ms | PostgreSQL median, ms |
| --- | ---: | ---: | ---: |
| Snapshot, no unrelated facts | 3,330 → 99 | 42.26 → 24.23 | 1,144.92 → 62.96 |
| Lexical selection, no unrelated facts | 3,331 → 83 | 54.78 → 34.19 | 1,334.34 → 75.64 |
| Point review, no unrelated facts | 3,392 → 225 | 41.51 → 21.45 | 1,201.33 → 106.47 |
| Snapshot, 2,000 unrelated facts | 3,330 → 99 | 59.24 → 44.06 | 1,176.68 → 85.06 |
| Lexical selection, 2,000 unrelated facts | 3,331 → 83 | 70.80 → 52.67 | 1,135.65 → 86.86 |
| Point review, 2,000 unrelated facts | 3,392 → 225 | 42.95 → 21.08 | 1,234.57 → 109.18 |

The declared backing-read budgets were met: fewer than 200 for whole-corpus reads and fewer than 300 for targeted review. The sought 2x median speedup was **not reached by most SQLite operations**; their shared-root improvement was about 1.34–2.04x, versus about 11.28–18.19x on this PostgreSQL setup. No timing sample was rerun or dropped to meet that target.

The root-free control also stays visible in the [summary](performance/context-read-reuse.json). Its improvement was smaller, down to 1.05x for PostgreSQL snapshot with 2,000 unrelated facts. Full scans, connection costs, hashing and host/cache variation remain. Five samples do not establish stable tail latency, concurrent fleet capacity or production throughput.

All 24 operation/configuration pairs returned the same public-result hash before and after, including exact evidence refs, content, ordering and issue metadata. Those pairs are deterministic workload checks, not 24 independent model tasks or evidence of improved answer accuracy. The earlier [repository quality result](team-repository-evaluation.md) remains unchanged and showed no memory advantage over the cheap handoff control.

## Reproduce and verify

```sh
PYTHONPATH=src python -m tests.eval.context_read_profile \
  --backend sqlite --output /tmp/context-reads.json
```

Use `--roots 0` for the flat control. For `--backend postgres`, provide `AI_KNOT_TEST_PG_DSN` for a local test server. The harness creates/drops only its own UUID schema or temporary SQLite file. It performs no inference and never needs the OpenAI key.

The [raw archive](performance/context-read-reuse-raw.json.gz) retains exact UTF-8 source report bytes and hashes. The [validation](superpowers/validation/2026-10-28-context-read-reuse.json) records local SQL, compatibility and full-CI checks. [Astra/Ponytail review](superpowers/reviews/2026-10-28-context-read-reuse-astra-ponytail.md) independently examines source behavior, accounting and minimality; it does not turn a serial profile into a quality or capacity claim.
