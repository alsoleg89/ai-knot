# Task-source scan — independent Astra / Ponytail predesign review

2026-10-28, base `53d72fbc290486060bad6c052f6c353516f43993`. Read the research note, supplied baseline, task applicability/acquisition, ReadCache, SQL session seam, backend loaders/decoders and source/slot validation. Read-only design assessment; no code changes, SQL, provider/API/credential access, tests or extra agents. Only this note was written.

**The proposed shape is a reasonable minimal optimization.** Add the internal filtered-load seam to the existing SQL loaders and use it only for `task_sources` candidate acquisition. Leave ordinary whole/point/slot loads, lexical selection, Python admission, post-provider snapshots, action guards and public output limits alone. No new index/schema, paging protocol, service or default budget is needed for the demonstrated issue.

The retained baseline contains 12 SQLite measurements, three timing samples per row. With 10,000 unreferenced other-task rows, snapshot and semantic acquisition each decode 10,002 Facts and peak around 36.5 MB of traced Python allocations, returning only `generic` and `target`. Same-task overflow also decodes 10,002 before returning its explicit limit outcome. These establish a decoder/materialization opportunity; they do not measure PostgreSQL improvement or prove a bounded retained-data scan.

## Correctness conditions before implementation

1. **SQL must be a conservative candidate filter.** Every Fact that Python `task_applicable` would admit must survive it. Generic means absent task_key or the empty string; the requested task key also qualifies. Python still rejects raw episodes, unsupported/inactive data and all failed basis/ACL/freshness/lineage checks. Bind task keys as SQL parameters, use a fixed JSON path/key, and keep type exclusions narrow. Do not add time/support/ACL predicates merely because the same query is being edited. Dates are stored as TEXT and interpreted by `_parse_datetime`/`Fact.is_active`; moving those rules into text comparisons would be a different compatibility change.

2. **Match the decoder's legacy JSON handling.** Both `_fact_from_row` implementations treat empty/falsy stored qualifiers as `{}`. Direct JSON extraction/casting on `''` can introduce failures where the old decoder succeeded; normalize that case consistently. A JSON null/non-string task_key is not a generic string key under Python, even if a coarse COALESCE/text extraction admits it; retaining Python filtering is essential. Include empty qualifiers, missing/empty task keys, JSON null/boolean/number values, quoted/escaped/unicode task labels and malformed input behavior in parity tests. Avoid silently changing the legacy TEXT JSON contract by choosing a stricter JSON representation without checking it.

3. **Duplicate JSON keys are a false-negative trap.** Existing `json.loads` uses the last duplicate task_key. A backend's scalar JSON extractor need not select the same duplicate. A row such as `{"task_key":"other","task_key":"restore"}` must not be excluded when the requested key is restore. A conservative any-matching-key predicate (JSON iteration/EXISTS) can overinclude and let Python decide the last-key result; that is preferable to excluding a valid candidate. Test both duplicate orders on both engines. Normal dict writes do not emit duplicates, but the stored TEXT/old decoder currently accepts them; an optimization should not accidentally redefine such legacy rows.

4. **A filtered list is never a complete namespace or slot.** Prime only known positive exact `(namespace, fact_id, no-slot-filter)` entries. Do not populate the complete `_ids`/`_slots` structures, infer negative ID results, or satisfy slot queries from this partial list. A conflicting/latest slot peer may carry another task label, and exact lineage roots/intermediates may be other-task, archived or episodic. Those unrestricted reads remain required for rejection/currentness. Include same-ID cross-namespace cases and a selected fact whose hidden other-task slot peer changes its admission, plus an other-task/episodic lineage root.

5. **Ordering needs a precise claim.** Preserve the current `ORDER BY created_at` clause and no LIMIT/prefix selection. It specifies no deterministic tie order. A changed predicate/query plan/cardinality can reorder equal-timestamp rows even with that clause unchanged. Snapshot output is finally sorted by namespace/ID, but semantic source insertion order feeds embedding inputs and tied ranking. Test equal timestamps with nonmonotonic IDs against the baseline for selected outputs and callback input order. Do not quietly introduce a new ID tiebreaker in this batch or claim a universal old tie-order guarantee the backend never provided.

## Minimality and measurement tradeoffs

Filtering before Python decoding is a better match for the demonstrated problem than decoding the full namespace and filtering later. A streaming Python row prefilter could avoid full Fact construction without SQL JSON semantics, but still transfers every selected column and parses every irrelevant row in Python. It is a fallback if conservative SQL compatibility becomes disproportionately complicated, not a reason to add a second general retrieval layer. Paging/limits would change completeness/overflow behavior and are unnecessary here.

Partial-cache correctness has a cost: same-task slotted candidates now need unrestricted slot queries that the old complete namespace cache could answer locally. Recheck the earlier 64-lessons/16-shared-roots workload, especially on PostgreSQL, alongside the sparse 10,000-other-task profile. Record query count, decoder rows, peak allocation and latency separately; do not hide a regression behind one sparse workload. Cross-task roots or slot peers may legitimately still decode other-task Facts.

The intended claim should stay narrow: fewer Python Fact decodes/materialized rows for unrelated-task candidate acquisition, with equivalent admitted context/status on tested valid/legacy inputs. SQL still scans/evaluates JSON absent a new suitable index, same-task/generic growth still overflows after substantial work, lineage/slot reads remain, and lexical selection is intentionally unchanged. No capacity, model-quality, context-sufficiency or lossless-compaction result follows.

Predesign disposition: proceed with this small seam after resolving the JSON/partial-cache cases above. Final source, parity tests and measured before/after evidence remain to be reviewed; no implementation or passing-test claim is made here.

## Design revision after rejected SQL predicates

Parent reports two concrete failures of the SQL-predicate approach: a focused 135-case run had 132 passes and three PostgreSQL semantic input-order parity failures at tied created_at timestamps, despite equal final public outputs; separately, PostgreSQL JSON/JSONB extraction rejected a legacy-valid JSON string containing an unused `\u0000` value that Python accepts. This note records those as parent-reported rejected-attempt evidence, not as independently executed SQL checks.

The chosen stdlib fallback is the smaller compatible design: keep the original SQL text and ordering literally unchanged, iterate raw cursor rows only on the new task-key path, and apply one shared Python metadata predicate before constructing a complete Fact. Keep the internal load_task_facts/positive-ID cache seam. JSON decoding twice for the small selected subset is a reasonable deliberate cost; do not add a predecoded DTO or another cache merely to remove it. Use `task_key is not None` so the empty-string generic-only task also takes the filtered path, and preserve stored-empty qualifiers handling and exact Python task-value comparison without string coercion.

This narrows the claim further. It reduces full Fact decoding and retained Python object materialization, not the rows scanned, columns transferred or database work. Ordinary PostgreSQL cursors may buffer result data natively; Python cursor iteration is not a server-side streaming or bounded-total-memory guarantee. Label tracemalloc observations as peak traced Python allocations, not total process memory. Same-task/generic overflow and dense slot-query tradeoffs remain unsolved and must stay visible.

Keep both failed SQL attempts in the evidence record and distinguish their source states from the final fallback. No new SQL/API/test was run by this reviewer. Final fallback source and sparse/dense measurements remain pending independent review.

## Fallback source and complete-cache count proof

Reviewed the current fallback implementation read-only. Both loaders keep the original SELECT columns, namespace/optional point/slot clauses and `ORDER BY created_at`; only the new task-key path iterates cursor rows and applies the shared stdlib JSON predicate before Fact construction. Empty stored qualifiers, duplicate keys, escaped labels and unused encoded NUL therefore use the existing Python interpretation. Ordinary full/point/slot loads retain their existing path. Python task applicability, support, activity and full evidence admission remain authoritative. No SQL predicate, schema, index, public selection or date rule was added.

The complete-cache proof is sound for the present SQL backends. The loader produces exactly one Fact per selected raw row, with no JOIN, LIMIT or deduplication; `(agent_id, id)` is the primary key. It fully consumes that subset before counting all rows of the same namespace. `SQLSession.load_task_facts` validates the namespace and uses the same connection as the following COUNT. SQLite opens an explicit read transaction (`BEGIN`, `query_only=ON`); PostgreSQL uses `REPEATABLE READ READ ONLY`. Consequently, equality between subset length and namespace COUNT proves that no row was omitted in that snapshot, including the empty namespace case. Only then may `_remember_namespace` seed complete ID/slot indexes and the whole-namespace cache. A strict subset still seeds only positive exact-ID entries, so other-task conflicting slot peers and cross-task/episodic lineage roots remain unrestricted reads.

This is a small, justified repair for the observed dense-workload regression: reuse the existing complete indexes when completeness is proven, adding one namespace count rather than a new completeness DTO, loader flag or second cache. The proof relies on these transaction and one-row/one-Fact invariants; do not reuse it with a paged/deduplicating loader or a count from another connection. The helper remains synchronous and local to the already existing read snapshot.

The new source tests cover complete-result slot/namespace reuse, the partial-result conflicting peer, and unrestricted lineage roots. The added concurrent-deletion test specifically loads one selected row out of two, deletes the excluded newer peer on a separate connection before COUNT, then requires the old snapshot to retain both slot rows while the next transaction sees the deletion. This addresses the exact cardinality race that could otherwise cause a false completeness proof. I inspected this test but did not execute it or any SQL.

Keep the initial fallback regression in the evidence. Parent reports 139 focused passes (including frozen input-order and Unicode NUL parity) and all 32 before/after public-output pairs equal, but also reports a real dense PostgreSQL regression before the COUNT repair: 99 to 163 backing reads, approximately 109 to 152 ms without noise and 119 to 172 ms with 2,000 noise rows. These are supplied results, not reviewer-run measurements. Final repaired profiles and test completion are pending in this note.

The performance claim remains narrow. Full Fact decoding and peak traced Python allocations for unrelated-task candidate acquisition can fall; the original SQL query still transfers all selected columns and cursor/native buffering can remain O(N). COUNT adds database work. Selected same-task/generic rows are still fully materialized before the first yield and before the public overflow check, so this does not bound same-task overflow, total scans or total process memory. Exact public/input parity on measured cases does not establish model-quality benefit.

Disposition: no blocking correctness or minimality finding in the count-based repair as currently implemented. Accept source shape; wait for the repaired sparse/dense profiles before claiming its performance outcome. No API, credentials, providers, SQL or tests were invoked by this reviewer.

Reviewed source SHA-256 (test file may still be formatted/extended by the parent):

- `src/ai_knot/always_on/reads.py`: `866c701ace372ffc67c830d1aac140126e5cdeb408f920877f7d9ab2c19d0ca3`
- `src/ai_knot/always_on/storage.py`: `9d29e7bc8698494d37ea359e448605a02fffd7de2552c71f600c2ff810183137`
- `src/ai_knot/always_on/context.py`: `29c9d66c6d822aa16298f176b89f8292698b35122ca1a3ac9eb987770ef976ef`
- `src/ai_knot/storage/sqlite_storage.py`: `b7ba58eea902603819b6bcdc2e01e1218ca287483fa742d552b74278fd9efb8e`
- `src/ai_knot/storage/postgres_storage.py`: `7441cc855e28946af08a9ab0f35c767c8db4c2b1b7891df3966cdd5e9828b709`
- `tests/always_on/test_task_source_scan.py`: `ecd98f25063ed874067ea7691a27c896b91d4c567be88ac713d78eb4cb9ff187`

## Final fallback evidence and closure

Independently inspected the final source/tests, profile driver, direct before/final JSON reports, intermediate regression reports, guide and owner-run logs. I performed read-only file/hash/comparison calculations; I did not run SQL, tests, providers, API calls or credentials. This closes the source/method review, not an independent rerun of the experiment.

Source provenance checks: both final reports record 80 imported package-source hashes, all matching the current checkout. Both baseline reports' 80 hashes match the corresponding files at `53d72fb`. All before/final/intermediate reports identify the same driver SHA-256, `e0ef9109ce1af8fd680c08401b7cbb171e999298fc3bd8873e3b22e5cd58f2c3`, which also matches the inspected driver file. The reused fixture and backend-isolation helpers (`tests/eval/context_read_profile.py` and `tests/eval/task_context/runner.py`) are unchanged from that base. The initially empty invocation-relative `source_sha256` field in baseline reports is therefore not being treated as a source freeze; the actual imported-package hashes supply the checked provenance.

Comparison reproduced by parsing the supplied JSON: 16 keyed operations per backend, 32 total before/final public result pairs, all exactly equal. Every stored median equals the median of its five retained timing samples. The driver takes one instrumented counts/tracemalloc pass, then five uninstrumented timing passes, asserting the same result each time. This is a small local repeated/warm profile, not a cold-start, parallel-load or statistically controlled deployment benchmark. Backing reads here mean Fact-loader calls plus SQLSession cursor reads; the counter is not a tally of every transaction/setup command or server plan operation.

| Snapshot workload | Backend | Median ms before → final | Fact decodes | Backing reads | Raw Fact rows |
|---|---|---:|---:|---:|---:|
| 2 relevant/generic + 10,000 other-task | SQLite | 146.53 → 55.84 | 10,002 → 2 | 3 → 4 | 10,002 → 10,002 |
| Same | PostgreSQL | 185.60 → 65.95 | 10,002 → 2 | 3 → 4 | 10,002 → 10,002 |
| 64 lessons / 16 roots, no noise | SQLite | 32.37 → 30.96 | 96 → 80 | 99 → 101 | 96 → 96 |
| Same | PostgreSQL | 109.19 → 90.33 | 96 → 80 | 99 → 101 | 96 → 96 |
| 64 lessons / 16 roots + 2,000 noise | SQLite | 70.78 → 43.35 | 2,096 → 80 | 99 → 101 | 2,096 → 2,096 |
| Same | PostgreSQL | 119.30 → 98.42 | 2,096 → 80 | 99 → 101 | 2,096 → 2,096 |

The sparse 10,000-record snapshot peaks are 36,493,751 → 17,384 traced Python bytes on SQLite and 36,489,752 → 20,126 on PostgreSQL. These are one instrumented allocation peak per row, not RSS/total memory or a bounded SQL buffer result. Same-task 10,000-record cases still decode all 10,002 Facts and retain the same explicit snapshot-too-large/input-limit statuses. Some small/no-noise medians increase, and count queries always add one read per scanned namespace. The guide appropriately presents selected examples while retaining all rows and these limits; it does not assert uniform speedup.

The intermediate dense regression is independently visible in the retained after-initial reports: 163 reads, PostgreSQL snapshot 152.00 ms without noise and 172.35 ms with noise. Final count-based reuse reduces that to 101 reads and the medians shown above. Keep this source state distinct from the final implementation rather than replacing its report. Both failed SQL-JSON approaches remain separately described.

Behavioral test review: the new nonmonotonic-ID test does actually produce tied timestamps because the existing `clock` fixture patches Fact defaults; it compares exact callback input order and public acquisition against the frozen historical selector in both insertion directions. The existing historical selector is only that archived module, with current shared admission dependencies; it is not a whole historical-runtime comparison. The separate source profiles do use the baseline package. The complete/partial-cache tests and separate-connection deletion-between-load-and-count regression exercise the new completeness boundary. Existing source/ACL and post-await revalidation paths remain intact.

Owner-run log evidence inspected: focused cache suite 141 passed; later nonmonotonic ties 4 passed; later cardinality race 2 passed. The full PostgreSQL gate log records **2648 passed, 5 skipped, 1 warning**, 89.99% coverage, 190.92 s. The race was added after that run's collection and passed separately; do not call the full run 2650 passed. Python 3.11.15 and 3.13.14 logs each now record 147 focused passes (38.38 s and 29.87 s). These are read logs, not reviewer-executed checks.

Final disposition: no blocking correctness or minimality finding. The shared raw-row predicate, existing internal session seam, positive-ID fallback and same-snapshot count proof are sufficient for the demonstrated decoder cost without a new cache/framework/schema. `docs/task-source-performance.md` accurately separates fewer Fact decodes/traced allocations from unchanged raw SQL rows, remaining same-task overflow, native buffers and lack of capacity/model-quality evidence. Its artifact links still require the parent's pending archive assembly/provenance verification; this review compared the direct temporary JSON reports before that archive existed.

Final checked raw-report SHA-256:

- `task-source-profile-before-sqlite.json`: `c77a83886102c079c9f68900ce37ed8a158b59345f46a5987ef4ace8aa1db6c5`
- `task-source-profile-before-postgres.json`: `68a251fb5974dd86aae682eebd29ded08180d5498bf61cc25014c5a5fbe3f631`
- `task-source-profile-after-initial-sqlite.json`: `21958782672a0d01ba0b43c829be01863b456f819a9c92f13281f2661e725a3a`
- `task-source-profile-after-initial-postgres.json`: `78ff2100ac04a201e35c6748eac904bdc41d2186269bc84c12ec7f0e71a65b8a`
- `task-source-profile-final-sqlite.json`: `9ae2c83cadb1356e27e43ac30677b5015b3d0b5eb32c54553901241f2e0815dd`
- `task-source-profile-final-postgres.json`: `fb2aad250ab24a4a48405d3b6e01cdacde54a0f9c0941161a344dd427e8f4d68`

Final checked changed source/test SHA-256:

- `src/ai_knot/always_on/context.py`: `29c9d66c6d822aa16298f176b89f8292698b35122ca1a3ac9eb987770ef976ef`
- `src/ai_knot/always_on/reads.py`: `866c701ace372ffc67c830d1aac140126e5cdeb408f920877f7d9ab2c19d0ca3`
- `src/ai_knot/always_on/storage.py`: `9d29e7bc8698494d37ea359e448605a02fffd7de2552c71f600c2ff810183137`
- `src/ai_knot/storage/sqlite_storage.py`: `b7ba58eea902603819b6bcdc2e01e1218ca287483fa742d552b74278fd9efb8e`
- `src/ai_knot/storage/postgres_storage.py`: `7441cc855e28946af08a9ab0f35c767c8db4c2b1b7891df3966cdd5e9828b709`
- `tests/always_on/test_task_source_scan.py`: `ecd98f25063ed874067ea7691a27c896b91d4c567be88ac713d78eb4cb9ff187`
- `tests/always_on/test_task_snapshot_equivalence.py`: `5c51fc205bee950fcce53141ba890e2569a37c2a37a7fb308e1aee01061d4459`
- `tests/always_on/test_context_read_reuse.py`: `c440250e413fa09845fd707d5850d3260578e2273d08610e0f739ad2dc231db1`
