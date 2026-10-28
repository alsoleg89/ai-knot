# Less decoding when retained memory contains other tasks

Task snapshots and semantic acquisition now check task/type metadata before constructing full `Fact` objects. This is an internal optimization of the existing SQLite/PostgreSQL path: public APIs, default budgets, source admission and ordering are unchanged. Lexical selection and ordinary KB reads keep their existing paths.

An always-on team can retain many tasks while one returning operator needs only a few observations. Previously `task_sources` materialized every Fact in each namespace before checking task labels or excluding raw episodes. A small output did not imply a small decoding cost.

Both backends retain the exact existing SQL query and its `ORDER BY created_at`. The task path iterates raw cursor rows and uses the same stdlib JSON rules before full decoding. Missing/empty task keys remain generic; Unicode, escaped values and duplicate-key last-value behavior are retained. Python still checks supported/active status, source ACL, exact lineage, current slot and freshness. No SQL JSON dialect, time comparison, index, schema, dependency or model call was added.

Fields of rows already excluded by task/type metadata are no longer fully decoded or validated by this candidate read. Direct storage reads and required root/slot lookups still use their full decoders. This is not a whole-database corruption validator; malformed metadata that must be parsed still fails.

The transaction-local cache normally records only known positive IDs from a filtered result. It does not infer that an omitted root or conflicting slot peer is absent. If the filtered row count equals the namespace's total row count in the **same SQL snapshot**, nothing was excluded, and the existing complete namespace/slot cache is safe to retain. This adds one native COUNT query per namespace, preserving cheap repeated slot checks on dense workloads.

## Local before/after profile

Baseline `53d72fb`; same model-free driver/data, fixed clock, isolated databases, five uninstrumented timing samples per row. Snapshot and semantic fitting-snapshot/overflow paths were measured with 0/1,000/10,000 retained records, plus the existing 64-lessons/16-shared-roots workload with 0/2,000 unrelated records. All 32 baseline/final public result pairs match exactly. Separate historical-selector tests also preserve callback input order, cache behavior and budgets, including equal timestamps with nonmonotonic IDs.

Snapshot examples, medians in milliseconds:

| Workload | SQL | Before → after ms | Full Facts decoded | Backing reads |
|---|---|---:|---:|---:|
| 2 current/generic facts +10,000 other-task records | SQLite | 146.53 →55.84 | 10,002 →2 | 3 →4 |
| Same | PostgreSQL | 185.60 →65.95 | 10,002 →2 | 3 →4 |
| 64 lessons/16 roots, no extra records | SQLite | 32.37 →30.96 | 96 →80 | 99 →101 |
| Same | PostgreSQL | 109.19 →90.33 | 96 →80 | 99 →101 |
| 64 lessons/16 roots +2,000 other-task records | SQLite | 70.78 →43.35 | 2,096 →80 | 99 →101 |
| Same | PostgreSQL | 119.30 →98.42 | 2,096 →80 | 99 →101 |

In the sparse 10,000-record snapshot, peak **traced Python allocations** fell from about 36.5MB to 17–21KB. This is not a total process/RSS measurement: PostgreSQL's regular cursor/native result buffers and SQLite's native query/sort memory are outside that number. Raw fact rows returned by SQL remain 10,002 in both sparse runs. There is no server-side streaming/paging claim or bound on total SQL/network work. Local medians are not deployment latency guarantees.

Same-task/generic overflow still fully decodes applicable candidates before the existing limit outcome; timings can be slightly worse because selected qualifiers are parsed twice and the count check adds work. The profile retains those cases. This change reduces an identified cost of unrelated tasks, not corpus growth, semantic sufficiency, lossless compaction, answer accuracy or fleet capacity.

## Verification and rejected approaches

The first SQLJSON prefilter changed three PostgreSQL historical embedding-input orders under equal timestamps. SQL JSON extraction also rejected an escaped U+0000 value accepted by the old Python parser. The final implementation avoids SQL JSON entirely and keeps those parity checks rather than changing their expectations.

The first partial-cache-only implementation added 64 slot reads on the dense fixture and slowed PostgreSQL. Its measured regression is retained separately. The same-snapshot count proof restores complete cache reuse only when justified; roots and slot peers outside a partial task result still use unrestricted source checks. Concurrent deletion between filtered read and count is tested with separate connections: the first transaction sees its old full snapshot, and the next read sees deletion.

Reproduction source, exact baseline/final source hashes, raw results, the intermediate regression and local logs are in the [compressed evidence](performance/task-source-scan-raw.json.gz); the [summary](performance/task-source-scan.json), [validation](superpowers/validation/2026-10-28-task-source-scan.json) and [independent review](superpowers/reviews/2026-10-28-task-source-scan.md) separate source/method review from owner-run tests. No provider or embedding requests were made for this profile.

See [growing task memory](research/2026-10-28-growing-task-memory.md) for the separate selection/compaction question. Planning from a selected subset must preserve detection of new relevant conditions; it cannot silently discard the full-corpus check simply to avoid overflow.
