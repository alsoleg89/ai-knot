# Independent Astra / Ponytail review: transaction-local context read reuse

Reviewed 2026-10-28 against base `48b99d549bf8610dba8ecb047789033b1dc3a88c` in `/Users/alsoleg/.codex/worktrees/shared-memory-return/ai-knot`. Read-only source/artifact review; no provider/API/Keychain access, no product edits, no additional agents. Only this note was written.

**Disposition: no correctness blocker and no worthwhile simplification found.** The 63-line private adapter is an appropriate small change for the measured duplicate reads. Keep it local to the existing read paths; no extra cache service, TTL, bulk loader, persistent index or action integration is justified by this evidence.

## Correctness review

- Fact lookup keys include namespace, ID and slot. After an already requested whole-namespace load, the indexes preserve distinct private/shared IDs, slot members' scan order, the intersection of ID and slot predicates, and the distinction between `None` and empty strings. Missing results are cached too. Point-only review never asks the adapter to preload a namespace. The real-SQL filter matrix exercises these cases both before and after a namespace scan.
- Reuse is confined to the existing read-only transaction: task snapshot / semantic initial acquisition through `task_sources`, lexical selection, `review_context`, and semantic final revalidation. No cache is installed on memory/storage instances or action/write paths. SQLite's existing explicit read transaction and PostgreSQL's existing repeatable-read read-only transaction are unchanged; negative results and reused grants/rows therefore do not introduce an additional staleness window.
- The source validator still evaluates its original revision, root/intermediate lineage, slot ambiguity, ACL and freshness conditions. The adapter caches raw data, not a previous admission decision. Read time is captured as before. A new operation gets a new adapter, so source changes, revocations and expiry are checked anew. The new concurrent-write regression explicitly preserves the current transaction's old snapshot and verifies rejection in the next read.
- Semantic provider I/O occurs after the initial SQL transaction has ended. Final validation opens a new transaction and creates a new adapter. Previously captured source values are not used to admit post-provider results; the new transaction reloads and validates refs and current applicability. Existing semantic tests cover mutations/revocation during the provider callback. Provider request construction and accounting were not changed by this patch.
- Returned lists and SQL row dictionaries are copied. Current internal consumers read, rather than mutate, loaded `Fact` objects; selected public results are text/ref projections. The adapter explicitly rejects write methods and retains the underlying SQL read-only enforcement. Backing exceptions occur before cache insertion. Early overflow/return exits the original transaction scope and leaves no persistent cache; the change does not add eager whole-namespace acquisition to point review.

One documentation precision issue was sent to the parent: `context-read-reuse.json` originally said Fact objects remain in “synchronous admission.” `SemanticContext._snapshot` already returns internal source Fact references which remain in `_acquire` across its await; the *lookup cache* does not survive/re-enter the post-provider transaction. Suggested wording: “Fact instances remain internal and are not mutated; read caches are discarded before provider I/O, and revalidation uses a new snapshot.” This is a wording correction, not a functional defect or request to copy every Fact.

## Independent accounting checks

Verified all 81 source/profile hashes in `context-read-profile-source.json` against the reviewed checkout: zero mismatches. Verified the compressed raw archive's recorded SHA256; all eight embedded reports match their original UTF-8 bytes and their individual SHA256 values; the embedded source manifest matches the original.

Recomputed all 24 before/after pairs (two backends × two root settings × two retained-noise settings × three operations): public result hashes match, each side contains five samples, medians match their samples, total reads equal the two counters, and summary counts/medians/speedup ratios match the original reports.

The shared-root workload reports snapshot `3330 → 99`, lexical `3331 → 83`, and point review `3392 → 225` backing reads on both SQL backends. These are fact-loader calls plus SQLSession read calls, excluding transaction setup/connection negotiation; the artifact states this correctly. Point review's separate `<300` threshold preserves targeted reads. It should not be tightened by replacing them with an unbounded namespace scan.

Observed shared-root median speedups span 1.34–2.04× for SQLite and 11.28–18.19× for local PostgreSQL. Five of six SQLite shared-root measurements miss the original 2× latency goal. The report preserves that failure rather than redefining success. The root-free control also shows the remaining scan cost: with 2,000 unrelated rows, PostgreSQL snapshot improves only 1.05×. No observed summary median regresses, but five serial local samples do not establish tail latency, concurrent-agent capacity, production bounds or model/task quality. No model or embedding calls were involved in this profile. Namespace scanning, hashing and per-ref admission CPU remain.

## Validation evidence and provenance

Inspected the completed parent CI log `/tmp/ai-knot-product-quality/context-read-full-pg.log`: `2464 passed, 5 skipped, 1 warning`, coverage `89.97%`, elapsed `91.48s`. This is inspected parent-run evidence, not an independently repeated full suite. A single SQLite-only narrow pytest invocation was attempted by this reviewer, but its terminal output was not retained across tool-output truncation, so no independent pass count is claimed from that attempt. No extra test rerun was performed.

Reviewed hashes:

- `reads.py`: `2a892316ede07bbe5c4b2ee20f997f50328eea07076ba71f17d95348f85f86d0`
- `context.py`: `07ee7e6fe23385f9b28073a87666c8c88fb6154e70173483e9abf644b9154f08`
- `memory.py`: `942598d88151ec61e30975b8e7bdbfd8389180f318b736c8ea6440588ef20954`
- `semantic.py`: `34dc1f00b7539c780bde82666eaa90d36e981558790a2fc759a75a2ebef435d8`
- `test_context_read_reuse.py`: `2cc511d998f0a03e194576dae0cee4e52417b1fd055cb5f1dab8c063e769448b`
- `context_read_profile.py`: `6efa67e6c0d86b24a56e61232c8c28f34893f8a15f6681e29a103b30f86abe7c`

Ponytail minimality verdict: Lean already. Ship.

Closure addendum: independently read the parent's final wording in `docs/context-read-performance.md:9`, `docs/performance/context-read-reuse.json:331`, and plan architecture. It now accurately distinguishes captured internal source objects from reusable SQL lookups/admission decisions and states the fresh post-provider snapshot. The sole documentation precision comment is resolved; no open review finding remains.
