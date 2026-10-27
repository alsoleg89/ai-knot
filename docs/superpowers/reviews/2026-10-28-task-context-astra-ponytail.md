# Independent task-context review

What this change does: It selects a bounded set of current facts for explicit task requirements. Task labels restrict applicability, role labels affect ranking, and the existing SQL evidence guards still determine whether facts can be used.

Reviewed the uncommitted milestone against `fcbcad5`, including the new selector, shared assembler and callers, typed derivation, SQL snapshots, lineage/freshness validation, HTTP/MCP decoding, TypeScript routing, and the design/research documents. No source files were edited by this reviewer.

## Must fix

1. **Unrelated words or hidden metadata can report complete evidence** (`src/ai_knot/always_on/context.py`, original lines 89–120).
   - **What this is:** Any positive BM25 score becomes requirement coverage, although BM25 includes stopwords and metadata that the returned `ContextFact` does not expose.
   - **Problem:** With only `Fact('How to rotate passwords')`, the requirement `How to restore certificates` returns that fact and `complete=True`. The content-bearing token sets are disjoint. Likewise, `Fact('Lunch starts at noon', tags=['certificate', 'expiry'])` satisfies `certificate expiry`; the same happens through `canonical_surface` or `source_snippets` alone.
   - **Fix:** Require original content-bearing token overlap with the emitted `Fact.content` before admitting or certifying a candidate. Filter raw BM25 results through that gate before deciding whether ubiquitous-term fallback is necessary; metadata may still affect ranking.
   - **If we skip it:** Missing task evidence is presented as a complete working set. This is a lexical error, independent of the deliberately unsupported claim of semantic entailment.

2. **Already covered requirements can consume the last fact slot** (`src/ai_knot/multi_agent/assembly.py`, original lines 99–130).
   - **What this is:** The coverage phase initially marks only the facet currently being processed. Other facets covered by the same selected fact are counted only after selection finishes.
   - **Problem:** Store A=`'alpha beta ' + 'details '*100`, B=`'beta'`, C=`'gamma'`, D=`'gamma alternate'`, E=`'gamma more'`. `select_context(('alpha','beta','gamma'), max_facts=2)` selects A and B and misses gamma, although A and C fit and cover all three. The same issue wastes a character budget. This assembler defect existed before the milestone but directly affects its new coverage-first contract.
   - **Fix:** Union positive facet evidence by fact identity, including separate candidate objects produced by the legacy caller, and update all those facets immediately when a fact is selected.
   - **If we skip it:** A tight context budget can be spent on redundant evidence while an available prerequisite is omitted.

## Should fix

3. **A full character budget still causes one SQL admission per remaining match** (`src/ai_knot/multi_agent/assembly.py`, original lines 81–88).
   - **What this is:** Admission runs before the size check for every candidate in the coverage and backfill loops.
   - **Problem:** With 1,000 facts from 1,000 publishers, all containing `Rollback checkpoint`, `max_chars=19` and `max_facts=8`, the selector returns one fact but makes 1,001 fact-load calls and 1,000 freshness queries. This took 0.751 seconds on the dedicated local PostgreSQL instance and 0.046 seconds on SQLite; 5,000 SQLite facts took 0.241 seconds. Each run used an isolated database/schema.
   - **Fix:** Once an admissible oversized candidate has established `budget_limited=True`, reject subsequent oversized candidates before admission. This preserves the flag's meaning and still checks every candidate that can fit, without adding a cache or another retrieval layer.
   - **If we skip it:** A small output budget does not bound database work, and a common full-budget request makes thousands of avoidable queries at the expected publisher scale.

4. **The Python facade accepts a string as separate character requirements** (`src/ai_knot/always_on/context.py`, original lines 31–39).
   - **What this is:** Validation checks length and iterates the input without first checking its collection type.
   - **Problem:** `select_context('cpu')` accepts three requirements, `('c','p','u')`. A matching fact can return `complete=True` for those characters. HTTP and MCP correctly reject the same malformed input.
   - **Fix:** Require a tuple/list of requirement strings before checking individual entries.
   - **If we skip it:** An easy Python calling mistake silently changes the request instead of failing validation.

## Verification and deliberate limits

The findings above were reproduced through the public SQLite selector; the multi-facet issue was also reproduced directly in the assembler. The query-count finding was reproduced on both SQL backends, with the PostgreSQL review schema removed afterward. The implementation owner accepted all four findings, added focused regressions, and applied the fixes. The findings are retained here as the original RED review record; none remains open in the reviewed code.

Independent post-fix verification:

- **48 passed** for task-context tests on both SQL backends plus shared assembly tests. Log: `/tmp/ai-knot-task-context/astra-core-fixed.log`.
- **19 passed** for real HTTP/MCP boundaries on both SQL backends, including subprocess MCP. Log: `/tmp/ai-knot-task-context/astra-surfaces.log`.
- The exact PostgreSQL multi-facet repro now selects A and C and returns `complete=True`.
- The exact 1,000-fact PostgreSQL budget repro now makes **3 fact-load calls and 2 freshness queries**, returns one fact with `budget_limited=True`, and took **0.042 seconds** in the local check (previously 0.751 seconds). This is one diagnostic run, not a production benchmark.
- An initial mixed-directory pytest argument ordering produced missing-fixture setup errors for later `always_on` files; grouping those files in their own run resolved collection, and all 19 passed. That initial log is preserved at `/tmp/ai-knot-task-context/astra-targeted.log`; it is not claimed as a passing gate.

No new source/ACL bypass was found. Selection uses the existing SQLite query-only transaction or PostgreSQL repeatable-read, read-only transaction. Namespace collision handling retains original evidence references, and filtering occurs before indexing for activity, support, raw episodes, task applicability, shared visibility and exact lineage. Existing admission covers slot conflicts and freshness. Task/role metadata does not grant authorization.

The full snapshot/index is still proportional to retained history. This review establishes the concrete avoidable SQL fanout above, not a production throughput guarantee for indefinitely growing histories. Lexical overlap remains weaker than semantic evidence, and a greedy assembler does not guarantee the mathematically optimal set under every budget; those are deliberate limits, not additional findings.

Verdict: Ship the reviewed code. Findings 1–4 are fixed and independently verified; the milestone's full-suite and frozen-trace gates remain the implementation owner's separate release criteria.

Not checked: real-model answer quality, production network latency, or the in-progress frozen trace harness; those require their separate validation reports.
