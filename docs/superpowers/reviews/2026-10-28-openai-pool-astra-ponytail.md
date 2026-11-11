What this change does: SharedMemoryPool now accepts the same explicit embedding URL, model and API key already supported by KnowledgeBase. Document and query embeddings both use those settings through the existing embedder; default Ollama behavior stays unchanged, and an empty URL disables embedding HTTP.

Looks good. Ship.

No concrete correctness, safety, regression or unnecessary-complexity finding was found in the scoped diff against HEAD `2f47c0436321c3fcbcfd04cf1d610b2e1eb9f158`.

Checked:
- Read the constructor, both async embedding paths, all repository callers found by search, the existing embedder, dense/hybrid retrieval, synchronous recall and its query-vector clearing, KnowledgeBase configuration, shared-pool tests, test fixtures, and the new usage section.
- New parameters are keyword-only; every old caller remains valid. Defaults exactly match the former embedder defaults. API key handling and response parsing remain in the existing helper; no dependency or extra configuration layer was added.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /tmp/ai-knot-pr-ci/py312/bin/python -m pytest tests/test_pool_openai_embeddings.py tests/test_embedder.py tests/test_shared_pool.py tests/test_air_gapped.py --no-cov -p no:cacheprovider -q`: 87 passed in 0.79s.
- A separate offline `httpx.MockTransport` probe passed: URL is exactly `https://api.openai.com/v1/embeddings` even with a trailing base-URL slash; Authorization is `Bearer review-sentinel`; model is `text-embedding-3-small`; first recall sends documents then query; later recall reuses document vectors; a query HTTP 503 clears the dense query vector and still returns the lexical match; synchronous recall and an empty-URL pool send no HTTP request.
- `git diff --check` passed. Reviewed file hashes were unchanged after tests. No release metadata is changed in the tracked diff.

Scope and limits:
- Assumed the current bounded local quality-evaluation workload, with existing multi-agent SQLite/PostgreSQL behavior preserved. This change is provider selection, not a capacity or concurrency redesign.
- The old benchmark multi-agent backend still constructs the pool with defaults. An OpenAI quality runner must explicitly pass the new settings, as the new documentation shows; this patch alone does not change historical benchmark defaults or select the host chat model.
- The existing embedder catches ConnectError, TimeoutException, HTTPStatusError and KeyError. Other transport/JSON-shape failures are outside that preexisting guarantee; `src/ai_knot/embedder.py` is unchanged by this diff. No broader all-failures fallback claim was validated.
- Full PostgreSQL verification is owned by the parent task. No API keys were accessed, no external model provider was contacted, and no live OpenAI quality or latency result is claimed.

Reviewed SHA-256:

| File | SHA-256 |
| --- | --- |
| src/ai_knot/pool.py | f23bdc3a1bf122c65a98794f64fcb7151a97f37b6d34c9895a5968456a4ce33f |
| src/ai_knot/_pool_recall.py | eb3c5f9faa3034b85731247750b769e3cf8f702d9e99cdfb270cd056379c8160 |
| tests/test_pool_openai_embeddings.py | d3341ddcf752d9618e6cac98821b28b2ff7ecc4649f26da8c08bb2fd3ebc9ae2 |
| docs/usage.md | 294e870ddccb5d69ecf281fb041f39752fbade479ea50ee8f10e29945727561f |

Verdict: Ship.
Not checked: live OpenAI credentials/provider behavior, answer quality, or the parent-owned full PostgreSQL run.


Follow-up review: shared-maintenance quota-test wall allowance

What this change does: The shared daily-call quota test now passes `max_wall_seconds=10`, matching neighboring shared-maintenance tests. Its model-call limits and assertions are unchanged.

Looks good. Ship. The adjustment preserves the intended assertion and does not mask a demonstrated quota-correctness failure.

Evidence:
- Read the reported failure at `/tmp/ai-knot-product-quality/ci-py312-pg.log:4016-4061`. The raised-limit run had reserved one call and stopped with `reasons=("wall_budget",)`, rather than `workspace_model_budget`. That is the explicit elapsed-time gate in `run_maintenance`, which is checked again after SQL reservation and before constructing the verifier coroutine.
- The test still requires the first verification, then `workspace_model_budget` and exactly one provider call after reapplying the limit of one, then one verification and exactly two total provider calls after raising the limit to two. The wider time allowance cannot satisfy those assertions if quota enforcement or verification fails.
- Production `MaintenanceBudget.max_wall_seconds` remains 1.0. The durable run expiry remains 60 seconds for both values because it uses `max(60, budget.max_wall_seconds)`; this test adjustment does not extend that protection.
- The separate deterministic wall regression still sets a one-second wall allowance, advances the injected monotonic clock to two seconds after reservation, asserts `budget_exhausted`/`wall_budget`, checks the durable reservation, and asserts the verifier was not called.
- Existing neighboring shared-maintenance tests already use ten seconds. The PostgreSQL CI pytest timeout remains twenty seconds. `git diff --check` passes.

Reviewed SHA-256 for `tests/always_on/test_shared_maintenance.py`: `b1e18f861e9a4054caaa6a3fc3454744b24c6bfc949afc0dcf202ad0c1e35543`.

Verdict: Ship.
Not checked: fresh focused and full CI execution are parent-owned. The failure log establishes the elapsed-time gate that fired, but does not independently measure which local PostgreSQL operation consumed that time.
