# OpenAI decision-quality evaluation

The broader product goal remains active. User steering on 2026-10-28 selects GPT-4.1 mini and OpenAI embeddings instead of local Ollama. Use `gpt-4.1-mini` (record the returned snapshot) and `text-embedding-3-small`; no model substitution. Official model pages were fetched through OpenAI Docs. The user says a key existed previously; its current configuration location is being resolved. Never log/persist the key.

## First implementation slice

- [x] RED: shared-pool async retrieval must pass explicit embedding endpoint/model/key to both document and query calls, retain no-network disabled mode, and preserve existing defaults.
- [x] GREEN: add the same opt-in embedding settings already used by KnowledgeBase to SharedMemoryPool and use them in the common async paths. No new dependency or automatic network worker.
- [x] Verify affected pool/embedder tests and static checks; document the OpenAI setup without touching deferred releases. [Validation](../validation/2026-10-28-openai-pool.json): 1996 tests passed with PostgreSQL required, 89.68% coverage; Astra/Ponytail Ship. This accepts provider configuration only; live model quality and the broader product goal remain open.

## Product-evaluation continuation

- Reproduced S9: correct_at3=1, stale-rival suppression/conflict resolution=.333; S19: evidence recall=1, keyword precision=.615. Preserve original data/labels and inspect actual decisions before tuning retrieval. Existing task traces store safe_next_action but do not score it.
- Build a metered decision/task runner using the requested OpenAI models, bounded output/concurrency/calls, identical available observations and total inference budgets, strong existing-memory/raw-history controls, and objective simulator/receipt outcomes. Include sleeps, handoffs, corrections, unknown effects and stale-policy distractors. Count embedding formation/retrieval and all model calls/tokens; cache only with model/input identity and record failures explicitly.
- Use pilot diagnostics to identify real gaps; do not mistake a low keyword score for a wrong decision, fabricate model results while key access is missing, or close the product goal when another engineering batch passes.

All ordinary correctness tests remain local; live OpenAI evaluation is explicitly authorized by the user. Preserve the original dirty release checkout and 0.12.0 source version metadata. Independent review and SQLite/PostgreSQL acceptance remain required for product code changes.

## Runtime preparation and limits

- [Usage configuration](../../usage.md#multi-agent) passes the OpenAI endpoint, model and key to both private KB and shared pool. The host's chat call separately selects `gpt-4.1-mini`. `select_context()` and `review_context()` remain model-free; configuring an embedding provider does not make those selectors use dense vectors.
- A temporary S9 diagnostic driver at `/tmp/ai-knot-product-quality/run_openai_probe.py` compares full raw history, BM25 and explicitly configured pool async retrieval. Its offline HTTP wiring check made nine mocked reader calls and four mocked embedding calls, verified model/authentication/usage recording, and rejected embedding failure instead of reporting a successful hybrid run. These are mocked results, not OpenAI quality/latency measurements. The three-question pilot cannot establish end-task benefit.
- The historical benchmark runner still defaults to Ollama. Do not claim its defaults changed or run it with a cloud URL alone. The live driver uses explicit `SharedMemoryPool` settings; a broader metered task runner remains the next evaluation step.
- Named-model references: [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small).
