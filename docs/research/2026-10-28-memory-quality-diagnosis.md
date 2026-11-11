# Memory quality: source selection, decision errors and misleading proxies

Research/repair date: 2026-10-28, starting from `987f885`. The five retained advisory misses are not five demonstrated losses of necessary evidence. This audit traces each metric back to its fixture and scorer, repairs false-positive evaluation, and retains fresh ranked-source diagnostics on SQLite and PostgreSQL. It changes no production retrieval algorithm, case text, numeric gate target or previous quality result.

## What the five advisory misses mean

| Retained observation | Supported interpretation | Unsupported inference |
| --- | --- | --- |
| S9 conflict resolution and wrong suppression each 1/3 | Correct claims are present; older contradictory unstructured claims remain alongside them | Two-thirds of required current evidence is missing, or structured CAS is broken |
| S19 keyword precision 8/13 | Surplus off-keyword results coexist with all three required keyword probes | Required source recall is below one, or precision has a universal 0.60 ceiling |
| S26 exact-source recall 5/15 at 1,000 publishers | The markerless query does not identify one particular peer among 50 same-domain sources | Two-thirds of the requested technical knowledge is absent, or 1/3 is a universal information-theoretic maximum |
| S26 exact-target distractor rate 45/50 | All rows other than the arbitrary exact source IDs are counted, including useful same-domain implementations | 90% of returned content is useless, or every result list has a fixed 0.70 floor |

The additional old PostgreSQL latency miss is environment-sensitive and separate from these source/decision questions. New traces are instrumented and cannot establish a latency regression or improvement.

## S9: current claims beside unstructured rivals

The [fixture](../../tests/eval/benchmark/fixtures.py) publishes 12 unstructured claims. A later `add_structured` creates the current alert slot for the first time: the fixture never inserts `competing_slot_v1`. The old unstructured 10-minute alert stays active, the new structured record has no superseded source, and another publisher already contains the 4-minute keyword. The legacy `supersession_propagation` probe checks keyword presence, not an observed CAS replacement. Its description is corrected; its name/value and all source inputs remain intact. Actual CAS checks live separately in S10 and the SQL protocol tests.

Fresh diagnostics reproduce correct-at-three = 1 and suppression/conflict = 1/3 on both backends. The older REST-support claim ranks first while deprecation ranks third; commander 24/7 coverage ranks first and the old weekdays claim second. Both rival pairs are unslotted with different claim keys and no explicit supersession link. Current evidence is available, but suppression of textual disagreement remains a semantic/source-model problem. Deleting an older-looking unrelated statement or trusting the last publisher is not a justified generic repair.

The [scenario source](../../tests/eval/benchmark/scenarios/s9_ma_pool_publish.py) now states its actual boundary. The raw archive includes the separate detailed SQLite publication-state diagnostic as supporting evidence; it is not a reconstruction of an old retained run.

## S19: surplus is real, shrinking top-k is unsafe

The [incident scorer](../../tests/eval/benchmark/scenarios/s19_incident_reconstruction.py) checks three expected keywords in three corresponding queries. Its precision is the fraction of returned rows containing any of those keywords, not query-specific causal relevance or a source-ID judgment. Fresh SQL diagnostics reproduce all three required probes and 8 keyword-positive rows among 13 results.

The off-keyword results are certificate rotation, an older similar incident and a release on another service. In the second query, the desired deployment source is fifth, behind two such surplus rows. A blanket top-k reduction to three would remove needed evidence. Some historical context may also be useful in a real incident; the fixture's word “noise” alone does not prove that it should be suppressed universally.

`top_k` is a maximum, and the scorer can return fewer rows. Therefore 3/5 is not a universal ceiling: the retained observation 8/13 already exceeds 0.60, and three relevant returned rows can yield precision one. Explanatory gate notes are corrected without lowering its 0.70 advisory target.

## S26: fix a binding false positive before trusting coverage

The [generator](../../tests/eval/benchmark/scenarios/s26_sparse_assembly.py) repeats 20 technical domains. At 1,000 publishers, each target has 49 same-domain peers with equivalent implementation text apart from its unique module marker. Swapping `[0,333,666]` for `[20,353,686]` produces the same conceptual query and different exact source IDs. The marker-named variant makes the exact source request identifiable.

The earlier `equivalence_recall` implementation credited any fact whose publisher had a requested domain. A generic operations note or a near-miss overview could therefore produce a binding PASS without an implementation shard. Its text-only branch, conversely, never credited domain coverage. Text-only duplicate rows could multiply one source into several target hits; prefix matches and a publisher carrying another target's marker could also produce false credit.

Source scoring v2 uses whole-word known fixture markers, requires matching publisher identity when Fact provenance is supplied, and deduplicates source identities. Generic/near-miss-only returns fail the binding coverage test. Genuine same-domain peer shards cover the conceptual facet without becoming exact targets. Repeated returned rows do not increase recall or first-hit rank. The existing per-row exact-target distractor rate is still not a duplicate-waste metric.

A separate regression covers duration checkpoint N=60: targets `[0,20,40]` share a domain. Coverage is defined per facet; one real implementation covers all three repeated conceptual requirements while exact-source recall stays 1/3. The final one-line overlap correction was made after both full diagnostics terminated. Replaying their default N=10/100/1000 rows gives identical measured scores. The archive preserves measured and final scorer source separately.

Markers identify generated fixture sources; they do not prove entailment, authenticity or source independence. A quoted/copied marker remains a limit of this proxy. For the fresh diagnostics, an additional independent audit compares the full body against the exact generated implementation plus its publisher, rather than trusting the marker helper alone.

### Fresh 1,000-publisher evidence

Both SQL diagnostics give the same source-count table over five conceptual queries:

| Observation | Count |
| --- | ---: |
| Ranked rows | 50 |
| Exact generated implementation bodies with matching publisher | 47 |
| Implementation rows in a requested target domain | 36 |
| Arbitrary exact target rows | 5 |
| Required facets with an actual implementation source | 15/15 |
| Exact-source recall when its marker is named | 15/15 |

Thus the corrected binding coverage passes on these fresh real retrieval traces; the old 1/3 exact-source score and 0.9 exact-target surplus rate do not establish a missing-domain problem here. The 14 rows outside requested-domain implementations still motivate studying context economy, not declaring all returned peers irrelevant.

These are separate retrieval runs, not paired snapshots of the earlier aggregates. Exact-peer outcomes at other checkpoints can differ; they do not establish a scorer gain or a database effect. Replaying identical newly returned rows through the legacy, measured and final scorers isolates the scoring correction and preserves every recorded default-scale non-timing value.

This fixture also includes each query concept literally in the shard's “enabling” clause. It does not isolate paraphrase retrieval, despite an older code comment claiming otherwise; that comment is corrected without modifying any generated text.

## Real-reader errors are a different layer

The earlier [16-case reader study](../semantic-context-evaluation.md) recovered every positive required evidence group with semantic selection and the cheap scoped bundle. In `release-two-prerequisites`, the semantic reader cited both required sources but chose the wrong action. In `release-expired-approval`, all readers treated absence of a current approval as a recorded current rejection. Neither remaining error demonstrates an omitted required source. Distractor text, order and model variation were not isolated, so their causal contributions remain unknown. The expired negative case's empty gold groups make group visibility vacuously true; that is not evidence of a supported decision.

The [continued repository experiment](../team-repository-evaluation.md) likewise delivered equal evidence sets to the cheap handoff control and task memory, with different ordering and arm-specific carried code. Its failures do not justify tuning a retriever on those three inspected cases. All original paid failures and charges remain unchanged.

## Evidence, reproduction and next work

The [summary](../performance/memory-quality-diagnostic.json) links exact raw report bytes, ranked texts/source metadata, body-level audit, unchanged corpus/gate proof and measured/final runtime states. Fresh diagnostics select S9/S19/S26 only: **3/3 applicable binding checks per backend**, not a fresh run of all 19 MA scenarios or all 11 binding checks. Publishers are populated sequentially; this is not 1,000 concurrently running model agents. Deterministic test vectors and a mock judge make no OpenAI requests. Native retrieval scores and facet/router attribution are absent, so no stage-specific ranking cause is claimed.

The first SQLite diagnostic did not intercept `socket.connect_ex`; its zero network-attempt count means zero intercepted attempts. Provider ports were stubbed and HTTP sending blocked. The gap was then closed with a red/green guard regression that also checks retained partial evidence. PostgreSQL uses the final guard; its explicitly configured local libpq connection is allowed. The original SQLite report was retained, not rerun for a stronger isolation claim.

```sh
PYTHONPATH=src python -m tests.eval.sparse_source_diagnostic \
  --backend sqlite --output /tmp/memory-quality.json
```

For PostgreSQL, supply a dedicated local `AI_KNOT_TEST_PG_DSN` and select `--backend postgres`; only the diagnostic's UUID schema is created/dropped. The command requires the source Git checkout for its pinned legacy-scorer comparison, but no API key.

The next product work should exercise an end-to-end always-on host that can assess current scoped evidence when lexical requirements are paraphrased, under an explicit inference budget and fresh action guards. A source-availability flag must not become a claim of satisfied requirements, and lexical mismatch must not silently become permanent inability to plan. Validate on independent longitudinal tasks with service/time/source constraints and strong cheap controls; preserve decision, retrieval, spending and source-safety outcomes separately. New graphs, blanket history suppression or more embeddings are not justified by these advisory values alone.

See [local validation](../superpowers/validation/2026-10-28-sparse-source-scoring.json) and [independent Astra/Ponytail review](../superpowers/reviews/2026-10-28-sparse-source-scoring-astra-ponytail.md). This repairs evaluation and changes the next engineering question; it does not demonstrate a new model-accuracy gain or completion of the product goal.
