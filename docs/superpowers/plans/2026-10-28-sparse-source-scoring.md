# Sparse-source scoring correction

**Goal:** Make the S26 binding evidence score require a known implementation shard rather than just a same-domain publisher, so quality work starts from trustworthy evidence.

**Basis:** HEAD `987f885`. Retained S19 has all three required keyword probes but surplus results. Retained S26 has no per-query source traces, and its equivalence score currently counts generic and near-miss facts by publisher domain. A generic-only return can therefore pass the binding domain gate. Exact text-only marker hits also count duplicate rows as distinct targets; a publisher can be credited with another target's marker.

**Design:** Keep the generated corpus, questions, all numeric thresholds and binding/advisory assignments unchanged. Use the existing unique shard markers as source identities: tokenize full marker words, map them to known fixture source indices, and require publisher/marker agreement when Fact provenance exists. Text-only results use unique known marker identities without inventing a publisher. Count exact targets and equivalent domains from those identities; a generic note with no marker contributes neither. This remains a source-identity proxy on generated fixtures, not an entailment judge. Correct unsupported universal ceiling/floor claims in explanatory notes only.

## Work and checks

- [x] Write failing async regressions for generic-only/near-miss-only false positives, valid equivalent peers, duplicate text hits, borrowed markers, prefix collisions, and real target sources. Include a negative binding-gate assertion.
- [x] Implement one shared marker-identity helper for conceptual and identifiable queries. Keep all fractions bounded and source count deduplicated; preserve top-k and timing measurement boundaries.
- [x] Run focused tests, Python compatibility and full local tests with PostgreSQL. Run fresh offline S9/S19/S26 diagnostics on both SQL backends with real retrieval, blocked network and deterministic embedding stubs; retain ranked source traces and distinguish them from earlier aggregate-only results. Do not rescore or overwrite earlier artifacts.
- [x] Independently review with Astra/Ponytail. Record which misses are retrieval, reader errors or insufficient/ambiguous evidence. Preserve original checkout/release files, make a local commit and keep the product goal active.

If corrected S26 coverage misses the unchanged binding target, retain that failure and investigate actual retrieval; do not lower the target or redefine a generic overview as a shard.

## Outcome and limits

Both fresh SQL diagnostics passed 3/3 binding checks for the three selected scenarios. A body-level audit found actual implementation sources for 15/15 required facets at each default checkpoint. All recorded default non-timing scores replay unchanged through legacy/measured/final scorer versions; no earlier result was rescored. Final local CI passed 2,476 tests with PostgreSQL required.

The initial SQLite diagnostic blocked provider/HTTP ports but missed Python `connect_ex`; preserve that scope limit and its exact runtime. The final guard and red/green partial-progress regression close the gap without rerunning SQLite for a stronger claim. Local PostgreSQL libpq connections remain explicitly allowed. Repeated-domain N60 coverage was corrected after measured runs and verified by regression and same-row default replay. Instrumented p95 and marker identities do not establish product latency, entailment, real-model quality or concurrent-fleet capacity. The broad product goal stays active.
