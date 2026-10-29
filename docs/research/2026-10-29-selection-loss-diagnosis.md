# Why bounded semantic selection lost task conditions

This is a post-hoc diagnosis of the [frozen growing-context study](../growing-context-evaluation.md), not a new quality trial. Product source is unchanged from `ece236a`. No provider requests, embeddings, SQL operations or new reader answers were generated. Original scores and inputs remain unchanged.

## Reproduction

The diagnostic reconstructs each final stage's available vectors from that case's recorded embedding calls before its recorded completion. It normalizes them using the original Euclidean norm, uses the exact original request and joined application rules, and scores the current whole-task reference's source texts. Those sources come from the corresponding frozen full-reference stage; this is a controlled reconstruction from the known exogenous history, not a new live observation of a multi-agent database.

It feeds these scores into the unchanged `CoverageAwareAssembler` with the manifest's original limits and requires exact **ordered** equality with the actual semantic selected IDs. All 20 final selections reproduce. It does not rerank with gold: frozen required-source IDs are consulted only after selection to describe omissions and calculate whether the required set would fit.

| Diagnostic | Final stages |
|---|---:|
| Exact ordered selection reproduced | 20/20 |
| Every source has positive cosine to every query facet | 20/20 |
| Assembler reports every facet covered | 20/20 |
| Required source missing despite all facets covered | 15/20 |
| Entire frozen required-source set fits eight facts / 2,000 evidence characters | 20/20 |

These counts concern the semantic arm. They do not establish that a new method can identify those sets without gold, or that the same phenomenon explains every lexical miss.

## Mechanism and implications

`semantic.py` assigns facet membership whenever cosine is positive. `assembly.py` counts all positive facet memberships on a selected fact as covered. Since every candidate in these recordings belongs to both facets, the first selected fact can cover both; later choices backfill by global relevance score. The two facets represent the request and the whole policy, not an enumerated set of all applicable decision conditions.

The existing coverage implementation is consistent with that map. The map is too weak to stand for prerequisite completeness. This diagnosis does not turn a relevance diagnostic into a new task-readiness failure or change the product's documented contract. Source guards and ACL/lineage checks address different questions and still apply.

The required sets fitting the budget demonstrates an opportunity in these particular cases, not an implementable oracle. Raising a threshold from observed scores, keeping every generic fact or adding a redundancy penalty has no demonstrated general solution. An explicit condition map can be wrong or incomplete; permission and its exception can be similar while both matter. The [primary-source research](2026-10-29-budgeted-evidence-selection.md) separates budgeted coverage, MMR, robust objectives and complementary prerequisites, including where their guarantees do not apply.

The next product candidate should distinguish trusted host-declared obligations or companion groups from optional relevance. First test whole-group inclusion, source withdrawal, revision changes and explicit budget infeasibility without inference. Then measure on new independently authored tasks, including unknown obligations, against the unchanged full-context and lexical controls with equal offered total model budgets. An explicit include set must not be filled from evaluation gold, automatically inferred from provenance roots or advertised as universal semantic completeness. A ranking change without an obligation map remains a separate hypothesis.

## Artifacts and checks

[Diagnostic source](growing_context_rank_diagnosis.py) checks the exact case hash and all retained runtime/meter source hashes before reconstructing selections. It reads the original manifest limits. It rejects a changed source freeze or a different selected-ID sequence instead of silently continuing under different code.

[Retained diagnostic evidence](../performance/growing-context-rank-diagnosis.json.gz) contains the script, per-source scores/ranks/lengths, selected flags, frozen-gold diagnostic labels, aggregate counts and fault-check results. Its input is the original [study archive](../performance/task-context-guard-raw.json.gz), which is unchanged. Extract its `growing-context-live-20261029`, `growing-context-frozen-20261029` entries before running; use the exact retained runtime or checkout `ece236a` and its existing dependencies.

```sh
PYTHONPATH=src python docs/research/growing_context_rank_diagnosis.py \
  /path/to/growing-context-live-20261029 \
  /path/to/growing-context-frozen-20261029/growing-context-cases.json \
  /tmp/growing-context-rank-diagnosis.json
```

The diagnostic passed exact reconstruction of 20 stages and three negative checks: tampered selection, changed case bytes and changed runtime hash are rejected. Its source passes Ruff format/lint. The application source has not changed; the previously retained 2,737-test PostgreSQL-inclusive acceptance belongs to `ece236a`, not a newly executed test run for this research note. No Astra was used. Deferred release changes remain protected, and the product goal remains active.
