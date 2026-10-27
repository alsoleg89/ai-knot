# Grounded memory, coverage and task context

The user authorized implementation of the three corrections identified in
`docs/research/reports/Проверка идей MemCurrent для ai knot.md`, with three
implementers, local SQLite/PostgreSQL tests and final Astra/Ponytail reviews.
This is the implementation specification for that request.

## Required behavior

1. An unsupported extracted ADD/UPDATE must not close or reinforce accepted
   knowledge or become a supported aggregate. Preserve operation-specific DELETE
   behavior and the existing trusted-host manual ingestion contract.
2. Automatic aggregates and episodic summaries must retain exact source revision
   roots and dependencies. Correction, deletion and source read revocation must
   invalidate their future recall and guarded evidence use. Copies must not add
   independent roots. Archiving an episode may preserve historical evidence;
   deleting or changing that evidence must invalidate descendants.
3. Rebuilding an aggregate must close its old version when its accepted source
   group becomes too small. Identifiable old automatic derivatives without exact
   lineage must not be silently promoted to current verified evidence. Manual
   facts without lineage remain supported. Do not invent historical revisions.
4. A multi-facet recall must preserve the assembler's actual covered/uncovered
   facets through the public pool metadata, including partial and empty results.
   A positive RRF score without relevance evidence must not count as facet
   coverage. Missing evidence must remain visible to abstention decisions.
5. A vague follow-up such as `Continue.` or `Продолжай` must be able to use bounded
   recent task context. The host may provide explicit task context. Latest
   standalone tasks must not be polluted by older unrelated tasks. Context is a
   retrieval signal, not permission or a new source of facts. Do not retain one
   session's context in another session or introduce model calls on reads.

## Architecture and choices

Reuse existing fact revision/lineage functions and transactional mutations rather
than adding a second memory engine. Share derivation validation between legacy
and always-on paths without requiring `AlwaysOnMemory` or SQL tables for ordinary
`KnowledgeBase`; YAML compatibility remains required. Preserve fact-list recall
and existing pool service callers while carrying richer assembly metadata.

Task context belongs at the host/integration query-building seam. Prefer a small
bounded deterministic helper and the existing adapters over changing HTTP/MCP
protocols or adding a catalog, scheduler, model, dependency or new identity layer.
Read the sibling adapters before deciding which callers need the same correction.

Alternatives considered: adding a catalog does not fix early write corruption or
discarded coverage; replacing the retrieval engine is broader than necessary;
making every legacy call initialize always-on SQL breaks supported backends.

## Compatibility and preservation

- Python >=3.11; no new dependencies. Existing TypeScript clients must still pass.
- Public manual operations and historic audit records keep their meaning.
- Preserve archived-source freshness deadlines and revision semantics.
- Do not rewrite existing report/probe evidence or deferred release files.
- Do not edit versions, changelog, release workflows or publish packages.
- No push, PR or merge is part of this request.

## Acceptance

Regression tests must first reproduce the diagnosed failures, then pass on the
fix. Cover supported/unsupported replacements, aggregate shrink/delete, source
correction and revocation, legacy/modern consolidation, manual facts, YAML,
SQLite, real isolated PostgreSQL, partial/zero/full facet coverage, irrelevant RRF
fallbacks, English/Russian follow-ups, task switching, bounded inputs and stateless
adapter composition. Test real public paths rather than only mocked metadata.

Run the full local Python suite, lint, format, mypy, TypeScript typecheck/build/
tests, always-on recovery and relevant multi-agent acceptance gates. Fix findings
from independent Astra and Ponytail reviews and rerun affected checks. Record
actual results and limits; these checks do not measure real-LLM answer accuracy.
