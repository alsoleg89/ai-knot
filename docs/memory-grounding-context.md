# Grounded memory, coverage and task context

This guide describes the corrected memory paths in this checkout. It introduces
no scheduler, model worker, package dependency or new transport protocol.

## Derived memory follows its sources

`KnowledgeBase.learn()` no longer lets an unsupported extracted ADD or UPDATE
close or reinforce accepted knowledge. DELETE remains an operation with its own
contract: its target does not have to repeat the fact being deleted. Explicit
manual host writes retain their existing meaning.

Entity aggregates include only accepted current sources. They record exact
revision roots and intermediate dependencies, just like explicitly derived facts.
Changing or deleting an input prevents future recall of the old aggregate; when
the source group shrinks below two fields, its old aggregate is closed. Optional
aggregation must not discard an accepted primary write when its complete lineage
cannot fit the existing 128-reference bound; roots are never truncated to fit.
Future-dated inputs do not enter a current aggregate, and an aggregate cannot
outlive the earliest finite expiry of its contributing current facts.

Publishing a derived fact preserves its finite validity. Under the owner/shared
SQL transaction, the pool checks the current private revision, live validity,
source revisions and grants before changing a shared slot or statistics. A stale
or expired replay cannot close its current replacement. Manual publication keeps
its existing trusted-host contract.

Legacy episodic consolidation also records roots and dependencies in the commit
that archives its inputs. Archiving can preserve a precise historical source;
correction and deletion invalidate descendants. An existing matching freshness
attestation follows the archived revision without extending its deadline.

Recognizable older automatic aggregates/summaries without exact roots are not
returned as current evidence. Normal manual facts without roots are still valid.
An aggregate can be rebuilt on a later write from its retained accepted sources.
An old summary whose original revisions are unavailable needs revalidation or
reconstruction; current revisions are not invented as historical evidence.
`recall_by_tag(tag, now=...)` also applies the requested active-time window;
use `list_facts()` or `get()` when inspecting stored history rather than current
recall. Source-access and revision checks continue to apply at historical times.

These rules record declared source dependencies. They do not prove that generated
text follows logically from a source or that a host's claimed action succeeded.
Already delivered model context cannot be erased by a later source revocation.
YAML remains usable for ordinary KnowledgeBase operations with its existing
single-writer limitations; always-on coordination requires SQLite or PostgreSQL.

## Coverage describes the selected evidence

For decomposed multi-source queries, the pool retains covered and uncovered
facets from the selected, readable facts. One selected facet out of three means
coverage of one third, even if the result list is nonempty. Rank-only RRF scores
cannot turn unrelated facts into coverage; facet admission requires lexical
relevance from the original facet query. This is a relevance gate, not a semantic
entailment check.

Existing `SharedMemoryPool.recall()` callers still receive `(Fact, score)` pairs.
`last_recall_risk` and `last_recall_abstains` use the actual coverage. The internal
multi-agent service exposes `recall_with_coverage()` for callers needing the
selected pairs and immutable covered/uncovered facet sets; its `recall()` list
wrapper remains compatible.

Shared recall applies support, memory-type, source permissions and the requested
time window before assembling evidence. The `now` argument is honored on SQL
backends as well as YAML. Trust credit follows the existing flat-path rule: at
most one publisher credit per recall, with separate per-fact consumer events.

## Follow-up queries can use their task

OpenAI Agents, the OpenAI wrapper, AutoGen and LlamaIndex use a shared deterministic
query builder. A short English/Russian follow-up such as `Continue.` or
`Продолжай` can include the previous standalone user task. A new standalone query
uses its own text. Empty current user input produces no retrieval query.

The host can pass context for the current run:

```python
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory

memory = AiKnotAgentsMemory(kb)
run_config = memory.build_run_config(
    task_context="Review the production rollout of build 42.",
    workflow_name="rollout-review",
)
```

`task_context` is consumed by the adapter, not forwarded as an unknown SDK option.
It takes precedence over inferred history for recognized follow-ups. Construct a
fresh run configuration when its task changes. Equivalent per-call context is
available on `MemoryEnabledOpenAI.enrich_messages`, AutoGen
`AiKnotAutoGenMemory.extract_query` / `update_context`, and the LlamaIndex retrieval seam.

The helper keeps at most eight user-message payloads; LlamaIndex retains its
existing smaller default. It scans message role metadata backwards, so long tool
loops do not push the current user task out of an eight-item raw-message tail.
Assistant, developer and tool payloads are not task authority. Each extracted
payload and final query is limited to 4096 Unicode characters, with at most 32
content nodes and nesting depth four. Metadata scanning is linear in the supplied
history. Character bounds are not tokenizer bounds.

There is no stored session history in the helper and no model call for query
construction. Its conservative follow-up recognition is a heuristic, not general
anaphora understanding. The caller remains responsible for supplying the correct
session, identity, current task and tool permissions.

## Local verification

Regression tests cover ordinary YAML calls and real SQLite/PostgreSQL schemas,
including source retraction, archived freshness, concurrent corrections, partial
coverage and adapter composition. A combined integration test checks contextual
recall, shared-source ACL revocation/deletion, uncovered facets and blocked action
admission together.

The local load harness uses eight independent processes and 200 publications plus
200 recalls per backend. It checks exact usage counts, resource exclusion, shared
verification and zero idle provider calls. Reports are retained separately from
earlier measurements:

- [SQLite load](performance/memory-grounding-context-sqlite.json)
- [PostgreSQL load](performance/memory-grounding-context-postgres.json)

These are correctness and local workload measurements. They do not establish a
real-LLM answer-accuracy improvement or production capacity estimate.

Final local acceptance used a clean feature-only export at the committed 0.12.0
version, excluding the separate deferred release preparation:

| Check | Result |
|---|---|
| Python 3.12, PostgreSQL required | 1784 passed, 2 expected skips; coverage 88.77% |
| Python 3.11 and 3.13 | 1583 passed each; PostgreSQL cases skipped in these runtimes |
| TypeScript | 56 passed; typecheck and ESM/CJS build passed |
| Lint, formatting, strict mypy | Passed; 76 source files type checked |
| Process recovery | 12/12 on SQLite and 12/12 on PostgreSQL; all safety counters zero |
| Multi-agent acceptance | 11/11 binding thresholds passed |
| Independent reviews | Astra ready; Ponytail Ship; all material findings closed |

The five existing advisory benchmark misses remain unchanged: S9 conflict and
wrong suppression, S19 evidence precision, and S26 target recall/distractor rate
at 1000 publishers. They are not binding acceptance thresholds or measured
accuracy improvements from these changes.

The [validation record](superpowers/validation/2026-10-27-memory-grounding-context.json)
includes source/test checksums and result scope. Review summaries:
[Astra](superpowers/reviews/2026-10-27-memory-grounding-astra.md) and
[Ponytail](superpowers/reviews/2026-10-27-memory-grounding-ponytail.md).
