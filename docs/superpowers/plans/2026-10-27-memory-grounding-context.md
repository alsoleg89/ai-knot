# Memory grounding and task context implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Repair derived-memory provenance, truthful multi-facet coverage and contextual follow-up retrieval.

**Architecture:** Keep existing storage and retrieval modules. Unify derivation
invariants, preserve assembly metadata and construct bounded task queries at the
adapter seam. Three implementers own disjoint source areas; integration and final
reviews cover the combined behavior.

**Tech Stack:** Python >=3.11, existing SQLite/PostgreSQL/YAML backends, pytest,
existing Node/TypeScript package. No added dependencies.

**Spec:** `docs/superpowers/specs/2026-10-27-memory-grounding-context-design.md`

## Global Constraints

- Do not edit any existing deferred release or research files; their byte hashes
  are preserved in `/tmp/ai-knot-memory-grounding-checks/preexisting.json`.
- Work on `codex/memory-grounding-context`, based on main `84fbc7a`.
- No package publication, version/changelog edits, pushes or merges.
- Ordinary KnowledgeBase calls must not require AlwaysOnMemory or SQL tables.
- No new dependencies or model calls on the read path.
- SQL regressions run against isolated schemas; no user data is used.

## Review Focus

- A failed candidate must not invalidate a valid slot or launder itself through an aggregate.
- Historical archive is different from correction/delete/ACL revoke.
- Empty/missing facets and rank-only scores cannot produce complete coverage.
- Follow-up context must not leak across sessions or override a new independent task.
- Legacy manual facts, YAML and list-returning callers remain compatible.

### Task 1: Derived-memory grounding

**Owner:** implementation agent 1.

**Files:** `src/ai_knot/learning.py`, `knowledge.py`, `always_on/lineage.py`,
`always_on/maintenance.py`; narrowly related lineage helpers if necessary.
Create regression tests in `tests/always_on/test_learning_lineage.py` and/or
`tests/test_learning_grounding.py`. Do not edit integration or coverage files.

**Interfaces:** Existing public learn/add_resolved/recall/derive_fact APIs remain.
Internal accepted-candidate and exact-root validation may be shared. Coordinate
any required pool-side filtering with task 2 before editing its files.

- [x] Add failing public-path tests for unsupported supersession/aggregate,
  aggregate shrink/delete, legacy summary retraction, archived sources and manual
  facts. Include real SQLite/PostgreSQL fixtures and YAML compatibility.
- [x] Confirm regressions fail for the expected reason before source changes.
- [x] Make the smallest compatible acceptance and derivation changes; handle
  identifiable legacy automatic derivatives without fabricated source revisions.
- [x] Verify correction/revoke/delete invalidates descendants and old callbacks
  cannot commit against changed inputs. Preserve legitimate DELETE/manual paths.
- [x] Run focused tests, lint and self-review; report exact commands and changes.

### Task 2: Truthful facet coverage

**Owner:** implementation agent 2.

**Files:** `src/ai_knot/multi_agent/recall_service.py`, `models.py`, `assembly.py`,
`src/ai_knot/_pool_recall.py`, `_pool_helpers.py` and relevant pool metadata seam.
Tests in `tests/test_multi_agent_recall_service.py`, `test_multi_agent_assembly.py`
and a new `tests/always_on/test_recall_coverage.py` if SQL fixtures are needed.

**Interfaces:** Preserve existing list-returning service callers. Carry immutable
assembly results or equivalent explicit metadata without mutable side-channel
state. Public pool pairs and trust accounting remain compatible.

- [x] Reproduce real pool metadata reporting 1.0 for one of three facets, and
  irrelevant fallback filling a missing facet. Cover zero/full/partial results.
- [x] Confirm failing tests before changing source.
- [x] Preserve covered/uncovered facets through service/pool metadata and use
  them for low-coverage/abstention. Admit facet evidence using real relevance
  signals rather than positive rank-only scores.
- [x] Verify fallback queries, existing breadth/diversity behavior, ACL/support/
  temporal filtering on touched paths and SQLite/PostgreSQL parity.
- [x] Run focused tests and existing multi-agent acceptance; self-review.

### Task 3: Bounded task-context retrieval

**Owner:** implementation agent 3.

**Files:** `src/ai_knot/integrations/openai_agents.py` and sibling integration
adapters with the same actual gap; optionally a small shared context helper.
Tests under `tests/test_integrations_*.py`. Update relevant integration docs,
excluding all files recorded in the preservation snapshot.

**Interfaces:** Existing constructors/callers work. Add optional host-supplied
task context only where useful. Context building is bounded, deterministic,
stateless per request, Unicode-safe and performs no model/network calls itself.
Do not modify `knowledge.py` owned by task 1.

- [x] Add failing real adapter tests for English/Russian vague follow-ups with
  previous task input, explicit host context, empty input, task switching and
  composed filters. Demonstrate no cross-session accumulation.
- [x] Confirm failures before source changes.
- [x] Build retrieval queries from current input and bounded relevant task
  context; do not interpret prior assistant text as authority or query instructions.
- [x] Test strict bounds and existing adapter behavior; preserve input objects
  and optional-dependency import safety.
- [x] Run focused tests and self-review; document the public behavior and limits.

### Integration and acceptance

- [x] Review each task against this spec and resolve overlapping interfaces.
- [x] Run full Python/PostgreSQL, lint/format/mypy and TypeScript checks locally.
- [x] Run recovery on SQLite/PostgreSQL and relevant multi-agent gates.
- [x] Run independent Astra and Ponytail reviews covering all three tasks.
- [x] Fix findings, rerun affected checks, verify preserved file hashes and record
  completed behavior/tests in a new standalone document.
