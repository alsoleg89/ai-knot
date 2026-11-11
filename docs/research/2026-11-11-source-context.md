# Source views for shared always-on memory

Access date: 2026-11-11. Primary sources and local evidence only; no adoption or
accuracy ranking is inferred from another system's feature documentation.

## What current systems expose

[LangGraph's BaseStore reference](https://reference.langchain.com/python/langgraph.store/base/BaseStore)
describes a persistent key/value interface shared across threads with application
chosen namespaces. Semantic indexing is optional. Its
[memory overview](https://docs.langchain.com/oss/python/concepts/memory) distinguishes
thread checkpoints from cross-session store data and describes both application-path
and background memory updates. This supports keeping recovery state and shared
knowledge as separate concerns; it does not establish how to validate generated facts.

The current [Letta shared-memory guide](https://docs.letta.com/concepts/shared-memory)
describes organization-owned Git repositories attached to cloud agents, with
commit/push/pull synchronization. It distinguishes these repositories from each
agent's MemFS and calls shared memory blocks a legacy API pattern. The
[memory configuration guide](https://docs.letta.com/configuration/memory) describes
background conversation review/consolidation, including an optional second agent
review before applying updates. This is stronger evidence for the current design
than older summaries that describe shared blocks as the whole current product.
The documentation is a feature description, not a comparative quality evaluation.

## Local evidence and decision

AI Knot already has owner-scoped recovery state, shared observations, exact
derivation roots, typed source guards and a whole-task read. The
[continued-work pilot](../team-work-evaluation.md) found a 1284-character generated
handoff outside its 1200-character contract. It also found that requiring every
cited root overflowed the bounded context. These do not justify another shared
storage system, a default summarizer or automatically increasing output budgets.

The design choice here is an additive query policy: let a host ask the existing
snapshot for admitted non-derived facts, excluding generated summaries/lessons
before output budgeting. The default still includes current grounded derivations.
The source view retains all currentness, visibility and task rules, never mutates
memory, and reports overflow without a partial prefix. A task guard still covers
the full admissible set. No inference or scheduler is introduced.

This decision is an engineering inference from the local failure and existing
architecture, not a result claimed by LangGraph or Letta. A non-derived annotation
does not attest truth, human authorship, sufficiency or action permission. Filtering
can omit useful learned procedures. Hosts keep the original task requirements and
may refuse or choose an explicitly bounded selector when raw observations are
insufficient or do not fit. Real longitudinal task benefit remains to be measured.
