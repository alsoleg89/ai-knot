# Explicit task working sets

The autonomous product goal is common memory for long-lived teams. Reliable return is committed at `fcbcad5`; this milestone makes current task context useful and bounded. Research: [task/role selection](../../research/2026-10-27-task-memory-selection.md). User delegated design/execution decisions; no approval pause is needed.

## Approach

Appending a role to a vague query can let role vocabulary masquerade as task evidence. A graph/LLM projector adds cost before proving selection value. Selected approach: opt-in explicit requirements, exact host-declared task applicability, soft recipient-role preference, existing lexical retrieval and coverage assembly, followed by existing current-evidence guards. Ordinary recall remains unchanged.

## Public contract

`AlwaysOnMemory.select_context(requirements: tuple[str,...], *, task_key: str = "", role: str = "", max_facts: int = 8, max_chars: int = 12000) -> TaskContext`.

- One to five unique, nonempty requirements, at most 4096 Unicode characters each, with content-bearing query tokens. Preserve request order. `max_facts` is an integer 1..128; `max_chars` integer 1..65536. Task/role labels are exact strings, at most 256 characters, with no leading/trailing whitespace; empty means unspecified.
- Read own private facts and the attached persistent shared pool in one read-only SQL snapshot. No new storage/schema, polling, model/embedding call, counters or intent mutation.
- Fact applicability is `qualifiers["task_key"]`: matching task or an absent/empty label is eligible; a different task is excluded. With no task_key, only unscoped records are eligible. These are host-declared retrieval constraints, never tenant isolation/ACLs or proof of universal applicability.
- `qualifiers["for_role"]` is a soft hint. A matching nonempty recipient role multiplies a task-relevant score by 1.25. Other/unlabelled roles remain eligible, especially for prerequisites. Role cannot create original-query relevance or coverage.
- Early active/support/type/ACL/lineage filters precede indexing. Exclude raw episodic records. Use one existing BM25F index, but coverage/admission requires content-bearing query words in emitted `Fact.content`. Tags, canonical text and snippets may rank; hidden metadata and stopwords cannot certify coverage. If no content-gated BM25 scores survive ubiquitous-term suppression, direct original-content overlap supplies relevance. This is not semantic entailment. Preserve namespaced identity internally even when private/shared fact IDs collide.
- Reuse coverage assembly with optional whole-fact character ceiling and an admission callback. Current revision, slot conflict, source lineage and freshness are checked before admission, cached per candidate for this selection. Requirement coverage takes priority over publisher quotas; rare prerequisites are not removed by expertise routing. Ties are deterministic. Budget constraints apply in coverage and backfill.
- Return immutable `TaskContext(facts, covered_requirements, missing_requirements, budget_limited, checked_at, complete)`. Facts are existing `ContextFact`s with original exact refs/text/slot. Coverage describes selected original-query lexical evidence after budget/admission; `complete` means every supplied requirement has such evidence, not task completion or permission to act. Read again on return and use action-time guards for effects.

Extend guarded `derive_fact` with optional `memory_type=MemoryType.SEMANTIC`, `task_key=""`, `for_role=""`. Persist the selected type/labels in the SAME transaction as exact roots/dependencies. Default calls retain existing meaning and qualifiers. Callers cannot overwrite internal lineage. This permits host-authored conditional procedural knowledge; it does not automatically learn a lesson from a successful retrieval or assert an external outcome.

Expose `memory_context_select` through strict HTTP `/v1/always-on/evidence/select` and TypeScript `selectContext`. Existing transport identity and permissions apply. CLI expansion beyond prior recovery is not required. No release/version/changelog or publication changes; original checkout's 30 protected files remain untouched.

## Acceptance

SQL tests: wrong-task high-overlap facts; role-only distractors; unlabelled necessary facts; rare publishers; source corrections/ACL revoke/freshness; shared/private ID collision; whole-fact budget and truthful missing coverage; ubiquitous-token query; deterministic order and source permutations; default derive compatibility and persisted procedural metadata. Assembly tests cover admission/budget in both phases without breaking old callers. Real strict HTTP/MCP/TS boundaries.

Freeze 16 task traces and four checkpoints each (64 per arm): initial, peer update, fresh-process handoff and source/outcome correction/revocation. Compare ordinary recall given task/role/requirements, task-only selection and task+role selection with the SAME corpus, context ceiling and zero selection inference calls. Report required/acceptable/forbidden evidence, missing requirements, irrelevant characters and latency. Include scoped-baseline ablation so role is not credited with task filtering. Synthetic selection quality is not LLM answer accuracy. Feature remains opt-in regardless of trace gains.

Run local full required-PostgreSQL Python suite, formatting/lint/strict types, TS tests/build/typecheck, existing recovery/MA gates, independent Astra/Ponytail review and original-byte verification. Keep unresolved quality limitations explicit rather than tuning old benchmark labels.
