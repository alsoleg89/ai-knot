# Task working sets for always-on teams

When several agents share memory, a relevant-looking procedure can still belong to another environment, and a generic role query can displace an older prerequisite. `AlwaysOnMemory.select_context` is an opt-in read-only selector for explicit task requirements. Start with the existing [team configuration](multi-agent-always-on.md). Ordinary `KnowledgeBase`/pool recall keeps its interface and routing; shared-assembler fixes can change tied or multi-facet selections under tight caps.

```python
context = memory.select_context(
    ("rollback checkpoint", "certificate rotation", "restore warning"),
    task_key="release42", role="operator", max_facts=3, max_chars=600,
)
for item in context.facts:
    print(item.content, item.ref)
print(context.missing_requirements)
```

Requirements are supplied by the host rather than inferred from punctuation or expanded by a model. Return values contain exact namespaced revision references, selected text, covered/missing requirement strings, a budget-pressure flag and check time. Selection makes no model/embedding call, changes no counters or intent state and starts no worker. SQLite/PostgreSQL use one read-only snapshot for the private/shared corpus, grants, source revisions and freshness checks.

For a small whole task corpus, use the queryless [task snapshot](task-snapshot.md): all current admissible facts or an explicit overflow error, with no lexical coverage claim. This avoids constructing search words merely to load known task-scoped memory.

## Applicability before role preference

Task applicability uses `Fact.qualifiers["task_key"]`. An unlabelled record is a generic candidate; a labelled record must exactly match the requested task. With no task key, only generic records are eligible. Use stable business keys that identify the task/environment/version boundary you actually need.

`Fact.qualifiers["for_role"]` is a recipient hint. Matching the supplied role boosts an already task-relevant score by 25%; it cannot introduce relevance or satisfy a requirement by itself. Other-role and unlabelled facts remain eligible, so a necessary prerequisite is not sacrificed to a role quota. Publisher expertise and volume do not remove the only matching source from this path.

These are host-declared applicability labels, not permission, tenant isolation or automatic semantic understanding. Read ACLs and source dependencies still apply. Labels do not change slot identity/CAS: include the business entity/task in slot addresses when storing separate states, for example `release42::result`. A role label never grants execute/maintain authority.

## Store conditional procedures with their sources

```python
from ai_knot.types import MemoryType

procedure = memory.derive_fact(
    "For release42, check the verified backup before restoring.",
    basis=observed_result_refs,
    memory_type=MemoryType.PROCEDURAL,
    task_key="release42", for_role="operator",
)
```

The type and labels are persisted atomically with exact source roots and intermediate revisions. Existing derive calls remain semantic and unlabelled. Manual host writes can also use ordinary `Fact(type=..., qualifiers=...)` followed by the existing ingestion API. Internal lineage fields cannot be overridden through these options.

For a shared procedure, its readers must also be able to read its roots. Publish an appropriately authorized source first and derive from its shared reference; a public projection of somebody else's private source does not bypass source access. Shared derivation still requires the workspace's maintain grant. Source correction/deletion/revocation retracts dependent procedures on subsequent selection/review.

The host supplies the observed evidence and lesson text. These options do not infer that an action succeeded, promote an assembled answer into a successful experience, prove the procedure follows from its source, or establish that one success generalizes. Use [record_experience](team-experience.md) when a lesson must also depend on an action's current recorded outcome. Raw episodic records stay outside ordinary task context; a grounded procedural projection can reference retained outcome evidence. Copies retain roots and do not become independent confirmations.

Run the model-free two-role example:

```sh
PYTHONPATH=src python examples/task_context.py
```

It shares one source corpus with operator/reviewer procedures, selects different role-appropriate instructions, then corrects the outcome and verifies that the old projections cannot be reused.

## Honest coverage and budgets

Coverage requires content-bearing requirement words in the text actually returned. Stopwords, role text, tags, canonical aliases or hidden snippets cannot certify it. Metadata can rank eligible facts. When BM25 suppresses ubiquitous terms, direct overlap with original content-bearing words supplies a conservative fallback.

The existing greedy assembler selects evidence for missing requirements before backfill. If one fact covers multiple requirements, that coverage is recognized immediately. Whole facts must fit both count and character budgets; text is never cut midway. `budget_limited=True` means an otherwise admissible candidate was omitted for size. Once that is established, further oversized candidates need no repeated SQL admission. It does not enumerate every omitted candidate or promise that more budget would supply a missing requirement.

`complete=True` means all supplied requirements have selected lexical evidence after currentness/admission and budget checks. It is not semantic entailment, proof that the task is complete, or an action permit. Greedy selection is deterministic but does not guarantee the optimal set for every possible corpus/budget. Full corpus indexing scales with retained history, and rejected policies can require multiple checks. This is a small working-set interface, not a production-capacity guarantee.

Limits: one to five unique content-bearing requirements, 4096 Unicode characters each; exact task/role labels at most 256 characters without surrounding whitespace; `max_facts` 1..128 (default 8), `max_chars` 1..65536 (default 12000). Character limits exclude host formatting and are not tokenizer limits.

On restart, use saved refs with `review_context`, and reselect for the current task/role when those inputs change. Resume durable work with `resume(intent_id)` and use `begin_action` guards before effects. Already delivered model text cannot be erased; replace stale context in your host.

## Existing transports

| Surface | Operation |
|---|---|
| Python | `memory.select_context(requirements, task_key=..., role=...)` |
| HTTP | `POST /v1/always-on/evidence/select` with the same named JSON fields |
| MCP | `memory_context_select` |
| TypeScript | `client.alwaysOn.selectContext({requirements, task_key, role, max_facts, max_chars})` |

HTTP/MCP retain configured-owner identity, strict types, authentication and source ACLs. These additions create no package dependency, storage schema or release/version change. See [research](research/2026-10-27-task-memory-selection.md), [implementation contract](superpowers/specs/2026-10-28-task-context-design.md) and [shared-memory roadmap](shared-memory-roadmap.md).

## Fixed-data comparison

The committed harness freezes 16 traces (four task families × two roles × matching/mismatching applicability), with initial, peer-update, fresh-process handoff and source/outcome correction checkpoints. Six variants run at the same three-fact/600-character ceiling, on the same facts, grants and sources. Each backend produced 384 result records. English/Russian requirements, old prerequisites, failed/unknown observations, copied lessons and withdrawn sources are included. Selection HTTP/embedding endpoints are blocked.

Of 192 requirement positions per variant, 112 have current applicable gold evidence; the remaining 80 intentionally lack it. Both SQL backends produced identical selected evidence and quality metrics.

That sentence describes the retained milestone runs. The final whole-program reproduction retained every quality metric, gold label and corpus/grant/usage control, and the selectors/strongest scoped control retained their evidence identities. Two PostgreSQL records in the older unscoped `Integrate` baseline chose the other role-labelled source with identical emitted text; its forbidden-source explanation changed only that ID. This is tied-source variation, not a retrieval/safety gain or a promise of bit-identical source IDs across all runs. See the [reproduction record](performance/program-trace-reproduction.json).

| Variant | Available evidence groups found | Wrong-applicability facts | Preferred role present, matching checkpoints |
|---|---:|---:|---:|
| Ordinary recall, task/role/requirements in query | 56/112 | 96 | 12/32 |
| Same recall after explicit applicability projection | 96/112 | 0 | 24/32 |
| Ordinary recall with explicit `Integrate` conjunction | 112/112 | 80 | 16/32 |
| Applicability projection + `Integrate` conjunction | 112/112 | 0 | 24/32 |
| Task selector, role preference disabled | 112/112 | 0 | 24/32 |
| Task selector with role preference | 112/112 | 0 | 32/32 |

The strongest existing scoped facet path already matches selector recall. The new API makes applicability, explicit missing requirements and role preference directly accessible; it does not establish a better underlying retrieval algorithm. The role preference increases matching-role presence in this fixture, while the proportion of selected role-labelled facts matching the recipient rises from 50% to 66.7% because useful opposite-role backfill remains allowed. These are separate metadata measures, not answer-quality scores.

The role alternatives in this control deliberately share valid procedure text: the measured gain is recipient-source preference. The executable two-role example separately checks selection of visibly different operator/reviewer instructions. Failed/unknown observations and unsupported projections are explicit fixture labels, not proof that this selector can infer whether an external operation succeeded.

The strongest scoped baseline and both selectors each return 11,200 content characters across 64 checkpoints. All variants make zero selection model/embedding calls; corrected/unsupported/raw-episode/role-only selection counters are zero. Both selectors have zero missed or falsely reported lexical coverage on this fixture. Ordinary recall lacks explicit supplied-requirement report fields; the harness records those metrics as unavailable rather than inventing them.

Synthetic editorial gold does not establish real-model accuracy, universal applicability, end-task success or production capacity. Keep the selector opt-in. Reproduce with:

```sh
PYTHONPATH=src python -m tests.eval.task_context.runner --backend sqlite --output-dir /tmp/task-context-sqlite
# Set AI_KNOT_TEST_PG_DSN for a disposable-schema PostgreSQL run:
PYTHONPATH=src python -m tests.eval.task_context.runner --backend postgres --output-dir /tmp/task-context-postgres
```

The CLI owns only its temporary database/schema. Its worker jobs may briefly contain the local test DSN, use restricted file permissions and are removed; they do not enter result reports. Full reports include the frozen corpus SHA, per-case gold, reasons, exact revisions and controls.

Retained results: [SQLite summary](performance/task-context-sqlite.json), [PostgreSQL summary](performance/task-context-postgres.json), compressed raw [SQLite](performance/task-context-sqlite.json.gz)/[PostgreSQL](performance/task-context-postgres.json.gz) reports and the [frozen manifest](performance/task-context-manifest.json.gz). Compression keeps the complete reproducibility evidence small; all files use standard JSON/gzip. [Independent source review](superpowers/reviews/2026-10-28-task-context-astra-ponytail.md) records the coverage/budget defects found and their verified fixes.

Local acceptance: 1,889 Python tests passed with PostgreSQL required, two expected skips, 89.24% coverage; 58 TypeScript tests, build/typecheck and Python static checks passed. Existing recovery passed 12/12 scenarios on each SQL backend and all 11 binding multi-agent gates passed. See [validation record](superpowers/validation/2026-10-28-task-context.json) and [independent method audit](superpowers/reviews/2026-10-28-task-context-method.md) for scope and limitations.
