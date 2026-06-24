"""S25 — Knowledge Conflict Resolution at Scale (10 Slots, 4 Wrong + 1 Canonical).

Verifies that the CAS mechanism correctly collapses conflicting facts from
4 agents down to a single canonical version per slot when an authoritative
agent publishes corrections.

Flow:
  Phase 1: Agents A–D each publish facts for all 10 entity slots (40 total).
           Each agent has different (wrong) values.
  Phase 2: Agent E (authoritative) publishes canonical values for all 10 slots.
           CAS supersedes whoever was last-writer for each slot.
  Phase 3: Verify exactly 10 active facts remain (1 per slot), all from agent_e.
  Phase 4: Agent F queries all 10 topics — must find agent_e's canonical values.

Metrics:
  resolution_correctness — 1.0 if exactly 10 active facts after resolution
  canonical_coverage     — fraction of queries returning agent_e's canonical keyword
  conflict_collapse      — 1.0 if all 10 active facts are from agent_e

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s25_conflict_resolution"

_QUERIER = "agent_f"
_WRONG_AGENTS = ["agent_a", "agent_b", "agent_c", "agent_d"]
_CANONICAL = "agent_e"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    slots = fixture.conflict_slots
    values_by_agent = [
        ("agent_a", fixture.conflict_agent_a_values),
        ("agent_b", fixture.conflict_agent_b_values),
        ("agent_c", fixture.conflict_agent_c_values),
        ("agent_d", fixture.conflict_agent_d_values),
    ]

    # Phase 1: 4 agents each publish wrong values for all 10 slots.
    for agent_id, values in values_by_agent:
        for (entity, attr), val in zip(slots, values, strict=True):
            await backend.add_structured(agent_id, val, entity=entity, attribute=attr)
        await backend.publish_to_pool(agent_id)

    # Phase 2: Agent E publishes canonical values.
    for (entity, attr), val in zip(slots, fixture.conflict_canonical_values, strict=True):
        await backend.add_structured(_CANONICAL, val, entity=entity, attribute=attr)
    await backend.publish_to_pool(_CANONICAL)

    # Phase 3: Count active facts per slot — expect exactly 1 each.
    total_active = 0
    for entity, attr in slots:
        count = await backend.pool_count_active_for_entity(entity, attr)
        total_active += count
    resolution_correctness = 1.0 if total_active == len(slots) else 0.0

    # Phase 4: Agent F queries all topics.
    canonical_hits = 0
    for query, kw in fixture.conflict_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=3)
        if any(kw.lower() in t.lower() for t in r.texts):
            canonical_hits += 1
    canonical_coverage = canonical_hits / len(fixture.conflict_queries)

    # Conflict collapse: verify all active facts are from agent_e.
    pool = getattr(backend, "_pool", None)
    collapse_ok = 1.0
    if pool is not None:
        for entity, attr in slots:
            el = entity.lower().strip()
            al = attr.lower().strip()
            for f in pool.list_shared_facts():
                if (
                    f.entity.lower().strip() == el
                    and f.attribute.lower().strip() == al
                    and f.is_active()
                    and f.origin_agent_id != _CANONICAL
                ):
                    collapse_ok = 0.0
                    break

    notes = (
        f"slots=10, wrong_agents=4, canonical=agent_e, "
        f"total_active={total_active}, resolution_correctness={resolution_correctness:.0%}, "
        f"canonical_coverage={canonical_coverage:.0%}, collapse_ok={collapse_ok:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "resolution_correctness": [resolution_correctness],
            "canonical_coverage": [canonical_coverage],
            "conflict_collapse": [collapse_ok],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
