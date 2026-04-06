"""S10 — MESI Entity-Addressed CAS (Conflict Resolution).

Verifies that when two agents publish facts about the same entity+attribute,
the shared pool retains exactly ONE active version — the latest one.

This tests the entity-addressed Compare-And-Swap (CAS) in SharedMemoryPool:
  - Agent A publishes salary v1 ($95k) for Jordan Lee.
  - Agent B publishes salary v2 ($140k) for the same person.
  - Expected: pool has exactly 1 active fact for (Jordan Lee, annual_salary).
  - Expected: that fact contains the v2 value ($140k).
  - Old fact: valid_until is set (MESI state = INVALID).

Metrics (deterministic):
  cas_correctness    — 1.0 if exactly 1 active fact for the entity+attribute, else 0.0
  latest_surfaced    — 1.0 if the retrieved fact contains the v2 keyword, else 0.0

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s10_ma_mesi_cas"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
    top_k: int = 3,
) -> ScenarioResult:
    await backend.reset()

    await backend.add_structured(
        "agent_a",
        fixture.cas_fact_v1,
        entity=fixture.cas_entity,
        attribute=fixture.cas_attribute,
    )
    await backend.publish_to_pool("agent_a")

    count_after_v1 = await backend.pool_count_active_for_entity(
        fixture.cas_entity, fixture.cas_attribute
    )

    await backend.add_structured(
        "agent_b",
        fixture.cas_fact_v2,
        entity=fixture.cas_entity,
        attribute=fixture.cas_attribute,
    )
    await backend.publish_to_pool("agent_b")

    count_after_v2 = await backend.pool_count_active_for_entity(
        fixture.cas_entity, fixture.cas_attribute
    )
    cas_correctness = 1.0 if count_after_v2 == 1 else 0.0

    r = await backend.pool_retrieve("agent_c", fixture.cas_query, top_k=top_k)
    latest_surfaced = 1.0 if any(fixture.cas_v2_keyword in t for t in r.texts) else 0.0

    notes = (
        f"active_after_v1={count_after_v1}, "
        f"active_after_v2={count_after_v2} (expected=1), "
        f"cas_correctness={cas_correctness:.0%}, "
        f"latest_surfaced={latest_surfaced:.0%} (keyword={fixture.cas_v2_keyword!r})"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "cas_correctness": [cas_correctness],
            "latest_surfaced": [latest_surfaced],
        },
        insert_result=None,
        retrieval_result=r,
        notes=notes,
    )
