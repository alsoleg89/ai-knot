"""S10 — MESI Entity-Addressed CAS (Conflict Resolution).

Verifies that when four agents publish facts about the same entity+attribute
sequentially, the shared pool retains exactly ONE active version — the last one.

This tests the entity-addressed Compare-And-Swap (CAS) in SharedMemoryPool:
  - Agent A publishes salary v1 ($95k).
  - Agent B publishes salary v2 ($110k) — supersedes v1.
  - Agent C publishes salary v3 ($125k) — supersedes v2.
  - Agent D publishes salary v4 ($140k, promotion to Staff) — supersedes v3.

Expected after all 4 publishes:
  - Exactly 1 active fact for (Jordan Lee, annual_salary).
  - That fact contains the v4 keyword ("Staff").
  - All previous versions have valid_until set (MESI state = INVALID).

Metrics (deterministic):
  cas_correctness  — 1.0 if exactly 1 active fact after all 4 publishes
  latest_surfaced  — 1.0 if the retrieved fact contains the v4 keyword

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

    # Publish 4 versions sequentially — each supersedes the previous.
    versions = [
        ("agent_a", fixture.cas_fact_v1),
        ("agent_b", fixture.cas_fact_v2),
        ("agent_c", fixture.cas_fact_v3),
        ("agent_d", fixture.cas_fact_v4),
    ]

    for agent_id, fact in versions:
        await backend.add_structured(
            agent_id,
            fact,
            entity=fixture.cas_entity,
            attribute=fixture.cas_attribute,
        )
        await backend.publish_to_pool(agent_id)

    # After all 4 publishes: exactly 1 active fact, containing v4 keyword.
    active_count = await backend.pool_count_active_for_entity(
        fixture.cas_entity, fixture.cas_attribute
    )
    cas_correctness = 1.0 if active_count == 1 else 0.0

    # Verify v4 is the active version.
    r = await backend.pool_retrieve("agent_a", fixture.cas_query, top_k=top_k)
    latest_surfaced = (
        1.0 if any(fixture.cas_v4_keyword.lower() in t.lower() for t in r.texts) else 0.0
    )

    notes = (
        f"versions=4, "
        f"active_count={active_count} (expected=1), "
        f"cas_correctness={cas_correctness:.0%}, "
        f"latest_surfaced={latest_surfaced:.0%} (v4_keyword={fixture.cas_v4_keyword!r})"
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
