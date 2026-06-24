"""S20 — Belief Revision (Multi-Round Contradiction Resolution).

Verifies that agents can converge on a shared belief through a multi-round
process of contradiction, authoritative correction, and supersession.

This tests a realistic knowledge-management pattern: agents start with
conflicting beliefs, an authoritative source publishes the correct value,
agents sync and align, and the system tracks further updates correctly.

Flow:
  Round 1: Agent A publishes "headcount = 5"; Agent B publishes "headcount = 8".
           (Conflict: two active facts for the same slot — last-write wins via CAS.)
  Round 2: Agent C (authoritative HR system) publishes "headcount = 7".
           Supersedes whichever of A/B was last; 1 active fact remains.
  Round 3: Agent A syncs → detects its fact was superseded → republishes "7".
           Agent B syncs → detects its fact was superseded → republishes "7".
  Round 4: Agent D queries pool → must retrieve a fact containing "7".
  Round 5: Agent C publishes correction "headcount = 6" (one resignation).
           Agent D queries again → must retrieve a fact containing "6".

Metrics (deterministic):
  convergence_ok  — 1.0 if exactly 1 active fact for the slot after round 3
  round4_recall   — 1.0 if D retrieves the value "7" in round 4
  round5_recall   — 1.0 if D retrieves the value "6" in round 5
  final_consensus — 1.0 if exactly 1 active fact for the slot after round 5

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s20_belief_revision"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    entity = fixture.belief_entity
    attribute = fixture.belief_attribute

    # Round 1: A and B publish conflicting beliefs.
    await backend.add_structured("agent_a", fixture.belief_v_a, entity=entity, attribute=attribute)
    await backend.publish_to_pool("agent_a")
    await backend.add_structured("agent_b", fixture.belief_v_b, entity=entity, attribute=attribute)
    await backend.publish_to_pool("agent_b")

    # Round 2: C (authoritative) publishes correct value — supersedes last writer.
    await backend.add_structured("agent_c", fixture.belief_v_c, entity=entity, attribute=attribute)
    await backend.publish_to_pool("agent_c")

    # Round 3: A and B sync, detect supersession, align with authoritative value.
    await backend.sync_dirty("agent_a")
    await backend.add_structured("agent_a", fixture.belief_v_c, entity=entity, attribute=attribute)
    await backend.publish_to_pool("agent_a")

    await backend.sync_dirty("agent_b")
    await backend.add_structured("agent_b", fixture.belief_v_c, entity=entity, attribute=attribute)
    await backend.publish_to_pool("agent_b")

    # Check convergence: exactly 1 active fact after round 3.
    active_after_r3 = await backend.pool_count_active_for_entity(entity, attribute)
    convergence_ok = 1.0 if active_after_r3 == 1 else 0.0

    # Round 4: D queries → must find "7".
    r4 = await backend.pool_retrieve(_QUERIER, fixture.belief_query, top_k=5)
    round4_recall = 1.0 if any(fixture.belief_keyword_round4 in t for t in r4.texts) else 0.0

    # Round 5: C publishes correction (one resignation → headcount = 6).
    await backend.add_structured(
        "agent_c", fixture.belief_v_final, entity=entity, attribute=attribute
    )
    await backend.publish_to_pool("agent_c")

    r5 = await backend.pool_retrieve(_QUERIER, fixture.belief_query, top_k=5)
    round5_recall = 1.0 if any(fixture.belief_keyword_round5 in t for t in r5.texts) else 0.0

    active_after_r5 = await backend.pool_count_active_for_entity(entity, attribute)
    final_consensus = 1.0 if active_after_r5 == 1 else 0.0

    notes = (
        f"rounds=5, entity={entity}, attribute={attribute}, "
        f"active_after_r3={active_after_r3}, convergence_ok={convergence_ok:.0%}, "
        f"round4_recall={round4_recall:.0%} (keyword={fixture.belief_keyword_round4!r}), "
        f"round5_recall={round5_recall:.0%} (keyword={fixture.belief_keyword_round5!r}), "
        f"final_consensus={final_consensus:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "convergence_ok": [convergence_ok],
            "round4_recall": [round4_recall],
            "round5_recall": [round5_recall],
            "final_consensus": [final_consensus],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
