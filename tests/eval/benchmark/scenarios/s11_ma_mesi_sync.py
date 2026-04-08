"""S11 — MESI Lazy Sync (Incremental Dirty Pull).

Verifies token-efficient incremental synchronisation: sync_dirty() returns
only facts that changed since the last sync, not the entire pool.

Flow:
  1. Agent A publishes 5 facts to the pool.
  2. Agent B calls sync_dirty() → receives all 5 facts (first sync).
  3. Agent A publishes an updated version of 1 fact (same entity+attribute).
  4. Agent B calls sync_dirty() again → receives only the 1 changed fact.

Metrics (deterministic):
  initial_sync_completeness  — fraction of initial facts received in first sync
                                (should be 1.0)
  incremental_efficiency     — 1 - (facts_returned_in_second_sync / pool_size)
                                (1.0 = only dirty facts returned; 0.0 = full broadcast)

arXiv 2603.15183 reports 95% token savings with MESI lazy invalidation vs
broadcast.  This scenario measures that savings empirically.

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s11_ma_mesi_sync"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    for fact in fixture.sync_initial_facts:
        await backend.insert_for_agent("agent_a", fact)
    # The updatable fact is added with entity+attribute so CAS can track its version.
    await backend.add_structured(
        "agent_a",
        fixture.sync_fact_v1,
        entity=fixture.sync_update_entity,
        attribute=fixture.sync_update_attribute,
    )
    await backend.publish_to_pool("agent_a")

    pool_size_initial = len(fixture.sync_initial_facts) + 1  # +1 for the structured fact

    first_sync = await backend.sync_dirty("agent_b")
    initial_sync_completeness = min(1.0, len(first_sync) / max(pool_size_initial, 1))

    await backend.add_structured(
        "agent_a",
        fixture.sync_fact_v2,
        entity=fixture.sync_update_entity,
        attribute=fixture.sync_update_attribute,
    )
    await backend.publish_to_pool("agent_a")

    second_sync = await backend.sync_dirty("agent_b")
    # 1.0 = only the 1 changed fact returned; 0.0 = entire pool re-broadcast.
    incremental_efficiency = max(
        0.0, min(1.0, 1.0 - len(second_sync) / max(pool_size_initial + 1, 1))
    )

    # fixture.sync_v2_keyword is unique to v2 (absent from v1 and initial facts).
    v2_in_sync = any(fixture.sync_v2_keyword.lower() in t.lower() for t in second_sync)

    notes = (
        f"pool_size={pool_size_initial}, "
        f"first_sync_count={len(first_sync)}, "
        f"initial_sync_completeness={initial_sync_completeness:.2%}, "
        f"second_sync_count={len(second_sync)}, "
        f"incremental_efficiency={incremental_efficiency:.2%}, "
        f"v2_in_second_sync={v2_in_sync}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "initial_sync_completeness": [initial_sync_completeness],
            "incremental_efficiency": [incremental_efficiency],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
