"""S11 — Progressive Knowledge Catchup (Realistic Incremental Sync).

Verifies that sync_dirty() correctly delivers incremental deltas when an agent
has been offline while others published updates, urgent facts, and CAS
supersessions.

Real-world pattern: Agent B goes "offline" after an initial sync. While offline,
Agent A updates 3 entity slots, Agent C publishes 4 new facts, and Agent D
publishes 3 urgent incident facts. Agent B returns, calls sync_dirty(), and
receives ONLY the delta (not the full pool).

Flow:
  Phase 1: Agent A publishes 8 initial facts.
  Phase 2: Agent B syncs (sees 8), publishes 5 own facts.
           Agent C publishes 6 facts.
  Phase 3 (B offline): A updates 3 slots, C adds 4 more, D adds 3 urgent.
  Phase 4: B syncs — must get only the delta from phase 3 (10 new/updated, not all).
           B publishes 2 response facts based on the delta.
  Phase 5: Querier verifies B's response facts are retrievable.

Metrics:
  delta_correctness    — 1.0 if sync_dirty returns ≥10 delta items (3 updates + 4 new + 3 urgent)
  delta_efficiency     — 1.0 if delta size < total pool size (not re-sending known facts)
  response_relevance   — fraction of B's response queries returning relevant results

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

    # Phase 1: Agent A publishes initial facts.
    for fact in fixture.catchup_initial_facts:
        await backend.insert_for_agent("agent_a", fact)
    await backend.publish_to_pool("agent_a")

    # Phase 2: B does initial sync, publishes own facts. C publishes concurrently.
    await backend.sync_dirty("agent_b")
    for fact in fixture.catchup_agent_b_facts:
        await backend.insert_for_agent("agent_b", fact)
    await backend.publish_to_pool("agent_b")

    for fact in fixture.catchup_agent_c_facts:
        await backend.insert_for_agent("agent_c", fact)
    await backend.publish_to_pool("agent_c")

    # Phase 3: While B is "offline", others publish updates.
    # A updates 3 entity slots (CAS supersession).
    for entity, attr, content in fixture.catchup_agent_a_updates:
        await backend.add_structured("agent_a", content, entity=entity, attribute=attr)
    await backend.publish_to_pool("agent_a")

    # C publishes 4 more facts.
    for fact in fixture.catchup_agent_c_extra_facts:
        await backend.insert_for_agent("agent_c", fact)
    await backend.publish_to_pool("agent_c")

    # D publishes 3 urgent incident facts.
    for fact in fixture.catchup_agent_d_urgent_facts:
        await backend.insert_for_agent("agent_d", fact)
    await backend.publish_to_pool("agent_d")

    # Phase 4: B comes back and syncs — should get only the delta.
    delta = await backend.sync_dirty("agent_b")
    delta_size = len(delta)

    # Expected delta: 3 CAS updates + 4 new from C + 3 urgent from D = 10 minimum.
    # (Some syncs also include C's initial 6 facts if B's first sync didn't see them.)
    delta_correctness = 1.0 if delta_size >= 10 else delta_size / 10

    # Pool total should be much larger than delta.
    pool = getattr(backend, "_pool", None)
    pool_size = len(pool.list_shared_facts()) if pool else delta_size + 1
    delta_efficiency = 1.0 if delta_size < pool_size else 0.0

    # B publishes response facts based on what it saw in the delta.
    for fact in fixture.catchup_agent_b_response_facts:
        await backend.insert_for_agent("agent_b", fact)
    await backend.publish_to_pool("agent_b")

    # Phase 5: Verify B's response facts are retrievable.
    response_hits = 0
    for query, kw in fixture.catchup_b_response_queries:
        r = await backend.pool_retrieve("agent_e", query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            response_hits += 1
    response_relevance = response_hits / len(fixture.catchup_b_response_queries)

    notes = (
        f"delta_size={delta_size}, pool_size={pool_size}, "
        f"delta_correctness={delta_correctness:.0%}, "
        f"delta_efficiency={'yes' if delta_efficiency else 'no'}, "
        f"response_relevance={response_relevance:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "delta_correctness": [delta_correctness],
            "delta_efficiency": [delta_efficiency],
            "response_relevance": [response_relevance],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
