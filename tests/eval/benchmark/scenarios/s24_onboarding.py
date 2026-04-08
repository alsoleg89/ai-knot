"""S24 — Multi-Round Knowledge Onboarding (KB Absorption).

Verifies that an agent can absorb knowledge from the shared pool into its
private KB and later answer questions from its own KB without pool access.

This tests a realistic onboarding pattern: a new team member queries shared
knowledge, saves key facts locally, and can later reference them independently.

Flow:
  Phase 1: Agents A–C publish 15 domain facts to the shared pool.
  Phase 2 (Round 1): Agent D queries pool for 5 topics, retrieves answers.
  Phase 3: Agent D absorbs retrieved facts into its private KB (uses actual
           retrieval results, not pre-scripted facts).
  Phase 4 (Round 3): Agent D queries its own KB for 3 previously learned topics.
           Must retrieve the absorbed facts from private KB (not pool).

Metrics:
  pool_retrieval_recall — fraction of round 1 pool queries that found relevant results
  kb_absorption         — fraction of round 3 KB queries answered from private KB
  retention_coverage    — fraction of all queries (pool + KB) that found relevant results

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s24_onboarding"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 1: Team agents publish baseline knowledge to pool.
    n = len(fixture.onboarding_pool_facts)
    third = n // 3
    agent_slices = [
        ("agent_a", fixture.onboarding_pool_facts[:third]),
        ("agent_b", fixture.onboarding_pool_facts[third : 2 * third]),
        ("agent_c", fixture.onboarding_pool_facts[2 * third :]),
    ]
    for agent_id, facts in agent_slices:
        for fact in facts:
            await backend.insert_for_agent(agent_id, fact)
        await backend.publish_to_pool(agent_id)

    # Phase 2 (Round 1): Agent D queries pool — learning phase.
    # Collect retrieved Facts for later structured absorption into private KB.
    from ai_knot.types import Fact as _Fact  # local import to avoid base.py cycle

    pool_hits = 0
    absorbed_facts: list[_Fact] = []
    seen_ids: set[str] = set()
    for query, kw in fixture.onboarding_round1_queries:
        r = await backend.pool_retrieve("agent_d", query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            pool_hits += 1
        for f in r.facts:
            if f.id not in seen_ids:
                seen_ids.add(f.id)
                absorbed_facts.append(f)
    pool_retrieval_recall = pool_hits / len(fixture.onboarding_round1_queries)

    # Phase 3: Agent D absorbs retrieved Facts into private KB with full metadata.
    await backend.absorb_from_pool("agent_d", absorbed_facts)

    # Phase 4 (Round 3): Agent D queries its own KB — retention test.
    kb_hits = 0
    for query, kw in fixture.onboarding_round3_kb_queries:
        r = await backend.retrieve_for_agent("agent_d", query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            kb_hits += 1
    kb_absorption = kb_hits / len(fixture.onboarding_round3_kb_queries)

    retention_coverage = (pool_hits + kb_hits) / (
        len(fixture.onboarding_round1_queries) + len(fixture.onboarding_round3_kb_queries)
    )

    notes = (
        f"pool_facts=15, absorbed={len(absorbed_facts)}, "
        f"pool_recall={pool_retrieval_recall:.0%}, "
        f"kb_absorption={kb_absorption:.0%}, "
        f"retention_coverage={retention_coverage:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "pool_retrieval_recall": [pool_retrieval_recall],
            "kb_absorption": [kb_absorption],
            "retention_coverage": [retention_coverage],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
