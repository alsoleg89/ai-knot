"""S16 — Knowledge Relay (Chain Publishing).

Verifies that agents can build knowledge on top of what other agents published —
the core "collaborative team" pattern in multi-agent systems.

Each agent calls sync_dirty() to read what the previous agent published, then
inserts new facts that reference those concepts, and publishes to the shared pool.
Agent D (empty KB) must retrieve facts from all three layers.

Flow:
  Round 0: Agent A inserts 5 infra facts (argocd, grafana, redis, istio, helm)
           and publishes to pool.
  Round 1: Agent B calls sync_dirty() — sees A's facts.
           B inserts 5 API facts that reference A's service names (redis, argocd).
           B publishes to pool.
  Round 2: Agent C calls sync_dirty() — sees A + B facts.
           C inserts 5 frontend facts that reference B's API patterns (openapi).
           C publishes to pool.
  Query:   Agent D (empty KB) queries pool with 3 targeted questions:
             Q1 → expects A's fact  (keyword: argocd)
             Q2 → expects B's fact  (keyword: redis)
             Q3 → expects C's fact  (keyword: openapi)

Metrics (deterministic):
  layer_a_recall  — 1.0 if D finds A's fact
  layer_b_recall  — 1.0 if D finds B's fact
  layer_c_recall  — 1.0 if D finds C's fact
  chain_depth     — fraction of 3 layers present (0.33 / 0.67 / 1.0)

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

import asyncio

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s16_knowledge_relay"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Round 0: Agent A publishes infra facts.
    await asyncio.gather(
        *[backend.insert_for_agent("agent_a", fact) for fact in fixture.relay_infra_facts]
    )
    await backend.publish_to_pool("agent_a")

    # Round 1: Agent B syncs, then publishes API facts that reference A's services.
    await backend.sync_dirty("agent_b")
    await asyncio.gather(
        *[backend.insert_for_agent("agent_b", fact) for fact in fixture.relay_api_facts]
    )
    await backend.publish_to_pool("agent_b")

    # Round 2: Agent C syncs, then publishes frontend facts that reference B's API.
    await backend.sync_dirty("agent_c")
    await asyncio.gather(
        *[backend.insert_for_agent("agent_c", fact) for fact in fixture.relay_frontend_facts]
    )
    await backend.publish_to_pool("agent_c")

    # Query: Agent D (empty KB) retrieves from pool.
    results = await asyncio.gather(
        *[backend.pool_retrieve(_QUERIER, query, top_k=5) for query, _ in fixture.relay_queries]
    )

    layer_hits = [
        any(keyword.lower() in t.lower() for t in r.texts)
        for r, (_, keyword) in zip(results, fixture.relay_queries, strict=True)
    ]
    layer_a_recall = 1.0 if layer_hits[0] else 0.0
    layer_b_recall = 1.0 if layer_hits[1] else 0.0
    layer_c_recall = 1.0 if layer_hits[2] else 0.0
    chain_depth = sum(layer_hits) / 3

    notes = (
        f"layers=3 (A=infra, B=api, C=frontend), querier=agent_d, "
        f"layer_a={layer_a_recall:.0%}, layer_b={layer_b_recall:.0%}, "
        f"layer_c={layer_c_recall:.0%}, chain_depth={chain_depth:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "layer_a_recall": [layer_a_recall],
            "layer_b_recall": [layer_b_recall],
            "layer_c_recall": [layer_c_recall],
            "chain_depth": [chain_depth],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
