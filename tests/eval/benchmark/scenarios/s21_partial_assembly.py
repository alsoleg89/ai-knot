"""S21 — Distributed Product Knowledge Assembly (5 Specialists + 1 Querier).

Verifies that a shared pool correctly aggregates knowledge from 5 specialist
agents with partially overlapping domains, and that a querier agent can
retrieve cross-domain answers.

Each specialist publishes 5 facts about a different aspect of "Nexus Analytics
Platform".  The querier agent_f must find answers that require knowledge from
multiple agents.

Metrics:
  coverage           — fraction of 5 queries returning ≥1 relevant result
  cross_agent_recall — fraction of cross-domain queries returning keywords
                       from ≥2 different agents
  assembly_depth     — mean number of distinct keyword groups found per query

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s21_partial_assembly"

_QUERIER = "agent_f"
_AGENTS = ["agent_a", "agent_b", "agent_c", "agent_d", "agent_e"]


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 1: All 5 specialists publish their domain facts.
    facts_by_agent = [
        ("agent_a", fixture.assembly_technical_facts),
        ("agent_b", fixture.assembly_business_facts),
        ("agent_c", fixture.assembly_ops_facts),
        ("agent_d", fixture.assembly_historical_facts),
        ("agent_e", fixture.assembly_integration_facts),
    ]
    for agent_id, facts in facts_by_agent:
        for fact in facts:
            await backend.insert_for_agent(agent_id, fact)
        await backend.publish_to_pool(agent_id)

    # Phase 2: Querier agent_f retrieves cross-domain answers.
    coverage_hits = 0
    cross_agent_hits = 0
    total_depth = 0.0

    for query, relevant_kws in fixture.assembly_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        texts_lower = [t.lower() for t in r.texts]

        # Coverage: at least one keyword found in top-5.
        kw_found = [kw for kw in relevant_kws if any(kw.lower() in t for t in texts_lower)]
        if kw_found:
            coverage_hits += 1

        # Cross-agent recall: ≥2 different keywords found (from ≥2 agents).
        if len(kw_found) >= 2:
            cross_agent_hits += 1

        total_depth += len(kw_found)

    total = len(fixture.assembly_queries)
    coverage = coverage_hits / total
    cross_agent_recall = cross_agent_hits / total
    assembly_depth = total_depth / total

    notes = (
        f"agents=5, facts=25, queries={total}, "
        f"coverage={coverage:.0%}, cross_agent_recall={cross_agent_recall:.0%}, "
        f"assembly_depth={assembly_depth:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "coverage": [coverage],
            "cross_agent_recall": [cross_agent_recall],
            "assembly_depth": [assembly_depth],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
