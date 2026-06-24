"""S8 — Multi-Team Knowledge Commons (Cross-Domain Pool Retrieval).

Verifies that a shared pool correctly aggregates knowledge from 4 teams with
deliberately overlapping domains, and returns multi-perspective answers for
overlapping topics while maintaining exclusivity for team-specific knowledge.

Four teams publish to the shared pool:
  Agent A (Platform):  infrastructure + shared API limits
  Agent B (Backend):   services + shared API limits (different angle)
  Agent C (Frontend):  UI + deployment facts
  Agent D (Data/ML):   pipelines + monitoring (shared tools)

Key design: domains INTENTIONALLY overlap (e.g., both Platform and Backend
know about API rate limits but from different perspectives; both Platform and
Data use Grafana for monitoring).

Sub-tests:
  A) Overlap coverage: queries on shared topics return facts from ≥2 teams.
  B) Exclusive recall: team-specific queries return only the owning team's facts.
  C) Cross-team depth: mean distinct keyword groups per overlap query.

Metrics:
  overlap_coverage   — fraction of overlap queries returning keywords from ≥2 agents
  exclusivity_recall — fraction of exclusive queries finding the correct team's keyword
  cross_team_depth   — mean keyword-group count across overlap queries

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s8_ma_isolation"

_QUERIER = "agent_e"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 1: All 4 teams publish their domain knowledge to the shared pool.
    teams = [
        ("agent_a", fixture.commons_platform_facts),
        ("agent_b", fixture.commons_backend_facts),
        ("agent_c", fixture.commons_frontend_facts),
        ("agent_d", fixture.commons_data_facts),
    ]
    for agent_id, facts in teams:
        for fact in facts:
            await backend.insert_for_agent(agent_id, fact)
        await backend.publish_to_pool(agent_id)

    # Phase 2: Overlap queries — expect facts from ≥2 agents per query.
    overlap_hits = 0
    total_depth = 0.0
    for query, kw_groups in fixture.commons_overlap_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        texts_lower = [t.lower() for t in r.texts]
        found_groups = [kw for kw in kw_groups if any(kw.lower() in t for t in texts_lower)]
        if len(found_groups) >= 2:
            overlap_hits += 1
        total_depth += len(found_groups)

    n_overlap = len(fixture.commons_overlap_queries)
    overlap_coverage = overlap_hits / n_overlap
    cross_team_depth = total_depth / n_overlap

    # Phase 3: Exclusive queries — only one team can answer.
    exclusive_hits = 0
    for query, kw in fixture.commons_exclusive_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            exclusive_hits += 1

    n_exclusive = len(fixture.commons_exclusive_queries)
    exclusivity_recall = exclusive_hits / n_exclusive

    notes = (
        f"teams=4, facts={sum(len(f) for _, f in teams)}, "
        f"overlap_coverage={overlap_coverage:.0%}, "
        f"exclusivity_recall={exclusivity_recall:.0%}, "
        f"cross_team_depth={cross_team_depth:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "overlap_coverage": [overlap_coverage],
            "exclusivity_recall": [exclusivity_recall],
            "cross_team_depth": [cross_team_depth],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
