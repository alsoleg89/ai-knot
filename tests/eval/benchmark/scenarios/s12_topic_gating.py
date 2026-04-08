"""S12 — Priority Triage Under Load (Dynamic Utility Threshold).

Verifies that publish gating correctly filters facts by urgency level during
an incident, and that lowering the threshold after resolution surfaces routine
facts previously excluded.

Four agents each publish 6 facts across 3 urgency levels:
  Critical (importance=0.85): active outages, service failures
  Routine  (importance=0.45): scheduled deployments, planned reviews
  Noise    (importance=0.15): office logistics, minor UI tickets

Flow:
  Phase 1: All agents insert 6 facts each (24 total) with tiered importance.
  Phase 2 (Incident): Publish with utility_threshold=0.6 → only critical facts enter pool.
  Phase 3: Incident commander queries pool — must find critical facts, not routine/noise.
  Phase 4 (Post-incident): Republish with threshold=0.3 → routine facts now enter.
  Phase 5: Routine queries — should now find scheduled items.

Metrics:
  triage_precision   — fraction of Phase 2 pool facts that are critical
  escalation_recall  — fraction of incident queries finding critical-keyword results
  post_incident_coverage — fraction of routine queries finding results after threshold drop

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s12_topic_gating"

_AGENTS = ["agent_a", "agent_b", "agent_c", "agent_d"]
_QUERIER = "agent_e"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    n_agents = len(_AGENTS)

    # Phase 1: Each agent inserts 2 critical + 2 routine + 2 noise facts.
    for i, agent_id in enumerate(_AGENTS):
        # Critical (importance=0.85)
        for fact in fixture.triage_critical_facts[i * 2 : (i + 1) * 2]:
            await backend.insert_for_agent_with_meta(agent_id, fact, importance=0.85)
        # Routine (importance=0.45)
        for fact in fixture.triage_routine_facts[i * 2 : (i + 1) * 2]:
            await backend.insert_for_agent_with_meta(agent_id, fact, importance=0.45)
        # Noise (importance=0.15)
        for fact in fixture.triage_noise_facts[i * 2 : (i + 1) * 2]:
            await backend.insert_for_agent_with_meta(agent_id, fact, importance=0.15)

    # Phase 2 (Incident): Publish with high threshold — only critical passes.
    total_published = 0
    for agent_id in _AGENTS:
        n = await backend.publish_to_pool(agent_id, utility_threshold=0.6)
        total_published += n

    # Expected: 8 critical facts pass (2 per agent × 4 agents).
    expected_critical = n_agents * 2
    triage_precision = min(1.0, expected_critical / max(total_published, 1))

    # Phase 3: Incident commander queries — must find critical facts.
    escalation_hits = 0
    for query, kw in fixture.triage_incident_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            escalation_hits += 1
    escalation_recall = escalation_hits / len(fixture.triage_incident_queries)

    # Phase 4 (Post-incident): Republish with lower threshold.
    for agent_id in _AGENTS:
        await backend.publish_to_pool(agent_id, utility_threshold=0.3)

    # Phase 5: Routine queries — should now find results.
    routine_hits = 0
    for query, kw in fixture.triage_routine_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        if any(kw.lower() in t.lower() for t in r.texts):
            routine_hits += 1
    post_incident_coverage = routine_hits / len(fixture.triage_routine_queries)

    notes = (
        f"agents={n_agents}, published_phase2={total_published}, "
        f"triage_precision={triage_precision:.0%}, "
        f"escalation_recall={escalation_recall:.0%}, "
        f"post_incident_coverage={post_incident_coverage:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "triage_precision": [triage_precision],
            "escalation_recall": [escalation_recall],
            "post_incident_coverage": [post_incident_coverage],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
