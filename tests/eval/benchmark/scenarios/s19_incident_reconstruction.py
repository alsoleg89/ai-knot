"""S19 — Incident Reconstruction (Multi-Phase Investigation with Noise).

Verifies that a shared pool can serve as the basis for collaborative incident
investigation with realistic noise: red herring facts from unrelated services,
coincidental timelines, and historical incidents are present alongside the
real evidence.

Flow:
  Phase 0 — Noise injection:
    5 red herring facts are published by agents before the real incident data.
  Phase 1 — Alert:
    Agent A inserts and publishes the alert observation.
  Phase 2 — Investigation:
    Agent B calls sync_dirty() (sees A's alert), inserts the deployment fact,
    and publishes.
    Agent C calls sync_dirty() (sees A + B), inserts the DB migration fact,
    and publishes.
  Phase 3 — Root cause query:
    Agent D (empty KB) queries the pool with 3 targeted questions.
    Must retrieve the 3 real evidence facts despite noise.

Metrics:
  alert_recall       — 1.0 if D finds A's alert fact (keyword: 14:32)
  deploy_recall      — 1.0 if D finds B's deployment fact (keyword: 14:28)
  migration_recall   — 1.0 if D finds C's migration fact (keyword: migration)
  evidence_recall    — fraction of 3 evidence types retrieved (0.33/0.67/1.0)
  evidence_precision — fraction of top-5 results containing relevant keywords

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

import asyncio

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s19_incident_reconstruction"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Phase 0 — Noise injection: red herring facts from various sources.
    for i, noise_fact in enumerate(fixture.incident_red_herring_facts):
        noise_agent = f"agent_{'abc'[i % 3]}"
        await backend.insert_for_agent(noise_agent, noise_fact)
    for agent_id in ("agent_a", "agent_b", "agent_c"):
        await backend.publish_to_pool(agent_id)

    # Reset agents A–C KBs for the actual incident data (new reset per agent
    # isn't available, but fresh facts get new IDs so the old noise stays in pool).

    # Phase 1 — Alert: Agent A publishes the initial alert.
    await backend.insert_for_agent("agent_a", fixture.incident_alert_fact)
    await backend.publish_to_pool("agent_a")

    # Phase 2 — Investigation: B and C each sync_dirty then contribute context.
    await backend.sync_dirty("agent_b")
    await backend.insert_for_agent("agent_b", fixture.incident_deploy_fact)
    await backend.publish_to_pool("agent_b")

    await backend.sync_dirty("agent_c")
    await backend.insert_for_agent("agent_c", fixture.incident_migration_fact)
    await backend.publish_to_pool("agent_c")

    # Phase 3 — Root cause queries by agent_d.
    results = await asyncio.gather(
        *[backend.pool_retrieve(_QUERIER, query, top_k=5) for query, _ in fixture.incident_queries]
    )

    keywords = [
        fixture.incident_alert_keyword,
        fixture.incident_deploy_keyword,
        fixture.incident_migration_keyword,
    ]
    hits = [
        any(kw.lower() in t.lower() for t in r.texts)
        for r, kw in zip(results, keywords, strict=True)
    ]
    alert_recall = 1.0 if hits[0] else 0.0
    deploy_recall = 1.0 if hits[1] else 0.0
    migration_recall = 1.0 if hits[2] else 0.0
    evidence_recall = sum(hits) / 3

    # Evidence precision: of all top-5 results across queries, how many are relevant?
    relevant_kws = {kw.lower() for kw in fixture.incident_relevant_keywords}
    total_results = 0
    relevant_results = 0
    for r in results:
        for t in r.texts:
            total_results += 1
            if any(kw in t.lower() for kw in relevant_kws):
                relevant_results += 1
    evidence_precision = relevant_results / max(total_results, 1)

    notes = (
        f"phases=3+noise (5 red herrings), querier=agent_d, "
        f"alert_recall={alert_recall:.0%}, deploy_recall={deploy_recall:.0%}, "
        f"migration_recall={migration_recall:.0%}, evidence_recall={evidence_recall:.2f}, "
        f"evidence_precision={evidence_precision:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "alert_recall": [alert_recall],
            "deploy_recall": [deploy_recall],
            "migration_recall": [migration_recall],
            "evidence_recall": [evidence_recall],
            "evidence_precision": [evidence_precision],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
