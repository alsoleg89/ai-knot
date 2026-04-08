"""S22 — Temporal Staleness Detection (Fresh Facts Must Win).

Verifies that when facts are updated across multiple rounds via CAS
(entity+attribute slots), queries consistently return the LATEST version.

Flow:
  Round 1: Agent A publishes v1 for 5 product config entities.
  Round 2: Agent B updates 3 of them to v2 (supersedes A via CAS).
  Round 3: Agent C updates 1 of them to v3 (supersedes B via CAS).
  Query:   Agent D queries all 5 entities — must get v3/v2/v1 as appropriate.

Metrics:
  freshness_recall     — fraction of queries returning the LATEST version keyword
  staleness_rejection  — fraction of queries NOT containing a stale keyword
  version_chain_ok    — 1.0 if each slot has exactly 1 active fact (CAS frontier correct)

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s22_temporal_staleness"

_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    # Round 1: Agent A publishes v1 for all 5 entities.
    for entity, attr, content, _kw in fixture.staleness_v1_facts:
        await backend.add_structured("agent_a", content, entity=entity, attribute=attr)
    await backend.publish_to_pool("agent_a")

    # Round 2: Agent B updates 3 entities to v2.
    for entity, attr, content, _kw in fixture.staleness_v2_updates:
        await backend.add_structured("agent_b", content, entity=entity, attribute=attr)
    await backend.publish_to_pool("agent_b")

    # Round 3: Agent C updates 1 entity to v3.
    for entity, attr, content, _kw in fixture.staleness_v3_updates:
        await backend.add_structured("agent_c", content, entity=entity, attribute=attr)
    await backend.publish_to_pool("agent_c")

    # Query phase: Agent D queries all 5 entities.
    fresh_hits = 0
    stale_misses = 0
    total = len(fixture.staleness_queries)

    for query, latest_kw, stale_kw in fixture.staleness_queries:
        r = await backend.pool_retrieve(_QUERIER, query, top_k=5)
        texts_lower = [t.lower() for t in r.texts]
        if any(latest_kw.lower() in t for t in texts_lower):
            fresh_hits += 1
        if not stale_kw or not any(stale_kw.lower() in t for t in texts_lower):
            stale_misses += 1

    freshness_recall = fresh_hits / total
    staleness_rejection = stale_misses / total

    # Verify version chain depth: entity 1 should have 3 versions (v1+v2+v3),
    # entities 2-3 should have 2 (v1+v2), entities 4-5 should have 1 (v1 only).
    depth_ok = 0
    for entity, attr, _content, _kw in fixture.staleness_v1_facts:
        count = await backend.pool_count_active_for_entity(entity, attr)
        if count == 1:  # CAS: only 1 active per slot regardless of version count
            depth_ok += 1
    version_chain_ok = 1.0 if depth_ok == 5 else depth_ok / 5

    notes = (
        f"rounds=3, entities=5, freshness_recall={freshness_recall:.0%}, "
        f"staleness_rejection={staleness_rejection:.0%}, "
        f"version_chain_ok={version_chain_ok:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "freshness_recall": [freshness_recall],
            "staleness_rejection": [staleness_rejection],
            "version_chain_ok": [version_chain_ok],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
