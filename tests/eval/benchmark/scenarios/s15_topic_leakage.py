"""S15 — Cross-Team Signal Contamination (Shared Terminology Isolation).

Verifies that per-channel pool retrieval returns facts only from the
requested channel, even when multiple channels contain the same terminology
(e.g., "deployment", "latency", "monitoring" appear in devops, backend,
and data channels).

Three agents publish to 3 channels with 2 "shared-term" facts each:
  Agent A → devops  (6 facts, 2 mention "deployment" and "latency")
  Agent B → backend (6 facts, 2 mention "deployment" and "latency")
  Agent C → data    (6 facts, 2 mention "monitoring" and "latency")

Queries test both channel-specific and shared-term retrieval.

Metrics:
  channel_precision       — fraction of results from correct channel (standard queries)
  shared_term_isolation   — fraction of shared-term queries returning correct channel
  cross_contamination_rate — fraction of results from WRONG channel (lower=better)

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s15_topic_leakage"

_CHANNELS = {
    "agent_a": "devops",
    "agent_b": "backend",
    "agent_c": "data",
}
_QUERIER = "agent_d"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    channel_facts = {
        "agent_a": (fixture.contamination_devops_facts, "devops"),
        "agent_b": (fixture.contamination_backend_facts, "backend"),
        "agent_c": (fixture.contamination_data_facts, "data"),
    }

    # Phase 1: Insert facts with channel metadata.
    for agent_id, (facts, channel) in channel_facts.items():
        for fact in facts:
            await backend.insert_for_agent_with_meta(
                agent_id, fact, topic_channel=channel, importance=0.8
            )

    # Publish sequentially — each team publishes its own channel.
    for agent_id in channel_facts:
        await backend.publish_to_pool(agent_id)

    # Phase 2: Standard channel-specific queries.
    channel_correct = 0
    channel_total = 0
    for query, channel, kw in fixture.contamination_channel_queries:
        r = await backend.pool_retrieve_for_channel(_QUERIER, query, top_k=5, topic_channel=channel)
        channel_total += 1
        if any(kw.lower() in t.lower() for t in r.texts):
            channel_correct += 1

    channel_precision = channel_correct / max(channel_total, 1)

    # Phase 3: Shared-term queries — terms that exist in multiple channels.
    shared_correct = 0
    shared_total = 0
    cross_contamination = 0
    total_results = 0

    for query, channel, excl_kw in fixture.contamination_shared_term_queries:
        r = await backend.pool_retrieve_for_channel(_QUERIER, query, top_k=5, topic_channel=channel)
        shared_total += 1
        if any(excl_kw.lower() in t.lower() for t in r.texts):
            shared_correct += 1

        # Count cross-contamination: results that clearly belong to another channel.
        other_channel_kws = {
            kw.lower() for q, ch, kw in fixture.contamination_channel_queries if ch != channel
        }
        for t in r.texts:
            total_results += 1
            if any(ok in t.lower() for ok in other_channel_kws):
                cross_contamination += 1

    shared_term_isolation = shared_correct / max(shared_total, 1)
    cross_contamination_rate = cross_contamination / max(total_results, 1)

    notes = (
        f"channels=3, shared_terms=3, "
        f"channel_precision={channel_precision:.0%}, "
        f"shared_term_isolation={shared_term_isolation:.0%}, "
        f"cross_contamination_rate={cross_contamination_rate:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "channel_precision": [channel_precision],
            "shared_term_isolation": [shared_term_isolation],
            "cross_contamination_rate": [cross_contamination_rate],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
