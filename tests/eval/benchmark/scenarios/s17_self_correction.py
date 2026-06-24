"""S17 — Self-Correction via Sync.

Verifies that an agent can detect its own supersession via sync_dirty() and
publish a corrected version, completing a full self-correction loop.

This tests a core real-world pattern: an agent discovers that its published
knowledge was overridden by a peer, understands it was wrong, and corrects itself.

Flow:
  Step 1: Agent A publishes fact_v1 for (entity, attribute) — the wrong value.
  Step 2: Agent B detects the error, publishes fact_v2 for the same slot.
          This supersedes A's fact (MESI CAS: A's fact → INVALID).
  Step 3: Agent A calls sync_dirty() and sees its own fact was invalidated.
          A publishes fact_v3 — a corrected version aligned with B's direction.
          (v3 supersedes v2: only one active fact remains.)
  Step 4: Agent C queries pool → must retrieve fact_v3 (A's corrected version).
          Agent A's trust after self-correction must exceed trust after step 2.

Metrics (deterministic):
  correction_surfaced — 1.0 if C's query returns a fact containing v3_keyword
  trust_recovery      — 1.0 if trust_a after step 4 > trust_a after step 2
  version_count       — 1.0 if exactly 3 versions exist in the slot chain

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.fixtures import MULTI_AGENT_FIXTURE, MultiAgentFixture
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s17_self_correction"


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    fixture: MultiAgentFixture = MULTI_AGENT_FIXTURE,
) -> ScenarioResult:
    await backend.reset()

    pool = getattr(backend, "_pool", None)

    # Step 1: Agent A publishes wrong initial value.
    await backend.add_structured(
        "agent_a",
        fixture.self_corr_v1,
        entity=fixture.self_corr_entity,
        attribute=fixture.self_corr_attribute,
    )
    await backend.publish_to_pool("agent_a")

    # Step 2: Agent B corrects it — supersedes A's fact.
    await backend.add_structured(
        "agent_b",
        fixture.self_corr_v2,
        entity=fixture.self_corr_entity,
        attribute=fixture.self_corr_attribute,
    )
    await backend.publish_to_pool("agent_b")

    trust_a_after_supersession = pool.get_trust("agent_a") if pool else 1.0

    # Step 3: Agent A syncs, detects supersession, publishes corrected version.
    await backend.sync_dirty("agent_a")
    await backend.add_structured(
        "agent_a",
        fixture.self_corr_v3,
        entity=fixture.self_corr_entity,
        attribute=fixture.self_corr_attribute,
    )
    await backend.publish_to_pool("agent_a")

    # Step 4: Agent C queries pool — must find A's corrected fact.
    result = await backend.pool_retrieve("agent_c", fixture.self_corr_query, top_k=5)
    correction_surfaced = (
        1.0 if any(fixture.self_corr_v3_keyword.lower() in t.lower() for t in result.texts) else 0.0
    )

    trust_a_after_correction = pool.get_trust("agent_a") if pool else 1.0
    trust_recovery = 1.0 if trust_a_after_correction > trust_a_after_supersession else 0.0

    # Count versions in the slot chain.
    version_count_ok = 0.0
    if pool is not None:
        entity_lower = fixture.self_corr_entity.lower().strip()
        attr_lower = fixture.self_corr_attribute.lower().strip()
        total_versions = sum(
            1
            for f in pool.list_shared_facts()
            if f.entity.lower().strip() == entity_lower
            and f.attribute.lower().strip() == attr_lower
        )
        version_count_ok = 1.0 if total_versions == 3 else 0.0

    notes = (
        f"trust_after_supersession={trust_a_after_supersession:.3f}, "
        f"trust_after_correction={trust_a_after_correction:.3f}, "
        f"correction_surfaced={correction_surfaced:.0%}, "
        f"trust_recovery={trust_recovery:.0%}, "
        f"version_count_ok={version_count_ok:.0%}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores={
            "correction_surfaced": [correction_surfaced],
            "trust_recovery": [trust_recovery],
            "version_count": [version_count_ok],
        },
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
