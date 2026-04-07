"""S13 — Concurrent Writers (Slot-Key Integrity Under Parallel Publish).

Verifies that when multiple agents publish facts for the same entity+attribute
slot in rapid succession, the shared pool maintains a correct version chain:
exactly one active fact remains and all superseded versions are properly closed.

Flow:
  1. N_AGENTS agents each insert a fact for the same entity+attribute slot
     (salary update) with a distinct value, plus NOISE_PER_AGENT unrelated facts.
  2. All agents publish to the pool sequentially (simulating serial resolution
     of concurrent intent — the threading.Lock in SharedMemoryPool guarantees
     this serialization in production; here we test the semantic invariants).
  3. After all publishes:
     - Count active facts for the slot → must be exactly 1 (no_lost_updates).
     - Count total facts (active + invalid) for the slot → must equal N_AGENTS
       (version_chain_integrity: every write was recorded, none silently dropped).

Metrics (deterministic):
  no_lost_updates          — 1.0 if exactly 1 active fact for the slot, else 0.0
  version_chain_integrity  — 1.0 if total slot versions == N_AGENTS, else 0.0

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

import asyncio

from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s13_concurrent_writers"

_N_AGENTS = 4
_NOISE_PER_AGENT = 5

_SLOT_ENTITY = "Jordan Lee"
_SLOT_ATTRIBUTE = "annual_salary"

# Each agent publishes a different salary value for the same person.
_SALARY_FACTS = [
    "Jordan Lee's annual salary is $95,000.",
    "Jordan Lee's annual salary is $110,000.",
    "Jordan Lee's annual salary is $125,000.",
    "Jordan Lee's annual salary is $140,000.",
]

_NOISE_FACTS = [
    "The deployment pipeline uses GitHub Actions.",
    "Python 3.12 is the standard runtime for all services.",
    "All services expose Prometheus /metrics endpoints.",
    "Database migrations run via Alembic before each release.",
    "Feature flags are managed through LaunchDarkly.",
    "Redis is used for rate limiting at the API gateway.",
    "All logs are shipped to OpenSearch via Fluent Bit.",
    "gRPC is preferred over REST for inter-service calls.",
    "Secrets are stored in HashiCorp Vault.",
    "Load tests run nightly against the staging environment.",
    "The monorepo uses Bazel for hermetic builds.",
    "All APIs require mTLS in production.",
    "Database connection pools are capped at 20 per service.",
    "SLO alerts page via PagerDuty.",
    "Canary deployments use a 5% traffic split initially.",
    "Container images are scanned with Trivy before push.",
    "All S3 buckets enforce server-side AES-256 encryption.",
    "Service mesh is Istio 1.20.",
    "Code coverage must stay above 80% on the main branch.",
    "GraphQL subscriptions are served via WebSockets.",
]


async def run(backend: MultiAgentMemoryBackend, judge: BaseJudge) -> ScenarioResult:
    await backend.reset()

    agent_ids = [f"agent_{chr(ord('a') + i)}" for i in range(_N_AGENTS)]

    # Each agent inserts its salary fact (structured) + noise facts.
    for i, agent_id in enumerate(agent_ids):
        await backend.add_structured(
            agent_id,
            _SALARY_FACTS[i],
            entity=_SLOT_ENTITY,
            attribute=_SLOT_ATTRIBUTE,
        )
        offset = (i * _NOISE_PER_AGENT) % len(_NOISE_FACTS)
        for j in range(_NOISE_PER_AGENT):
            await backend.insert_for_agent(agent_id, _NOISE_FACTS[(offset + j) % len(_NOISE_FACTS)])

    # All agents publish concurrently — tests real slot-key integrity under parallel writes.
    await asyncio.gather(*[backend.publish_to_pool(agent_id) for agent_id in agent_ids])

    # Measure slot integrity.
    active_count = await backend.pool_count_active_for_entity(_SLOT_ENTITY, _SLOT_ATTRIBUTE)
    total_count = await _count_all_slot_versions(backend, _SLOT_ENTITY, _SLOT_ATTRIBUTE)

    no_lost_updates = 1.0 if active_count == 1 else 0.0
    # version_chain_integrity requires access to pool internals; skip for backends
    # that don't expose _pool (total_count == -1).
    version_chain_integrity = (
        1.0 if total_count == _N_AGENTS else 0.0 if total_count != -1 else None
    )

    notes = (
        f"agents={_N_AGENTS}, "
        f"active_slot_count={active_count} (expected=1), "
        f"total_slot_versions={total_count} (expected={_N_AGENTS}), "
        f"no_lost_updates={no_lost_updates:.0%}"
        + (
            f", version_chain_integrity={version_chain_integrity:.0%}"
            if version_chain_integrity is not None
            else ""
        )
    )

    judge_scores: dict[str, list[float]] = {"no_lost_updates": [no_lost_updates]}
    if version_chain_integrity is not None:
        judge_scores["version_chain_integrity"] = [version_chain_integrity]

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )


async def _count_all_slot_versions(
    backend: MultiAgentMemoryBackend, entity: str, attribute: str
) -> int:
    """Count active + invalidated versions for a slot.

    pool_count_active_for_entity counts only active facts; we need ALL versions
    (including INVALID) to verify every write was recorded. No public API exists
    for this, so we access _pool.list_shared_facts() directly.
    Returns -1 when pool internals are not accessible (non-ai-knot backends).
    """
    pool = getattr(backend, "_pool", None)
    if pool is None:
        return -1

    entity_lower = entity.lower().strip()
    attribute_lower = attribute.lower().strip()
    return sum(
        1
        for f in pool.list_shared_facts()
        if f.entity.lower().strip() == entity_lower
        and f.attribute.lower().strip() == attribute_lower
    )
