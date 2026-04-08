"""S13 — Concurrent Writers (Slot-Key Integrity Under Parallel Publish).

Verifies that when multiple agents publish facts for the same entity+attribute
slot in rapid succession, the shared pool maintains a correct version chain:
exactly one active fact remains and all superseded versions are properly closed.

Flow:
  1. N_AGENTS agents each insert a fact for the same entity+attribute slot
     (salary update) with a distinct value, plus NOISE_PER_AGENT unrelated facts.
  2. All agents publish to the pool concurrently via ThreadPoolExecutor — each
     call runs in its own OS thread, genuinely contending for
     SharedMemoryPool._publish_lock.  This exercises the real serialisation
     path (threading.Lock + AtomicUpdateCapable/BEGIN EXCLUSIVE where available).
  3. After all publishes:
     - Count active facts for the slot → must be exactly 1 (no_lost_updates).
     - Count total facts (active + invalid) for the slot → must equal N_AGENTS
       (version_chain_integrity: every write was recorded, none silently dropped).

Why threading and not asyncio.gather:
  pool.publish() is synchronous.  asyncio.gather on async wrappers around it
  gives the event loop no await points to switch on — calls execute in strict
  sequence and _publish_lock is never contended.  ThreadPoolExecutor forces
  N_AGENTS OS threads to race for the lock simultaneously.

  For backends that do not expose _pool/_kbs, the test falls back to sequential
  publish_to_pool() calls (semantic invariants are still verified).

Metrics (deterministic):
  no_lost_updates          — 1.0 if exactly 1 active fact for the slot, else 0.0
  version_chain_integrity  — 1.0 if total slot versions == N_AGENTS, else 0.0
  used_threads             — 1.0 if real OS threads were used, 0.0 if fallback

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

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

    # Publish concurrently via real OS threads so _publish_lock is genuinely contended.
    # Backends that expose _pool/_kbs get the threaded path; others fall back to sequential.
    pool = getattr(backend, "_pool", None)
    kbs: dict = getattr(backend, "_kbs", {})
    used_threads = pool is not None and bool(kbs)

    if used_threads:
        barrier = threading.Barrier(_N_AGENTS)

        def _publish_sync(agent_id: str) -> None:
            kb = kbs[agent_id]
            fact_ids = [f.id for f in kb.list_facts() if f.is_active()]
            barrier.wait()  # all threads start publish() simultaneously
            pool.publish(agent_id, fact_ids, kb=kb, utility_threshold=0.0)  # type: ignore[union-attr]

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=_N_AGENTS) as executor:
            await asyncio.gather(
                *[loop.run_in_executor(executor, _publish_sync, aid) for aid in agent_ids]
            )
    else:
        for agent_id in agent_ids:
            await backend.publish_to_pool(agent_id)

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
        f"threads={'yes' if used_threads else 'no (fallback)'}, "
        f"active_slot_count={active_count} (expected=1), "
        f"total_slot_versions={total_count} (expected={_N_AGENTS}), "
        f"no_lost_updates={no_lost_updates:.0%}"
        + (
            f", version_chain_integrity={version_chain_integrity:.0%}"
            if version_chain_integrity is not None
            else ""
        )
    )

    judge_scores: dict[str, list[float]] = {
        "no_lost_updates": [no_lost_updates],
        "used_threads": [1.0 if used_threads else 0.0],
    }
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
