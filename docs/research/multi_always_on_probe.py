"""Code-grounded multi-agent observations; not a production feature or test suite."""

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ai_knot import KnowledgeBase
from ai_knot.always_on import AlwaysOnMemory, VerificationOutcome, VerificationResult
from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

NOW = datetime(2035, 1, 1, tzinfo=UTC)


def fact(content, **kwargs):
    return Fact(content, created_at=NOW, last_accessed=NOW, valid_from=NOW, **kwargs)


def main():
    with tempfile.TemporaryDirectory() as folder:
        store = SQLiteStorage(str(Path(folder) / "memory.db"))
        pool = SharedMemoryPool(store, persist_stats=True, clock=lambda: NOW)
        agents = {
            name: AlwaysOnMemory(
                KnowledgeBase(name, storage=store, embed_url=""), pool=pool, clock=lambda: NOW
            )
            for name in ("researcher", "operator")
        }
        permits = []
        for name, memory in agents.items():
            item = memory.create_intent(client_key="deploy:42", goal="Deploy build 42")
            lease = memory.claim(item.id, worker_id=name + "-worker")
            action = memory.prepare_action(
                item.id,
                lease_token=lease.token,
                action_key="deploy",
                request={"target": "prod", "build": "42"},
            )
            permits.append(memory.begin_action(action.id, lease_token=lease.token))
        results = {
            "cross_agent_action": {
                "permits": len(permits),
                "distinct_idempotency_keys": len({p.idempotency_key for p in permits}),
            }
        }
        same_owner = agents["operator"]
        second = same_owner.create_intent(client_key="other-plan:42", goal="Deploy same build 42")
        lease = same_owner.claim(second.id, worker_id="another-worker")
        action = same_owner.prepare_action(
            second.id,
            lease_token=lease.token,
            action_key="deploy",
            request={"target": "prod", "build": "42"},
        )
        same_owner.begin_action(action.id, lease_token=lease.token)
        results["cross_intent_resource"] = {"second_intent_permitted_despite_other_started": True}
        shared = fact(
            "Deployment target",
            id="target",
            visibility_scope="team:ops",
            origin_agent_id="publisher",
            memory_tier="pool",
        )
        store.save("__shared__", [shared])
        for name in agents:
            pool.grant_read(name, "team:ops")
        a, b = agents["researcher"], agents["operator"]
        ar = a.capture_evidence((("__shared__", "target"),))[0]
        br = b.capture_evidence((("__shared__", "target"),))[0]
        a.set_freshness(ar, source_ref="primary-source", recheck_after_seconds=60)
        a.record_verification(
            ar,
            result=VerificationResult(VerificationOutcome.CONFIRMED, {"source": "primary"}, "v1"),
        )
        b.set_freshness(br, source_ref="operator-source", recheck_after_seconds=3600)
        results["shared_policy"] = {
            "source_seen_by_researcher": a.due_verifications()[0].source_ref,
            "policy_reset_by_other_reader": True,
        }
        calls = []
        entered = asyncio.Event()

        class Verifier:
            config_key = "same-verifier"
            model_cost = 1

            async def verify(self, request):
                calls.append(request.source_ref)
                if len(calls) == 2:
                    entered.set()
                await asyncio.wait_for(entered.wait(), timeout=5)
                return VerificationResult(
                    VerificationOutcome.CONFIRMED, {"source": request.source_ref}, "v2"
                )

        async def concurrent_checks():
            return await asyncio.gather(
                *(
                    run_maintenance(
                        memory,
                        run_key="same-tick",
                        budget=MaintenanceBudget(max_model_calls=1, max_wall_seconds=5),
                        verifier=Verifier(),
                    )
                    for memory in agents.values()
                )
            )

        reports = asyncio.run(concurrent_checks())
        results["duplicate_verification"] = {
            "provider_calls": len(calls),
            "statuses": [r.status for r in reports],
            "reserved_calls": sum(r.model_calls_reserved for r in reports),
            "committed_verifications": sum(r.verified for r in reports),
        }
        pools = [SharedMemoryPool(store, persist_stats=True, clock=lambda: NOW) for _ in range(2)]
        for p in pools:
            p.register("researcher")
            p.register("operator")
        for i, name in enumerate(("researcher", "operator")):
            kb = KnowledgeBase(name + "-stats", storage=store, embed_url="")
            item = fact("Independent publish " + name, id="stats-" + name)
            store.save(name + "-stats", [item])
            pools[i].publish(name, [item.id], kb=kb)
        results["pool_stats"] = {
            "durable_publish_count": store.load_pool_stats().get("publish_count"),
            "durable_publish_events": len(
                [e for e in store.load_trust_events() if e["event_type"] == "publish"]
            ),
        }
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
