#!/usr/bin/env python3
"""Local multi-agent load probe; fresh SQLite database or PostgreSQL schema only.

Uses independent Python processes, SQL connections, synchronous BM25 recall and a
scripted verifier. No embedding/model endpoint is contacted. Latencies are local
observations, never pass/fail thresholds. PostgreSQL CLI creates/drops an isolated
schema; the benchmark() API expects an already isolated, empty store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, RevisionConflict
from ai_knot.always_on.maintenance import MaintenanceBudget, run_maintenance
from ai_knot.always_on.models import VerificationOutcome, VerificationResult
from ai_knot.always_on.shared_maintenance import configure_workspace_budget
from ai_knot.always_on.workspace import WorkspaceMemory
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact


def storage(backend, location):
    if backend == "sqlite":
        return SQLiteStorage(location)
    from ai_knot.storage.postgres_storage import PostgresStorage

    return PostgresStorage(location)


def wait_for(predicate):
    deadline = time.monotonic() + 60
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("benchmark process rendezvous timed out")
        time.sleep(0.005)


def worker(config):
    backend, location = config["backend"], config["location"]
    index, agents, operations = config["index"], config["agents"], config["operations"]
    rendezvous = Path(config["rendezvous"])
    store = storage(backend, location)
    owner = f"agent-{index}"
    kb = KnowledgeBase(owner, storage=store)
    facts = [Fact(f"Deployment detail agent {index} sequence {op}") for op in range(operations)]
    store.save(owner, facts)
    pool = SharedMemoryPool(store, persist_stats=True)
    pool.register(owner)
    workspace = WorkspaceMemory(store, "benchmark")
    memory = AlwaysOnMemory(kb, workspace=workspace)
    intent = memory.create_intent(client_key="deploy", goal="Deploy benchmark")
    lease = memory.claim(intent.id, worker_id=owner, ttl_seconds=3600)
    (rendezvous / f"ready-{index}").touch()
    wait_for(lambda: (rendezvous / "start").exists())
    started = time.monotonic()
    publish, recall = [], []
    for fact in facts:
        before = time.monotonic()
        assert len(pool.publish(owner, [fact.id], kb=kb)) == 1
        publish.append(time.monotonic() - before)
        before = time.monotonic()
        assert len(pool.recall(fact.content, f"observer-{index}", top_k=1)) == 1
        recall.append(time.monotonic() - before)
    pool.flush_stats()
    pool.flush_stats()
    ended = time.monotonic()
    # Every contender has a different operation but the same resource. Hold the
    # winning permit until every contender attempted admission.
    permit = None
    before = time.monotonic()
    try:
        action = memory.prepare_action(
            intent.id,
            lease_token=lease.token,
            action_key="deploy",
            operation_key=f"deploy-{index}",
            resource_key="production",
            request={"v": index},
        )
        permit = memory.begin_action(action.id, lease_token=lease.token)
    except RevisionConflict:
        pass
    resource_ended = time.monotonic()
    resource_latency = resource_ended - before
    resource_started = before
    (rendezvous / f"contended-{index}").touch()
    wait_for(lambda: all((rendezvous / f"contended-{i}").exists() for i in range(agents)))
    if permit is not None:
        memory.finish_action(
            action.id, lease_token=lease.token, outcome="succeeded", result={"ok": True}
        )
    return {
        "publish": publish,
        "recall": recall,
        "started": started,
        "ended": ended,
        "permits": int(permit is not None),
        "resource_seconds": resource_latency,
        "resource_started": resource_started,
        "resource_ended": resource_ended,
    }


async def refresh_fleet(backend, location, agents):
    store = storage(backend, location)
    workspace = WorkspaceMemory(store, "benchmark")
    configure_workspace_budget(workspace, max_model_calls_per_day=1)
    now = datetime.now(UTC)
    fact = Fact("Benchmark shared source", created_at=now, valid_from=now)
    store.save("__shared__", [*store.load("__shared__"), fact])
    memories = []
    for index in range(agents):
        owner = f"agent-{index}"
        workspace.grant(owner, "maintain")
        connection = storage(backend, location)
        memories.append(
            AlwaysOnMemory(
                KnowledgeBase(owner, storage=connection),
                pool=SharedMemoryPool(connection, persist_stats=True),
                workspace=WorkspaceMemory(connection, "benchmark"),
            )
        )
    (ref,) = memories[0].capture_evidence((("__shared__", fact.id),))
    memories[0].set_freshness(ref, source_ref="scripted-source", recheck_after_seconds=3600)

    class Verifier:
        config_key = "benchmark-v1"
        model_cost = 1
        calls = 0

        async def verify(self, request):
            self.calls += 1
            await asyncio.sleep(0.02)  # Keep first claim open while peers check it.
            return VerificationResult(VerificationOutcome.CONFIRMED, {"ok": True}, self.config_key)

    verifier = Verifier()
    budget = MaintenanceBudget(max_model_calls=1, max_wall_seconds=60)
    started = time.monotonic()
    results = await asyncio.gather(
        *(
            run_maintenance(memory, run_key=f"check-{index}", budget=budget, verifier=verifier)
            for index, memory in enumerate(memories)
        )
    )
    elapsed = time.monotonic() - started
    calls = verifier.calls
    idle = await asyncio.gather(
        *(
            run_maintenance(memory, run_key=f"idle-{index}", budget=budget, verifier=verifier)
            for index, memory in enumerate(memories)
        )
    )
    return {
        "provider_calls": calls,
        "idle_calls": verifier.calls - calls,
        "verified": sum(result.verified for result in results),
        "idle_verified": sum(result.verified for result in idle),
        "seconds": elapsed,
        "checks": agents,
    }


def summarize(samples, elapsed):
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "p50_ms": ordered[math.ceil(len(ordered) * 0.5) - 1] * 1000,
        "p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1] * 1000,
        "throughput_per_second": len(samples) / elapsed,
    }


def benchmark(backend, location, *, agents=4, operations=10):
    if agents < 2 or operations < 1:
        raise ValueError("agents must be >=2 and operations >=1")
    store = storage(backend, location)
    if store.load("__shared__") or store.load_pool_stats():
        raise ValueError("benchmark requires an isolated empty store")
    workspace = WorkspaceMemory(store, "benchmark")
    for index in range(agents):
        workspace.grant(f"agent-{index}", "execute")
    with tempfile.TemporaryDirectory(prefix="ai-knot-load-") as rendezvous:
        processes = []
        try:
            for index in range(agents):
                config = dict(
                    backend=backend,
                    location=location,
                    agents=agents,
                    operations=operations,
                    index=index,
                    rendezvous=rendezvous,
                )
                env = {**os.environ, "AI_KNOT_BENCHMARK_WORKER": json.dumps(config)}
                processes.append(
                    subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), "--worker"],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            root = Path(rendezvous)
            wait_for(lambda: all((root / f"ready-{i}").exists() for i in range(agents)))
            (root / "start").touch()
            results = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=90)
                if process.returncode:
                    raise RuntimeError(f"worker failed: {stderr}")
                results.append(json.loads(stdout))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
    elapsed = max(row["ended"] for row in results) - min(row["started"] for row in results)
    stats = store.load_pool_stats()
    expected = agents * operations
    actual = len(store.load("__shared__"))
    used = sum(stats["used_count"].values())
    permits = sum(row["permits"] for row in results)
    assert actual == expected == sum(stats["publish_count"].values()), "lost/double publications"
    assert stats["publish_count"] == {f"agent-{index}": operations for index in range(agents)}
    assert used == expected, "lost/double recall usage"
    consumers = set().union(*(set(value) for value in stats["fact_consumers"].values()))
    assert consumers == {f"observer-{index}" for index in range(agents)}, "lost consumer union"
    assert permits == 1, "resource admitted extra execution permits"
    refresh = asyncio.run(refresh_fleet(backend, location, agents))
    assert refresh["provider_calls"] == refresh["verified"] == 1, "duplicate shared refresh"
    assert refresh["idle_calls"] == refresh["idle_verified"] == 0, "idle called provider"
    resource_elapsed = max(row["resource_ended"] for row in results) - min(
        row["resource_started"] for row in results
    )
    backend_version = sqlite3.sqlite_version
    if backend == "postgres":
        with store._get_conn() as connection:
            backend_version = connection.execute("SHOW server_version").fetchone()[0]
    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "backend_version": backend_version,
        },
        "backend": backend,
        "workload": {
            "agents": agents,
            "operations_per_agent": operations,
            "processes": agents,
            "retrieval": "BM25, no network embeddings",
            "refresh": "async fleet, independent connections, scripted verifier",
        },
        "shared_phase_seconds": elapsed,
        "combined_throughput_per_second": expected * 2 / elapsed,
        "publish": summarize([x for row in results for x in row["publish"]], elapsed),
        "recall": summarize([x for row in results for x in row["recall"]], elapsed),
        "resource": summarize([row["resource_seconds"] for row in results], resource_elapsed),
        "refresh": refresh,
        "invariants": {
            "published": actual,
            "recalled": expected,
            "usage_counter": used,
            "consumer_agents": len(consumers),
            "permits": permits,
            "refresh_provider_calls": refresh["provider_calls"],
            "idle_provider_calls": refresh["idle_calls"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--operations", type=int, default=10)
    parser.add_argument("--dsn", default=os.environ.get("AI_KNOT_TEST_PG_DSN"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(os.environ["AI_KNOT_BENCHMARK_WORKER"]))))
        return
    if args.backend == "sqlite":
        with tempfile.TemporaryDirectory(prefix="ai-knot-benchmark-") as directory:
            report = benchmark(
                "sqlite",
                str(Path(directory) / "load.db"),
                agents=args.agents,
                operations=args.operations,
            )
    else:
        if not args.dsn:
            parser.error("PostgreSQL requires --dsn or AI_KNOT_TEST_PG_DSN")
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        schema = "benchmark_" + uuid4().hex
        with psycopg.connect(args.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            location = make_conninfo(args.dsn, options=f"-csearch_path={schema}")
            report = benchmark("postgres", location, agents=args.agents, operations=args.operations)
        finally:
            with psycopg.connect(args.dsn, autocommit=True) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
