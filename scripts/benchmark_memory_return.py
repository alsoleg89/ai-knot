#!/usr/bin/env python3
"""Independent-process sleep/return workload in a disposable SQL store.

No model is connected. Wake decisions are a deterministic host policy; latency
is a local observation, not an SLO or a measurement of LLM answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, CursorExpired, EvidenceRef, WatchSelector
from ai_knot.types import Fact


def storage(backend, location):
    if backend == "sqlite":
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        return SQLiteStorage(location)
    from ai_knot.storage.postgres_storage import PostgresStorage

    return PostgresStorage(location)


def memory(backend, location, index):
    store = storage(backend, location)
    return AlwaysOnMemory(
        KnowledgeBase(f"reader-{index}", storage=store, embed_url=""),
        pool=SharedMemoryPool(store, persist_stats=True),
    )


def reader(config):
    backend, location, index, old_ref = config
    ao = memory(backend, location, index)
    started = time.perf_counter()
    pages = empty_pages = recovered = stale_rejected = 0
    if old_ref is not None:
        stale = ao.review_context((EvidenceRef(**old_ref),))
        assert not stale.accepted and not stale.complete
        stale_rejected = 1
        try:
            ao.poll("deployment")
        except CursorExpired:
            batch = ao.resync("deployment", limit=2)
            recovered = 1
        else:
            raise AssertionError("Pruned unobserved update must require recovery")
    else:
        batch = ao.poll("deployment", limit=1)
    observed = None
    while batch.delivery_id is not None:
        assert not batch.resync_required and len(batch.events) <= 2
        pages += 1
        empty_pages += not batch.events
        for event in batch.events:
            assert event.fact_id == "target" and event.after_revision
            observed = EvidenceRef(event.namespace, event.fact_id, event.after_revision)
        # The workload's processing result is returned to its parent. Delivery
        # receipts/crash safety are exercised separately by the integration suite.
        ao.ack("deployment", delivery_id=batch.delivery_id)
        if not batch.has_more:
            break
        batch = ao.poll("deployment", limit=1)
    assert observed is not None
    context = ao.review_context((observed,))
    expected = "Deploy in US" if old_ref is not None else "Deploy in EU"
    assert context.complete and [item.content for item in context.accepted] == [expected]
    elapsed = time.perf_counter() - started
    for _ in range(5):
        idle = ao.poll("deployment")
        assert idle.delivery_id is None and not idle.events and not idle.has_more
    return {
        "ref": asdict(observed),
        "seconds": elapsed,
        "pages": pages,
        "empty_pages": empty_pages,
        "recovered": recovered,
        "stale_rejected": stale_rejected,
        "useful_wake_decisions": 1,
        "idle_checks": 5,
    }


def benchmark(backend, location, agents, noise):
    """Requires an empty, already isolated store; CLI owns its store/schema."""
    assert 1 <= agents <= 32 and 1000 <= noise <= 100000
    for index in range(agents):
        memory(backend, location, index).subscribe(
            subscription_id="deployment",
            selector=WatchSelector("__shared__", slot_keys=("deployment::region",)),
        )
    store = storage(backend, location)
    target = Fact("Deploy in EU", id="target", slot_key="deployment::region", value_text="EU")
    unrelated = [Fact("Unrelated update", id=f"noise-{i:06}") for i in range(noise)]
    store.save("__shared__", [*unrelated, target])
    with ProcessPoolExecutor(
        max_workers=agents, mp_context=multiprocessing.get_context("spawn")
    ) as workers:
        first = list(workers.map(reader, [(backend, location, i, None) for i in range(agents)]))
        assert all(item["empty_pages"] >= 1 for item in first)
        store.save(
            "__shared__", [*unrelated, replace(target, content="Deploy in US", value_text="US")]
        )
        memory(backend, location, 0).prune_changes("__shared__", force=True)
        returned = list(
            workers.map(reader, [(backend, location, i, first[i]["ref"]) for i in range(agents)])
        )
    assert sum(item["recovered"] for item in returned) == agents
    assert sum(item["stale_rejected"] for item in returned) == agents
    durations = sorted(item["seconds"] for item in [*first, *returned])
    return {
        "backend": backend,
        "agents": agents,
        "irrelevant_updates": noise,
        "process_model": "spawned processes, independent SQL connections",
        "python": platform.python_version(),
        "platform": platform.system(),
        "target_observations": agents * 2,
        "snapshots_recovered": agents,
        "stale_contexts_rejected": agents,
        "missed_target_updates": 0,
        "model_calls": 0,
        "useful_wake_decisions": agents * 2,
        "idle_checks": agents * 10,
        "idle_wake_decisions": 0,
        "filtered_empty_pages": sum(item["empty_pages"] for item in first),
        "return_seconds_p50": durations[len(durations) // 2],
        "return_seconds_p95": durations[min(len(durations) - 1, int(len(durations) * 0.95))],
        "scope": "Protocol correctness/local timing; scripted host; LLM accuracy unmeasured",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), required=True)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--noise", type=int, default=2500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.agents <= 32 or not 1000 <= args.noise <= 100000:
        parser.error("agents must be 1..32; noise must be 1000..100000")
    if args.backend == "sqlite":
        with tempfile.TemporaryDirectory(prefix="knot-return-load-") as directory:
            report = benchmark(
                "sqlite", str(Path(directory) / "memory.db"), args.agents, args.noise
            )
    else:
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo

        dsn = os.environ["AI_KNOT_TEST_PG_DSN"]
        schema = "return_load_" + uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            report = benchmark(
                "postgres",
                make_conninfo(dsn, options=f"-csearch_path={schema}"),
                args.agents,
                args.noise,
            )
        finally:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
