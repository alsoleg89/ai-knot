"""Run deterministic recovery checks; PostgreSQL mode requires a real DSN."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_knot.always_on import AlwaysOnMemory, WatchSelector
from ai_knot.always_on.storage import AlwaysOnStorageCapable, SQLStatement, sql
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.base import StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact

from .scenarios import (
    consolidation_race,
    crash_recovery,
    owner_and_scope_isolation,
    snapshot_history,
    stale_plan,
    verifier_outage,
)


@contextmanager
def backend(kind: str, directory: Path) -> Iterator[StorageBackend]:
    if kind == "sqlite":
        yield SQLiteStorage(str(directory / "memory.db"))
        return
    dsn = os.environ.get("AI_KNOT_TEST_PG_DSN")
    if not dsn:
        raise ValueError("AI_KNOT_TEST_PG_DSN is required for PostgreSQL acceptance")
    import psycopg
    from psycopg import sql as pgsql
    from psycopg.conninfo import make_conninfo

    from ai_knot.storage.postgres_storage import PostgresStorage

    schema = "ao_gate_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(pgsql.SQL("CREATE SCHEMA {}").format(pgsql.Identifier(schema)))
    try:
        yield PostgresStorage(make_conninfo(dsn, options=f"-csearch_path={schema}"))
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(pgsql.SQL("DROP SCHEMA {} CASCADE").format(pgsql.Identifier(schema)))


def acceptance(storage: StorageBackend, work: Path) -> list[dict[str, Any]]:
    records = []
    checks: list[tuple[str, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], bool]]] = []
    for point in (
        "after_poll",
        "after_checkpoint",
        "after_begin",
        "after_external",
        "after_finish",
    ):
        checks.append(
            (
                point,
                lambda point=point: crash_recovery(storage, work, crash_at=point),
                lambda r: (
                    r["completed"]
                    and r["external_effects"] == 1
                    and all(
                        r[k] == 0
                        for k in (
                            "missing_committed_changes",
                            "unintended_duplicate_effects",
                            "accepted_stale_lease_mutations",
                            "false_auto_retries_unknown_actions",
                        )
                    )
                ),
            )
        )
    checks.extend(
        [
            (
                "owner_and_scope_isolation",
                lambda: owner_and_scope_isolation(storage),
                lambda r: (
                    r["cross_owner_scope_leaks"] == 0
                    and r["replay_identity_preserved"]
                    and r["revoked_resync"]
                ),
            ),
            (
                "consolidation_race",
                lambda: consolidation_race(storage),
                lambda r: (
                    r["consolidated"] == 0
                    and r["foreground_preserved"]
                    and r["source_preserved"]
                    and r["stale_summary_absent"]
                ),
            ),
            (
                "unknown_without_proof",
                lambda: crash_recovery(storage, work, crash_at="after_external", reconcile=False),
                lambda r: (
                    not r["completed"]
                    and r["unknown_actions"] == 1
                    and r["external_effects"] == 1
                    and r["false_auto_retries_unknown_actions"] == 0
                ),
            ),
            ("snapshot_restore", lambda: snapshot_history(storage), lambda r: all(r.values())),
            (
                "stale_plan",
                lambda: stale_plan(storage),
                lambda r: r["stale_action_starts"] == 0 and r["blocked"],
            ),
            (
                "deleted_basis",
                lambda: stale_plan(storage, deleted=True),
                lambda r: r["stale_action_starts"] == 0 and r["blocked"],
            ),
            (
                "verifier_outage",
                lambda: verifier_outage(storage),
                lambda r: (
                    r["fact_preserved"]
                    and r["validity_not_extended"]
                    and r["outcome"] == "unavailable"
                    and r["failure_count"] == 1
                ),
            ),
        ]
    )
    for name, run, valid in checks:
        started = time.perf_counter()
        try:
            result = run()
            passed = bool(valid(result))
        except Exception as exc:
            result = {"error_type": type(exc).__name__, "message": str(exc)}
            passed = False
        records.append(
            dict(
                name=name,
                passed=passed,
                latency_seconds=time.perf_counter() - started,
                result=result,
            )
        )
    return records


def percentile(samples: list[float], p: float) -> float:
    return sorted(samples)[max(0, min(len(samples) - 1, math.ceil(len(samples) * p) - 1))]


def measure(
    storage: StorageBackend, count: int, subscriptions: int, *, large_replay: int = 0
) -> dict[str, Any]:
    owner = f"perf-{count}-{subscriptions}"
    facts = [
        Fact(id=f"f{i}", content=f"Project {i} deploy automation uses Python") for i in range(count)
    ]
    storage.save(owner, facts)
    kb = KnowledgeBase(owner, storage=storage, embed_url="")
    memory = AlwaysOnMemory(kb)
    for i in range(subscriptions):
        memory.subscribe(subscription_id=f"w{i}", selector=WatchSelector(namespace=owner))
    write_times = []
    poll_times = []
    resume_times = []
    raw_events_scanned = 0
    largest_scan = 0
    delivered_batches = 0

    def count_scan(subscription_id: str) -> None:
        nonlocal raw_events_scanned, largest_scan
        with memory._transaction(write=False) as session:
            row = session.fetch_one(
                sql("SELECT cursor FROM ao_subscriptions WHERE owner=? AND id=?"),
                (owner, subscription_id),
            )
            assert row is not None
            scan = session.fetch_one(
                sql(
                    "SELECT COUNT(*) AS n FROM (SELECT seq FROM ao_changes "
                    "WHERE namespace=? AND seq>? ORDER BY seq LIMIT 1000) AS page"
                ),
                (owner, row["cursor"]),
            )
            assert scan is not None
            number = int(scan["n"] or 0)
            raw_events_scanned += number
            largest_scan = max(largest_scan, number)

    intent = memory.create_intent(client_key="perf", goal="Measure resume")
    for repeat in range(5):
        start = time.perf_counter()
        facts[0].content = f"Updated deployment {repeat}"
        storage.save(owner, facts)
        write_times.append(time.perf_counter() - start)
        for i in range(subscriptions):
            count_scan(f"w{i}")
            start = time.perf_counter()
            batch = memory.poll(f"w{i}")
            poll_times.append(time.perf_counter() - start)
            replay = AlwaysOnMemory(KnowledgeBase(owner, storage=storage)).poll(f"w{i}")
            if replay != batch:
                raise AssertionError("restart replay changed a delivery")
            if batch.delivery_id:
                delivered_batches += 1
                memory.ack(f"w{i}", delivery_id=batch.delivery_id)
        start = time.perf_counter()
        memory.resume(intent.id)
        resume_times.append(time.perf_counter() - start)
    if large_replay:
        # Batch inserting the explicitly requested large journal avoids O(n²) test setup.
        memory.subscribe(subscription_id="large", selector=WatchSelector(namespace=owner))
        facts.extend(Fact(id=f"large{i}", content=f"large event {i}") for i in range(large_replay))
        storage.save(owner, facts)
        delivered = 0
        while True:
            count_scan("large")
            batch = memory.poll("large", limit=500)
            if not batch.delivery_id:
                break
            delivered += len(batch.events)
            delivered_batches += 1
            memory.ack("large", delivery_id=batch.delivery_id)
        if delivered != large_replay:
            raise AssertionError(f"large replay lost events: {delivered}/{large_replay}")
    assert isinstance(storage, AlwaysOnStorageCapable)
    with storage.always_on_transaction((owner,), write=False) as session:
        stats = session.fetch_one(
            sql("SELECT COUNT(*) AS rows FROM ao_changes WHERE namespace=?"), (owner,)
        )
        receipts = session.fetch_one(
            sql(
                "SELECT COUNT(*) AS rows, COALESCE(SUM(length(delivery_id)),0) AS bytes, "
                "COALESCE(SUM(length(owner)+length(subscription_id)+length(incarnation)"
                "+length(delivery_id)+length(CAST(through_seq AS TEXT))),0) AS logical_bytes "
                "FROM ao_delivery_receipts WHERE owner=?"
            ),
            (owner,),
        )
        journal = session.fetch_one(
            sql(
                "SELECT COALESCE(SUM(length(fact_id)+length(kind)"
                "+COALESCE(length(old_access_json),0)+COALESCE(length(new_access_json),0)"
                "+COALESCE(length(before_revision),0)+COALESCE(length(after_revision),0)),0) "
                "AS bytes FROM ao_changes WHERE namespace=?"
            ),
            (owner,),
        )
        plan = session.fetch_all(
            SQLStatement(
                "EXPLAIN QUERY PLAN SELECT * FROM ao_changes "
                "WHERE namespace=? AND seq>? ORDER BY seq LIMIT 1000",
                'EXPLAIN SELECT * FROM "ai-knot_ao_changes" '
                "WHERE namespace=%s AND seq>%s ORDER BY seq LIMIT 1000",
            ),
            (owner, 0),
        )
    return dict(
        facts=count,
        subscriptions=subscriptions,
        write_p50=statistics.median(write_times),
        write_p95=percentile(write_times, 0.95),
        poll_p50=statistics.median(poll_times),
        poll_p95=percentile(poll_times, 0.95),
        resume_p50=statistics.median(resume_times),
        resume_p95=percentile(resume_times, 0.95),
        journal_rows=stats["rows"] if stats else 0,
        journal_logical_bytes=journal["bytes"] if journal else 0,
        receipt_rows=receipts["rows"] if receipts else 0,
        receipt_token_bytes=receipts["bytes"] if receipts else 0,
        receipt_logical_bytes=receipts["logical_bytes"] if receipts else 0,
        raw_events_scanned=raw_events_scanned,
        largest_scan=largest_scan,
        delivered_batches=delivered_batches,
        scan_limit=1000,
        query_plan=plan,
        large_replay_events=large_replay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--facts", default="1000,10000")
    parser.add_argument("--subscriptions", default="1,10,100")
    parser.add_argument("--large-replay", type=int, default=0)
    args = parser.parse_args()
    if args.storage == "postgres" and not os.environ.get("AI_KNOT_TEST_PG_DSN"):
        parser.error("AI_KNOT_TEST_PG_DSN is required for PostgreSQL acceptance")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai-knot-ao-gate-") as temp:
        work = Path(temp)
        with backend(args.storage, work) as storage:
            scenarios = acceptance(storage, work)
            timings = []
            if args.benchmark:
                for n in map(int, args.facts.split(",")):
                    for subs in map(int, args.subscriptions.split(",")):
                        timings.append(measure(storage, n, subs, large_replay=args.large_replay))
    report = dict(
        setup=dict(
            storage=args.storage, postgres_required=args.storage == "postgres", offline=True
        ),
        scenarios=scenarios,
        aggregate=dict(
            passed=all(s["passed"] for s in scenarios),
            passed_scenarios=sum(s["passed"] for s in scenarios),
            total_scenarios=len(scenarios),
            invariants={
                name: sum(s["result"].get(name, 0) for s in scenarios)
                for name in (
                    "missing_committed_changes",
                    "stale_action_starts",
                    "accepted_stale_lease_mutations",
                    "unintended_duplicate_effects",
                    "false_auto_retries_unknown_actions",
                    "cross_owner_scope_leaks",
                )
            },
        ),
        benchmarks=timings,
    )
    target = args.output_dir / "always_on_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(dict(report=str(target), **report["aggregate"])))
    if not report["aggregate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
