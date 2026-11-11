#!/usr/bin/env python3
"""Bounded real-process outcome/lesson consistency workload; no model is connected.

The CLI owns a temporary SQLite database or PostgreSQL schema. Latencies describe
this scripted local workload, not service capacity, remote success or LLM accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, EvidenceRef, WorkspaceMemory
from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.storage import sql
from ai_knot.types import Fact, MemoryType

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
WORKSPACE = "experience-load"
SCOPE = "experience-receipts"
MAX_FACTS, MAX_CHARS = 3, 600


@contextmanager
def offline():
    with (
        patch("ai_knot.embedder.embed_texts", side_effect=AssertionError("embedding call")),
        patch("httpx.Client.send", side_effect=AssertionError("inference HTTP call")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("inference HTTP call")),
    ):
        yield


def storage(backend, location):
    if backend == "sqlite":
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        return SQLiteStorage(location)
    from ai_knot.storage.postgres_storage import PostgresStorage

    return PostgresStorage(location)


def memory(store, owner, *, pool=True, maintain=False):
    return AlwaysOnMemory(
        KnowledgeBase(owner, storage=store, embed_url=""),
        pool=SharedMemoryPool(store, persist_stats=True, clock=lambda: NOW) if pool else None,
        workspace=WorkspaceMemory(store, WORKSPACE) if maintain else None,
        clock=lambda: NOW,
    )


def wait_for(predicate):
    deadline = time.monotonic() + 60
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("experience workload process rendezvous timed out")
        time.sleep(0.005)


def bounded_context(reader, task_key, role, allowed):
    started = time.perf_counter()
    view = reader.select_context(
        ("checkpoint verification",),
        task_key=task_key,
        role=role,
        max_facts=MAX_FACTS,
        max_chars=MAX_CHARS,
    )
    elapsed = time.perf_counter() - started
    selected = {item.ref.fact_id for item in view.facts}
    assert selected <= set(allowed), "stale, inapplicable or unexpected lesson selected"
    assert bool(selected) == bool(allowed), "current available lessons were missed"
    assert len(view.facts) <= MAX_FACTS
    assert sum(len(item.content) for item in view.facts) <= MAX_CHARS
    assert reader.review_context(tuple(item.ref for item in view.facts)).complete
    return elapsed, len(selected)


def worker(config):
    with offline():
        return _worker(config)


def _worker(config):
    store = storage(config["backend"], config["location"])
    mode = config["mode"]
    if mode == "reconcile":
        results = []
        for index, record in enumerate(config["actions"]):
            # A new process and facade deliberately have no shared-pool attachment.
            reopened = memory(store, record["owner"], pool=False)
            outcome = "succeeded" if index % 2 == 0 else "retryable"
            corrected = reopened.reconcile_action(
                record["action_id"],
                lease_token=record["lease_token"],
                outcome=outcome,
                evidence={"independent_observation": "host corrected the recorded result"},
            )
            results.append(
                {
                    "action_id": corrected.id,
                    "state": corrected.state.value,
                    "requested_outcome": outcome,
                }
            )
        return {"without_pool": len(results), "corrected": results, "pid": os.getpid()}

    owner = config.get("owner", "peer")
    reader = memory(store, owner, maintain=mode == "record")
    if mode == "return":
        refs = tuple(EvidenceRef(**ref) for ref in config["saved_refs"])
        reviewed = reader.review_context(refs, max_chars=65536)
        accepted = {item.ref.fact_id for item in reviewed.accepted}
        assert accepted == set(config["allowed_review"]), "saved-ref admission mismatch"
        samples = []
        selected = 0
        for task_key, allowed in config["allowed_tasks"].items():
            for role in ("operator", "reviewer"):
                elapsed, count = bounded_context(reader, task_key, role, allowed)
                samples.append(elapsed)
                selected += count
        historical = reader.review_context((EvidenceRef(**config["historical_ref"]),))
        return {
            "return": samples,
            "selected": selected,
            "accepted": sorted(accepted),
            "historical_survives": historical.complete,
            "pid": os.getpid(),
        }

    rendezvous = Path(config["rendezvous"])
    (rendezvous / f"ready-{config['index']}").touch()
    wait_for(lambda: (rendezvous / "start").exists())
    started = time.monotonic()
    durations, returns, outputs = [], [], {}
    for _ in range(config["replays"]):
        for record in config["actions"]:
            for key in range(config["keys"]):
                before = time.perf_counter()
                result = reader.record_experience(
                    record["action_id"],
                    expected_action_revision=record["revision"],
                    experience_key=f"lesson-{key}",
                    content="Checkpoint verification: inspect retained receipt; "
                    f"conditional step {key}.",
                    basis=(EvidenceRef(**record["source_ref"]),),
                    namespace="__shared__",
                    task_key=record["task_key"],
                    for_role="operator" if key % 2 == 0 else "reviewer",
                )
                durations.append(time.perf_counter() - before)
                identity = f"{record['index']}:{key}"
                ref = asdict(result.lesson.ref)
                assert identity not in outputs or outputs[identity] == ref, "replay changed output"
                outputs[identity] = ref
        first = config["actions"][0]
        allowed = [
            ref["fact_id"]
            for identity, ref in outputs.items()
            if identity.startswith(f"{first['index']}:")
        ]
        for role in ("operator", "reviewer"):
            elapsed, _ = bounded_context(reader, first["task_key"], role, allowed)
            returns.append(elapsed)
    return {
        "record": durations,
        "return": returns,
        "outputs": outputs,
        "pid": os.getpid(),
        "started": started,
        "ended": time.monotonic(),
    }


def processes(configs, work, *, simultaneous=False):
    children = []
    jobs = []
    try:
        for config in configs:
            job = work / f"job-{uuid4().hex}.json"
            job.touch(mode=0o600)
            job.write_text(json.dumps(config))
            jobs.append(job)
            children.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(job)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        if simultaneous:
            wait_for(
                lambda: all((work / f"ready-{index}").exists() for index in range(len(configs)))
            )
            (work / "start").touch()
        rows = []
        for child in children:
            stdout, stderr = child.communicate(timeout=180)
            if child.returncode:
                safe_error = redact_worker_error(stderr, configs)
                raise RuntimeError(f"experience workload worker failed:\n{safe_error[-6000:]}")
            rows.append(json.loads(stdout))
        return rows
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        for job in jobs:
            job.unlink(missing_ok=True)


def redact_worker_error(stderr, configs):
    """Redact before truncation, including libpq's reformatted connection errors."""
    secrets = set()
    for config in configs:
        for action in config.get("actions", ()):
            secrets.add(action.get("lease_token", ""))
        if config.get("backend") == "postgres":
            location = config.get("location", "")
            secrets.add(location)
            with suppress(ValueError):
                secrets.add(unquote(urlsplit(location).password or ""))
            try:
                from psycopg import ProgrammingError
                from psycopg.conninfo import conninfo_to_dict
            except ImportError:
                pass
            else:
                # Invalid diagnostic strings still get the field redaction below.
                with suppress(ValueError, ProgrammingError):
                    secrets.add(conninfo_to_dict(location).get("password", ""))
    forms = {
        form
        for secret in secrets
        if secret
        for form in (
            secret,
            repr(secret)[1:-1],
            json.dumps(secret)[1:-1],
            quote(secret, safe=""),
        )
    }
    for private in sorted(forms, key=len, reverse=True):
        stderr = stderr.replace(private, "<redacted>")
    stderr = re.sub(r"(?i)\bpostgres(?:ql)?://[^\s'\"<>]+", "<redacted PostgreSQL DSN>", stderr)
    return re.sub(
        r"(?i)(['\"]?(?:password|lease_token)['\"]?\s*[:=]\s*)"
        r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[^\s,}]+)",
        r"\1<redacted>",
        stderr,
    )


def summarize(samples):
    ordered = sorted(samples)
    assert ordered
    return {
        "samples": len(ordered),
        "seconds_p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "seconds_p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def benchmark(backend, location, *, owners=4, workers=8, actions=8, keys=8, replays=3, noise=500):
    """Caller must supply an empty disposable store; CLI creates and removes its own."""
    assert 1 <= owners <= 4 and 2 * owners <= workers <= 8 and workers % owners == 0
    assert 4 <= actions <= 8 and actions % owners == 0
    assert 2 <= keys <= 8 and 1 <= replays <= 3 and 0 <= noise <= 500
    with offline(), tempfile.TemporaryDirectory(prefix="knot-experience-workers-") as directory:
        return _benchmark(
            backend, location, owners, workers, actions, keys, replays, noise, Path(directory)
        )


def _benchmark(backend, location, owners, workers, actions, keys, replays, noise, work):
    store = storage(backend, location)
    assert not store.load("__shared__"), "workload requires an empty isolated store"
    workspace = WorkspaceMemory(store, WORKSPACE)
    pool = SharedMemoryPool(store, persist_stats=True, clock=lambda: NOW)
    for owner in [*(f"writer-{i}" for i in range(owners)), "peer"]:
        workspace.grant(owner, "maintain")
        pool.grant_read(owner, SCOPE)
    fixed = dict(created_at=NOW, last_accessed=NOW, valid_from=NOW)
    sources = [
        Fact(
            f"Host retained tool observation for operation {i}.",
            id=f"receipt-{i}",
            type=MemoryType.EPISODIC,
            visibility_scope=SCOPE,
            origin_agent_id="observer",
            **fixed,
        )
        for i in range(actions)
    ]
    attempt = Fact(
        "Historical attempt one timed out before eventual reconciliation.",
        id="attempt-history",
        type=MemoryType.EPISODIC,
        visibility_scope=SCOPE,
        origin_agent_id="observer",
        **fixed,
    )
    unrelated = [
        Fact(f"Unrelated inventory bulletin {i}.", id=f"noise-{i}", **fixed) for i in range(noise)
    ]
    store.save("__shared__", [*sources, attempt, *unrelated])
    source_hash = hashlib.sha256(
        canonical_json([*sources, attempt, *unrelated]).encode()
    ).hexdigest()
    manifests = []
    for index, source in enumerate(sources):
        owner = f"writer-{index % owners}"
        host = memory(store, owner, maintain=True)
        intent = host.create_intent(client_key=f"run-{index}", goal="Scripted observed operation")
        lease = host.claim(intent.id, worker_id="host", ttl_seconds=3600)
        action = host.prepare_action(
            intent.id, lease_token=lease.token, action_key="operation", request={"index": index}
        )
        host.begin_action(action.id, lease_token=lease.token)
        action = host.finish_action(
            action.id,
            lease_token=lease.token,
            outcome="succeeded" if index % 2 == 0 else "failed",
            result={"retained_observation": source.id},
        )
        manifests.append(
            {
                "index": index,
                "owner": owner,
                "action_id": action.id,
                "revision": action.revision,
                "lease_token": lease.token,
                "task_key": f"task-{index}",
                "source_ref": asdict(host.capture_evidence((("__shared__", source.id),))[0]),
            }
        )
    config = {"backend": backend, "location": location}
    configs = [
        {
            **config,
            "mode": "record",
            "index": index,
            "owner": f"writer-{index % owners}",
            "actions": [r for r in manifests if r["owner"] == f"writer-{index % owners}"],
            "keys": keys,
            "replays": replays,
            "rendezvous": str(work),
        }
        for index in range(workers)
    ]
    recorded = processes(configs, work, simultaneous=True)
    actual_calls = sum(len(row["record"]) for row in recorded)
    expected_calls = actions * keys * replays * (workers // owners)
    assert actual_calls == expected_calls
    outputs = {}
    for row in recorded:
        for identity, ref in row["outputs"].items():
            assert identity not in outputs or outputs[identity] == ref, (
                "concurrent duplicate output"
            )
            outputs[identity] = ref
    expected_lessons = actions * keys
    assert len(outputs) == expected_lessons
    host = memory(store, "writer-0", maintain=True)
    independent_sources = host.independent_evidence(
        tuple(EvidenceRef(**ref) for ref in outputs.values())
    )
    assert len(independent_sources) == actions, "lesson keys inflated or lost independent sources"
    with host._transaction(("__shared__",), write=False) as session:
        rows = session.fetch_all(sql("SELECT * FROM ao_experiences"))
    unique_bindings = {
        (r["owner"], r["action_id"], r["namespace"], r["experience_key"]) for r in rows
    }
    duplicate_bindings = len(rows) - len(unique_bindings)
    assert len(rows) == expected_lessons and duplicate_bindings == 0
    facts = store.load("__shared__")
    assert len(facts) == noise + actions + 1 + expected_lessons, "lost or duplicate shared facts"
    assert {ref["fact_id"] for ref in outputs.values()} == {
        fact.id for fact in facts if "_ai_knot_action_binding" in fact.qualifiers
    }
    assert len({row["pid"] for row in recorded}) == workers
    overlapping = max(row["started"] for row in recorded) < min(row["ended"] for row in recorded)
    assert overlapping, "recording processes did not overlap"

    original = EvidenceRef(**outputs["0:0"])
    copy_refs = []
    for _ in range(2):
        copied = host.derive_fact(
            "Checkpoint verification: copied source-dependent procedure.",
            basis=(copy_refs[-1] if copy_refs else original,),
            namespace="__shared__",
            memory_type=MemoryType.PROCEDURAL,
            task_key="task-0",
            for_role="operator",
        )
        copy_refs.append(host.capture_evidence((("__shared__", copied.id),))[0])
    roots = host.independent_evidence((original, *copy_refs))
    assert len(roots) == 1 and roots[0].fact_id == "receipt-0"
    warning = host.derive_fact(
        "Historical attempt warning: retain the timed-out attempt observation.",
        basis=host.capture_evidence((("__shared__", attempt.id),)),
        namespace="__shared__",
        memory_type=MemoryType.PROCEDURAL,
        task_key="history",
    )
    warning_ref = asdict(host.capture_evidence((("__shared__", warning.id),))[0])
    all_refs = list(outputs.values())
    current_ids = {ref["fact_id"] for ref in all_refs}
    task_ids = {
        f"task-{i}": [
            ref["fact_id"] for identity, ref in outputs.items() if identity.startswith(f"{i}:")
        ]
        for i in range(actions)
    }
    task_ids["task-0"].extend(ref.fact_id for ref in copy_refs)
    return_rows = []

    def check_return(allowed, allowed_tasks):
        (result,) = processes(
            [
                {
                    **config,
                    "mode": "return",
                    "saved_refs": all_refs,
                    "allowed_review": sorted(allowed),
                    "allowed_tasks": allowed_tasks,
                    "historical_ref": warning_ref,
                }
            ],
            work,
        )
        return_rows.append(result)
        return result

    first = check_return(current_ids, task_ids)
    assert first["historical_survives"]
    pool.revoke_read("peer", SCOPE)
    withdrawn = check_return(set(), {task: [] for task in task_ids})
    assert not withdrawn["historical_survives"]
    pool.grant_read("peer", SCOPE)
    store.save(
        "__shared__",
        [
            replace(fact, content="Host withdrew the original operation observation.", version=1)
            if fact.id == "receipt-0"
            else fact
            for fact in store.load("__shared__")
        ],
    )
    source_ids = {ref["fact_id"] for identity, ref in outputs.items() if identity.startswith("0:")}
    current_ids -= source_ids
    task_ids["task-0"] = []
    after_source = check_return(current_ids, task_ids)
    assert not host.review_context((original, *copy_refs)).accepted

    failed = [record for record in manifests if record["index"] % 2]
    (corrected,) = processes([{**config, "mode": "reconcile", "actions": failed}], work)
    failed_indices = {record["index"] for record in failed}
    outcome_ids = {
        ref["fact_id"]
        for identity, ref in outputs.items()
        if int(identity.split(":")[0]) in failed_indices
    }
    current_ids -= outcome_ids
    for index in failed_indices:
        task_ids[f"task-{index}"] = []
    final = check_return(current_ids, task_ids)
    assert final["historical_survives"] and corrected["without_pool"] == len(failed)
    assert {item["requested_outcome"] for item in corrected["corrected"]} == {
        "succeeded",
        "retryable",
    }
    assert {item["state"] for item in corrected["corrected"]} == {"succeeded", "prepared"}
    final_facts = store.load("__shared__")
    assert len(final_facts) == noise + actions + 1 + expected_lessons + 3
    assert all(not fact.supported for fact in final_facts if fact.id in outcome_ids)
    with host._transaction(("__shared__",), write=False) as session:
        assert len(session.fetch_all(sql("SELECT * FROM ao_experiences"))) == expected_lessons
    return_samples = [sample for row in [*recorded, *return_rows] for sample in row["return"]]
    return {
        "backend": backend,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "workload": {
            "owners": owners,
            "workers": workers,
            "actions": actions,
            "keys": keys,
            "replays_per_worker": replays,
            "noise": noise,
            "max_facts": MAX_FACTS,
            "max_chars": MAX_CHARS,
            "process_model": "simultaneous processes; independent SQL connections",
            "correction_schedule": "between fresh-process return phases",
        },
        "timing_scope": {
            "record": "record_experience calls, including idempotent replays",
            "return": "select_context only; subsequent review/assertions excluded",
        },
        "fixed_source_corpus_sha256": source_hash,
        "expected": {"record_calls": expected_calls, "durable_lessons": expected_lessons},
        "record": summarize([sample for row in recorded for sample in row["record"]]),
        "return": summarize(return_samples),
        "invariants": {
            "record_calls": actual_calls,
            "durable_lessons": len(rows),
            "unique_returned_lessons": len(outputs),
            "duplicate_bindings": duplicate_bindings,
            "independent_roots_for_original_and_copies": len(roots),
            "independent_sources_for_all_lessons": len(independent_sources),
            "stale_selections": 0,
            "acl_rejected": len(first["accepted"]) - len(withdrawn["accepted"]),
            "source_rejected": len(first["accepted"]) - len(after_source["accepted"]),
            "outcome_rejected": len(after_source["accepted"]) - len(final["accepted"]),
            "historical_attempt_warning_survives": final["historical_survives"],
            "reopened_without_pool": corrected["without_pool"],
            "distinct_record_processes": len({row["pid"] for row in recorded}),
            "overlapping_worker_intervals": overlapping,
            "model_calls": 0,
            "inference_endpoints_blocked": True,
        },
        "return_stages": [
            {
                "stage": stage,
                "accepted_lessons": len(row["accepted"]),
                "selected_facts": row["selected"],
                "historical_survives": row["historical_survives"],
            }
            for stage, row in zip(
                ("before_changes", "acl_revoked", "source_corrected", "outcome_reconciled"),
                return_rows,
                strict=True,
            )
        ],
        "scope": "Consistency/local timing only; no capacity, comparison or model-accuracy claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    for name, default in (
        ("owners", 4),
        ("workers", 8),
        ("actions", 8),
        ("keys", 8),
        ("replays", 3),
        ("noise", 500),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker.read_text()))))
        return
    params = {
        key: getattr(args, key)
        for key in ("owners", "workers", "actions", "keys", "replays", "noise")
    }
    if args.backend == "sqlite":
        with tempfile.TemporaryDirectory(prefix="knot-experience-load-") as directory:
            report = benchmark("sqlite", str(Path(directory) / "memory.db"), **params)
    else:
        import psycopg
        from psycopg import sql as pgsql
        from psycopg.conninfo import make_conninfo

        dsn = os.environ.get("AI_KNOT_TEST_PG_DSN")
        if not dsn:
            parser.error("PostgreSQL requires AI_KNOT_TEST_PG_DSN")
        schema = "experience_load_" + uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(pgsql.SQL("CREATE SCHEMA {}").format(pgsql.Identifier(schema)))
        try:
            report = benchmark(
                "postgres", make_conninfo(dsn, options=f"-csearch_path={schema}"), **params
            )
        finally:
            with psycopg.connect(dsn, autocommit=True) as connection:
                connection.execute(
                    pgsql.SQL("DROP SCHEMA {} CASCADE").format(pgsql.Identifier(schema))
                )
    output = args.output or Path("artifacts") / f"team-experience-{args.backend}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
