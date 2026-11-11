"""Budgeted repository experiment, with a fresh SQL-backed process for each stage."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from tests.eval.task_context.runner import _reopen, isolated_backend, storage_descriptor
from tests.eval.team_decisions.openai import MeteredOpenAI

from . import runner
from .protocol import (
    ARMS,
    CODER_OUTPUT,
    MAX_CALLS,
    MAX_CONTEXT_CHARS,
    MAX_TOKENS,
    RESEARCH_OUTPUT,
    context_chars,
    handoff_schema,
    parse_handoff,
    researcher_messages,
)

MAX_WORKER_BYTES = 2 * 1024 * 1024
MODULE = "tests.eval.team_repos.experiment"
CASE_MANIFEST_SHA256 = "148d2368d1c841fd5005879118b041d6d4ce7b9cfe574c8038a4c65af1ffc656"


def cleanup_run(run_id):
    # Imported here so ordinary orchestration tests need no Docker installation.
    from .sandbox import cleanup_run as cleanup

    cleanup(run_id)


async def form_notes(case, client_factory):
    """Form the one common public-only handoff sequence before any acquisition arm."""
    result = {"status": "failed", "notes": [], "calls": [], "spent_tokens": 0}
    client = None
    try:
        client = client_factory(
            max_calls=3, max_tokens=18000, max_output_tokens_per_call=RESEARCH_OUTPUT
        )
        async with client:
            for index, stage in enumerate(case.stages):
                facts = [
                    f
                    for f in runner.projected_documents(case, index)
                    if f.qualifiers.get("task_key", "") in ("", case.task_key)
                ]
                visible = runner.entries(facts)
                if context_chars(visible) > MAX_CONTEXT_CHARS:
                    raise ValueError("Full context exceeds the frozen ceiling")
                raw = await client.complete(
                    researcher_messages(case, stage, visible),
                    max_output_tokens=RESEARCH_OUTPUT,
                    response_schema=handoff_schema(len(visible)),
                )
                summary, refs = parse_handoff(raw, visible)
                result["notes"].append({"summary": summary, "basis": [asdict(r) for r in refs]})
        result["status"] = "complete"
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    finally:
        if client is not None:
            result.update(calls=client.calls, spent_tokens=client.spent_tokens)
    return result


def _prepare_stage(storage, case, index, intent_id, note):
    runner.seed_stage(storage, case, index, note)
    memory = runner.memory_for(storage, index)
    if intent_id is None:
        intent_id = memory.create_intent(client_key="repository-task", goal=case.goal).id
    packet = memory.resume(intent_id)
    if {"unknown_action", "host_blocked", "deadline_exceeded", "live_lease"} & set(packet.reasons):
        raise ValueError("Work is blocked")
    progress = dict(memory.get_intent(intent_id).progress)
    history = list(progress.get("history", []))
    seen = {tuple(sorted(item["ref"].items())) for item in history}
    current = runner.entries(
        [
            f
            for f in storage.load("__shared__")
            if f.qualifiers.get("task_key", "") in ("", case.task_key)
        ]
    )
    for item in current:
        key = tuple(sorted(item["ref"].items()))
        if key not in seen:
            history.append(item)
            seen.add(key)
    progress.update(history=history, stage_event=index)
    if index == 0:
        progress["code"] = case.initial_code
    lease = memory.claim(intent_id, worker_id=f"stage-event-{index}", ttl_seconds=300)
    memory.checkpoint(
        intent_id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent_id).revision,
        state="ready",
        progress=progress,
        basis=(),
    )
    return intent_id


def run_arm(
    case,
    arm,
    formation,
    work,
    *,
    backend="sqlite",
    keychain_service="",
    worker=None,
    on_update=None,
):
    """Retain every charge and prior result, including after a process/cleanup failure."""
    allocated = arm != "direct_sources"
    calls = list(formation["calls"]) if allocated else []
    tokens = formation["spent_tokens"] if allocated else 0
    result = {
        "case": case.id,
        "arm": arm,
        "status": "failed",
        "stages": [],
        "calls": calls,
        "spent_tokens": tokens,
        "allocated_formation_tokens": tokens,
        "allocated_formation_calls": len(calls),
        "worker_spent_tokens": 0,
        "worker_calls": [],
    }
    worker = worker or run_worker
    try:
        work.mkdir(parents=True, exist_ok=True)
        if arm not in ARMS:
            raise ValueError("Unknown acquisition policy")
        if allocated and formation["status"] != "complete":
            raise ValueError("Common formation failed")
        with isolated_backend(backend, work) as storage:
            intent_id = None
            for index, _ in enumerate(case.stages):
                note = formation["notes"][index] if allocated else None
                intent_id = _prepare_stage(storage, case, index, intent_id, note)
                remaining_calls = MAX_CALLS - len(result["calls"])
                remaining_tokens = MAX_TOKENS - result["spent_tokens"]
                if remaining_calls <= 0 or remaining_tokens <= 0:
                    raise ValueError("Arm budget exhausted")
                job = {
                    "storage": storage_descriptor(storage),
                    "case_id": case.id,
                    "index": index,
                    "arm": arm,
                    "intent_id": intent_id,
                    "remaining_calls": remaining_calls,
                    "remaining_tokens": remaining_tokens,
                    "keychain_service": keychain_service,
                    "run_id": uuid4().hex,
                }
                try:
                    row = worker(job)
                    _validate_worker(row, job)
                except Exception as exc:
                    row = {
                        "stage": index,
                        "arm": arm,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "ledger_lost": True,
                        "calls": [
                            {
                                "kind": "unknown_worker",
                                "status": "reserved",
                                "charged_tokens": remaining_tokens,
                            }
                        ]
                        + [
                            {"kind": "unknown_worker", "status": "reserved", "charged_tokens": 0}
                            for _ in range(remaining_calls - 1)
                        ],
                        "spent_tokens": remaining_tokens,
                    }
                result["stages"].append(row)
                result["calls"].extend(row["calls"])
                result["spent_tokens"] += row["spent_tokens"]
                result["worker_calls"].extend(row["calls"])
                result["worker_spent_tokens"] += row["spent_tokens"]
                if on_update is not None and on_update(result) is False:
                    raise OSError("Report could not be persisted")
                if row["status"] != "complete":
                    break
            else:
                result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
    if on_update is not None:
        on_update(result)
    return result


def _validate_worker(row, job):
    if (
        not isinstance(row, dict)
        or type(row.get("stage")) is not int
        or row.get("stage") != job["index"]
        or row.get("arm") != job["arm"]
        or row.get("status") not in ("complete", "failed")
        or not isinstance(row.get("calls"), list)
        or len(row["calls"]) > job["remaining_calls"]
        or type(row.get("spent_tokens")) is not int
        or row["spent_tokens"] < 0
    ):
        raise ValueError("Invalid worker result")
    if any(
        not isinstance(call, dict)
        or call.get("kind") not in ("complete", "embed")
        or type(call.get("charged_tokens")) is not int
        or call["charged_tokens"] < 0
        for call in row["calls"]
    ):
        raise ValueError("Invalid worker ledger")
    if sum(call["charged_tokens"] for call in row["calls"]) != row["spent_tokens"]:
        raise ValueError("Invalid worker charge total")
    if row["spent_tokens"] > job["remaining_tokens"]:
        row["status"] = "failed"
        row["error_type"] = "BudgetExceeded"


def run_worker(job):
    """Secrets never cross argv, environment, or a job file; stdout has a hard read cap."""
    root = Path(__file__).resolve().parents[3]
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(root / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    row = None
    try:
        with tempfile.TemporaryFile() as output:
            completed = subprocess.run(
                [sys.executable, "-m", MODULE, "--worker"],
                input=json.dumps(job).encode("utf-8"),
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=180,
                check=False,
                env=environment,
                cwd=root,
            )
            output.seek(0)
            payload = output.read(MAX_WORKER_BYTES + 1)
        if completed.returncode != 0 or len(payload) > MAX_WORKER_BYTES:
            raise ValueError("Worker did not return a bounded result")
        from .protocol import loads_strict

        row = loads_strict(payload.decode("utf-8"))
        _validate_worker(row, job)
    finally:
        try:
            cleanup_run(job["run_id"])
        except Exception as exc:
            if row is None:
                raise
            row.update(status="failed", cleanup_error_type=type(exc).__name__)
    return row


def _keychain(service):
    if not isinstance(service, str) or not service or len(service) > 256 or "\n" in service:
        raise ValueError("Invalid Keychain service")
    result = subprocess.run(
        ["security", "find-generic-password", "-a", getpass.getuser(), "-s", service, "-w"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Credential unavailable")
    key = result.stdout.decode("utf-8").rstrip("\n")
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Credential unavailable")
    return key


async def worker_stage(job, case, *, client_factory=None, sandbox_factory=None):
    """Reopen state, read credentials locally, and preserve the ledger through close errors."""
    result = {
        "stage": job["index"],
        "arm": job["arm"],
        "status": "failed",
        "calls": [],
        "spent_tokens": 0,
    }
    client = None
    try:
        storage = _reopen(job["storage"])
        if client_factory is None:
            key = _keychain(job["keychain_service"])

            def client_factory(**limits):
                return MeteredOpenAI(key, **limits)

        if sandbox_factory is None:
            from .sandbox import DockerSandbox

            sandbox_factory = DockerSandbox
        client = client_factory(
            max_calls=job["remaining_calls"],
            max_tokens=job["remaining_tokens"],
            max_output_tokens_per_call=CODER_OUTPUT,
        )
        async with client:
            result = await runner.run_stage(
                storage,
                case,
                job["index"],
                job["arm"],
                job["intent_id"],
                client,
                sandbox_factory(run_id=job["run_id"]),
            )
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
    finally:
        if client is not None:
            result.update(calls=client.calls, spent_tokens=client.spent_tokens)
    return result


def _validate_job(job):
    expected = {
        "storage",
        "case_id",
        "index",
        "arm",
        "intent_id",
        "remaining_calls",
        "remaining_tokens",
        "keychain_service",
        "run_id",
    }
    if (
        not isinstance(job, dict)
        or set(job) != expected
        or job.get("arm") not in ARMS
        or type(job.get("index")) is not int
        or not 0 <= job["index"] < 3
        or type(job.get("remaining_calls")) is not int
        or not 1 <= job["remaining_calls"] <= MAX_CALLS
        or type(job.get("remaining_tokens")) is not int
        or not 1 <= job["remaining_tokens"] <= MAX_TOKENS
        or any(
            not isinstance(job.get(key), str)
            or not job[key]
            or len(job[key]) > 256
            or "\n" in job[key]
            or "\r" in job[key]
            for key in ("case_id", "intent_id", "keychain_service", "run_id")
        )
        or not re.fullmatch(r"[a-f0-9]{32}", job["run_id"])
    ):
        raise ValueError("Invalid worker job")
    descriptor = job["storage"]
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("backend") not in ("sqlite", "postgres")
        or set(descriptor) != {"backend", "path" if descriptor["backend"] == "sqlite" else "dsn"}
        or any(not isinstance(value, str) or not value for value in descriptor.values())
    ):
        raise ValueError("Invalid worker storage")


def load_cases():
    """CLI-only fixture import; generic runtime tests never invoke it."""
    module = importlib.import_module("tests.eval.team_repos.cases")
    if module.manifest_sha256() != CASE_MANIFEST_SHA256:
        raise ValueError("Sealed case manifest differs from the authored manifest")
    return module.CASES


def source_hashes():
    """Hash runtime and sealed bytes as metadata; never render or import the sealed cases."""
    root = Path(__file__).resolve().parents[3]
    files = [
        *sorted((root / "src" / "ai_knot").rglob("*.py")),
        *[
            root / "tests/eval/team_repos" / name
            for name in ("experiment.py", "runner.py", "protocol.py", "sandbox.py", "cases.py")
        ],
        root / "tests/eval/team_decisions/openai.py",
        root / "tests/eval/task_context/runner.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def _write_report(output, report):
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(output)


async def run_experiment(
    cases,
    work,
    *,
    backend="sqlite",
    output=None,
    keychain_service="",
    client_factory=None,
    worker=None,
    form=form_notes,
):
    cases = tuple(cases)
    before = source_hashes()
    report = {
        "status": "running",
        "backend": backend,
        "records": [],
        "formations": [],
        "source_sha256": before,
        "source_unchanged_during_run": True,
        "models_requested": ["gpt-4.1-mini", "text-embedding-3-small"],
        "per_case_arm_budget": {"calls": MAX_CALLS, "total_tokens": MAX_TOKENS},
        "case_manifest_sha256": CASE_MANIFEST_SHA256,
        "blinding_note": "Orchestration contributor accidentally saw two keyword-matched fixture "
        "lines before freeze; the primary implementer did not. No double-blind claim.",
        "interface_correction_note": "An explicit Case.task_key corrected repository identity "
        "versus task applicability after the initial freeze and public-brief access, before "
        "provider calls. This narrows the blinding claim. The original freeze is retained "
        "and a second freeze precedes live runs.",
        "unique_calls_plus_reservations": [],
        "unique_tokens_plus_reservations": 0,
    }

    write_failed = False

    def update(_=None):
        nonlocal write_failed
        ledger = [call for row in report["formations"] for call in row["calls"]]
        total = sum(row["spent_tokens"] for row in report["formations"])
        for row in report["records"]:
            ledger.extend(row["worker_calls"])
            total += row["worker_spent_tokens"]
        report["unique_calls_plus_reservations"] = ledger
        report["unique_tokens_plus_reservations"] = total
        known = [
            call
            for call in ledger
            if isinstance(call.get("usage"), dict)
            and type(call["usage"].get("total_tokens")) is int
        ]
        worker_unknown = [call for call in ledger if call["kind"] == "unknown_worker"]
        provider_unknown = [
            call for call in ledger if call not in known and call["kind"] != "unknown_worker"
        ]
        report["known_unique_calls"] = known
        report["known_unique_spent_tokens"] = sum(call["usage"]["total_tokens"] for call in known)
        report["unknown_worker_reserved_calls"] = len(worker_unknown)
        report["unknown_worker_reserved_tokens"] = sum(c["charged_tokens"] for c in worker_unknown)
        report["unknown_provider_calls"] = provider_unknown
        report["unknown_provider_reserved_tokens"] = sum(
            c["charged_tokens"] for c in provider_unknown
        )
        try:
            after = source_hashes()
            report["source_sha256_after"] = after
            report["source_unchanged_during_run"] &= before == after
            _write_report(output, report)
            return True
        except Exception as exc:
            write_failed = True
            report.update(status="failed", error_type=type(exc).__name__)
            return False

    if not update():
        return report
    try:
        if not cases:
            raise ValueError("An experiment requires authored cases")
        for case in cases:
            common = await form(case, client_factory)
            report["formations"].append({"case": case.id, **common})
            if not update():
                return report
            for arm in ARMS:
                attached = False

                def arm_update(row):
                    nonlocal attached
                    if not attached:
                        report["records"].append(row)
                        attached = True
                    return update()

                run_arm(
                    case,
                    arm,
                    common,
                    work / case.id / arm,
                    backend=backend,
                    keychain_service=keychain_service,
                    worker=worker,
                    on_update=arm_update,
                )
                if write_failed:
                    return report
        report["status"] = (
            "complete"
            if len(report["records"]) == len(cases) * len(ARMS)
            and all(row["status"] == "complete" for row in report["records"])
            else "failed"
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    update()
    if not report["source_unchanged_during_run"]:
        report["status"] = "invalidated_by_source_change"
        update()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keychain-service")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            from .protocol import loads_strict

            raw = sys.stdin.read(MAX_WORKER_BYTES + 1)
            if len(raw) > MAX_WORKER_BYTES:
                raise ValueError("Oversized worker job")
            job = loads_strict(raw)
            _validate_job(job)
            case = next(case for case in load_cases() if case.id == job["case_id"])
            result = asyncio.run(worker_stage(job, case))
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
            return 0
        except Exception as exc:
            print(json.dumps({"error_type": type(exc).__name__}))
            return 1
    if args.output is None or not args.keychain_service:
        parser.error("--output and --keychain-service are required")
    report = None
    try:
        key = _keychain(args.keychain_service)
        cases = load_cases()
        with tempfile.TemporaryDirectory(prefix="ai-knot-team-repositories-") as temporary:
            report = asyncio.run(
                run_experiment(
                    cases,
                    Path(temporary),
                    backend=args.backend,
                    output=args.output,
                    keychain_service=args.keychain_service,
                    client_factory=lambda **limits: MeteredOpenAI(key, **limits),
                )
            )
        _write_report(args.output, report)
        return 0 if report["status"] == "complete" else 1
    except Exception as exc:
        if report is None and args.output.exists():
            try:
                report = json.loads(args.output.read_text())
            except (OSError, ValueError):
                report = None
        if not isinstance(report, dict):
            report = {}
        report.update(status="failed", error_type=type(exc).__name__)
        try:
            _write_report(args.output, report)
        except Exception as write_error:
            print(json.dumps({"status": "failed", "error_type": type(write_error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
