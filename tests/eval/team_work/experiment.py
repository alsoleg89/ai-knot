"""One frozen paid run, fresh stage workers, and credential-free exact replay.

Offline mode uses reference solutions solely to test the wiring. It is never a
quality measurement and is not a fallback for live or replay failures.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from ai_knot.always_on.context import task_applicable
from tests.eval.task_context.runner import _reopen, isolated_backend, storage_descriptor
from tests.eval.team_decisions.openai import BudgetExceeded, MeteredOpenAI
from tests.eval.team_repos.experiment import _keychain
from tests.eval.team_repos.experiment import source_hashes as old_source_hashes
from tests.eval.team_repos.protocol import (
    context_chars,
    handoff_schema,
    parse_handoff,
    researcher_messages,
    safe_output,
)
from tests.eval.team_repos.runner import entries
from tests.eval.team_repos.sandbox import DEFAULT_IMAGE, DockerSandbox, cleanup_run

from .corpus import load_corpus
from .provider import PrefixClient, RecordingClient, ReplayClient, transport_failure, truncated_text
from .runner import ARMS, CODER_OUTPUT, publish_handoff, render_entries, run_stage
from .state import memory_for, publish_stage, source_facts

ROOT = Path(__file__).resolve().parents[3]
MODULE = "tests.eval.team_work.experiment"
CORPUS_SHA256 = "0b11a369992ee24052b69c014a8dcc4bf8e1e4bb2cd6b4a7ec40df5d997d782e"
MAX_CALLS, MAX_TOKENS = 12, 60000
SERVICE = "ai-knot.openai-api-key"
MAX_RESULT_BYTES = 8 * 1024 * 1024


class InvalidHandoff(ValueError):
    """A returned researcher proposal failed the public formation contract."""


def live_client(**authority):
    return MeteredOpenAI(_keychain(SERVICE), **authority)


def source_hashes():
    result = old_source_hashes()
    for path in sorted(Path(__file__).parent.glob("*")):
        if path.suffix in (".py", ".json"):
            result[str(path.relative_to(ROOT))] = sha_bytes(path.read_bytes())
    return result


def sha_bytes(raw):
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    """Atomic, owner-readable ledger; caller owns the output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def empty_meter():
    return dict(
        requests=[], meter_calls=[], spent_tokens=0, accounted_tokens=0, actual_provider_calls=0
    )


def costs(snapshot):
    # Pending reservations count as a call even before the wrapped meter enters.
    entered = len(snapshot["meter_calls"])
    pending = sum(
        r["status"] == "pending" and r["meter_start"] == entered for r in snapshot["requests"]
    )
    return entered + pending, snapshot["accounted_tokens"]


class Probe:
    """Observe transaction lifetime without altering storage or locking rules."""

    def __init__(self, storage):
        self.storage, self.depth = storage, 0

    def __getattr__(self, name):
        return getattr(self.storage, name)

    @contextmanager
    def always_on_transaction(self, *args, **kwargs):
        with self.storage.always_on_transaction(*args, **kwargs) as session:
            self.depth += 1
            try:
                yield session
            finally:
                self.depth -= 1


class OfflineClient:
    """Explicit plumbing oracle, not a model, never constructed in live/replay."""

    actual_provider_calls = 0

    def __init__(self, code="", **limits):
        self.code, self.calls, self.spent_tokens = code, [], 0
        self.limits = limits

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def charge(self, kind):
        if len(self.calls) >= self.limits["max_calls"]:
            raise BudgetExceeded("Offline call ceiling")
        self.calls.append(dict(kind=kind, status="ok", charged_tokens=1))
        self.spent_tokens += 1

    async def complete(self, messages, **kwargs):
        self.charge("complete")
        if "summary" in kwargs["response_schema"]["properties"]:
            return json.dumps({"summary": "Use the current source rules.", "basis": [1]})
        payload = json.loads(messages[-1]["content"])
        return json.dumps(
            {"action": "patch", "code": self.code, "basis": [len(payload["entries"])]}
        )

    async def embed(self, texts):
        self.charge("embed")
        # Deterministic identical vectors exercise acquisition, not relevance.
        return [[1.0, 0.0] for _ in texts]


async def form_notes(case, client, update):
    notes = []
    async with client:
        for index, stage in enumerate(case.stages):
            facts = [f for f in source_facts(case, index) if task_applicable(f, case.task_key)]
            visible = render_entries(entries(facts))
            if context_chars(visible) > 24000:
                raise ValueError("Formation context overflow")
            messages = researcher_messages(case, stage, visible)
            messages[0]["content"] += "\nThe summary must be at most 1200 characters."
            raw = await client.complete(
                messages, max_output_tokens=768, response_schema=handoff_schema(len(visible))
            )
            try:
                if safe_output(raw)[1]:
                    raise ValueError("Unsafe formation output")
                summary, refs = parse_handoff(raw, visible)
                if len(summary) > 1200:
                    raise ValueError("Formation summary overflow")
            except ValueError:
                raise InvalidHandoff("Researcher output rejected") from None
            notes.append(dict(summary=summary, basis=[asdict(r) for r in refs]))
            update()
    return notes


def initialize(storage, case):
    memory = memory_for(storage, 0)
    intent = memory.create_intent(client_key="continued-team-work", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="initialize")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress=dict(code=case.initial_code, last_receipt=None, history=[]),
        basis=(),
    )
    return intent.id


async def worker_stage(job, case, *, client_factory=None, sandbox_factory=DockerSandbox):
    result = dict(
        stage=job["index"],
        arm=job["arm"],
        status="running",
        pid=os.getpid(),
        provider=empty_meter(),
    )
    client = None
    replayer = None

    def save_stage(row):
        result["result"] = copy.deepcopy(row)
        write(job["result_path"], result)

    def save_provider(snapshot):
        result["provider"] = snapshot
        write(job["result_path"], result)

    try:
        storage = Probe(_reopen(job["storage"]))
        if job["mode"] == "replay":
            replayer = ReplayClient(
                job["recording"],
                allow_transport_failure=True,
                allow_truncated=True,
                allow_budget_refusal=True,
            )
            transport = replayer
        else:
            limits = dict(
                max_calls=job["remaining_calls"],
                max_tokens=job["remaining_tokens"],
                max_output_tokens_per_call=CODER_OUTPUT,
                completion_timeout=90,
            )

            def factory(**authority):
                if client_factory is not None:
                    return client_factory(**authority)
                if job["mode"] == "offline":
                    return OfflineClient(
                        code=case.stages[job["index"]].reference_solution, **authority
                    )
                if job["mode"] == "live":
                    return MeteredOpenAI(_keychain(SERVICE), **authority)
                raise ValueError("Unknown worker mode")

            transport = (
                PrefixClient(job["prefix_recording"], factory, **limits)
                if "prefix_recording" in job
                else factory(**limits)
            )
        client = RecordingClient(
            transport,
            on_update=save_provider,
            boundary=lambda: storage.depth == 0,
            retry_transport=True,
            allow_truncated=True,
        )
        async with client:
            row = await run_stage(
                storage,
                case,
                job["index"],
                job["arm"],
                job["intent_id"],
                job["note"],
                client,
                sandbox_factory(run_id=job["run_id"]),
                on_update=save_stage,
                observed_history=job.get("observed_history"),
            )
        result["result"] = row
        if replayer is not None:
            replayer.assert_consumed()
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        if not isinstance(exc, Exception):
            raise
    finally:
        if client is not None:
            result["provider"] = client.snapshot()
        write(job["result_path"], result)
    return result


def run_worker(job):
    """Restricted environment; SQL locator on stdin, no API secret propagation."""
    environment = dict(
        PATH=os.environ.get("PATH", os.defpath),
        HOME=str(Path.home()),
        PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
    )
    try:
        with tempfile.TemporaryFile() as output:
            done = subprocess.run(
                [sys.executable, "-m", MODULE, "--worker"],
                input=json.dumps(job).encode(),
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=420,
                env=environment,
                cwd=ROOT,
                check=False,
            )
        if done.returncode != 0:
            raise RuntimeError("Worker exited unsuccessfully")
        path = Path(job["result_path"])
        if path.stat().st_size > MAX_RESULT_BYTES:
            raise ValueError("Worker report exceeds byte ceiling")
        result = json.loads(path.read_text())
        if (
            result.get("status") != "complete"
            or result.get("stage") != job["index"]
            or result.get("arm") != job["arm"]
            or result.get("pid") == os.getpid()
        ):
            raise ValueError("Invalid worker receipt")
        return result
    finally:
        cleanup_run(job["run_id"])


def account(report):
    snapshots = [r["provider"] for r in report["formations"]] + [
        r["worker"]["provider"] for r in report["stages"] if "worker" in r
    ]
    lost = [r for r in report["stages"] if r.get("worker", {}).get("status") == "lost"]
    return dict(
        accounted_tokens=sum(s["accounted_tokens"] for s in snapshots),
        actual_provider_calls=sum(s["actual_provider_calls"] for s in snapshots),
        logical_calls=sum(costs(s)[0] for s in snapshots),
        unknown_worker_count=len(lost),
        actual_provider_calls_complete=not lost,
    )


def quality(report, cases):
    return {
        arm: dict(
            passed_stages=sum(
                r.get("worker", {}).get("result", {}).get("tests_passed", False)
                for r in report["stages"]
                if r["arm"] == arm
            ),
            planned_stages=len(cases) * 3,
            all_stages_passed=sum(
                all(
                    r.get("worker", {}).get("result", {}).get("tests_passed", False)
                    for r in report["stages"]
                    if r["arm"] == arm and r["case"] == case.id
                )
                for case in cases
            ),
            planned_cases=len(cases),
            final_stage_passed=sum(
                r.get("worker", {}).get("result", {}).get("tests_passed", False)
                for r in report["stages"]
                if r["arm"] == arm and r["index"] == 2
            ),
        )
        for arm in ARMS
    }


def healthy(snapshot):
    requests = snapshot["requests"]
    for index, request in enumerate(requests):
        if request["status"] == "ok":
            continue
        if request["status"] == "truncated":
            indexes = request["meter_call_indexes"]
            if (
                len(indexes) != 1
                or truncated_text(snapshot, indexes[0], request["max_output_tokens"]) is None
            ):
                return False
            continue
        if (
            request["status"] != "failed"
            or request.get("error_type") != "EvaluationError"
            or index + 1 == len(requests)
            or requests[index + 1]["status"] not in ("ok", "truncated")
        ):
            return False
        following = requests[index + 1]
        if any(
            request[key] != following[key]
            for key in ("kind", "inputs", "response_schema", "max_output_tokens")
        ):
            return False
        indexes = request["meter_call_indexes"]
        if len(indexes) != 1 or not transport_failure(snapshot["meter_calls"][indexes[0]]):
            return False
    return all(
        c["status"] == "ok" or transport_failure(c) or truncated_text(snapshot, index) is not None
        for index, c in enumerate(snapshot["meter_calls"])
    )


async def run_experiment(
    cases,
    work,
    output,
    *,
    mode,
    backend="sqlite",
    replay=None,
    prefix=None,
    client_factory=None,
    worker=run_worker,
):
    output, work = Path(output), Path(work)
    # A previous paid run must never be silently overwritten or resumed.
    with output.open("x"):
        pass
    if mode not in ("offline", "live", "replay") or (mode == "replay" and replay is None):
        raise ValueError("Explicit run mode required")
    cases = tuple(cases)
    report = dict(
        status="running",
        mode=mode,
        backend=backend,
        quality=None,
        formations=[],
        corpus_sha256=CORPUS_SHA256,
        source_sha256=source_hashes(),
        models=["gpt-4.1-mini", "text-embedding-3-small"],
        python=sys.version,
        sandbox_image=DEFAULT_IMAGE,
        protocol=dict(
            temperature=0,
            researcher_output_tokens=768,
            summary_chars=1200,
            coder_output_tokens=CODER_OUTPUT,
            attempts_per_stage=2,
            bounded_facts=8,
            bounded_content_chars=2000,
            full_json_chars=24000,
            coder_transport_retries=1,
            coder_timeout_seconds=90,
        ),
        per_case_arm_budget=dict(calls=MAX_CALLS, tokens=MAX_TOKENS),
        stages=[
            dict(case=c.id, arm=a, index=i, status="not_run")
            for c in cases
            for a in ARMS
            for i in range(3)
        ],
    )

    def update():
        report["accounting"] = account(report)
        write(output, report)

    update()
    try:
        if prefix is not None:
            if (
                mode != "live"
                or prefix["mode"] != "live"
                or prefix["corpus_sha256"] != report["corpus_sha256"]
                or prefix["models"] != report["models"]
                or prefix["per_case_arm_budget"] != report["per_case_arm_budget"]
                or any(report["protocol"].get(k) != v for k, v in prefix["protocol"].items())
            ):
                raise ValueError("Incompatible continuation prefix")
            report["prefix_source_sha256"] = prefix["source_sha256"]
            report["prefix_report_sha256"] = sha_bytes(json.dumps(prefix, sort_keys=True).encode())
        if replay is not None and (
            replay["status"] != "complete"
            or replay["mode"] != "live"
            or replay["source_sha256"] != report["source_sha256"]
        ):
            raise ValueError("Replay requires complete frozen live run")
        for case_number, case in enumerate(cases):
            common = dict(case=case.id, status="running", provider=empty_meter(), notes=[])
            report["formations"].append(common)

            def save_common(snapshot, common=common):
                common["provider"] = snapshot
                update()

            client = None
            try:
                if mode == "replay":
                    client = ReplayClient(replay["formations"][case_number]["provider"])
                else:
                    limits = dict(max_calls=3, max_tokens=18000, max_output_tokens_per_call=768)
                    factory = client_factory or (
                        OfflineClient if mode == "offline" else live_client
                    )
                    prior = (
                        next((r for r in prefix["formations"] if r["case"] == case.id), None)
                        if prefix
                        else None
                    )
                    if prior is not None:
                        transport = PrefixClient(prior["provider"], factory, **limits)
                    else:
                        transport = factory(**limits)
                    client = RecordingClient(transport, on_update=save_common)
                common["notes"] = await form_notes(case, client, update)
                common["status"] = "complete"
            except InvalidHandoff:
                common.update(status="failed", error_type="InvalidHandoff")
            except BaseException:
                common["status"] = "failed"
                raise
            finally:
                if client is not None:
                    common["provider"] = client.snapshot()
                update()
                if mode == "replay" and client is not None:
                    client.assert_consumed()
            for arm in ARMS:
                directory = work / case.id / arm
                directory.mkdir(parents=True, exist_ok=True)
                allocated_calls, allocated_tokens = (
                    costs(common["provider"]) if arm != "direct_sources" else (0, 0)
                )
                if arm != "direct_sources" and common["status"] != "complete":
                    for row in report["stages"]:
                        if row["case"] == case.id and row["arm"] == arm:
                            row.update(
                                status="formation_failed",
                                allocated_calls=allocated_calls,
                                allocated_tokens=allocated_tokens,
                            )
                    update()
                    continue
                with isolated_backend(backend, directory) as storage:
                    intent_id = initialize(storage, case)
                    observed_history = []
                    budget_exhausted = False
                    for index in range(3):
                        row = next(
                            r
                            for r in report["stages"]
                            if (r["case"], r["arm"], r["index"]) == (case.id, arm, index)
                        )
                        note = common["notes"][index] if arm != "direct_sources" else None
                        publish_stage(storage, case, index)
                        publish_handoff(storage, case, index, note)
                        if (
                            budget_exhausted
                            or allocated_calls >= MAX_CALLS
                            or allocated_tokens >= MAX_TOKENS
                        ):
                            budget_exhausted = True
                            row.update(
                                status="budget_exhausted",
                                allocated_calls=allocated_calls,
                                allocated_tokens=allocated_tokens,
                            )
                            update()
                            continue
                        job = dict(
                            storage=storage_descriptor(storage),
                            case_id=case.id,
                            index=index,
                            arm=arm,
                            intent_id=intent_id,
                            note=note,
                            remaining_calls=MAX_CALLS - allocated_calls,
                            remaining_tokens=MAX_TOKENS - allocated_tokens,
                            mode=mode,
                            run_id=uuid4().hex,
                            result_path=str((directory / f"stage-{index}.json").resolve()),
                            observed_history=observed_history,
                        )
                        if mode == "replay":
                            source = next(
                                r
                                for r in replay["stages"]
                                if (r["case"], r["arm"], r["index"]) == (case.id, arm, index)
                            )
                            job["recording"] = source["worker"]["provider"]
                        if prefix is not None:
                            prior = next(
                                r
                                for r in prefix["stages"]
                                if (r["case"], r["arm"], r["index"]) == (case.id, arm, index)
                            )
                            if "worker" in prior:
                                job["prefix_recording"] = prior["worker"]["provider"]
                        row["status"] = "running"
                        update()
                        try:
                            row["worker"] = worker(job)
                        except BaseException:
                            # The last durable pre-I/O reservation is retained, but
                            # an unknown worker may have entered another request.
                            # Reserve its entire remaining authority and stop.
                            row["worker"] = dict(
                                status="lost",
                                provider=dict(
                                    requests=[],
                                    meter_calls=[
                                        dict(kind="unknown_worker")
                                        for _ in range(job["remaining_calls"])
                                    ],
                                    spent_tokens=job["remaining_tokens"],
                                    accounted_tokens=job["remaining_tokens"],
                                    actual_provider_calls=0,
                                ),
                            )
                            row["status"] = "failed"
                            update()
                            partial = Path(job["result_path"])
                            try:
                                if partial.stat().st_size <= MAX_RESULT_BYTES:
                                    row["partial_worker"] = json.loads(partial.read_text())
                            except (OSError, ValueError):
                                row["partial_unreadable"] = True
                            raise
                        snapshot = row["worker"]["provider"]
                        count, tokens = costs(snapshot)
                        allocated_calls += count
                        allocated_tokens += tokens
                        row.update(
                            status="complete",
                            allocated_calls=allocated_calls,
                            allocated_tokens=allocated_tokens,
                        )
                        update()
                        stage = row["worker"].get("result", {})
                        observed_history = stage.get("observed_history", observed_history)
                        if (
                            stage.get("error_type") == "BudgetExceeded"
                            and allocated_calls <= MAX_CALLS
                            and allocated_tokens <= MAX_TOKENS
                        ):
                            budget_exhausted = True
                            row["status"] = "budget_exhausted"
                            update()
                            continue
                        receipts = [
                            a["receipt"] for a in stage.get("attempts", []) if "receipt" in a
                        ]
                        if (
                            row["worker"]["status"] != "complete"
                            or not healthy(snapshot)
                            or allocated_calls > MAX_CALLS
                            or allocated_tokens > MAX_TOKENS
                            or stage.get("status") in ("failed", "context_changed")
                            or not stage.get("source_unchanged", False)
                            or any(r["status"] not in ("pass", "fail") for r in receipts)
                        ):
                            row["status"] = "failed"
                            update()
                            raise RuntimeError("Worker or provider infrastructure failed")
        if source_hashes() != report["source_sha256"]:
            raise ValueError("Runtime changed during execution")
        report["status"] = "complete"
        if mode != "offline":
            report["quality"] = quality(report, cases)
    except BaseException as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, quality=None)
        if not isinstance(exc, Exception):
            raise
    finally:
        update()
    return report


def parity_projection(report):
    """Semantic outcomes only: no backend, process IDs, action IDs or latency."""
    stages = []
    for row in report["stages"]:
        stage = copy.deepcopy(row.get("worker", {}).get("result", {}))
        for attempt in stage.get("attempts", []):
            if "receipt" in attempt:
                attempt["receipt"].pop("elapsed_seconds", None)
        stages.append(
            dict(
                case=row["case"],
                arm=row["arm"],
                index=row["index"],
                status=row["status"],
                result=stage,
                allocated_calls=row.get("allocated_calls"),
                allocated_tokens=row.get("allocated_tokens"),
            )
        )
    return dict(
        status=report["status"],
        quality=report["quality"],
        notes=[r["notes"] for r in report["formations"]],
        stages=stages,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=("offline", "live", "replay"))
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--prefix", type=Path)
    args = parser.parse_args()
    cases = load_corpus(Path(__file__).with_name("cases.json"), expected_sha256=CORPUS_SHA256)
    if args.worker:
        job = json.load(sys.stdin)
        case = next(c for c in cases if c.id == job["case_id"])
        result = asyncio.run(worker_stage(job, case))
        return 0 if result["status"] == "complete" else 1
    if args.mode is None or args.output is None or args.work is None:
        parser.error("Explicit --mode, --output and --work required")
    replay = json.loads(args.replay.read_text()) if args.replay else None
    prefix = json.loads(args.prefix.read_text()) if args.prefix else None
    report = asyncio.run(
        run_experiment(
            cases,
            args.work,
            args.output,
            mode=args.mode,
            backend=args.backend,
            replay=replay,
            prefix=prefix,
        )
    )
    if replay is not None and report["status"] == "complete":
        report["parity"] = parity_projection(report) == parity_projection(replay)
        write(args.output, report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                accounting=report["accounting"],
                quality=report["quality"],
                parity=report.get("parity"),
            )
        )
    )
    return 0 if report["status"] == "complete" and report.get("parity", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
