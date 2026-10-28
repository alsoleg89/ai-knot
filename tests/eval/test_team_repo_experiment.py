import asyncio
import io
import json
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest

from tests.eval.task_context.runner import _reopen
from tests.eval.team_decisions.openai import MeteredOpenAI
from tests.eval.team_repos import experiment, runner


def fixture():
    stages = []
    for index in range(3):
        docs = (
            SimpleNamespace(path="rule.md", text=f"Use rule {index}", task_key="sample"),
            SimpleNamespace(path="stable.md", text="Keep this rule", task_key=""),
            SimpleNamespace(path="other.md", text="UNRELATED", task_key="elsewhere"),
        )
        stages.append(
            SimpleNamespace(
                brief=f"Stage {index}",
                documents=docs,
                test_source="HIDDEN_TESTS",
                reference_solution="HIDDEN_REFERENCE",
            )
        )
    return SimpleNamespace(
        id="sample",
        task_key="sample",
        goal="Build value()",
        requirements=("value",),
        initial_code="INITIAL",
        stages=tuple(stages),
    )


def formation(tokens=90):
    return {
        "status": "complete",
        "notes": [None] * 3,
        "calls": [
            {
                "kind": "complete",
                "charged_tokens": tokens // 3,
                "usage": {"total_tokens": tokens // 3},
            }
        ]
        * 3,
        "spent_tokens": tokens,
    }


def passing_worker(seen):
    def worker(job):
        storage = _reopen(job["storage"])
        index = job["index"]
        memory = runner.memory_for(storage, index)
        intent = memory.get_intent(job["intent_id"])
        seen.append((job, intent.progress))
        lease = memory.claim(intent.id, worker_id="fake")
        progress = {**intent.progress, "code": f"CANDIDATE_{index}"}
        kwargs = {}
        if index < 2:
            kwargs["waiting_for"] = runner.WaitUntil(runner.moment(index + 1))
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state="waiting" if index < 2 else "completed",
            progress=progress,
            **kwargs,
        )
        return {
            "stage": index,
            "arm": job["arm"],
            "status": "complete",
            "calls": [{"kind": "complete", "charged_tokens": 100, "usage": {"total_tokens": 100}}],
            "spent_tokens": 100,
            "tests_passed": True,
        }

    return worker


def test_formation_projects_only_public_applicable_inputs_and_keeps_costs():
    requests = []

    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"summary":"Rules","basis":[1]}'},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            },
        )

    def factory(**kwargs):
        assert kwargs == {"max_calls": 3, "max_tokens": 18000, "max_output_tokens_per_call": 768}
        return MeteredOpenAI("offline", transport=httpx.MockTransport(response), **kwargs)

    result = asyncio.run(experiment.form_notes(fixture(), factory))
    assert result["status"] == "complete"
    assert len(result["notes"]) == 3 and len(result["calls"]) == 3
    assert result["spent_tokens"] == 33
    text = json.dumps(requests)
    assert "HIDDEN_TESTS" not in text and "HIDDEN_REFERENCE" not in text
    assert "UNRELATED" not in text
    assert all(note["basis"][0]["namespace"] == "__shared__" for note in result["notes"])


def test_business_task_key_drives_formation_and_all_four_arms_when_repository_id_differs(tmp_path):
    case = fixture()
    case.id = "repository-key"
    formation_inputs = []

    def response(request):
        payload = json.loads(request.content)
        formation_inputs.append(json.loads(payload["messages"][1]["content"])["entries"])
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"summary":"Current rule","basis":[1]}'},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
            },
        )

    def factory(**kwargs):
        return MeteredOpenAI("offline", transport=httpx.MockTransport(response), **kwargs)

    common = asyncio.run(experiment.form_notes(case, factory))
    assert common["status"] == "complete"
    for index, visible in enumerate(formation_inputs):
        assert f"rule.md\nUse rule {index}" in [entry["content"] for entry in visible]
        assert all("UNRELATED" not in entry["content"] for entry in visible)

    for arm in experiment.ARMS:
        contexts = []
        proceed = passing_worker([])

        def worker(job, contexts=contexts, proceed=proceed):
            storage = _reopen(job["storage"])
            memory = runner.memory_for(storage, job["index"])
            progress = memory.get_intent(job["intent_id"]).progress
            selected, _ = asyncio.run(
                runner.context_for(
                    storage, memory, case, job["index"], job["arm"], object(), progress["history"]
                )
            )
            contexts.append([entry["content"] for entry in selected])
            return proceed(job)

        row = experiment.run_arm(case, arm, common, tmp_path / arm, worker=worker)
        assert row["status"] == "complete"
        for index, content in enumerate(contexts):
            assert f"rule.md\nUse rule {index}" in content
            assert all("UNRELATED" not in text for text in content)


def test_arm_charges_full_common_formation_and_carries_code_and_changed_history(tmp_path):
    seen = []
    result = experiment.run_arm(
        fixture(), "raw_team_history", formation(), tmp_path, worker=passing_worker(seen)
    )
    assert result["status"] == "complete" and len(result["stages"]) == 3
    assert [item[0]["remaining_calls"] for item in seen] == [9, 8, 7]
    assert [item[0]["remaining_tokens"] for item in seen] == [59910, 59810, 59710]
    assert [item[1]["code"] for item in seen] == ["INITIAL", "CANDIDATE_0", "CANDIDATE_1"]
    assert [len(item[1]["history"]) for item in seen] == [2, 3, 4]
    assert result["spent_tokens"] == 390 and len(result["calls"]) == 6
    assert result["worker_spent_tokens"] == 300
    assert "UNRELATED" not in json.dumps(result)


def test_worker_lost_reserves_remaining_budget_retains_prior_stage_and_aborts(tmp_path):
    seen = []
    first = passing_worker(seen)

    def worker(job):
        if job["index"]:
            raise RuntimeError("private failure details")
        return first(job)

    result = experiment.run_arm(fixture(), "direct_sources", formation(), tmp_path, worker=worker)
    assert result["status"] == "failed" and len(result["stages"]) == 2
    assert result["spent_tokens"] == 60000 and len(result["calls"]) == 12
    assert result["allocated_formation_tokens"] == 0
    assert result["stages"][0]["tests_passed"]
    assert result["stages"][1]["error_type"] == "RuntimeError"
    assert "private failure details" not in json.dumps(result)


def test_cleanup_failures_preserve_formation_and_worker_charges(tmp_path, monkeypatch):
    class ClosingClient:
        calls = [{"kind": "complete", "charged_tokens": 73}]
        spent_tokens = 73

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise RuntimeError("secret close failure")

        async def complete(self, *args, **kwargs):
            return '{"summary":"Rules","basis":[1]}'

    notes = asyncio.run(experiment.form_notes(fixture(), lambda **kw: ClosingClient()))
    assert notes["status"] == "failed" and notes["spent_tokens"] == 73
    assert len(notes["calls"]) == 1
    original = experiment.isolated_backend

    @contextmanager
    def broken_close(*args):
        with original(*args) as storage:
            yield storage
        raise RuntimeError("secret DB cleanup failure")

    monkeypatch.setattr(experiment, "isolated_backend", broken_close)
    result = experiment.run_arm(
        fixture(), "task_memory", formation(), tmp_path, worker=passing_worker([])
    )
    assert result["status"] == "failed" and result["spent_tokens"] == 390
    assert len(result["stages"]) == 3 and len(result["calls"]) == 6
    assert "secret" not in json.dumps(result)


def test_process_contract_uses_stdin_bounded_output_and_no_credential_environment(monkeypatch):
    job = {
        "index": 0,
        "arm": "direct_sources",
        "remaining_calls": 12,
        "remaining_tokens": 60000,
        "keychain_service": "service",
        "run_id": "a" * 32,
    }
    monkeypatch.setenv("OPENAI_API_KEY", "secret-never-inherit")
    monkeypatch.setattr(experiment, "__name__", "__main__")

    def run(command, **kwargs):
        assert command[1:3] == ["-m", "tests.eval.team_repos.experiment"]
        assert command[-1] == "--worker" and "service" not in command
        assert json.loads(kwargs["input"]) == job
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert kwargs["timeout"] == 180
        kwargs["stdout"].write(
            json.dumps(
                {
                    "stage": 0,
                    "arm": "direct_sources",
                    "status": "complete",
                    "calls": [],
                    "spent_tokens": 0,
                }
            ).encode()
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", run)
    monkeypatch.setattr(experiment, "cleanup_run", lambda run_id: None)
    assert experiment.run_worker(job)["status"] == "complete"

    def too_large(command, **kwargs):
        kwargs["stdout"].write(b"x" * (experiment.MAX_WORKER_BYTES + 1))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", too_large)
    try:
        experiment.run_worker(job)
    except ValueError:
        pass
    else:
        raise AssertionError("oversized worker stdout accepted")


def test_timeout_cleans_only_the_owned_run(monkeypatch):
    cleaned = []

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 180)

    monkeypatch.setattr(experiment.subprocess, "run", timeout)
    monkeypatch.setattr(experiment, "cleanup_run", cleaned.append)
    try:
        experiment.run_worker({"run_id": "b" * 32})
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("lost worker accepted")
    assert cleaned == ["b" * 32]


def test_suite_deduplicates_actual_formation_and_writes_each_stage(tmp_path):
    updates = []

    async def notes(case, factory):
        return formation()

    result = asyncio.run(
        experiment.run_experiment(
            (fixture(),),
            tmp_path / "work",
            output=tmp_path / "report.json",
            form=notes,
            worker=passing_worker(updates),
        )
    )
    assert result["status"] == "complete"
    assert result["known_unique_spent_tokens"] == 1290
    assert len(result["known_unique_calls"]) == 15
    assert len(result["records"]) == 4
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["known_unique_spent_tokens"] == 1290
    assert saved["source_unchanged_during_run"]
    assert len(updates) == 12


def test_child_reopens_sql_and_keeps_ledger_when_client_close_fails(tmp_path):
    case = fixture()
    with experiment.isolated_backend("sqlite", tmp_path) as storage:
        intent_id = experiment._prepare_stage(storage, case, 0, None, None)
        descriptor = experiment.storage_descriptor(storage)

    class Client:
        calls = []
        spent_tokens = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise RuntimeError("secret close detail")

        async def complete(self, *args, **kwargs):
            self.calls.append({"kind": "complete", "charged_tokens": 17})
            self.spent_tokens += 17
            return '{"action":"patch","code":"CHILD_CANDIDATE","basis":[1]}'

    def factory(**kwargs):
        assert kwargs == {"max_calls": 9, "max_tokens": 59000, "max_output_tokens_per_call": 2048}
        return Client()

    class Sandbox:
        def __init__(self, **kwargs):
            assert kwargs == {"run_id": "c" * 32}

        def run(self, code, tests):
            return {
                "status": "pass",
                "tests_run": 1,
                "failures": 0,
                "errors": 0,
                "failed_test_names": [],
            }

    job = {
        "storage": descriptor,
        "case_id": case.id,
        "index": 0,
        "arm": "direct_sources",
        "intent_id": intent_id,
        "remaining_calls": 9,
        "remaining_tokens": 59000,
        "keychain_service": "service",
        "run_id": "c" * 32,
    }
    result = asyncio.run(
        experiment.worker_stage(job, case, client_factory=factory, sandbox_factory=Sandbox)
    )
    assert result["status"] == "failed" and result["spent_tokens"] == 17
    assert len(result["calls"]) == 1 and result["error_type"] == "RuntimeError"
    restored = runner.memory_for(_reopen(descriptor), 0).get_intent(intent_id)
    assert restored.progress["code"] == "CHILD_CANDIDATE"
    assert "secret close detail" not in json.dumps(result)


def test_unknown_effects_block_following_stage_without_resetting_progress(tmp_path):
    seen = []

    def worker(job):
        storage = _reopen(job["storage"])
        memory = runner.memory_for(storage, 0)
        lease = memory.claim(job["intent_id"], worker_id="effect")
        action = memory.prepare_action(
            job["intent_id"], lease_token=lease.token, action_key="effect", request={}, basis=()
        )
        memory.begin_action(action.id, lease_token=lease.token)
        seen.append(job)
        return {"stage": 0, "arm": job["arm"], "status": "complete", "calls": [], "spent_tokens": 0}

    result = experiment.run_arm(fixture(), "direct_sources", formation(), tmp_path, worker=worker)
    assert result["status"] == "failed" and len(seen) == 1
    assert result["spent_tokens"] == 0 and result["error_type"] == "ValueError"


def test_malformed_worker_ledger_reserves_budget_and_stops(tmp_path):
    def worker(job):
        return {"stage": 0, "arm": job["arm"], "status": "complete", "calls": [], "spent_tokens": 1}

    result = experiment.run_arm(fixture(), "task_memory", formation(), tmp_path, worker=worker)
    assert result["spent_tokens"] == 60000 and len(result["calls"]) == 12
    assert len(result["stages"]) == 1 and result["stages"][0]["ledger_lost"]


def test_worker_cli_loads_cases_once_and_emits_only_json(monkeypatch, capsys):
    loaded = []
    case = fixture()

    def load():
        loaded.append(True)
        return (case,)

    async def child(job, selected):
        assert selected is case and job["case_id"] == case.id
        return {
            "stage": 0,
            "arm": "direct_sources",
            "status": "failed",
            "calls": [],
            "spent_tokens": 0,
        }

    job = {
        "storage": {"backend": "sqlite", "path": "/tmp/owned.db"},
        "case_id": "sample",
        "index": 0,
        "arm": "direct_sources",
        "intent_id": "intent",
        "remaining_calls": 12,
        "remaining_tokens": 60000,
        "keychain_service": "service",
        "run_id": "d" * 32,
    }
    monkeypatch.setattr(experiment, "load_cases", load)
    monkeypatch.setattr(experiment, "worker_stage", child)
    monkeypatch.setattr(experiment.sys, "stdin", io.StringIO(json.dumps(job)))
    assert experiment.main(["--worker"]) == 0
    assert loaded == [True]
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_suite_invalidates_report_when_runtime_changes(tmp_path, monkeypatch):
    hashes = iter(({"runtime": "before"}, {"runtime": "after"}))
    monkeypatch.setattr(experiment, "source_hashes", lambda: next(hashes, {"runtime": "after"}))
    result = asyncio.run(experiment.run_experiment((), tmp_path))
    assert result["status"] == "invalidated_by_source_change"
    assert not result["source_unchanged_during_run"]


def test_unknown_worker_reservations_are_distinct_from_known_account_usage(tmp_path):
    async def notes(case, factory):
        return formation()

    def lost(job):
        raise RuntimeError("worker lost")

    result = asyncio.run(experiment.run_experiment((fixture(),), tmp_path, form=notes, worker=lost))
    assert result["known_unique_spent_tokens"] == 90
    assert len(result["known_unique_calls"]) == 3
    assert result["unknown_worker_reserved_tokens"] == 239730
    assert result["unknown_worker_reserved_calls"] == 39
    assert result["unique_tokens_plus_reservations"] == 239820
    assert len(result["unique_calls_plus_reservations"]) == 42


def test_formation_rejects_oversize_source_without_a_provider_call():
    case = fixture()
    # Source text fits; visible paths, indices, and metadata push the rendered context over.
    case.stages[0].documents[0].text = "x" * 23900

    class Client:
        calls = []
        spent_tokens = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def complete(self, *args, **kwargs):
            raise AssertionError("oversized context sent to provider")

    result = asyncio.run(experiment.form_notes(case, lambda **kwargs: Client()))
    assert result["status"] == "failed" and result["calls"] == []
    assert result["error_type"] == "ValueError"


def test_cli_temporary_cleanup_failure_never_erases_recorded_costs(tmp_path, monkeypatch):
    @contextmanager
    def broken_cleanup(**kwargs):
        yield str(tmp_path)
        raise RuntimeError("secret cleanup detail")

    async def suite(*args, **kwargs):
        report = {"status": "complete", "known_unique_spent_tokens": 145}
        experiment._write_report(kwargs["output"], report)
        return report

    monkeypatch.setattr(experiment.tempfile, "TemporaryDirectory", broken_cleanup)
    monkeypatch.setattr(experiment, "run_experiment", suite)
    monkeypatch.setattr(experiment, "_keychain", lambda service: "offline")
    monkeypatch.setattr(experiment, "load_cases", lambda: (fixture(),))
    output = tmp_path / "report.json"
    assert experiment.main(["--output", str(output), "--keychain-service", "service"]) == 1
    report = json.loads(output.read_text())
    assert report["known_unique_spent_tokens"] == 145 and report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"


def test_unknown_provider_billing_is_not_counted_as_known_usage(tmp_path):
    async def notes(case, factory):
        return {
            "status": "failed",
            "notes": [],
            "spent_tokens": 700,
            "calls": [{"kind": "complete", "charged_tokens": 700, "usage": None}],
        }

    result = asyncio.run(
        experiment.run_experiment((fixture(),), tmp_path, form=notes, worker=passing_worker([]))
    )
    assert result["known_unique_spent_tokens"] == 300
    assert result["unknown_provider_reserved_tokens"] == 700
    assert len(result["unknown_provider_calls"]) == 1
    assert result["unknown_worker_reserved_calls"] == 0
    assert result["unique_tokens_plus_reservations"] == 1000


def test_cli_final_write_failure_keeps_last_persisted_ledger(tmp_path, monkeypatch):
    original_write = experiment._write_report
    writes = 0

    def write(output, report):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("private write failure")
        original_write(output, report)

    async def suite(*args, **kwargs):
        row = {
            "status": "running",
            "known_unique_spent_tokens": 155,
            "records": [{"stages": [{"spent_tokens": 155}]}],
        }
        experiment._write_report(kwargs["output"], row)
        experiment._write_report(kwargs["output"], {**row, "status": "complete"})
        return row

    monkeypatch.setattr(experiment, "_write_report", write)
    monkeypatch.setattr(experiment, "run_experiment", suite)
    monkeypatch.setattr(experiment, "_keychain", lambda service: "offline")
    monkeypatch.setattr(experiment, "load_cases", lambda: (fixture(),))
    output = tmp_path / "report.json"
    assert experiment.main(["--output", str(output), "--keychain-service", "service"]) == 1
    saved = json.loads(output.read_text())
    assert saved["known_unique_spent_tokens"] == 155 and saved["error_type"] == "OSError"
    assert saved["records"][0]["stages"][0]["spent_tokens"] == 155


def test_cli_persistent_write_failure_is_sanitized_and_preserves_last_report(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "report.json"
    experiment._write_report(output, {"status": "running", "known_unique_spent_tokens": 166})

    def broken_write(*args):
        raise OSError("private filesystem diagnostics")

    async def suite(*args, **kwargs):
        raise OSError("private filesystem diagnostics")

    monkeypatch.setattr(experiment, "_write_report", broken_write)
    monkeypatch.setattr(experiment, "run_experiment", suite)
    monkeypatch.setattr(experiment, "_keychain", lambda service: "offline")
    monkeypatch.setattr(experiment, "load_cases", lambda: (fixture(),))
    assert experiment.main(["--output", str(output), "--keychain-service", "service"]) == 1
    assert json.loads(output.read_text())["known_unique_spent_tokens"] == 166
    assert json.loads(capsys.readouterr().out)["error_type"] == "OSError"


def test_stage_report_write_failure_returns_latest_ledger_and_stops_paid_work(
    tmp_path, monkeypatch
):
    writes = 0
    original_write = experiment._write_report

    def write(output, report):
        nonlocal writes
        writes += 1
        if writes >= 3:
            raise OSError("private write failure")
        original_write(output, report)

    async def notes(case, factory):
        return formation()

    seen = []
    monkeypatch.setattr(experiment, "_write_report", write)
    report = asyncio.run(
        experiment.run_experiment(
            (fixture(),),
            tmp_path / "work",
            output=tmp_path / "report.json",
            form=notes,
            worker=passing_worker(seen),
        )
    )
    assert report["status"] == "failed"
    assert report["known_unique_spent_tokens"] == 190
    assert len(seen) == 1 and len(report["records"][0]["stages"]) == 1
    assert report["error_type"] == "OSError"


def test_empty_experiment_cannot_report_verified_completion(tmp_path):
    result = asyncio.run(experiment.run_experiment((), tmp_path))
    assert result["status"] == "failed" and result["error_type"] == "ValueError"


def test_setup_failure_retains_allocated_formation_cost(tmp_path):
    target = tmp_path / "already-file"
    target.write_text("occupied")
    result = experiment.run_arm(
        fixture(), "task_memory", formation(), target, worker=passing_worker([])
    )
    assert result["status"] == "failed" and result["spent_tokens"] == 90
    assert result["error_type"] == "FileExistsError" and result["worker_calls"] == []


@pytest.mark.parametrize(
    "extra", [{"api_key": "forbidden"}, {"index": True}, {"keychain_service": "service\n"}]
)
def test_worker_cli_rejects_bad_jobs_before_case_loading(monkeypatch, capsys, extra):
    job = {
        "storage": {"backend": "sqlite", "path": "/tmp/owned.db"},
        "case_id": "sample",
        "index": 0,
        "arm": "direct_sources",
        "intent_id": "intent",
        "remaining_calls": 12,
        "remaining_tokens": 60000,
        "keychain_service": "service",
        "run_id": "d" * 32,
        **extra,
    }

    def forbidden_load():
        raise AssertionError("fixtures loaded before validating job")

    monkeypatch.setattr(experiment, "load_cases", forbidden_load)
    monkeypatch.setattr(experiment.sys, "stdin", io.StringIO(json.dumps(job)))
    assert experiment.main(["--worker"]) == 1
    assert json.loads(capsys.readouterr().out)["error_type"] == "ValueError"


def test_invalid_child_call_kind_cannot_silently_erase_accounting(tmp_path):
    def worker(job):
        return {
            "stage": 0,
            "arm": job["arm"],
            "status": "complete",
            "spent_tokens": 10,
            "calls": [{"charged_tokens": 10}],
        }

    result = experiment.run_arm(fixture(), "direct_sources", formation(), tmp_path, worker=worker)
    assert result["spent_tokens"] == 60000 and result["stages"][0]["ledger_lost"]


def test_source_hash_failure_after_stage_retains_latest_cost_and_stops(tmp_path, monkeypatch):
    checks = 0

    def hashes():
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise OSError("source cannot be inspected")
        return {"runtime": "fixed"}

    async def notes(case, factory):
        return formation()

    seen = []
    monkeypatch.setattr(experiment, "source_hashes", hashes)
    result = asyncio.run(
        experiment.run_experiment((fixture(),), tmp_path, form=notes, worker=passing_worker(seen))
    )
    assert result["status"] == "failed" and len(seen) == 1
    assert result["known_unique_spent_tokens"] == 190
