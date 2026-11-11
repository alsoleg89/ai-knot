import asyncio
import json
from pathlib import Path

import pytest

from tests.eval.test_team_work_state import sample


def test_schedule_is_saved_before_first_formation_and_failure_stops_all_calls(tmp_path):
    from tests.eval.team_work.experiment import run_experiment

    calls = []
    output = tmp_path / "report.json"

    def fail(**kwargs):
        saved = json.loads(output.read_text())
        assert len(saved["stages"]) == 54
        assert all(r["status"] == "not_run" for r in saved["stages"])
        calls.append(kwargs)
        raise RuntimeError("Do not leak provider diagnostics")

    result = asyncio.run(
        run_experiment(
            tuple(sample(i) for i in range(3)),
            tmp_path / "work",
            output,
            mode="offline",
            client_factory=fail,
        )
    )
    assert len(calls) == 1
    assert result["status"] == "incomplete"
    assert result["quality"] is None
    assert result["accounting"]["accounted_tokens"] == 0
    assert all(r["status"] == "not_run" for r in result["stages"])
    assert "Do not leak" not in output.read_text()


def test_offline_formation_is_public_and_citation_roots_are_current(tmp_path):
    from tests.eval.team_work.experiment import form_notes
    from tests.eval.team_work.provider import RecordingClient

    class Client:
        calls = []
        spent_tokens = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def complete(self, messages, **kwargs):
            text = json.dumps(messages)
            assert "PRIVATE_TEST_SENTINEL" not in text
            assert "PRIVATE_REFERENCE_SENTINEL" not in text
            assert "1200" in text
            self.calls.append(dict(kind="complete", status="ok", charged_tokens=1))
            self.spent_tokens += 1
            return json.dumps({"summary": "Value rule", "basis": [1]})

    client = RecordingClient(Client(), on_update=lambda snapshot: None)
    notes = asyncio.run(form_notes(sample(), client, lambda: None))
    assert len(notes) == 3
    assert notes[0]["basis"][0] != notes[1]["basis"][0]


def test_output_refuses_overwrite(tmp_path):
    from tests.eval.team_work.experiment import run_experiment

    output = tmp_path / "report.json"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        asyncio.run(run_experiment((sample(),), tmp_path / "work", output, mode="offline"))
    assert output.read_text() == "keep"


@pytest.mark.parametrize("failure", ["lost", "interrupt", "corrupt"])
def test_lost_worker_reserves_remaining_authority_and_stops_schedule(tmp_path, failure):
    from tests.eval.team_work.experiment import OfflineClient, run_experiment

    jobs = []

    def lost(job):
        jobs.append(job)
        if failure == "corrupt":
            Path(job["result_path"]).write_text("{")
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("worker vanished")

    kwargs = dict(mode="offline", client_factory=OfflineClient, worker=lost)
    output = tmp_path / "report.json"
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(run_experiment((sample(),), tmp_path / "work", output, **kwargs))
        report = json.loads(output.read_text())
    else:
        report = asyncio.run(run_experiment((sample(),), tmp_path / "work", output, **kwargs))
    assert len(jobs) == 1
    assert report["status"] == "incomplete" and report["quality"] is None
    assert report["accounting"]["accounted_tokens"] == 60003
    assert report["accounting"]["unknown_worker_count"] == 1
    assert not report["accounting"]["actual_provider_calls_complete"]
    assert sum(r["status"] == "not_run" for r in report["stages"]) == 17


def test_successful_run_charges_common_formation_to_each_consuming_arm(tmp_path):
    from tests.eval.team_work.experiment import OfflineClient, empty_meter, run_experiment

    def worker(job):
        meter = empty_meter()
        meter.update(
            meter_calls=[dict(kind="complete", status="ok", charged_tokens=2)],
            spent_tokens=2,
            accounted_tokens=2,
        )
        return dict(
            status="complete",
            provider=meter,
            result=dict(status="complete", source_unchanged=True, tests_passed=True, attempts=[]),
        )

    result = asyncio.run(
        run_experiment(
            (sample(),),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="offline",
            client_factory=OfflineClient,
            worker=worker,
        )
    )
    assert result["status"] == "complete" and result["quality"] is None
    assert result["accounting"]["accounted_tokens"] == 39
    for arm in {r["arm"] for r in result["stages"]}:
        final = next(r for r in result["stages"] if r["arm"] == arm and r["index"] == 2)
        assert final["allocated_calls"] == (3 if arm == "direct_sources" else 6)
        assert final["allocated_tokens"] == (6 if arm == "direct_sources" else 9)


def test_provider_failure_preserves_charge_and_all_unrun_denominators(tmp_path):
    from tests.eval.team_work.experiment import OfflineClient, empty_meter, run_experiment

    jobs = []

    def worker(job):
        jobs.append(job)
        meter = empty_meter()
        meter.update(
            requests=[dict(status="failed", meter_start=0)],
            meter_calls=[dict(kind="complete", status="error", charged_tokens=5)],
            spent_tokens=5,
            accounted_tokens=5,
            actual_provider_calls=1,
        )
        return dict(
            status="complete",
            provider=meter,
            result=dict(status="failed", source_unchanged=True, attempts=[]),
        )

    result = asyncio.run(
        run_experiment(
            tuple(sample(i) for i in range(3)),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="offline",
            client_factory=OfflineClient,
            worker=worker,
        )
    )
    assert len(jobs) == 1 and len(result["stages"]) == 54
    assert result["status"] == "incomplete" and result["quality"] is None
    assert result["accounting"]["accounted_tokens"] == 8
    assert result["accounting"]["actual_provider_calls"] == 1
    assert result["stages"][0]["status"] == "failed"
    assert sum(r["status"] == "not_run" for r in result["stages"]) == 53


def test_arm_budget_failure_does_not_prevent_other_arms_from_running(tmp_path):
    from tests.eval.team_work.experiment import OfflineClient, empty_meter, quality, run_experiment

    jobs = []

    def worker(job):
        jobs.append(job)
        refused = job["arm"] == "direct_sources"
        return dict(
            status="complete",
            provider=empty_meter(),
            result=dict(
                status="failed" if refused else "complete",
                error_type="BudgetExceeded" if refused else None,
                source_unchanged=True,
                tests_passed=not refused,
                attempts=[],
            ),
        )

    case = sample()
    result = asyncio.run(
        run_experiment(
            (case,),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="offline",
            client_factory=OfflineClient,
            worker=worker,
        )
    )
    assert result["status"] == "complete" and len(jobs) == 16
    assert sum(r["status"] == "budget_exhausted" for r in result["stages"]) == 3
    assert quality(result, (case,))["direct_sources"]["passed_stages"] == 0
    assert quality(result, (case,))["semantic"]["passed_stages"] == 3


def test_invalid_common_formation_fails_consumers_but_keeps_direct_control(tmp_path):
    from tests.eval.team_work.experiment import OfflineClient, empty_meter, quality, run_experiment

    class BadSummary(OfflineClient):
        async def complete(self, *args, **kwargs):
            self.charge("complete")
            return json.dumps({"summary": "x" * 1201, "basis": [1]})

    jobs = []

    def worker(job):
        jobs.append(job)
        return dict(
            status="complete",
            provider=empty_meter(),
            result=dict(status="complete", source_unchanged=True, tests_passed=True, attempts=[]),
        )

    case = sample()
    report = asyncio.run(
        run_experiment(
            (case,),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="offline",
            client_factory=BadSummary,
            worker=worker,
        )
    )
    assert report["status"] == "complete" and len(jobs) == 3
    assert all(j["arm"] == "direct_sources" for j in jobs)
    assert sum(r["status"] == "formation_failed" for r in report["stages"]) == 15
    assert report["formations"][0]["error_type"] == "InvalidHandoff"
    assert report["accounting"]["accounted_tokens"] == 1
    assert quality(report, (case,))["direct_sources"]["passed_stages"] == 3
    assert quality(report, (case,))["semantic"]["passed_stages"] == 0


def test_reported_overspend_is_not_mistaken_for_a_zero_charge_budget_refusal(tmp_path):
    from tests.eval.team_work.experiment import OfflineClient, empty_meter, run_experiment

    def worker(job):
        meter = empty_meter()
        meter.update(
            meter_calls=[dict(kind="complete", status="budget_exceeded", charged_tokens=60001)],
            spent_tokens=60001,
            accounted_tokens=60001,
        )
        return dict(
            status="complete",
            provider=meter,
            result=dict(
                status="failed", error_type="BudgetExceeded", source_unchanged=True, attempts=[]
            ),
        )

    report = asyncio.run(
        run_experiment(
            (sample(),),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="offline",
            client_factory=OfflineClient,
            worker=worker,
        )
    )
    assert report["status"] == "incomplete" and report["quality"] is None
    assert report["accounting"]["accounted_tokens"] == 60004
    assert sum(r["status"] == "not_run" for r in report["stages"]) == 17


def test_invalid_formation_replay_rejects_unconsumed_trailing_request(tmp_path):
    from tests.eval.team_work.experiment import (
        InvalidHandoff,
        OfflineClient,
        empty_meter,
        form_notes,
        run_experiment,
        source_hashes,
    )
    from tests.eval.team_work.provider import RecordingClient

    case = sample()

    class BadSummary(OfflineClient):
        async def complete(self, *args, **kwargs):
            self.charge("complete")
            return json.dumps({"summary": "x" * 1201, "basis": [1]})

    async def recording():
        client = RecordingClient(BadSummary(max_calls=3), on_update=lambda snapshot: None)
        with pytest.raises(InvalidHandoff):
            await form_notes(case, client, lambda: None)
        # A well-formed extra source request must not disappear when local
        # formation parsing rejects the first successful provider response.
        first = client.snapshot()["requests"][0]
        async with client:
            await client.complete(
                first["inputs"],
                max_output_tokens=first["max_output_tokens"],
                response_schema=first["response_schema"],
            )
        return client.snapshot()

    source = asyncio.run(recording())
    replay = dict(
        status="complete",
        mode="live",
        source_sha256=source_hashes(),
        formations=[dict(provider=source)],
        stages=[
            dict(case=case.id, arm="direct_sources", index=i, worker=dict(provider=empty_meter()))
            for i in range(3)
        ],
    )

    def worker(job):
        return dict(
            status="complete",
            provider=empty_meter(),
            result=dict(status="complete", source_unchanged=True, tests_passed=True, attempts=[]),
        )

    result = asyncio.run(
        run_experiment(
            (case,),
            tmp_path / "work",
            tmp_path / "report.json",
            mode="replay",
            replay=replay,
            worker=worker,
        )
    )
    assert result["status"] == "incomplete" and result["quality"] is None
    assert result["error_type"] == "ValueError"
    assert result["accounting"]["actual_provider_calls"] == 0
