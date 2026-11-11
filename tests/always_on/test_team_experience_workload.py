"""Executable team examples and bounded process workloads check durable invariants."""

import importlib.util
from pathlib import Path

import pytest


def load_module(relative):
    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_team_experience_example(tmp_path):
    report = load_module("examples/team_experience.py").example(tmp_path)
    assert report["observed_outcomes"] == ["succeeded", "failed"]
    assert report["peer_reused_current_lessons"] == 2
    assert report["independent_roots_for_original_and_copies"] == 1
    assert report["old_current_lesson_rejected_after_reconciliation"]
    assert report["historical_attempt_warning_survives"]
    assert report["model_calls"] == 0


def test_team_experience_workload_invariants(sql_storage):
    module = load_module("scripts/benchmark_team_experience.py")
    backend = "sqlite" if hasattr(sql_storage, "_db_path") else "postgres"
    location = str(sql_storage._db_path) if backend == "sqlite" else sql_storage._dsn
    report = module.benchmark(
        backend,
        location,
        owners=2,
        workers=4,
        actions=4,
        keys=2,
        replays=2,
        noise=12,
    )
    checks = report["invariants"]
    assert checks["record_calls"] == 32
    assert checks["durable_lessons"] == checks["unique_returned_lessons"] == 8
    assert checks["duplicate_bindings"] == 0
    assert checks["independent_roots_for_original_and_copies"] == 1
    assert checks["independent_sources_for_all_lessons"] == 4
    assert checks["stale_selections"] == 0
    assert checks["acl_rejected"] == 8
    assert checks["source_rejected"] == 2
    assert checks["outcome_rejected"] == 4
    assert checks["historical_attempt_warning_survives"]
    assert checks["reopened_without_pool"] == 2
    assert checks["distinct_record_processes"] == 4
    assert checks["overlapping_worker_intervals"]
    assert checks["model_calls"] == 0
    assert report["record"]["samples"] == 32
    assert report["return"]["samples"] >= 4


@pytest.mark.parametrize(
    "location,password",
    (
        ("postgresql://tester:pa%20ss%40word@localhost/test", "pa ss@word"),
        ("host=localhost dbname=test password='two words\\'\\\\value'", "two words'\\value"),
    ),
)
def test_worker_failure_redacts_credentials_and_removes_job(
    location, password, monkeypatch, tmp_path
):
    module = load_module("scripts/benchmark_team_experience.py")
    token = "synthetic-sensitive-lease-token"
    stderr = (
        f"connection={location}\npassword={password}\nlease_token='{token}'\n"
        "postgresql://other:unconfigured-password@localhost/test\n"
        "password='unconfigured conninfo password'\n"
        "lease_token='unconfigured-lease-token'\nUseful failure detail"
    )

    class FailedChild:
        returncode = 1

        def communicate(self, timeout):
            return "", stderr

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FailedChild())
    with pytest.raises(RuntimeError) as error:
        module.processes(
            [{"backend": "postgres", "location": location, "actions": [{"lease_token": token}]}],
            tmp_path,
        )
    message = str(error.value)
    for private in (
        location,
        password,
        token,
        "unconfigured-password",
        "unconfigured conninfo password",
        "unconfigured-lease-token",
    ):
        assert private not in message
    assert "Useful failure detail" in message
    assert not list(tmp_path.glob("job-*.json"))
