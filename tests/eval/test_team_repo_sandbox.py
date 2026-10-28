"""The evaluation sandbox fails closed without executing candidates on the host."""

import json
import os
import subprocess
from pathlib import Path

import pytest


def _receipt(**updates):
    return {
        "tests_run": 1,
        "failures": 0,
        "errors": 0,
        "failed_test_names": [],
        **updates,
    }


def _fake_run(monkeypatch, payload=None, returncode=0):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "run":
            mount = command[command.index("--mount") + 1]
            folder = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            calls[-1] += ({p.name: p.read_text() for p in folder.iterdir()},)
            data = json.dumps(_receipt()) if payload is None else payload
            kwargs["stdout"].write(data.encode())
        return subprocess.CompletedProcess(command, returncode, stdout=b"")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_candidate_is_only_staged_in_a_fresh_restricted_container(monkeypatch):
    from tests.eval.team_repos.sandbox import DockerSandbox

    calls = _fake_run(monkeypatch)
    code = "raise RuntimeError('never execute on host')"
    tests = "import unittest\nclass Check(unittest.TestCase):\n def test_one(self): pass\n"
    sandbox = DockerSandbox()
    first = sandbox.run(code, tests)
    second = sandbox.run(code, tests)

    assert first["status"] == second["status"] == "pass"
    assert first["tests_run"] == 1 and first["exit_code"] == 0
    assert first["elapsed_seconds"] >= 0
    assert set(first) == {
        "status",
        "tests_run",
        "failures",
        "errors",
        "failed_test_names",
        "elapsed_seconds",
        "exit_code",
    }
    command, options, files = calls[0]
    assert command[:2] == ["docker", "run"]
    for flag in (
        "--rm",
        "--read-only",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=128m",
        "--cpus=1",
        "--pull=never",
        "--log-driver=none",
    ):
        assert flag in command
    assert command[command.index("--user") + 1] == "65534:65534"
    assert command[command.index("--tmpfs") + 1] == "/tmp:rw,noexec,nosuid,size=16m"
    assert command[command.index("--mount") + 1].endswith(",dst=/work,readonly")
    assert command.count("--mount") == 1
    assert command[-4:] == ["python", "-I", "-B", "/work/safecheckrunner.py"]
    assert command[-5].startswith("sha256:") and len(command[-5]) == 71
    assert options["timeout"] == 10
    assert options["stderr"] == subprocess.DEVNULL
    assert options["stdin"] == subprocess.DEVNULL
    assert set(options["env"]) <= {"PATH", "HOME"}
    assert files["solution.py"] == code and files["test_source.py"] == tests
    assert set(files) == {"solution.py", "test_source.py", "safecheckrunner.py"}
    assert command[command.index("--name") + 1] != calls[1][0][calls[1][0].index("--name") + 1]
    assert not Path(
        command[command.index("--mount") + 1].split("src=")[1].split(",dst=")[0]
    ).exists()


@pytest.mark.parametrize(
    "payload,returncode,status",
    [
        (json.dumps(_receipt(failures=1, failed_test_names=["Check.test_one"])), 1, "fail"),
        (json.dumps(_receipt(errors=1, failed_test_names=["Check.test_one"])), 1, "fail"),
        (json.dumps(_receipt(tests_run=0)), 0, "fail"),
        (json.dumps(_receipt()), 137, "error"),
        ("not JSON containing SECRET", 0, "error"),
        (json.dumps(_receipt()) + "\n" + json.dumps(_receipt()), 0, "error"),
        (json.dumps(_receipt(tests_run=True)), 0, "error"),
        (json.dumps(_receipt(failed_test_names=["secret expected = 42"])), 1, "error"),
        (json.dumps({**_receipt(), "traceback": "SECRET"}), 0, "error"),
        ("x" * 70000, 0, "error"),
        ("[" * 8000 + "]" * 8000, 0, "error"),
    ],
    ids=[
        "failure",
        "test_error",
        "zero_tests",
        "crash",
        "malformed",
        "extra_output",
        "boolean_count",
        "unsafe_test_name",
        "unknown_field",
        "oversize",
        "deep_json",
    ],
)
def test_only_valid_nonempty_success_receipts_pass(monkeypatch, payload, returncode, status):
    from tests.eval.team_repos.sandbox import DockerSandbox

    _fake_run(monkeypatch, payload, returncode)
    result = DockerSandbox().run("pass", "pass")

    assert result["status"] == status
    assert "SECRET" not in json.dumps(result)
    assert "secret expected" not in json.dumps(result)


@pytest.mark.parametrize(
    "code,tests", [("x" * 65537, "pass"), ("pass", "é" * 32769)], ids=["code", "test_utf8_bytes"]
)
def test_oversize_sources_do_not_start_docker(monkeypatch, code, tests):
    from tests.eval.team_repos.sandbox import DockerSandbox

    calls = _fake_run(monkeypatch)
    result = DockerSandbox().run(code, tests)
    assert result["status"] == "invalid_input" and result["exit_code"] is None
    assert calls == []


@pytest.mark.parametrize("owned", [True, False])
def test_timeout_removes_only_full_id_with_matching_ownership_label(monkeypatch, owned):
    from tests.eval.team_repos.sandbox import DockerSandbox

    calls = []
    identifier = "a" * 64

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1:3] == ["container", "inspect"]:
            label = calls[0][calls[0].index("--label") + 1].split("=", 1)[1]
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{identifier} {label if owned else 'foreign'}".encode()
            )
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(subprocess, "run", run)
    result = DockerSandbox(timeout_seconds=0.01).run("pass", "pass")
    assert result["status"] == "timeout" and result["exit_code"] is None
    assert calls[1][1:3] == ["container", "inspect"]
    assert calls[1][3] == calls[0][calls[0].index("--name") + 1]
    if owned:
        assert calls[2] == ["docker", "container", "rm", "--force", identifier]
    else:
        assert len(calls) == 2


def test_missing_docker_is_a_sanitized_error(monkeypatch):
    from tests.eval.team_repos.sandbox import DockerSandbox

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("SECRET host path")

    monkeypatch.setattr(subprocess, "run", unavailable)
    result = DockerSandbox().run("pass", "pass")
    assert result["status"] == "error"
    assert "SECRET" not in json.dumps(result)


def test_optional_run_label_allows_parent_cleanup_after_worker_crash(monkeypatch):
    from tests.eval.team_repos.sandbox import DockerSandbox

    calls = _fake_run(monkeypatch)
    run_id = "c" * 32
    assert DockerSandbox(run_id=run_id).run("pass", "pass")["status"] == "pass"
    command = calls[0][0]
    labels = [command[index + 1] for index, item in enumerate(command) if item == "--label"]
    assert f"ai-knot.repo-eval-run={run_id}" in labels
    assert any(label.startswith("ai-knot.eval.sandbox=") for label in labels)


@pytest.mark.parametrize("run_id", ["", "a" * 31, "a" * 33, "A" * 32, "foreign", True])
def test_invalid_run_identifiers_never_reach_docker(monkeypatch, run_id):
    from tests.eval.team_repos.sandbox import DockerSandbox, cleanup_run

    calls = _fake_run(monkeypatch)
    with pytest.raises(ValueError):
        DockerSandbox(run_id=run_id)
    with pytest.raises(ValueError):
        cleanup_run(run_id)
    assert calls == []


def test_parent_cleanup_checks_exact_run_owner_name_and_full_id_before_removal(monkeypatch):
    from tests.eval.team_repos.sandbox import cleanup_run

    run_id, owner = "c" * 32, "f" * 32
    identifiers = [character * 64 for character in "abcde"]
    inspected = {
        identifiers[0]: f"{identifiers[0]} /ai-knot-eval-{owner} {owner} {run_id}",
        identifiers[1]: f"{identifiers[1]} /ai-knot-eval-{owner} {owner} {'d' * 32}",
        identifiers[2]: f"{identifiers[2]} /unrelated-container {owner} {run_id}",
        identifiers[3]: f"{identifiers[3]} /ai-knot-eval-foreign foreign {run_id}",
        identifiers[4]: f"{'9' * 64} /ai-knot-eval-{owner} {owner} {run_id}",
    }
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=("\n".join([*identifiers, "short-id"])).encode()
            )
        if command[1:3] == ["container", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=inspected[command[3]].encode())
        return subprocess.CompletedProcess(command, 0, stdout=b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert cleanup_run(run_id) == 1
    assert f"label=ai-knot.repo-eval-run={run_id}" in calls[0]
    assert "label=ai-knot.eval.sandbox" in calls[0]
    assert [command for command in calls if command[1:3] == ["container", "rm"]] == [
        ["docker", "container", "rm", "--force", identifiers[0]],
    ]
    assert {command[3] for command in calls if command[1:3] == ["container", "inspect"]} == set(
        identifiers
    )


def test_parent_cleanup_fails_closed_when_docker_is_unavailable(monkeypatch):
    from tests.eval.team_repos.sandbox import cleanup_run

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("SECRET host diagnostic")

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert cleanup_run("a" * 32) == 0


@pytest.mark.skipif(
    os.environ.get("AI_KNOT_TEST_DOCKER_SANDBOX") != "1",
    reason="explicit opt-in required for disposable Docker probes",
)
@pytest.mark.parametrize(
    "code,decorator,test_count",
    [
        ("import unittest\ndef solve(): raise unittest.SkipTest('private reason')", "", 2),
        ("def solve(): return 0", "@unittest.expectedFailure", 2),
        (
            "import sys\ndef solve():\n sys._getframe(1).f_locals['self']._outcome.result.stop()\n"
            " return 42",
            "",
            1,
        ),
    ],
    ids=["skip", "expected_failure", "partial_suite"],
)
def test_real_docker_rejects_skipped_expected_failure_and_partial_suites(
    code,
    decorator,
    test_count,
):
    from tests.eval.team_repos.sandbox import DockerSandbox

    tests = f"""import unittest
from solution import solve
class Check(unittest.TestCase):
    {decorator}
    def test_a(self):
        self.assertEqual(solve(), 42)
    {decorator}
    def test_b(self):
        self.assertEqual(solve(), 42)
"""
    result = DockerSandbox().run(code, tests)
    assert result["status"] == "fail"
    assert result["exit_code"] == 1 and result["tests_run"] == test_count
    assert result["errors"] + result["failures"] > 0
    assert "private reason" not in json.dumps(result)
