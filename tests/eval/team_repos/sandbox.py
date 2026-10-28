"""Disposable Docker execution for generated Python and private stdlib tests.

Only this module's three temporary files are mounted into a container. Receipts
contain counts and test names, never candidate output or assertion diagnostics.
Docker isolation protects the host; unittest in the same interpreter is not an
attestation mechanism against a candidate deliberately tampering with the grader.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

DEFAULT_IMAGE = "sha256:b64631e04e4920160c50fbe8d8df828f7f35f06f425cb44aa09bca53e708a35a"
MAX_SOURCE_BYTES = 65536
MAX_RECEIPT_BYTES = 16384
_OWNER_LABEL = "ai-knot.eval.sandbox"
_RUN_LABEL = "ai-knot.repo-eval-run"
_UUID_HEX = re.compile(r"[a-f0-9]{32}")
_TEST_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*")

_RUNNER = r"""
import json
import os
import re
import sys
import types
import unittest

# Save a private output descriptor before any candidate/test import; ordinary
# prints, direct writes to stdout/stderr, and child-process output go to /dev/null.
report_fd = os.dup(1)
sink = os.open(os.devnull, os.O_WRONLY)
os.dup2(sink, 1)
os.dup2(sink, 2)
os.close(sink)
emit, finish = os.write, os._exit
encode = json.JSONEncoder().encode
sys.path.insert(0, "/work")

class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.failure_count = 0
        self.error_count = 0
        self.names = []

    def record(self, test):
        name = type(test).__name__ + "." + getattr(test, "_testMethodName", "fixture")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*", name):
            name = "sandbox.fixture"
        if len(name) <= 128 and name not in self.names and len(self.names) < 64:
            self.names.append(name)

    def addFailure(self, test, error):
        self.failure_count += 1
        self.record(test)

    def addError(self, test, error):
        self.error_count += 1
        self.record(test)

    def addUnexpectedSuccess(self, test):
        self.failure_count += 1
        self.record(test)

    def addExpectedFailure(self, test, error):
        self.failure_count += 1
        self.record(test)

    def addSkip(self, test, reason):
        self.error_count += 1
        self.record(test)

    def addSubTest(self, test, subtest, error):
        if error is not None:
            if issubclass(error[0], test.failureException):
                self.addFailure(test, error)
            else:
                self.addError(test, error)

result = Result()
expected_count = 0
try:
    module = types.ModuleType("test_source")
    module.__file__ = "/work/test_source.py"
    sys.modules[module.__name__] = module
    with open(module.__file__, encoding="utf-8") as source:
        exec(compile(source.read(), module.__file__, "exec"), module.__dict__)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for value in list(module.__dict__.values()):
        if (isinstance(value, type) and issubclass(value, unittest.TestCase)
                and value is not unittest.TestCase and value.__module__ == module.__name__):
            suite.addTests(loader.loadTestsFromTestCase(value))
    expected_count = suite.countTestCases()
    suite.run(result)
except BaseException:
    result.error_count += 1
if result.testsRun != expected_count:
    result.error_count += 1

receipt = {"tests_run": result.testsRun, "failures": result.failure_count,
           "errors": result.error_count, "failed_test_names": result.names}
passed = result.testsRun > 0 and result.failure_count == 0 and result.error_count == 0
emit(report_fd, encode(receipt).encode("utf-8"))
finish(0 if passed else 1)
"""


class DockerSandbox:
    """Run each candidate in a new resource-limited container, without pulling."""

    def __init__(
        self, image: str = DEFAULT_IMAGE, timeout_seconds: float = 10, run_id: str | None = None
    ) -> None:
        if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]*", image):
            raise ValueError("invalid Docker image reference")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if run_id is not None and (not isinstance(run_id, str) or not _UUID_HEX.fullmatch(run_id)):
            raise ValueError("invalid evaluation run ID")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.run_id = run_id

    def run(self, code: str, test_source: str) -> dict:
        """Return a JSON-safe receipt; every incomplete or malformed run fails closed."""
        started = time.monotonic()
        receipt = {
            "status": "error",
            "tests_run": 0,
            "failures": 0,
            "errors": 0,
            "failed_test_names": [],
            "elapsed_seconds": 0.0,
            "exit_code": None,
        }
        try:
            if any(
                not isinstance(source, str) or len(source) > MAX_SOURCE_BYTES
                for source in (code, test_source)
            ):
                receipt["status"] = "invalid_input"
                return receipt
            sources = [source.encode("utf-8") for source in (code, test_source)]
            if any(len(source) > MAX_SOURCE_BYTES for source in sources):
                receipt["status"] = "invalid_input"
                return receipt
            token = uuid.uuid4().hex
            name = f"ai-knot-eval-{token}"
            # This is the Docker CLI's environment, never a container --env list.
            environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(Path.home())}
            with tempfile.TemporaryDirectory(prefix="ai-knot-eval-", dir="/tmp") as directory:
                folder = Path(directory)
                folder.chmod(0o755)
                for filename, content in zip(
                    ("solution.py", "test_source.py", "safecheckrunner.py"),
                    (*sources, _RUNNER.encode("utf-8")),
                    strict=True,
                ):
                    target = folder / filename
                    target.write_bytes(content)
                    target.chmod(0o644)
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--log-driver=none",
                    "--name",
                    name,
                    "--label",
                    f"{_OWNER_LABEL}={token}",
                ]
                if self.run_id is not None:
                    command += ["--label", f"{_RUN_LABEL}={self.run_id}"]
                command += [
                    "--network=none",
                    "--read-only",
                    "--user",
                    "65534:65534",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--pids-limit=64",
                    "--memory=128m",
                    "--cpus=1",
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,size=16m",
                    "--mount",
                    f"type=bind,src={folder},dst=/work,readonly",
                    "--workdir",
                    "/work",
                    self.image,
                    "python",
                    "-I",
                    "-B",
                    "/work/safecheckrunner.py",
                ]
                # A file sink avoids an unbounded communicate()/capture_output buffer.
                with tempfile.TemporaryFile() as output:
                    try:
                        completed = subprocess.run(
                            command,
                            stdin=subprocess.DEVNULL,
                            stdout=output,
                            stderr=subprocess.DEVNULL,
                            timeout=self.timeout_seconds,
                            env=environment,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        receipt["status"] = "timeout"
                        self._cleanup(name, token, environment)
                        return receipt
                    receipt["exit_code"] = completed.returncode
                    if completed.returncode not in (0, 1):
                        return receipt
                    output.seek(0)
                    payload = output.read(MAX_RECEIPT_BYTES + 1)
                if len(payload) > MAX_RECEIPT_BYTES:
                    return receipt
                result = json.loads(payload)
                if not self._valid_result(result):
                    return receipt
                receipt.update(result)
                receipt["status"] = (
                    "pass"
                    if (
                        completed.returncode == 0
                        and result["tests_run"] > 0
                        and result["failures"] == 0
                        and result["errors"] == 0
                    )
                    else "fail"
                )
        except UnicodeEncodeError:
            receipt["status"] = "invalid_input"
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        finally:
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return receipt

    @staticmethod
    def _valid_result(value: object) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "tests_run",
            "failures",
            "errors",
            "failed_test_names",
        }:
            return False
        if any(
            type(value[key]) is not int or not 0 <= value[key] <= 100000
            for key in ("tests_run", "failures", "errors")
        ):
            return False
        names = value["failed_test_names"]
        return (
            isinstance(names, list)
            and len(names) <= 64
            and all(
                isinstance(name, str) and len(name) <= 128 and _TEST_NAME.fullmatch(name)
                for name in names
            )
        )

    @staticmethod
    def _cleanup(name: str, token: str, environment: dict[str, str]) -> None:
        """After a timeout, remove only the full container ID bearing our token."""
        try:
            inspected = subprocess.run(
                [
                    "docker",
                    "container",
                    "inspect",
                    name,
                    "--format",
                    '{{.Id}} {{index .Config.Labels "' + _OWNER_LABEL + '"}}',
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=3,
                check=False,
            )
            identity = inspected.stdout.decode("ascii").strip().split()
            if (
                inspected.returncode == 0
                and len(identity) == 2
                and identity[1] == token
                and re.fullmatch(r"[a-f0-9]{64}", identity[0])
            ):
                subprocess.run(
                    ["docker", "container", "rm", "--force", identity[0]],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    timeout=3,
                    check=False,
                )
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass


def cleanup_run(run_id: str) -> int:
    """Clean containers from an interrupted worker, after verifying exact ownership.

    Only containers selected by the exact run label and bearing this sandbox's
    owner label, matching generated name, and matching full ID can be removed.
    """
    if not isinstance(run_id, str) or not _UUID_HEX.fullmatch(run_id):
        raise ValueError("invalid evaluation run ID")
    environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(Path.home())}
    try:
        listed = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"label={_RUN_LABEL}={run_id}",
                "--filter",
                f"label={_OWNER_LABEL}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            timeout=3,
            check=False,
        )
        if listed.returncode != 0:
            return 0
        identifiers = listed.stdout.decode("ascii").split()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 0
    removed = 0
    for identifier in identifiers:
        if not re.fullmatch(r"[a-f0-9]{64}", identifier):
            continue
        try:
            inspected = subprocess.run(
                [
                    "docker",
                    "container",
                    "inspect",
                    identifier,
                    "--format",
                    '{{.Id}} {{.Name}} {{index .Config.Labels "'
                    + _OWNER_LABEL
                    + '"}} {{index .Config.Labels "'
                    + _RUN_LABEL
                    + '"}}',
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=3,
                check=False,
            )
            identity = inspected.stdout.decode("ascii").split()
            if (
                inspected.returncode != 0
                or len(identity) != 4
                or identity[0] != identifier
                or identity[3] != run_id
                or not _UUID_HEX.fullmatch(identity[2])
                or identity[1] != f"/ai-knot-eval-{identity[2]}"
            ):
                continue
            result = subprocess.run(
                ["docker", "container", "rm", "--force", identifier],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=3,
                check=False,
            )
            removed += result.returncode == 0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return removed
