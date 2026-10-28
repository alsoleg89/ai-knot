"""An attempted diagnostic socket connection fails closed with retained progress."""

import json
import socket
import sys

from tests.eval import sparse_source_diagnostic


def test_connect_ex_is_blocked_and_partial_evidence_is_saved(tmp_path, monkeypatch):
    output = tmp_path / "result.json"

    async def attempted(kind, directory, report, target):
        report["scenarios"].append({"scenario": "completed-control"})
        with socket.socket() as connection:
            connection.settimeout(0.01)
            connection.connect_ex(("127.0.0.1", 0))
        report["gate"] = {"passed": True}

    monkeypatch.setattr(sparse_source_diagnostic, "diagnostic", attempted)
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--backend", "sqlite", "--output", str(output)])
    assert sparse_source_diagnostic.main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "AssertionError"
    assert report["network_attempts"] == 1
    assert report["scenarios"] == [{"scenario": "completed-control"}]
