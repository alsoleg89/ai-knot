"""Acceptance checks against public APIs and real process exits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.base import StorageBackend


@pytest.mark.parametrize(
    "crash_at", ["after_poll", "after_checkpoint", "after_begin", "after_external", "after_finish"]
)
def test_all_crash_windows(sql_storage: StorageBackend, tmp_path: Path, crash_at: str) -> None:
    from tests.eval.always_on.scenarios import crash_recovery

    result = crash_recovery(sql_storage, tmp_path, crash_at=crash_at)
    assert result["missing_committed_changes"] == 0
    assert result["unintended_duplicate_effects"] == 0
    assert result["accepted_stale_lease_mutations"] == 0
    assert result["false_auto_retries_unknown_actions"] == 0
    assert result["completed"] is True


def test_unknown_without_external_proof_stays_blocked(
    sql_storage: StorageBackend, tmp_path: Path
) -> None:
    from tests.eval.always_on.scenarios import crash_recovery

    result = crash_recovery(sql_storage, tmp_path, crash_at="after_external", reconcile=False)
    assert result["completed"] is False
    assert result["unknown_actions"] == 1
    assert result["external_effects"] == 1
    assert result["false_auto_retries_unknown_actions"] == 0


def test_snapshot_restore_does_not_rewind_action_or_delivery_history(
    sql_storage: StorageBackend,
) -> None:
    from tests.eval.always_on.scenarios import snapshot_history

    result = snapshot_history(sql_storage)
    assert result["action_survived"] is True
    assert result["old_ack_noop"] is True
    assert result["restore_change_delivered"] is True


def test_cross_owner_records_are_not_readable(sql_storage: StorageBackend) -> None:
    from ai_knot.always_on import AccessDenied, AlwaysOnMemory

    owner = AlwaysOnMemory(KnowledgeBase("alice", storage=sql_storage))
    foreign = AlwaysOnMemory(KnowledgeBase("bob", storage=sql_storage))
    intent = owner.create_intent(client_key="private", goal="Private work")
    with pytest.raises(AccessDenied):
        foreign.get_intent(intent.id)
    assert foreign.list_intents() == []


def test_benchmark_records_receipt_growth_and_index_plan(sql_storage: StorageBackend) -> None:
    from tests.eval.always_on.runner import measure

    report = measure(sql_storage, 10, 2)
    assert report["journal_rows"] == 6  # opt-in resync marker plus five fact changes
    assert report["receipt_rows"] == 10
    assert report["receipt_token_bytes"] > 0
    assert report["receipt_logical_bytes"] > report["receipt_token_bytes"]
    assert report["raw_events_scanned"] == 10
    assert report["largest_scan"] == 1
    assert report["delivered_batches"] == 10
    assert report["journal_logical_bytes"] > 0
    assert report["query_plan"]


def test_report_uses_nearest_rank_percentile() -> None:
    from tests.eval.always_on.runner import percentile

    assert percentile([1, 2, 3, 4, 5], 0.95) == 5


def test_acceptance_runner_emits_report(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "tests.eval.always_on.runner",
        "--storage",
        "sqlite",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads((tmp_path / "always_on_report.json").read_text())
    assert report["setup"]["storage"] == "sqlite"
    assert report["aggregate"]["passed"] is True
    assert len(report["scenarios"]) >= 10


def test_required_postgres_runner_fails_without_dsn(tmp_path: Path) -> None:
    import os

    env = dict(os.environ)
    env.pop("AI_KNOT_TEST_PG_DSN", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.eval.always_on.runner",
            "--storage",
            "postgres",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert result.returncode != 0
    assert "AI_KNOT_TEST_PG_DSN" in result.stderr
