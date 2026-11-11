import json
import subprocess
import sys
from pathlib import Path


def test_offline_recovery_twice(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "examples" / "always_on_recovery.py"),
        "--data-dir",
        str(tmp_path),
    ]
    for _ in range(2):
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "completed": True,
            "effects": 1,
            "stale_plan_blocked": True,
            "model_calls": 0,
            "no_idempotency_branch": "blocked",
        }


def test_two_agent_memory_return_example():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "shared_memory_return.py")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "current_context": ["Deploy in US"],
        "stale_context_rejected": True,
        "snapshot_recovered": True,
        "model_calls": 0,
    }


def test_recipient_roles_and_outcome_retraction_example():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "task_context.py")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "operator_role_fit": True,
        "reviewer_role_fit": True,
        "retracted_source_blocks_reuse": True,
        "snapshot_facts": 6,
        "snapshot_overflow_handled": True,
        "source_only_facts": 1,
        "model_calls": 0,
    }
