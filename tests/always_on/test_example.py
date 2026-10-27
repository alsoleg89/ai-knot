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
