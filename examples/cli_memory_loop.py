"""Repo-native CLI proof of the ai-knot memory loop."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADDED_FACT_RE = re.compile(r"Added fact ([^:]+):")


def _run_cli(data_dir: str, *args: str) -> str:
    command = [
        sys.executable,
        "-m",
        "ai_knot.cli",
        "--storage",
        "sqlite",
        "--data-dir",
        data_dir,
        *args,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    cli_preview = " ".join(args)
    print(f"$ ai-knot {cli_preview}")
    print(result.stdout.rstrip())
    print()
    return result.stdout


def _parse_fact_id(output: str) -> str:
    match = ADDED_FACT_RE.search(output)
    if match is None:
        raise RuntimeError(f"Could not parse fact ID from CLI output: {output!r}")
    return match.group(1)


def main() -> None:
    agent_id = "assistant"

    with tempfile.TemporaryDirectory(prefix="ai-knot-cli-demo-") as data_dir:
        noisy_fact = _run_cli(data_dir, "add", agent_id, "Team standup is at 10am")
        noisy_fact_id = _parse_fact_id(noisy_fact)

        _run_cli(data_dir, "add", agent_id, "User prefers TypeScript for frontend work")
        _run_cli(data_dir, "add", agent_id, "User deploys APIs with Docker Compose")

        _run_cli(data_dir, "search", agent_id, "what does the user deploy with?")
        _run_cli(data_dir, "list", agent_id)
        _run_cli(data_dir, "delete", agent_id, noisy_fact_id)
        _run_cli(data_dir, "list", agent_id)


if __name__ == "__main__":
    main()
