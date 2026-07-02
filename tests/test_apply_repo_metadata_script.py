"""Tests for the GitHub repo-metadata helper script."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "apply_repo_metadata.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("apply_repo_metadata", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_prints_dry_run_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert "Target repo: alsoleg89/ai-knot" in out
    assert "gh api repos/alsoleg89/ai-knot --method PATCH" in out
    assert "gh api repos/alsoleg89/ai-knot/topics --method PUT" in out
    assert "Homepage:" in out
    assert "https://alsoleg89.github.io/ai-knot/" in out
    assert "Dry run only." in out


def test_main_fails_apply_when_gh_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "_gh_available", lambda: False)

    code = module.main(["--apply"])
    err = capsys.readouterr().err

    assert code == 1
    assert "`gh` is not installed" in err


def test_main_runs_both_gh_commands_when_apply_is_requested(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    recorded: list[list[str]] = []

    monkeypatch.setattr(module, "_gh_available", lambda: True)
    monkeypatch.setattr(module, "_run", lambda command: recorded.append(command))

    code = module.main(["--apply", "--repo", "demo/repo"])
    out = capsys.readouterr().out

    assert code == 0
    assert len(recorded) == 3
    assert recorded[0][:4] == ["gh", "api", "repos/demo/repo", "--method"]
    assert recorded[1][:4] == ["gh", "api", "repos/demo/repo/topics", "--method"]
    assert recorded[2][:4] == ["gh", "api", "repos/demo/repo", "--method"]
    assert "GitHub repo metadata updated." in out


def test_main_reports_gh_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def boom(command: list[str]) -> None:
        raise subprocess.CalledProcessError(4, command)

    monkeypatch.setattr(module, "_gh_available", lambda: True)
    monkeypatch.setattr(module, "_run", boom)

    code = module.main(["--apply"])
    err = capsys.readouterr().err

    assert code == 4
    assert "gh api failed with exit code 4" in err
