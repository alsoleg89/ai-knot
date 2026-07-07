"""Tests for the GitHub release-body renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_github_release.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_github_release", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_release_body_uses_changelog_entry() -> None:
    module = _load_module()
    body = module.render_release_body("0.11.0")

    assert body.startswith("# ai-knot v0.11.0")
    assert "Deterministic, self-hosted long-term memory for AI agents." in body
    assert "## Start here" in body
    assert "docs/integrations.md" in body
    assert "## Changelog" in body
    assert "Typed, validated configuration object" in body
    assert "## [Unreleased]" not in body


def test_main_fails_when_changelog_version_is_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    code = module.main(["--version", "9.9.9"])
    err = capsys.readouterr().err

    assert code == 1
    assert "CHANGELOG entry for version '9.9.9' not found" in err
