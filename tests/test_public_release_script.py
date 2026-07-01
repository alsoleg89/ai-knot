"""Tests for the public launch-state verifier script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_public_release.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_release", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_versions_are_in_sync() -> None:
    module = _load_module()
    versions = module._local_versions(_SCRIPT_PATH.parent.parent)
    assert versions["pyproject"] == versions["init"] == versions["npm_package"] == "0.11.0"


def test_main_reports_failing_public_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def fake_fetch_json(url: str) -> dict[str, Any]:
        if url == module.PYPI_URL:
            return {"info": {"version": "0.11.0"}}
        if url == module.NPM_URL:
            return {"dist-tags": {"latest": "0.9.3"}}
        if url == module.API_REPO_URL:
            return {
                "default_branch": "main",
                "stargazers_count": 1,
                "updated_at": "2026-06-24T17:56:03Z",
            }
        raise AssertionError(f"unexpected URL: {url}")

    def fake_fetch_text(url: str) -> str:
        if url.endswith("/README.md"):
            return "Old README without launch markers"
        raise module.urllib.error.HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(module, "_fetch_text", fake_fetch_text)

    code = module.main()
    out = capsys.readouterr().out

    assert code == 1
    assert "PyPI latest: 0.11.0" in out
    assert "npm latest:  0.9.3" in out
    assert "[FAIL] npm matches local" in out
    assert "[FAIL] public docs file: docs/crewai-case-study.md" in out


def test_main_reports_green_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def fake_fetch_json(url: str) -> dict[str, Any]:
        if url == module.PYPI_URL:
            return {"info": {"version": "0.11.0"}}
        if url == module.NPM_URL:
            return {"dist-tags": {"latest": "0.11.0"}}
        if url == module.API_REPO_URL:
            return {
                "default_branch": "main",
                "stargazers_count": 5,
                "updated_at": "2026-07-01T00:00:00Z",
            }
        raise AssertionError(f"unexpected URL: {url}")

    def fake_fetch_text(url: str) -> str:
        if url.endswith("/README.md"):
            return "\n".join(module.README_MARKERS)
        if any(url.endswith(path) for path in module.DOC_MARKERS):
            return "ok"
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(module, "_fetch_text", fake_fetch_text)

    code = module.main()
    out = capsys.readouterr().out

    assert code == 0
    assert "Launch gate is green." in out
    assert "[PASS] npm matches local" in out


def test_main_reports_network_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def boom(url: str) -> dict[str, Any]:
        raise module.urllib.error.URLError("dns down")

    monkeypatch.setattr(module, "_fetch_json", boom)

    code = module.main()
    out = capsys.readouterr().out

    assert code == 2
    assert "FAIL network" in out
