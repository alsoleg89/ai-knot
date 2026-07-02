"""Tests for the public release-state verifier script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_public_release.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_release", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_versions_are_in_sync() -> None:
    module = _load_module()
    versions = module._local_versions(SCRIPT_PATH.parent.parent)
    assert (
        versions["pyproject"]
        == versions["init"]
        == versions["npm_package"]
        == versions["npm_lock"]
        == "0.11.0"
    )


def test_main_reports_failing_public_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def fake_fetch_json(url: str) -> dict[str, Any]:
        if url == module.PYPI_URL:
            return {"info": {"version": "0.11.0"}}
        if url == module.NPM_URL:
            return {
                "dist-tags": {"latest": "0.9.3"},
                "description": "Old npm description",
                "repository": {
                    "type": "git",
                    "url": "git+https://github.com/alsoleg89/ai_knot.git",
                    "directory": "npm",
                },
                "readme": "# stale",
            }
        if url == module.API_REPO_URL:
            return {
                "default_branch": "main",
                "stargazers_count": 1,
                "description": "Old repo description",
                "homepage": "",
                "has_pages": False,
                "topics": ["ai", "agents"],
                "updated_at": "2026-06-24T17:56:03Z",
            }
        raise AssertionError(f"unexpected URL: {url}")

    def fake_fetch_text(url: str) -> str:
        if url == f"{module.RAW_BASE_URL}/README.md":
            return "Old README without public markers"
        raise module.urllib.error.HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(module, "_fetch_text", fake_fetch_text)

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "[FAIL] npm matches local" in out
    assert "[FAIL] GitHub description" in out
    assert "[FAIL] public README marker: examples/function_calling_surface_demo.py" in out
    assert "[FAIL] public repo file: docs/memory-commands.md" in out
    assert "[FAIL] public repo file: scripts/render_github_release.py" in out
    assert "Likely next actions:" in out
    assert "Push or merge the current release-ready branch to public `main`." in out
    assert "Publish npm so the package page refreshes version" in out


def test_main_reports_green_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def fake_fetch_json(url: str) -> dict[str, Any]:
        if url == module.PYPI_URL:
            return {"info": {"version": "0.11.0"}}
        if url == module.NPM_URL:
            return {
                "dist-tags": {"latest": "0.11.0"},
                "description": json.loads(
                    (SCRIPT_PATH.parent.parent / "npm" / "package.json").read_text(encoding="utf-8")
                )["description"],
                "repository": {
                    "type": "git",
                    "url": "git+https://github.com/alsoleg89/ai-knot.git",
                    "directory": "npm",
                },
                "readme": "\n".join(module.NPM_README_MARKERS),
            }
        if url == module.API_REPO_URL:
            return {
                "default_branch": "main",
                "stargazers_count": 5,
                "description": module.RECOMMENDED_GITHUB_DESCRIPTION,
                "homepage": module.RECOMMENDED_GITHUB_HOMEPAGE,
                "has_pages": True,
                "topics": list(module.RECOMMENDED_GITHUB_TOPICS) + ["python", "typescript"],
            }
        raise AssertionError(f"unexpected URL: {url}")

    def fake_fetch_text(url: str) -> str:
        if url.endswith("/README.md"):
            return "\n".join(module.README_MARKERS)
        if any(url.endswith(path) for path in module.PUBLIC_FILE_MARKERS):
            return "ok"
        if url in {entry[0] for entry in module.PAGES_MARKERS}:
            for page_url, marker in module.PAGES_MARKERS:
                if url == page_url:
                    return marker
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(module, "_fetch_text", fake_fetch_text)

    code = module.main(["--require-pages"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Release gate is green." in out
    assert "[PASS] npm matches local" in out
    assert "[PASS] GitHub homepage" in out
    assert "[PASS] public npm README marker: ## Basic memory loop" in out
    assert "[PASS] GitHub Pages: https://alsoleg89.github.io/ai-knot/" in out


def test_main_writes_json_and_summary_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()

    def fake_fetch_json(url: str) -> dict[str, Any]:
        if url == module.PYPI_URL:
            return {"info": {"version": "0.11.0"}}
        if url == module.NPM_URL:
            return {
                "dist-tags": {"latest": "0.9.3"},
                "description": "Old npm description",
                "repository": {"type": "git", "url": "git+https://github.com/alsoleg89/ai_knot.git", "directory": "npm"},
                "readme": "# stale",
            }
        if url == module.API_REPO_URL:
            return {"description": "Old repo description", "homepage": "", "has_pages": False, "topics": []}
        raise AssertionError(f"unexpected URL: {url}")

    def fake_fetch_text(url: str) -> str:
        if url == f"{module.RAW_BASE_URL}/README.md":
            return "stale"
        raise module.urllib.error.HTTPError(url, 404, "not found", None, None)

    monkeypatch.setattr(module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(module, "_fetch_text", fake_fetch_text)

    json_out = tmp_path / "report.json"
    summary_out = tmp_path / "report.md"
    code = module.main(["--json-out", str(json_out), "--summary-out", str(summary_out)])

    assert code == 1
    report = json.loads(json_out.read_text(encoding="utf-8"))
    summary = summary_out.read_text(encoding="utf-8")
    assert report["local_version"] == "0.11.0"
    assert "npm matches local" in report["failures"]
    assert "Likely next actions:" in summary
