"""Tests for the local release preflight script."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_local_launch_ready.py"
NPM_DIST_MARKER = Path(__file__).resolve().parent.parent / "npm" / "dist" / "esm" / "index.js"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_local_launch_ready", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_reports_green_state_on_current_repo(
    capsys: pytest.CaptureFixture[str],
) -> None:
    if shutil.which("npm") is None:
        pytest.skip("npm is required for the local release preflight")
    if not NPM_DIST_MARKER.exists():
        pytest.skip("npm dist is not built in this environment")

    module = _load_module()

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 0
    assert "Local release preflight is green." in out
    assert "[PASS] local version sync" in out
    assert "[PASS] release notes render" in out
    assert "[PASS] site article render" in out
    assert "[PASS] whitepaper pdf render" in out
    assert "[PASS] npm package audit" in out


def test_main_reports_missing_required_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "REQUIRED_FILES", ["docs/definitely-missing.md"])

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "[FAIL] required file: docs/definitely-missing.md" in out
    assert "Local release preflight is NOT green." in out


def test_main_reports_release_render_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def boom(repo_root: Path, version: str) -> str:
        raise ValueError(f"broken release body for {version}")

    monkeypatch.setattr(module, "_render_release_body", boom)

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "[FAIL] release notes render: broken release body for 0.11.0" in out


def test_main_reports_site_article_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()

    def drift(repo_root: Path) -> dict[str, str]:
        return {"whitepaper.html": "bad", "developer-article.html": "bad"}

    monkeypatch.setattr(module, "_render_site_articles_artifacts", drift)

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert (
        "[FAIL] site article render: drift in ['whitepaper.html', 'developer-article.html']" in out
    )


def test_main_reports_npm_package_audit_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_check_npm_package_audit",
        lambda repo_root: ("npm package audit", False, "compiled tests leaked into tarball"),
    )

    code = module.main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "[FAIL] npm package audit: compiled tests leaked into tarball" in out


def test_npm_package_audit_builds_missing_dist_before_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    npm_dir = tmp_path / "npm"
    npm_dir.mkdir()
    (npm_dir / "dist" / "esm").mkdir(parents=True)
    (npm_dir / "dist" / "esm" / "index.js").write_text("", encoding="utf-8")
    (npm_dir / "dist" / "esm" / "index.d.ts").write_text("", encoding="utf-8")
    (npm_dir / "dist" / "cjs").mkdir(parents=True)
    (npm_dir / "dist" / "cjs" / "index.js").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        module.shutil, "which", lambda name: "/usr/bin/npm" if name == "npm" else None
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object):
        calls.append(cmd)
        if cmd == ["npm", "run", "build"]:
            (npm_dir / "dist" / "cjs" / "package.json").write_text(
                '{"type":"commonjs"}', encoding="utf-8"
            )
            return type(
                "Result", (), {"returncode": 0, "stdout": "> ai-knot@0.11.0 build\n", "stderr": ""}
            )()
        if cmd == ["npm", "run", "package:audit"]:
            return type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        "Tarball: ai-knot-0.11.0.tgz\n"
                        "Entries: 46\n"
                        "No compiled test files in tarball.\n"
                    ),
                    "stderr": "",
                },
            )()
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    label, ok, detail = module._check_npm_package_audit(tmp_path)

    assert label == "npm package audit"
    assert ok is True
    assert "built missing npm dist before audit" in detail
    assert calls == [["npm", "run", "build"], ["npm", "run", "package:audit"]]
