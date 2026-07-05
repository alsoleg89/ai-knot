"""Regression checks for package and repo metadata surfaces."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_RELEASE_SCRIPT = REPO_ROOT / "scripts" / "check_public_release.py"
NPM_DIST_MARKER = REPO_ROOT / "npm" / "dist" / "esm" / "index.js"


def _load_release_audit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_release", PUBLIC_RELEASE_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pyproject_metadata_matches_public_positioning() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["description"] == "Deterministic, self-hosted long-term memory for AI agents."
    assert {
        "agent-memory",
        "ai-memory",
        "long-term-memory",
        "mcp",
        "langgraph",
        "llamaindex",
        "crewai",
        "autogen",
        "openai-agents",
        "pydanticai",
    } <= set(project["keywords"])
    assert project["urls"]["Benchmarks"].endswith("/docs/benchmarks.md")
    assert project["urls"]["Integrations"].endswith("/docs/integrations.md")
    assert project["urls"]["Changelog"].endswith("/CHANGELOG.md")


def test_npm_metadata_matches_public_distribution_story() -> None:
    package = json.loads((REPO_ROOT / "npm" / "package.json").read_text(encoding="utf-8"))
    tsconfig_build = json.loads(
        (REPO_ROOT / "npm" / "tsconfig.build.json").read_text(encoding="utf-8")
    )

    assert "TypeScript" in package["description"]
    assert "HTTP" in package["description"]
    assert "MCP" in package["description"]
    assert "Vercel AI SDK" in package["description"]
    assert package["bin"]["ai-knot-demo"] == "./scripts/demo.mjs"
    assert package["bin"]["ai-knot-doctor"] == "./scripts/doctor.mjs"
    assert package["scripts"]["doctor"] == "node scripts/doctor.mjs"
    assert package["scripts"]["package:audit"] == "node scripts/package-audit.mjs"
    assert "scripts/demo.mjs" in package["files"]
    assert "scripts/doctor.mjs" in package["files"]
    assert "src/__tests__/**/*" in tsconfig_build["exclude"]


def test_npm_package_audit_passes_on_current_repo() -> None:
    if shutil.which("npm") is None:
        pytest.skip("npm is required for the npm package audit")
    if not NPM_DIST_MARKER.exists():
        pytest.skip("npm dist is not built in this environment")

    result = subprocess.run(
        ["npm", "run", "package:audit"],
        cwd=REPO_ROOT / "npm",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No compiled test files in tarball." in result.stdout
    assert "Required runtime files present." in result.stdout


def test_mcp_registry_manifest_and_repo_audit_metadata_are_in_sync() -> None:
    module = _load_release_audit_module()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    server_manifest = json.loads((REPO_ROOT / "server.json").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    expected_version = pyproject["project"]["version"]
    expected_name = "io.github.alsoleg89/ai-knot"

    assert "<!-- mcp-name: io.github.alsoleg89/ai-knot -->" in readme
    assert server_manifest["name"] == expected_name
    assert server_manifest["version"] == expected_version
    assert server_manifest["packages"][0]["registryType"] == "pypi"
    assert server_manifest["packages"][0]["identifier"] == "ai-knot"
    assert server_manifest["packages"][0]["version"] == expected_version
    assert server_manifest["packages"][0]["transport"]["type"] == "stdio"
    assert module.RECOMMENDED_GITHUB_DESCRIPTION.startswith("Deterministic, self-hosted")
    assert module.RECOMMENDED_GITHUB_HOMEPAGE == "https://alsoleg89.github.io/ai-knot/"


def test_public_release_audit_workflow_requires_live_pages() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "public-launch-audit.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/check_public_release.py" in workflow
    assert "--require-pages" in workflow
    assert "--json-out /tmp/ai-knot-public-release-audit.json" in workflow
    assert "--summary-out /tmp/ai-knot-public-release-audit.md" in workflow
