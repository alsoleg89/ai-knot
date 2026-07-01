#!/usr/bin/env python3
"""Check whether the public ai-knot release state matches the local branch."""

from __future__ import annotations

import json
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

REPO = "alsoleg89/ai-knot"
API_REPO_URL = f"https://api.github.com/repos/{REPO}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO}/main"
PYPI_URL = "https://pypi.org/pypi/ai-knot/json"
NPM_URL = "https://registry.npmjs.org/ai-knot"

README_MARKERS = [
    "Install by surface",
    "What it looks like in your stack",
    "skills/README.md",
    "Browser inspector",
    "examples/crewai_surface_demo.py",
    "docs/launch-checklist.md",
]

PUBLIC_FILE_MARKERS = [
    "docs/crewai-case-study.md",
    "docs/openclaw-case-study.md",
    "docs/claude-mcp-case-study.md",
    "docs/publish-ready-audit.md",
    "docs/readme-patterns.md",
    "docs/site/index.html",
    "examples/notebook_walkthrough.ipynb",
    "skills/README.md",
]


def _fetch_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _local_versions(repo_root: Path) -> dict[str, str]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    npm_package = json.loads((repo_root / "npm" / "package.json").read_text(encoding="utf-8"))

    init_text = (repo_root / "src" / "ai_knot" / "__init__.py").read_text(encoding="utf-8")
    prefix = '__version__ = "'
    start = init_text.index(prefix) + len(prefix)
    end = init_text.index('"', start)

    return {
        "pyproject": str(pyproject["project"]["version"]),
        "init": init_text[start:end],
        "npm_package": str(npm_package["version"]),
    }


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    versions = _local_versions(repo_root)
    target_version = versions["pyproject"]
    failures: list[str] = []

    try:
        pypi = _fetch_json(PYPI_URL)
        npm = _fetch_json(NPM_URL)
        repo = _fetch_json(API_REPO_URL)
        public_readme = _fetch_text(f"{RAW_BASE_URL}/README.md")
    except urllib.error.URLError as exc:
        print(f"FAIL network: {exc}")
        return 2

    pypi_version = str(pypi["info"]["version"])
    npm_version = str(npm["dist-tags"]["latest"])
    default_branch = str(repo["default_branch"])
    stars = int(repo["stargazers_count"])
    updated_at = str(repo["updated_at"])

    checks: list[tuple[str, bool, str]] = [
        (
            "local version sync",
            len(set(versions.values())) == 1,
            "pyproject="
            f"{versions['pyproject']} init={versions['init']} "
            f"npm/package.json={versions['npm_package']}",
        ),
        (
            "PyPI matches local",
            pypi_version == target_version,
            f"public PyPI={pypi_version} local={target_version}",
        ),
        (
            "npm matches local",
            npm_version == target_version,
            f"public npm={npm_version} local={target_version}",
        ),
        (
            "GitHub default branch",
            default_branch == "main",
            f"default_branch={default_branch}",
        ),
    ]

    for marker in README_MARKERS:
        present = marker in public_readme
        checks.append(
            (
                f"public README marker: {marker}",
                present,
                "present" if present else "missing from public README",
            )
        )

    for path in PUBLIC_FILE_MARKERS:
        try:
            _fetch_text(f"{RAW_BASE_URL}/{path}")
        except urllib.error.HTTPError:
            present = False
        else:
            present = True
        checks.append(
            (
                f"public repo file: {path}",
                present,
                "reachable on public main" if present else "missing on public main",
            )
        )

    print("== Local target ==")
    print(f"version: {target_version}")
    print()
    print("== Public state ==")
    print(f"PyPI latest: {pypi_version}")
    print(f"npm latest:  {npm_version}")
    print(f"GitHub stars: {stars}")
    print(f"GitHub default branch: {default_branch}")
    print(f"GitHub updated_at: {updated_at}")
    print()
    print("== Checks ==")

    for label, ok, detail in checks:
        print(f"[{_status(ok)}] {label}: {detail}")
        if not ok:
            failures.append(label)

    if failures:
        print()
        print("Launch gate is NOT green.")
        print("Fix the failing public-state items above before main launch.")
        return 1

    print()
    print("Launch gate is green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
