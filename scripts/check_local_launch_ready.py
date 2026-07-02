#!/usr/bin/env python3
"""Check whether the local repository is ready for a public release."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MCP_NAME = "io.github.alsoleg89/ai-knot"
NPM_RUNTIME_MARKERS = [
    "dist/esm/index.js",
    "dist/esm/index.d.ts",
    "dist/cjs/index.js",
    "dist/cjs/package.json",
]

README_MARKERS = [
    "memory-commands.md",
    "examples/README.md",
    "examples/function_calling_surface_demo.py",
    "examples/http_sidecar_surface_demo.py",
    "ai-knot serve-mcp assistant --port 8765",
    "ai-knot setup openclaw --agent-id assistant --storage sqlite",
    "hero-demo.gif",
]

DOC_INDEX_MARKERS = [
    "memory-commands.md",
    "integrations.md",
    "benchmarks.md",
    "whitepaper.md",
    "developer-article.md",
    "server.json",
    "RELEASE.md",
]

REQUIRED_FILES = [
    "README.md",
    "server.json",
    "docs/README.md",
    "docs/RELEASE.md",
    "docs/positioning.md",
    "docs/comparison.md",
    "docs/faq.md",
    "docs/benchmarks.md",
    "docs/integrations.md",
    "docs/memory-commands.md",
    "docs/troubleshooting.md",
    "docs/usage.md",
    "docs/whitepaper.md",
    "docs/developer-article.md",
    "docs/site/index.html",
    "docs/site/whitepaper.html",
    "docs/site/developer-article.html",
    "docs/assets/hero-demo.gif",
    "docs/assets/hero-demo-poster.png",
    "examples/README.md",
    "examples/cli_memory_loop.py",
    "examples/browser_inspector_demo.py",
    "examples/claude_mcp_setup.py",
    "examples/function_calling_surface_demo.py",
    "examples/http_sidecar_surface_demo.py",
    "examples/openclaw_integration.py",
    "npm/README.md",
    "npm/examples/basic-memory-loop.ts",
    ".github/workflows/release.yml",
    ".github/workflows/pages.yml",
    ".github/workflows/publish-mcp-registry.yml",
    "scripts/check_public_release.py",
    "scripts/render_github_release.py",
    "scripts/render_site_articles.py",
    "scripts/render_whitepaper_pdf.py",
]


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _load_versions(repo_root: Path) -> dict[str, str]:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    npm_package = json.loads((repo_root / "npm" / "package.json").read_text(encoding="utf-8"))
    npm_lock = json.loads((repo_root / "npm" / "package-lock.json").read_text(encoding="utf-8"))

    init_text = (repo_root / "src" / "ai_knot" / "__init__.py").read_text(encoding="utf-8")
    prefix = '__version__ = "'
    start = init_text.index(prefix) + len(prefix)
    end = init_text.index('"', start)

    return {
        "pyproject": str(pyproject["project"]["version"]),
        "init": init_text[start:end],
        "npm_package": str(npm_package["version"]),
        "npm_lock": str(npm_lock["version"]),
    }


def _load_module(repo_root: Path, relative_path: str, module_name: str) -> ModuleType:
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render_release_body(repo_root: Path, version: str) -> str:
    module = _load_module(repo_root, "scripts/render_github_release.py", "render_github_release")
    return module.render_release_body(version)


def _render_whitepaper_pdf_artifact(repo_root: Path, version: str) -> Path:
    module = _load_module(repo_root, "scripts/render_whitepaper_pdf.py", "render_whitepaper_pdf")
    fd, temp_path = tempfile.mkstemp(
        prefix=f"ai-knot-whitepaper-v{version}-",
        suffix=".pdf",
    )
    os.close(fd)
    output_path = Path(temp_path)
    markdown = (repo_root / "docs" / "whitepaper.md").read_text(encoding="utf-8")
    return module.render_whitepaper_pdf(markdown, output_path)


def _render_site_articles_artifacts(repo_root: Path) -> dict[str, str]:
    module = _load_module(repo_root, "scripts/render_site_articles.py", "render_site_articles")
    with tempfile.TemporaryDirectory(prefix="ai-knot-site-articles-") as tmp_dir:
        written = module.render_site_articles(Path(tmp_dir))
        return {path.name: path.read_text(encoding="utf-8") for path in written if path.exists()}


def _check_readme_markers(repo_root: Path) -> list[tuple[str, bool, str]]:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    return [
        (f"README marker: {marker}", marker in readme, "present" if marker in readme else "missing")
        for marker in README_MARKERS
    ]


def _check_doc_index_markers(repo_root: Path) -> list[tuple[str, bool, str]]:
    index_text = (repo_root / "docs" / "README.md").read_text(encoding="utf-8")
    return [
        (
            f"docs index marker: {marker}",
            marker in index_text,
            "present" if marker in index_text else "missing",
        )
        for marker in DOC_INDEX_MARKERS
    ]


def _check_required_files(repo_root: Path) -> list[tuple[str, bool, str]]:
    return [
        (
            f"required file: {relative_path}",
            (repo_root / relative_path).exists(),
            "present" if (repo_root / relative_path).exists() else "missing",
        )
        for relative_path in REQUIRED_FILES
    ]


def _check_server_manifest(repo_root: Path, version: str) -> list[tuple[str, bool, str]]:
    manifest = json.loads((repo_root / "server.json").read_text(encoding="utf-8"))
    package = manifest["packages"][0]
    return [
        (
            "server.json name",
            manifest.get("name") == EXPECTED_MCP_NAME,
            f"name={manifest.get('name')!r}",
        ),
        (
            "server.json version",
            manifest.get("version") == version,
            f"manifest={manifest.get('version')} local={version}",
        ),
        (
            "server.json package identifier",
            package.get("identifier") == "ai-knot",
            f"identifier={package.get('identifier')!r}",
        ),
        (
            "server.json transport type",
            package.get("transport", {}).get("type") == "stdio",
            f"transport={package.get('transport', {}).get('type')!r}",
        ),
    ]


def _check_npm_package_audit(repo_root: Path) -> tuple[str, bool, str]:
    npm_dir = repo_root / "npm"
    if shutil.which("npm") is None:
        return "npm package audit", False, "`npm` is not installed"

    missing_markers = [marker for marker in NPM_RUNTIME_MARKERS if not (npm_dir / marker).exists()]
    built_missing_dist = False
    if missing_markers:
        build_result = subprocess.run(
            ["npm", "run", "build"],
            cwd=npm_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if build_result.returncode != 0:
            detail = build_result.stderr.strip() or build_result.stdout.strip() or "build failed"
            return "npm package audit", False, f"npm build failed before audit: {detail}"
        built_missing_dist = True

    result = subprocess.run(
        ["npm", "run", "package:audit"],
        cwd=npm_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "package audit failed"
        return "npm package audit", False, detail

    detail = " | ".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    if built_missing_dist:
        detail = f"built missing npm dist before audit | {detail}"
    return "npm package audit", True, detail


def main(argv: list[str] | None = None) -> int:
    del argv
    repo_root = REPO_ROOT
    versions = _load_versions(repo_root)
    version = versions["pyproject"]

    print("== Local target ==")
    print(f"version: {version}")
    print()
    print("== Checks ==")

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "local version sync",
            len(set(versions.values())) == 1,
            (
                "pyproject={pyproject} init={init} npm/package.json={npm_package} "
                "npm/package-lock.json={npm_lock}"
            ).format(**versions),
        )
    )

    try:
        _render_release_body(repo_root, version)
        checks.append(("release notes render", True, "rendered from CHANGELOG.md"))
    except ValueError as exc:
        checks.append(("release notes render", False, str(exc)))

    try:
        written = _render_site_articles_artifacts(repo_root)
        expected = {
            "whitepaper.html": (repo_root / "docs" / "site" / "whitepaper.html").read_text(
                encoding="utf-8"
            ),
            "developer-article.html": (
                repo_root / "docs" / "site" / "developer-article.html"
            ).read_text(encoding="utf-8"),
        }
        drift = [name for name, text in expected.items() if written.get(name) != text]
        if drift:
            checks.append(("site article render", False, f"drift in {drift}"))
        else:
            checks.append(("site article render", True, "checked-in Pages articles are in sync"))
    except ValueError as exc:
        checks.append(("site article render", False, str(exc)))

    try:
        artifact = _render_whitepaper_pdf_artifact(repo_root, version)
        checks.append(("whitepaper pdf render", True, f"rendered {artifact.name}"))
    except ValueError as exc:
        checks.append(("whitepaper pdf render", False, str(exc)))

    checks.append(_check_npm_package_audit(repo_root))
    checks.extend(_check_server_manifest(repo_root, version))
    checks.extend(_check_readme_markers(repo_root))
    checks.extend(_check_doc_index_markers(repo_root))
    checks.extend(_check_required_files(repo_root))

    failures = 0
    for label, ok, detail in checks:
        if not ok:
            failures += 1
        print(f"[{_status(ok)}] {label}: {detail}")

    print()
    if failures:
        print("Local release preflight is NOT green.")
        return 1

    print("Local release preflight is green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
