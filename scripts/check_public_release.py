#!/usr/bin/env python3
"""Check whether the public ai-knot release state matches the local branch."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = "alsoleg89/ai-knot"
API_REPO_URL = f"https://api.github.com/repos/{REPO}"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPO}/main"
PAGES_BASE_URL = "https://alsoleg89.github.io/ai-knot"
RECOMMENDED_GITHUB_HOMEPAGE = PAGES_BASE_URL + "/"
PYPI_URL = "https://pypi.org/pypi/ai-knot/json"
NPM_URL = "https://registry.npmjs.org/ai-knot"
RECOMMENDED_GITHUB_DESCRIPTION = (
    "Deterministic, self-hosted long-term memory for AI agents: "
    "store facts instead of transcripts, recall only what matters."
)
RECOMMENDED_GITHUB_TOPICS = [
    "agent-memory",
    "ai-memory",
    "long-term-memory",
    "llm-memory",
    "memory",
    "ai-agents",
    "llm",
    "rag",
    "mcp",
    "model-context-protocol",
    "context-engineering",
    "langgraph",
    "llamaindex",
    "crewai",
    "openclaw",
    "self-hosted",
    "knowledge-base",
    "python",
    "typescript",
    "openai",
]
NPM_README_MARKERS = [
    "npx ai-knot-demo",
    "## Basic memory loop",
    "add -> search -> list -> delete",
    "npm run example:basic-memory-loop",
    "## Vercel AI SDK",
    "docs/memory-commands.md",
]
PAGES_MARKERS = [
    (PAGES_BASE_URL + "/", "<title>ai-knot | Deterministic memory for AI agents</title>"),
    (
        PAGES_BASE_URL + "/whitepaper.html",
        "<title>ai-knot Whitepaper | Agent memory as a knowledge layer</title>",
    ),
    (
        PAGES_BASE_URL + "/developer-article.html",
        "<title>ai-knot Developer Guide | Add deterministic memory in under 30 minutes</title>",
    ),
]
README_MARKERS = [
    "## Start here",
    "memory-commands.md",
    "ai-knot demo",
    "npx ai-knot-demo",
    "mcp-name: io.github.alsoleg89/ai-knot",
    "What it looks like in your stack",
    "ai-knot[langgraph]",
    "ai-knot[llamaindex]",
    "PydanticAI",
    "skills/README.md",
    "Browser inspector",
    "hero-demo.gif",
    "codespaces-quickstart.md",
    "examples/README.md",
    "examples/cli_memory_loop.py",
    "examples/crewai_surface_demo.py",
    "examples/function_calling_surface_demo.py",
    "examples/http_sidecar_surface_demo.py",
    "examples/llamaindex_integration.py",
    "examples/llamaindex_surface_demo.py",
    "examples/pydanticai_surface_demo.py",
    "ai-knot serve-mcp assistant --port 8765",
]
PUBLIC_FILE_MARKERS = [
    "docs/assets/hero-demo.gif",
    "docs/autogen-case-study.md",
    "docs/langgraph-case-study.md",
    "docs/llamaindex-case-study.md",
    "docs/openai-agents-case-study.md",
    "docs/openclaw-case-study.md",
    "docs/pydanticai-case-study.md",
    "docs/vercel-ai-sdk-case-study.md",
    "docs/claude-mcp-case-study.md",
    "docs/http-sidecar-case-study.md",
    "docs/codespaces-quickstart.md",
    "docs/memory-commands.md",
    "docs/site/index.html",
    "docs/site/whitepaper.html",
    "docs/site/developer-article.html",
    ".github/workflows/pages.yml",
    ".github/workflows/publish-mcp-registry.yml",
    "examples/README.md",
    "examples/cli_memory_loop.py",
    "examples/browser_inspector_demo.py",
    "examples/autogen_surface_demo.py",
    "examples/claude_mcp_setup.py",
    "examples/function_calling_surface_demo.py",
    "examples/http_sidecar_surface_demo.py",
    "examples/langgraph_surface_demo.py",
    "examples/llamaindex_integration.py",
    "examples/llamaindex_surface_demo.py",
    "examples/openclaw_integration.py",
    "examples/openai_agents_surface_demo.py",
    "examples/pydanticai_surface_demo.py",
    "npm/examples/basic-memory-loop.ts",
    "npm/scripts/demo.mjs",
    "scripts/render_github_release.py",
    "scripts/check_local_launch_ready.py",
    "scripts/render_site_articles.py",
    "scripts/render_whitepaper_pdf.py",
    "server.json",
    "skills/README.md",
]


def _fetch_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ai-knot-release-audit",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-knot-release-audit"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def _local_versions(repo_root: Path) -> dict[str, str]:
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


def _local_npm_package(repo_root: Path) -> dict[str, object]:
    return json.loads((repo_root / "npm" / "package.json").read_text(encoding="utf-8"))


def _normalize_repo_url(value: str) -> str:
    normalized = value.strip().lower()
    if normalized.startswith("git+"):
        normalized = normalized[4:]
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _has_prefix(failures: list[str], prefix: str) -> bool:
    return any(label.startswith(prefix) for label in failures)


def _build_next_actions(failures: list[str]) -> list[str]:
    actions: list[str] = []

    if "PyPI matches local" in failures:
        actions.append("Publish the missing PyPI version, then rerun this audit.")

    if _has_prefix(failures, "public README marker:") or _has_prefix(
        failures, "public repo file:"
    ):
        actions.append("Push or merge the current release-ready branch to public `main`.")

    npm_related = (
        "npm matches local" in failures
        or "npm description" in failures
        or "npm repository url" in failures
        or "npm repository directory" in failures
        or _has_prefix(failures, "public npm README marker:")
    )
    if npm_related:
        actions.append(
            "Publish npm so the package page refreshes version, description, repository metadata, and README."
        )

    if (
        "GitHub description" in failures
        or "GitHub topics" in failures
        or "GitHub homepage" in failures
    ):
        actions.append(
            "Apply the prepared GitHub repo metadata with `python scripts/apply_repo_metadata.py --apply`."
        )

    if "GitHub Pages enabled" in failures or _has_prefix(failures, "GitHub Pages:"):
        actions.append(
            "Enable or redeploy GitHub Pages for `docs/site/`, then rerun with `--require-pages`."
        )

    if actions:
        actions.append("Rerun `python scripts/check_public_release.py` until it returns green.")

    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-pages",
        action="store_true",
        help="Treat missing GitHub Pages deployment as a failure.",
    )
    parser.add_argument("--json-out", help="Optional path to write the machine-readable report.")
    parser.add_argument("--summary-out", help="Optional path to write a Markdown summary.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    local_versions = _local_versions(repo_root)
    local_version = local_versions["pyproject"]
    local_npm = _local_npm_package(repo_root)

    failures: list[str] = []
    lines: list[str] = []
    checks: list[dict[str, Any]] = []

    pypi = _fetch_json(PYPI_URL)
    npm = _fetch_json(NPM_URL)
    repo = _fetch_json(API_REPO_URL)

    lines.append("== Local target ==")
    lines.append(f"version: {local_version}")
    lines.append("")
    lines.append("== Public state ==")
    lines.append(f"PyPI latest: {pypi['info']['version']}")
    lines.append(f"npm latest:  {npm['dist-tags']['latest']}")
    lines.append("")
    lines.append("== Checks ==")

    def add_check(label: str, ok: bool, detail: str) -> None:
        checks.append({"label": label, "ok": ok, "detail": detail})
        if not ok:
            failures.append(label)
        lines.append(f"[{_status(ok)}] {label}: {detail}")

    add_check("local version sync", len(set(local_versions.values())) == 1, str(local_versions))
    add_check("PyPI matches local", pypi["info"]["version"] == local_version, f"public={pypi['info']['version']}")
    add_check(
        "npm matches local",
        npm["dist-tags"]["latest"] == local_version,
        f"public={npm['dist-tags']['latest']}",
    )
    add_check(
        "npm description",
        npm.get("description") == local_npm.get("description"),
        f"public={npm.get('description')!r}",
    )

    public_repo = npm.get("repository", {})
    local_repo = local_npm.get("repository", {})
    add_check(
        "npm repository url",
        _normalize_repo_url(str(public_repo.get("url", "")))
        == _normalize_repo_url(str(local_repo.get("url", ""))),
        f"public={public_repo.get('url')!r}",
    )
    add_check(
        "npm repository directory",
        public_repo.get("directory") == local_repo.get("directory"),
        f"public={public_repo.get('directory')!r}",
    )

    npm_readme = str(npm.get("readme", ""))
    for marker in NPM_README_MARKERS:
        add_check(
            f"public npm README marker: {marker}",
            marker in npm_readme,
            "present" if marker in npm_readme else "missing",
        )

    add_check(
        "GitHub description",
        repo.get("description") == RECOMMENDED_GITHUB_DESCRIPTION,
        f"public={repo.get('description')!r}",
    )
    public_topics = set(repo.get("topics", []))
    add_check(
        "GitHub topics",
        set(RECOMMENDED_GITHUB_TOPICS) <= public_topics,
        f"public={sorted(public_topics)}",
    )
    add_check(
        "GitHub homepage",
        str(repo.get("homepage", "")) == RECOMMENDED_GITHUB_HOMEPAGE,
        f"public={repo.get('homepage')!r}",
    )
    add_check(
        "GitHub Pages enabled",
        bool(repo.get("has_pages")),
        f"has_pages={repo.get('has_pages')!r}",
    )

    public_readme = _fetch_text(f"{RAW_BASE_URL}/README.md")
    for marker in README_MARKERS:
        add_check(
            f"public README marker: {marker}",
            marker in public_readme,
            "present" if marker in public_readme else "missing",
        )

    for path in PUBLIC_FILE_MARKERS:
        url = f"{RAW_BASE_URL}/{path}"
        try:
            _fetch_text(url)
            add_check(f"public repo file: {path}", True, "present")
        except urllib.error.HTTPError as exc:
            add_check(f"public repo file: {path}", False, f"HTTP {exc.code}")

    if args.require_pages:
        for url, marker in PAGES_MARKERS:
            try:
                page = _fetch_text(url)
                add_check(
                    f"GitHub Pages: {url}",
                    marker in page,
                    "present" if marker in page else "marker missing",
                )
            except urllib.error.HTTPError as exc:
                add_check(f"GitHub Pages: {url}", False, f"HTTP {exc.code}")

    lines.append("")
    if failures:
        lines.append("Release gate is NOT green.")
        actions = _build_next_actions(failures)
        if actions:
            lines.append("")
            lines.append("Likely next actions:")
            lines.extend(f"- {action}" for action in actions)
        exit_code = 1
    else:
        lines.append("Release gate is green.")
        exit_code = 0

    report = {
        "local_version": local_version,
        "failures": failures,
        "checks": checks,
    }
    summary = "\n".join(lines) + "\n"
    print(summary, end="")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.summary_out:
        Path(args.summary_out).write_text(summary, encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
