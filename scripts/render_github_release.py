#!/usr/bin/env python3
"""Render the GitHub Release body from CHANGELOG.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
REPO_BLOB_BASE = "https://github.com/alsoleg89/ai-knot/blob/main"

QUICK_LINKS = [
    ("README quickstart", f"{REPO_BLOB_BASE}/README.md#quickstart-30-seconds"),
    ("Integrations by surface", f"{REPO_BLOB_BASE}/docs/integrations.md"),
    ("Benchmarks", f"{REPO_BLOB_BASE}/docs/benchmarks.md"),
    ("Troubleshooting + doctor", f"{REPO_BLOB_BASE}/docs/troubleshooting.md"),
]


def _changelog_entry(version: str) -> str:
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\].*?\n(?P<body>.*?)(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(changelog)
    if match is None:
        raise ValueError(
            f"CHANGELOG entry for version {version!r} not found; update CHANGELOG.md first"
        )
    return match.group("body").strip()


def render_release_body(version: str) -> str:
    links = "\n".join(f"- [{label}]({url})" for label, url in QUICK_LINKS)
    return (
        f"# ai-knot v{version}\n\n"
        "Deterministic, self-hosted long-term memory for AI agents.\n\n"
        "## Start here\n\n"
        f"{links}\n\n"
        "## Changelog\n\n"
        f"{_changelog_entry(version)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version, e.g. 0.11.0")
    parser.add_argument("--output", help="Optional output path. Defaults to stdout.")
    args = parser.parse_args(argv)

    try:
        rendered = render_release_body(args.version)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
