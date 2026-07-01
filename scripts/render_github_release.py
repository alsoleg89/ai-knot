#!/usr/bin/env python3
"""Render the GitHub Release body from repo-owned launch copy + CHANGELOG."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANNOUNCE_PATH = _REPO_ROOT / "docs" / "announce.md"
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"
_REPO_BLOB_BASE = "https://github.com/alsoleg89/ai-knot/blob/main"

_QUICK_LINKS = [
    ("README quickstart", f"{_REPO_BLOB_BASE}/README.md#quickstart-30-seconds"),
    ("Integrations by surface", f"{_REPO_BLOB_BASE}/docs/integrations.md"),
    ("Benchmarks", f"{_REPO_BLOB_BASE}/docs/benchmarks.md"),
    ("Troubleshooting + doctor", f"{_REPO_BLOB_BASE}/docs/troubleshooting.md"),
]


def _extract_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"could not find section {heading!r}")
    return match.group("body").strip()


def _extract_marked_block(section: str, marker: str, *, stop_marker: str | None = None) -> str:
    lines = section.splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise ValueError(f"could not find marker {marker!r}") from exc

    collected: list[str] = []
    for line in lines[start:]:
        if stop_marker is not None and line == stop_marker:
            break
        if line.startswith("> "):
            collected.append(line[2:])
        elif line == ">" or line.strip() == "":
            collected.append("")
        else:
            collected.append(line)
    return "\n".join(collected).strip()


def _normalize_release_copy(text: str, version: str) -> str:
    return re.sub(r"ai-knot v\d+\.\d+\.\d+", f"ai-knot v{version}", text)


def _release_copy(version: str) -> str:
    announce = _ANNOUNCE_PATH.read_text(encoding="utf-8")
    release_section = _extract_section(announce, "## GitHub release / discussion / pinned post")
    raw = _extract_marked_block(
        release_section,
        "**Short release copy:**",
        stop_marker="**Pinned discussion opener:**",
    )
    return _normalize_release_copy(raw, version)


def _discussion_opener() -> str:
    announce = _ANNOUNCE_PATH.read_text(encoding="utf-8")
    release_section = _extract_section(announce, "## GitHub release / discussion / pinned post")
    block = _extract_marked_block(release_section, "**Pinned discussion opener:**")
    return re.sub(r"\n---\s*$", "", block).strip()


def _changelog_entry(version: str) -> str:
    changelog = _CHANGELOG_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\].*?\n(?P<body>.*?)(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(changelog)
    if match is None:
        raise ValueError(
            f"CHANGELOG entry for version {version!r} not found; update CHANGELOG.md first"
        )
    body = match.group("body").strip()
    return re.sub(r"\n---\s*$", "", body).strip()


def render_release_body(version: str) -> str:
    links = "\n".join(f"- [{label}]({url})" for label, url in _QUICK_LINKS)
    return (
        f"# ai-knot v{version}\n\n"
        f"{_release_copy(version)}\n\n"
        "## Start here\n\n"
        f"{links}\n\n"
        "## Changelog\n\n"
        f"{_changelog_entry(version)}\n\n"
        "## Feedback that helps most\n\n"
        f"{_discussion_opener()}\n"
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
