#!/usr/bin/env python3
"""Print or apply the recommended GitHub repo metadata for ai-knot."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_RELEASE_SCRIPT = REPO_ROOT / "scripts" / "check_public_release.py"


def _load_public_release_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_public_release", _PUBLIC_RELEASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load {_PUBLIC_RELEASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _description_command(repo: str, description: str) -> list[str]:
    return [
        "gh",
        "api",
        f"repos/{repo}",
        "--method",
        "PATCH",
        "--raw-field",
        f"description={description}",
    ]


def _topics_command(repo: str, topics: list[str]) -> list[str]:
    command = [
        "gh",
        "api",
        f"repos/{repo}/topics",
        "--method",
        "PUT",
        "--header",
        "Accept: application/vnd.github+json",
    ]
    for topic in topics:
        command.extend(["--raw-field", f"names[]={topic}"])
    return command


def _homepage_command(repo: str, homepage: str) -> list[str]:
    return [
        "gh",
        "api",
        f"repos/{repo}",
        "--method",
        "PATCH",
        "--raw-field",
        f"homepage={homepage}",
    ]


def _render_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print or apply the recommended GitHub description/topics/homepage "
            "for the ai-knot repo."
        )
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repository in owner/name form. Defaults to the public release audit target.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the metadata with gh api instead of only printing the commands.",
    )
    args = parser.parse_args(argv)

    module = _load_public_release_module()
    repo = args.repo or module.REPO
    description = str(module.RECOMMENDED_GITHUB_DESCRIPTION)
    topics = list(module.RECOMMENDED_GITHUB_TOPICS)
    homepage = str(module.RECOMMENDED_GITHUB_HOMEPAGE)

    description_command = _description_command(repo, description)
    topics_command = _topics_command(repo, topics)
    homepage_command = _homepage_command(repo, homepage)

    print(f"Target repo: {repo}")
    print()
    print("Description:")
    print(description)
    print()
    print("Topics:")
    for topic in topics:
        print(f"- {topic}")
    print()
    print("Homepage:")
    print(homepage)
    print()
    print("gh commands:")
    print(_render_command(description_command))
    print(_render_command(topics_command))
    print(_render_command(homepage_command))

    if not args.apply:
        print()
        print("Dry run only. Re-run with --apply to update the GitHub repo metadata.")
        return 0

    if not _gh_available():
        print("ERROR: `gh` is not installed or not on PATH.", file=sys.stderr)
        return 1

    try:
        _run(description_command)
        _run(topics_command)
        _run(homepage_command)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: gh api failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    print()
    print("GitHub repo metadata updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
