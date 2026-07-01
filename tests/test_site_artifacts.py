"""Checks for launch-site artifacts."""

from __future__ import annotations

import pathlib


def test_pages_site_exists_and_covers_core_message() -> None:
    site_path = pathlib.Path("docs/site/index.html")
    html = site_path.read_text(encoding="utf-8")

    assert "<title>ai-knot | Deterministic memory for AI agents</title>" in html
    assert "Store facts. Not transcripts." in html
    assert "browser inspector" in html.lower()
    assert "notebook walkthrough" in html.lower()
    assert 'href="../benchmarks.md"' in html
    assert 'href="../whitepaper.md"' in html


def test_pages_workflow_exists_and_targets_docs_site() -> None:
    workflow_path = pathlib.Path(".github/workflows/pages.yml")
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "actions/deploy-pages@v4" in workflow
    assert "docs/site" in workflow
    assert "github-pages" in workflow
