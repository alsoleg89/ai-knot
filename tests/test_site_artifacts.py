"""Checks for launch-site artifacts."""

from __future__ import annotations

import pathlib


def test_hero_demo_asset_exists_and_is_wired_into_readme_and_site() -> None:
    gif_path = pathlib.Path("docs/assets/hero-demo.gif")
    poster_path = pathlib.Path("docs/assets/hero-demo-poster.png")
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    site = pathlib.Path("docs/site/index.html").read_text(encoding="utf-8")

    assert gif_path.exists()
    assert poster_path.exists()
    assert "docs/assets/hero-demo.gif" in readme
    assert "../assets/hero-demo.gif" in site


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
