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
    assert 'rel="canonical" href="https://alsoleg89.github.io/ai-knot/"' in html
    assert 'property="og:title" content="ai-knot | Deterministic memory for AI agents"' in html
    assert 'name="keywords"' in html
    assert 'property="og:site_name" content="ai-knot"' in html
    assert "hero-demo-poster.png" in html
    assert 'name="twitter:card" content="summary_large_image"' in html
    assert 'name="twitter:image:alt"' in html
    assert "SoftwareSourceCode" in html
    assert "pypi.org/project/ai-knot" in html
    assert "npmjs.com/package/ai-knot" in html
    assert "Store facts. Not transcripts." in html
    assert "browser inspector" in html.lower()
    assert "notebook walkthrough" in html.lower()
    assert "examples/README.md" in html
    assert "serve-mcp assistant --port 8765" in html
    assert "write-default-config" in html
    assert "../memory-commands.md" in html
    assert "LangGraph" in html
    assert "LlamaIndex" in html
    assert "PydanticAI" in html
    assert "create_basic_memory_functions" in html
    assert "openai_agents_surface_demo.py" in html
    assert "function_calling_surface_demo.py" in html
    assert "http_sidecar_surface_demo.py" in html
    assert 'href="../benchmarks.md"' in html
    assert 'href="./whitepaper.html"' in html
    assert 'href="./developer-article.html"' in html


def test_whitepaper_page_exists_and_is_pages_ready() -> None:
    page_path = pathlib.Path("docs/site/whitepaper.html")
    html = page_path.read_text(encoding="utf-8")

    assert "<title>ai-knot Whitepaper | Agent memory as a knowledge layer</title>" in html
    assert 'rel="canonical" href="https://alsoleg89.github.io/ai-knot/whitepaper.html"' in html
    assert 'property="og:type" content="article"' in html
    assert "78.0%" in html
    assert "59.6%" in html
    assert "0.83" in html
    assert "facts instead of transcripts" in html
    assert "deterministic, self-hosted knowledge layer" in html
    assert "../whitepaper.md" in html


def test_developer_article_page_exists_and_is_pages_ready() -> None:
    page_path = pathlib.Path("docs/site/developer-article.html")
    html = page_path.read_text(encoding="utf-8")

    assert (
        "<title>ai-knot Developer Guide | Add deterministic memory in under 30 minutes</title>"
        in html
    )
    assert (
        'rel="canonical" href="https://alsoleg89.github.io/ai-knot/developer-article.html"' in html
    )
    assert 'property="og:type" content="article"' in html
    assert 'name="keywords"' in html
    assert 'name="twitter:image:alt"' in html
    assert "store facts, not transcripts" in html.lower()
    assert "LlamaIndex" in html
    assert "add → search → list → delete" in html
    assert "ai-knot delete assistant &lt;fact_id&gt;" in html
    assert "ai-knot serve-mcp assistant --port 8765" in html
    assert "write-default-config" in html
    assert "../memory-commands.md" in html
    assert "HttpKnowledgeBase" in html
    assert "ai-knot-doctor" in html
    assert "learn([...])" in html
    assert "/v1/facts/resolved" in html
    assert "create_basic_memory_functions(...)" in html
    assert "http_sidecar_surface_demo.py" in html
    assert "../developer-article.md" in html


def test_pages_workflow_exists_and_targets_docs_site() -> None:
    workflow_path = pathlib.Path(".github/workflows/pages.yml")
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "actions/deploy-pages@v4" in workflow
    assert "docs/site" in workflow
    assert "github-pages" in workflow
