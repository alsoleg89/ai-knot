"""Tests for the Pages article renderer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_site_articles.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_site_articles", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_site_articles_creates_expected_pages(tmp_path: Path) -> None:
    module = _load_module()

    written = module.render_site_articles(tmp_path)

    assert [path.name for path in written] == ["developer-article.html", "whitepaper.html"]
    developer = (tmp_path / "developer-article.html").read_text(encoding="utf-8")
    whitepaper = (tmp_path / "whitepaper.html").read_text(encoding="utf-8")

    assert (
        "<title>ai-knot Developer Guide | Add deterministic memory in under 30 minutes</title>"
        in developer
    )
    assert "../memory-commands.md" in developer
    assert "/v1/facts/resolved" in developer
    assert "learn([...])" in developer
    assert "ai-knot delete assistant &lt;fact_id&gt;" in developer

    assert "<title>ai-knot Whitepaper | Agent memory as a knowledge layer</title>" in whitepaper
    assert "../whitepaper.md" in whitepaper
    assert "78.0%" in whitepaper
    assert "facts instead of transcripts" in whitepaper


def test_checked_in_site_articles_match_renderer(tmp_path: Path) -> None:
    module = _load_module()
    module.render_site_articles(tmp_path)

    generated_whitepaper = (tmp_path / "whitepaper.html").read_text(encoding="utf-8")
    generated_article = (tmp_path / "developer-article.html").read_text(encoding="utf-8")

    assert generated_whitepaper == Path("docs/site/whitepaper.html").read_text(encoding="utf-8")
    assert generated_article == Path("docs/site/developer-article.html").read_text(encoding="utf-8")
