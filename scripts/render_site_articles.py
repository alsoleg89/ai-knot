#!/usr/bin/env python3
"""Render the Pages-ready long-form article pages from repo-owned markdown."""

from __future__ import annotations

import argparse
import html
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
except ImportError as exc:  # pragma: no cover - exercised through the CLI/preflight path
    raise ImportError(
        "markdown-it-py is required for site article rendering. "
        "Install the dev extras or add markdown-it-py to the environment."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
SITE_ROOT = DOCS_ROOT / "site"
HERO_IMAGE_URL = "https://alsoleg89.github.io/ai-knot/assets/hero-demo-poster.png"
TWITTER_IMAGE_ALT = "ai-knot poster showing deterministic long-term memory for AI agents"


@dataclass(frozen=True)
class Link:
    label: str
    href: str


@dataclass(frozen=True)
class Metric:
    value: str
    label: str


@dataclass(frozen=True)
class PageConfig:
    slug: str
    source_path: Path
    output_path: Path
    title: str
    description: str
    og_description: str
    twitter_description: str
    eyebrow_label: str
    hero_title: str
    hero_lede_html: str
    nav_links: list[Link]
    cta_links: list[Link]
    metrics: list[Metric]
    footer_links: list[Link]
    keywords: list[str] = field(default_factory=list)
    toc_limit: int = 5


COMMON_STYLE = """
:root {
  color-scheme: light;
  --bg: #f6f1e8;
  --page: radial-gradient(circle at top left, #fff9f0 0%, #f7f1e7 40%, #efe5d7 100%);
  --panel: rgba(255, 251, 245, 0.94);
  --panel-strong: #fffdfa;
  --ink: #1d1b18;
  --muted: #62584e;
  --line: rgba(95, 78, 59, 0.18);
  --accent: #9f4020;
  --accent-2: #155b52;
  --accent-soft: #efe2d8;
  --accent-2-soft: #dfeeea;
  --shadow: 0 20px 60px rgba(29, 27, 24, 0.08);
  --hero: linear-gradient(135deg, rgba(255, 248, 239, 0.96), rgba(245, 233, 218, 0.92));
  --mono: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
  --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
  --sans: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
}

* { box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: var(--serif);
  line-height: 1.68;
}

a {
  color: inherit;
  text-decoration-color: rgba(159, 64, 32, 0.42);
  text-underline-offset: 0.18em;
}

code {
  font-family: var(--mono);
  font-size: 0.92em;
  background: rgba(255, 255, 255, 0.78);
  padding: 0.12em 0.32em;
  border-radius: 0.35em;
}

pre {
  margin: 20px 0;
  padding: 18px 20px;
  overflow: auto;
  border-radius: 20px;
  border: 1px solid rgba(95, 78, 59, 0.14);
  background: #fffdf9;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
}

pre code {
  background: transparent;
  padding: 0;
  border-radius: 0;
  display: block;
  line-height: 1.52;
}

.shell {
  width: min(1080px, calc(100% - 32px));
  margin: 0 auto;
  padding: 24px 0 72px;
}

.nav {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: center;
  margin-bottom: 20px;
  font-family: var(--sans);
  font-size: 0.95rem;
}

.nav__brand {
  font-weight: 700;
  letter-spacing: 0.02em;
  text-decoration: none;
}

.nav__links {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  color: var(--muted);
}

.hero,
.prose,
.footer {
  padding: 34px;
  border-radius: 30px;
  border: 1px solid var(--line);
  background: var(--panel);
  box-shadow: var(--shadow);
}

.hero {
  background: var(--hero);
  margin-bottom: 18px;
}

.prose {
  background: rgba(255, 253, 250, 0.96);
}

.footer {
  margin-top: 18px;
  background: rgba(255, 253, 250, 0.9);
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 11px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(95, 78, 59, 0.14);
  font-family: var(--sans);
  font-size: 0.84rem;
  color: var(--muted);
  margin-bottom: 16px;
}

h1,
h2,
h3 {
  margin: 0;
  line-height: 1.06;
  letter-spacing: -0.035em;
  color: var(--ink);
}

h1 {
  font-size: clamp(2.5rem, 5.4vw, 4.8rem);
  max-width: 12ch;
  margin-bottom: 16px;
}

.prose h2 {
  font-size: clamp(1.7rem, 2.7vw, 2.7rem);
  margin: 42px 0 14px;
  scroll-margin-top: 16px;
}

.prose h2:first-child {
  margin-top: 0;
}

.prose h3 {
  font-size: 1.18rem;
  margin: 28px 0 10px;
}

p,
li {
  color: var(--muted);
  font-size: 1.06rem;
}

.lede {
  max-width: 66ch;
  font-size: 1.12rem;
  color: var(--muted);
}

.hero__cta,
.toc,
.footer__links {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 22px;
}

.button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 18px;
  border-radius: 999px;
  border: 1px solid transparent;
  text-decoration: none;
  font-family: var(--sans);
  font-size: 0.95rem;
  transition: transform 160ms ease, box-shadow 160ms ease;
}

.button:hover {
  transform: translateY(-1px);
  box-shadow: 0 12px 28px rgba(29, 27, 24, 0.12);
}

.button--primary {
  background: var(--accent);
  color: #fff9f4;
}

.button--secondary {
  background: rgba(255, 255, 255, 0.74);
  border-color: rgba(95, 78, 59, 0.14);
}

.metrics {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin: 18px 0 22px;
}

.metric {
  padding: 16px 18px;
  border-radius: 22px;
  border: 1px solid rgba(95, 78, 59, 0.12);
  background: rgba(255, 255, 255, 0.78);
}

.metric strong {
  display: block;
  font-family: var(--sans);
  font-size: 1.48rem;
  color: var(--accent);
}

.metric span {
  display: block;
  margin-top: 6px;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 0.95rem;
}

.prose ul,
.prose ol {
  margin: 18px 0 0;
  padding-left: 22px;
}

.prose li + li {
  margin-top: 10px;
}

.prose p + p {
  margin-top: 16px;
}

.prose blockquote {
  margin: 24px 0;
  padding: 18px 20px;
  border-left: 4px solid var(--accent);
  background: var(--accent-soft);
  border-radius: 18px;
}

.prose hr {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 30px 0;
}

.footer p {
  margin: 0;
}

@media (max-width: 760px) {
  .shell {
    width: min(100%, calc(100% - 20px));
  }

  .hero,
  .prose,
  .footer {
    padding: 22px;
  }

  .nav {
    flex-direction: column;
    align-items: flex-start;
  }
}
""".strip()


PAGES: dict[str, PageConfig] = {
    "whitepaper": PageConfig(
        slug="whitepaper",
        source_path=DOCS_ROOT / "whitepaper.md",
        output_path=SITE_ROOT / "whitepaper.html",
        title="ai-knot Whitepaper | Agent memory as a knowledge layer",
        description=(
            "Research-style paper for ai-knot: why agent memory should store facts "
            "instead of transcripts, and why deterministic, self-hosted recall "
            "is a real product wedge."
        ),
        og_description=(
            "Why agent memory should be treated as a deterministic, self-hosted "
            "knowledge layer instead of a growing prompt log."
        ),
        twitter_description=(
            "Why deterministic, self-hosted recall is a defensible agent-memory wedge."
        ),
        eyebrow_label="Whitepaper",
        hero_title="Agent memory should be a knowledge layer, not a log.",
        hero_lede_html=(
            "Most AI-agent stacks still store every message, then replay a growing slice "
            "of that transcript into future prompts. ai-knot takes the opposite position: "
            "<strong>store facts instead of transcripts</strong>, retrieve only what matters, "
            "and keep the read path deterministic, self-hosted, and testable."
        ),
        nav_links=[
            Link("Landing", "./index.html"),
            Link("GitHub", "https://github.com/alsoleg89/ai-knot"),
            Link("Benchmarks", "../benchmarks.md"),
            Link("Comparison", "../comparison.md"),
        ],
        cta_links=[
            Link("Open the repo", "https://github.com/alsoleg89/ai-knot"),
            Link("Read the markdown source", "../whitepaper.md"),
        ],
        metrics=[
            Metric("78.0%", "LoCoMo QA accuracy"),
            Metric("59.6%", "LongMemEval QA accuracy"),
            Metric("0.83", "Deterministic MRR"),
            Metric("0.26", "LoCoMo evidence_recall@5"),
        ],
        footer_links=[
            Link("Inspect benchmarks", "../benchmarks.md"),
            Link("Browse examples", "https://github.com/alsoleg89/ai-knot/blob/main/examples/README.md"),
        ],
    ),
    "developer-article": PageConfig(
        slug="developer-article",
        source_path=DOCS_ROOT / "developer-article.md",
        output_path=SITE_ROOT / "developer-article.html",
        title="ai-knot Developer Guide | Add deterministic memory in under 30 minutes",
        description=(
            "Practical developer guide for ai-knot: add deterministic, self-hosted "
            "long-term memory to an agent without replaying the whole transcript."
        ),
        og_description=(
            "A practical walkthrough for adding deterministic, self-hosted memory to "
            "agents across Python, MCP, TypeScript, CrewAI, LlamaIndex, "
            "LangGraph, PydanticAI, and more."
        ),
        twitter_description=(
            "Stop replaying the whole transcript. Store facts, recall the right few, "
            "and keep the read path deterministic."
        ),
        eyebrow_label="Developer article",
        hero_title="Stop replaying the whole transcript.",
        hero_lede_html=(
            "Most agent memory systems still start from the chat log. ai-knot takes a "
            "simpler view: <strong>store facts, not transcripts</strong>, recall the right few, "
            "and keep the read path deterministic so memory stays cheap, self-hosted, and testable."
        ),
        nav_links=[
            Link("Landing", "./index.html"),
            Link("Whitepaper", "./whitepaper.html"),
            Link("GitHub", "https://github.com/alsoleg89/ai-knot"),
            Link("Examples", "https://github.com/alsoleg89/ai-knot/blob/main/examples/README.md"),
        ],
        cta_links=[
            Link("Open the repo", "https://github.com/alsoleg89/ai-knot"),
            Link("Browse examples", "https://github.com/alsoleg89/ai-knot/blob/main/examples/README.md"),
            Link("Read the markdown source", "../developer-article.md"),
        ],
        metrics=[
            Metric("0 LLM", "Calls on the hot retrieval path by default"),
            Metric("3 stores", "YAML, SQLite, PostgreSQL"),
            Metric("8+ surfaces", "MCP, Python, TS, frameworks, HTTP"),
            Metric("30 min", "to get deterministic memory into an agent"),
        ],
        footer_links=[
            Link("Open quickstart", "https://github.com/alsoleg89/ai-knot/blob/main/examples/quickstart.py"),
            Link("Browse examples", "https://github.com/alsoleg89/ai-knot/blob/main/examples/README.md"),
            Link("Inspect benchmarks", "../benchmarks.md"),
        ],
        keywords=[
            "agent-memory",
            "ai-memory",
            "long-term-memory",
            "mcp",
            "model-context-protocol",
            "langgraph",
            "llamaindex",
            "crewai",
            "openclaw",
            "openai-agents",
            "pydanticai",
            "vercel-ai-sdk",
        ],
    ),
}


def _strip_preamble(text: str) -> str:
    marker = "\n---\n"
    if marker not in text:
        raise ValueError("expected markdown preamble separator '---'")
    return text.split(marker, 1)[1].strip() + "\n"


def _extract_updated_date(text: str) -> str:
    match = re.search(r"Updated:\s+\*\*(?P<date>[^*]+)\*\*", text)
    if match is None:
        raise ValueError("could not find updated date in markdown source")
    return match.group("date")


def _clean_heading_text(text: str) -> str:
    stripped = re.sub(r"^\d+(?:\.\d+)?\.?\s*", "", text).strip()
    return stripped or text.strip()


def _slugify(text: str) -> str:
    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return normalized or "section"


def _rewrite_href(href: str, *, source_path: Path, output_path: Path) -> str:
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return href

    base, anchor = href, ""
    if "#" in href:
        base, anchor = href.split("#", 1)
    if base == "":
        return href

    target = (source_path.parent / base).resolve()
    relative = os.path.relpath(target, output_path.parent.resolve())
    rewritten = Path(relative).as_posix()
    return f"{rewritten}#{anchor}" if anchor else rewritten


def _markdown_to_html(
    markdown: str,
    *,
    source_path: Path,
    output_path: Path,
) -> tuple[str, list[tuple[str, str]]]:
    renderer = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": True})
    tokens = renderer.parse(markdown)
    toc: list[tuple[str, str]] = []

    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            inline = tokens[index + 1] if index + 1 < len(tokens) else None
            if inline is None or inline.type != "inline":
                continue
            label = _clean_heading_text(inline.content)
            slug = _slugify(label)
            token.attrSet("id", slug)
            if token.tag == "h2":
                toc.append((slug, label))

    stack: list[Token] = list(tokens)
    while stack:
        token = stack.pop()
        if token.type == "link_open":
            href = token.attrGet("href")
            if href is not None:
                token.attrSet(
                    "href",
                    _rewrite_href(href, source_path=source_path, output_path=output_path),
                )
        if token.children:
            stack.extend(token.children)

    body_html = renderer.renderer.render(tokens, renderer.options, {})
    body_html = body_html.replace(" -&gt; ", " → ")
    return body_html.strip(), toc


def _render_links(links: list[Link], *, button_class: str | None = None) -> str:
    rendered: list[str] = []
    for link in links:
        class_attr = f' class="button {button_class}"' if button_class else ""
        href = html.escape(link.href, quote=True)
        label = html.escape(link.label)
        rendered.append(
            f'<a{class_attr} href="{href}">{label}</a>'
        )
    return "\n        ".join(rendered)


def _render_metrics(metrics: list[Metric]) -> str:
    if not metrics:
        return ""
    cards = "\n        ".join(
        (
            '<div class="metric">'
            f"<strong>{html.escape(metric.value)}</strong>"
            f"<span>{html.escape(metric.label)}</span>"
            "</div>"
        )
        for metric in metrics
    )
    return (
        '<div class="metrics" aria-label="Page highlights">\n'
        f"        {cards}\n"
        "      </div>"
    )


def _render_toc(toc: list[tuple[str, str]], *, limit: int) -> str:
    if not toc:
        return ""
    items = "\n        ".join(
        (
            '<a class="button button--secondary" '
            f'href="#{html.escape(slug, quote=True)}">{html.escape(label)}</a>'
        )
        for slug, label in toc[:limit]
    )
    return f'<div class="toc">\n        {items}\n      </div>'


def render_site_article(page: PageConfig, *, markdown_text: str | None = None) -> str:
    source_text = (
        markdown_text
        if markdown_text is not None
        else page.source_path.read_text(encoding="utf-8")
    )
    updated = _extract_updated_date(source_text)
    article_markdown = _strip_preamble(source_text)
    body_html, toc = _markdown_to_html(
        article_markdown,
        source_path=page.source_path,
        output_path=page.output_path,
    )
    keywords_meta = ""
    if page.keywords:
        keywords = ", ".join(page.keywords)
        keywords_meta = (
            f'  <meta name="keywords" content="{html.escape(keywords, quote=True)}">\n'
        )

    hero_cta = _render_links(page.cta_links, button_class="button--secondary")
    if hero_cta:
        hero_cta = hero_cta.replace(
            'class="button button--secondary"',
            'class="button button--primary"',
            1,
        )
        hero_cta = f'<div class="hero__cta">\n        {hero_cta}\n      </div>'

    footer_links = _render_links(page.footer_links, button_class="button--secondary")
    footer_links_html = (
        f'<div class="footer__links">\n        {footer_links}\n      </div>' if footer_links else ""
    )
    nav_links = _render_links(page.nav_links)
    canonical_url = f"https://alsoleg89.github.io/ai-knot/{page.output_path.name}"
    escaped_title = html.escape(page.title, quote=True)
    escaped_description = html.escape(page.description, quote=True)
    escaped_og_description = html.escape(page.og_description, quote=True)
    escaped_twitter_description = html.escape(page.twitter_description, quote=True)
    escaped_eyebrow = html.escape(page.eyebrow_label)
    escaped_updated = html.escape(updated)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{html.escape(page.title)}</title>\n"
        '  <meta name="application-name" content="ai-knot">\n'
        '  <meta name="theme-color" content="#9f4020">\n'
        f'  <link rel="canonical" href="{canonical_url}">\n'
        f'  <meta name="description" content="{escaped_description}">\n'
        f"{keywords_meta}"
        '  <meta property="og:type" content="article">\n'
        '  <meta property="og:site_name" content="ai-knot">\n'
        f'  <meta property="og:title" content="{escaped_title}">\n'
        f'  <meta property="og:description" content="{escaped_og_description}">\n'
        f'  <meta property="og:url" content="{canonical_url}">\n'
        f'  <meta property="og:image" content="{HERO_IMAGE_URL}">\n'
        f'  <meta property="og:image:alt" content="{TWITTER_IMAGE_ALT}">\n'
        '  <meta name="twitter:card" content="summary_large_image">\n'
        f'  <meta name="twitter:title" content="{escaped_title}">\n'
        f'  <meta name="twitter:description" content="{escaped_twitter_description}">\n'
        f'  <meta name="twitter:image" content="{HERO_IMAGE_URL}">\n'
        f'  <meta name="twitter:image:alt" content="{TWITTER_IMAGE_ALT}">\n'
        "  <style>\n"
        f"{COMMON_STYLE}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <div class="shell">\n'
        '    <nav class="nav" aria-label="Primary">\n'
        '      <a class="nav__brand" href="./index.html">ai-knot</a>\n'
        f'      <div class="nav__links">\n        {nav_links}\n      </div>\n'
        "    </nav>\n\n"
        '    <header class="hero">\n'
        f'      <div class="eyebrow">{escaped_eyebrow} · {escaped_updated}</div>\n'
        f"      <h1>{html.escape(page.hero_title)}</h1>\n"
        f'      <p class="lede">{page.hero_lede_html}</p>\n'
        f"      {hero_cta}\n"
        f"      {_render_metrics(page.metrics)}\n"
        f"      {_render_toc(toc, limit=page.toc_limit)}\n"
        "    </header>\n\n"
        '    <article class="prose">\n'
        f"{body_html}\n"
        "    </article>\n\n"
        '    <footer class="footer">\n'
        "      <p>\n"
        "        These Pages-ready long-form articles are generated from the repo-owned markdown "
        "sources so share links and public docs stay aligned.\n"
        "      </p>\n"
        f"      {footer_links_html}\n"
        "    </footer>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def render_site_articles(output_dir: Path | None = None) -> list[Path]:
    target_root = output_dir or SITE_ROOT
    target_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for page in PAGES.values():
        destination = target_root / page.output_path.name
        rendered = render_site_article(page)
        destination.write_text(rendered, encoding="utf-8")
        written.append(destination)

    return sorted(written)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to docs/site.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else SITE_ROOT
    written = render_site_articles(output_dir)
    print(f"Rendered {len(written)} site article pages to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
