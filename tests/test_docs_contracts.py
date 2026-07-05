"""Regression checks for the public docs surfaces."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_root_readme_keeps_core_memory_loop_visible() -> None:
    text = _read("README.md")

    assert "## Basic memory commands" in text
    assert "ai-knot demo" in text
    assert "ai-knot add    assistant" in text
    assert "ai-knot search assistant" in text
    assert "ai-knot list   assistant" in text
    assert "ai-knot delete assistant <fact_id>" in text
    assert "docs/memory-commands.md" in text
    assert "examples/function_calling_surface_demo.py" in text
    assert "examples/http_sidecar_surface_demo.py" in text
    assert "docs/comparison.md" in text
    assert "docs/competitive-analysis.md" not in text
    assert "docs/launch-checklist.md" not in text
    assert "scripts/run_competitor_bench_pack.py" not in text


def test_docs_index_lists_only_public_product_and_maintainer_docs() -> None:
    text = _read("docs/README.md")

    assert "usage.md" in text
    assert "memory-commands.md" in text
    assert "integrations.md" in text
    assert "benchmarks.md" in text
    assert "comparison.md" in text
    assert "whitepaper.md" in text
    assert "developer-article.md" in text
    assert "RELEASE.md" in text
    assert "launch-handoff.md" not in text
    assert "launch-day-runbook.md" not in text
    assert "output/launch-bundle-v0.11.0" not in text
    assert "launch-post.md" not in text
    assert "launch-plan.md" not in text
    assert "gtm-readiness.md" not in text
    assert "demo-script.md" not in text


def test_buyer_docs_keep_llamaindex_and_public_positioning() -> None:
    comparison = _read("docs/comparison.md")
    faq = _read("docs/faq.md")

    assert "I already use LlamaIndex" in comparison
    assert "What if I already use LlamaIndex?" in faq
    assert "AiKnotLlamaIndexMemory" in faq


def test_benchmarks_doc_uses_runner_not_removed_scorecard_script() -> None:
    text = _read("docs/benchmarks.md")

    assert "python -m tests.eval.benchmark.runner" in text
    assert "scripts/run_competitor_bench_pack.py" not in text
    assert "competitor-bench-pack.md" not in text


def test_release_doc_matches_slimmed_release_path() -> None:
    text = _read("docs/RELEASE.md")

    assert "Create Release" in text
    assert "render_github_release.py" in text
    assert "render_site_articles.py" in text
    assert "render_whitepaper_pdf.py" in text
    assert "render_launch_bundle.py" not in text
    assert "docs/announce.md" not in text


def test_contributing_and_pr_template_drop_internal_launch_docs() -> None:
    contributing = _read("CONTRIBUTING.md")
    pr_template = _read(".github/PULL_REQUEST_TEMPLATE.md")

    assert "docs/launch-checklist.md" not in contributing
    assert "docs/launch-day-runbook.md" not in contributing
    assert "docs/launch-handoff.md" not in contributing
    assert "docs/RELEASE.md" in contributing
    assert "docs/RELEASE.md" in pr_template
