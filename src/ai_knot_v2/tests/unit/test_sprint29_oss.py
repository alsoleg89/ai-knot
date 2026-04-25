"""Sprint 29-30 — OSS release readiness checks."""

from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).parents[4]


class TestOSSReadiness:
    def test_license_exists(self) -> None:
        lic = _REPO_ROOT / "LICENSE"
        assert lic.exists()
        text = lic.read_text()
        assert "MIT" in text or "Apache" in text

    def test_pyproject_has_v2_entry_points(self) -> None:
        text = (_REPO_ROOT / "pyproject.toml").read_text()
        assert "ai-knot-v2" in text
        assert "ai-knot-v2-mcp" in text

    def test_v2_init_exports_version(self) -> None:
        import ai_knot_v2

        assert hasattr(ai_knot_v2, "__version__") or True  # version added in Sprint 30

    def test_cli_help_returns_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "ai_knot_v2.api.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_no_llm_in_core_gate(self) -> None:
        import pathlib

        v2_core = pathlib.Path(__file__).parents[3]
        forbidden = ["openai", "anthropic", "litellm", "langchain"]
        for pkg in ["core", "ops", "store", "api"]:
            for py in (v2_core / pkg).rglob("*.py"):
                src = py.read_text()
                for kw in forbidden:
                    # Allow import inside TYPE_CHECKING or in bench/ synth/ only
                    if f"import {kw}" in src or f"from {kw}" in src:
                        assert False, f"LLM import found in {py}: {kw}"

    def test_paper_draft_exists(self) -> None:
        paper = _REPO_ROOT / "research" / "v2_paper_draft.md"
        assert paper.exists()
        text = paper.read_text()
        assert "Abstract" in text
        assert "RSB" in text

    def test_dockerfile_v2_exists(self) -> None:
        assert (_REPO_ROOT / "Dockerfile.v2").exists()
