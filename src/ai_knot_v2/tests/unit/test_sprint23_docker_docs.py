"""Sprint 23 — Docker + docs structural checks."""

from __future__ import annotations

import pathlib

_REPO_ROOT = pathlib.Path(__file__).parents[4]
_V2_ROOT = pathlib.Path(__file__).parents[2]


class TestDockerfile:
    def test_dockerfile_exists(self) -> None:
        assert (_REPO_ROOT / "Dockerfile.v2").exists()

    def test_dockerfile_references_mcp_server(self) -> None:
        text = (_REPO_ROOT / "Dockerfile.v2").read_text()
        assert "ai_knot_v2.api.mcp_server" in text

    def test_dockerfile_sets_env_vars(self) -> None:
        text = (_REPO_ROOT / "Dockerfile.v2").read_text()
        for var in ("AIKNOT_V2_DB_PATH", "AIKNOT_V2_AGENT_ID", "AIKNOT_V2_MAX_ATOMS"):
            assert var in text, f"missing env var: {var}"


class TestV2CLAUDEMd:
    def test_claude_md_exists(self) -> None:
        assert (_V2_ROOT / "CLAUDE.md").exists()

    def test_claude_md_has_invariants(self) -> None:
        text = (_V2_ROOT / "CLAUDE.md").read_text()
        assert "No LLM" in text
        assert "stop rule" in text.lower() or "stop-rule" in text.lower()
        assert "FORBIDDEN" in text
