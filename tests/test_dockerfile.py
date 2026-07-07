"""Container-hardening and security-policy regression checks."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_dockerfile_runs_as_non_root() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "useradd" in text
    assert "USER appuser" in text


def test_dockerfile_has_healthcheck() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text
    assert "/health" in text


def test_security_policy_is_not_stale() -> None:
    text = (REPO / ".github" / "SECURITY.md").read_text(encoding="utf-8")
    # The old table pinned 0.1.x forever; supported-versions must stay evergreen.
    assert "0.1.x" not in text
    assert "Latest release" in text
