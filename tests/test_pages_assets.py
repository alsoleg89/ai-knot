"""GitHub Pages must ship the assets its HTML references.

Only ``docs/site`` is published (see .github/workflows/pages.yml), so any
``../assets/...`` reference escapes the deploy root and 404s, and every
``assets/<file>`` reference must exist under ``docs/assets`` for the copy step to
ship it.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "docs" / "site"
ASSETS = REPO / "docs" / "assets"
_ASSET_REF = re.compile(r"assets/([A-Za-z0-9._-]+\.(?:gif|png|svg|jpe?g|webp))")


def test_site_html_has_no_parent_relative_asset_refs() -> None:
    for html in SITE.glob("*.html"):
        assert "../assets/" not in html.read_text(encoding="utf-8"), html.name


def test_referenced_site_assets_exist() -> None:
    for html in SITE.glob("*.html"):
        for name in _ASSET_REF.findall(html.read_text(encoding="utf-8")):
            assert (ASSETS / name).exists(), f"{html.name} references missing docs/assets/{name}"


def test_pages_workflow_copies_assets_into_site() -> None:
    workflow = (REPO / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
    assert "cp -r docs/assets docs/site/assets" in workflow
