"""Tests for the whitepaper PDF renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

# pypdf is only needed to read back and verify the generated PDF; the renderer
# itself has no PDF dependency. Skip in environments (e.g. CI) without pypdf.
PdfReader = pytest.importorskip("pypdf").PdfReader

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "render_whitepaper_pdf.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_whitepaper_pdf", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_whitepaper_pdf_creates_readable_multi_page_pdf(tmp_path: Path) -> None:
    module = _load_module()
    source = Path("docs/whitepaper.md").read_text(encoding="utf-8")
    output = tmp_path / "whitepaper.pdf"

    module.render_whitepaper_pdf(source, output)

    assert output.exists()
    reader = PdfReader(str(output))
    assert len(reader.pages) >= 2
    first_page = reader.pages[0].extract_text()
    last_page = reader.pages[-1].extract_text()
    assert "ai-knot whitepaper" in first_page
    assert "Treating agent memory as a knowledge layer" in first_page
    assert "product shape and measurement philosophy is a legitimate wedge." in last_page
