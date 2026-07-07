#!/usr/bin/env python3
"""Render the ai-knot whitepaper into a repo-native PDF artifact."""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "docs" / "whitepaper.md"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "pdf"
PAGE_WIDTH = 612
PAGE_HEIGHT = 792
TOP_MARGIN = 72
BOTTOM_MARGIN = 54
LEFT_MARGIN = 64
RIGHT_MARGIN = 64
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


class Block(NamedTuple):
    kind: str
    text: str = ""
    marker: str = ""


class Style(NamedTuple):
    font: str
    size: int
    leading: int
    max_chars: int
    space_before: int
    space_after: int
    indent: int = 0


STYLES = {
    "title": Style("Helvetica-Bold", 22, 28, 28, 0, 10),
    "subtitle": Style("Helvetica-Oblique", 12, 16, 58, 0, 18),
    "meta": Style("Helvetica-Oblique", 10, 13, 62, 0, 18),
    "h2": Style("Helvetica-Bold", 16, 21, 58, 12, 6),
    "h3": Style("Helvetica-Bold", 12, 16, 68, 10, 4),
    "body": Style("Helvetica", 11, 15, 82, 0, 6),
    "quote": Style("Helvetica-Oblique", 11, 15, 78, 0, 8, 16),
    "bullet": Style("Helvetica", 11, 15, 78, 0, 3, 14),
    "numbered": Style("Helvetica", 11, 15, 78, 0, 3, 14),
}

FONT_KEYS = {
    "Helvetica": "F1",
    "Helvetica-Bold": "F2",
    "Helvetica-Oblique": "F3",
}


def _current_version() -> str:
    init_text = (REPO_ROOT / "src" / "ai_knot" / "__init__.py").read_text(encoding="utf-8")
    prefix = '__version__ = "'
    start = init_text.index(prefix) + len(prefix)
    end = init_text.index('"', start)
    return init_text[start:end]


def _default_output_path() -> Path:
    return DEFAULT_OUTPUT_DIR / f"ai-knot-whitepaper-v{_current_version()}.pdf"


def _sanitize(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("**", "").replace("__", "")
    return text


def _collapse(text: str) -> str:
    return " ".join(_sanitize(text).split())


def parse_whitepaper_markdown(text: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph_lines: list[str] = []
    quote_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            blocks.append(Block("body", _collapse(" ".join(paragraph_lines))))
            paragraph_lines = []

    def flush_quote() -> None:
        nonlocal quote_lines
        if quote_lines:
            blocks.append(Block("quote", _collapse(" ".join(quote_lines))))
            quote_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_quote()
            continue
        if stripped == "---":
            flush_paragraph()
            flush_quote()
            blocks.append(Block("rule"))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flush_quote()
            blocks.append(Block("title", _collapse(stripped[2:])))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_quote()
            heading = _collapse(stripped[3:])
            if not any(block.kind == "subtitle" for block in blocks):
                blocks.append(Block("subtitle", heading))
            else:
                blocks.append(Block("h2", heading))
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_quote()
            blocks.append(Block("h3", _collapse(stripped[4:])))
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            quote_lines.append(stripped[2:])
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            flush_quote()
            blocks.append(Block("numbered", _collapse(numbered.group(2)), marker=f"{numbered.group(1)}."))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            flush_quote()
            blocks.append(Block("bullet", _collapse(stripped[2:]), marker="-"))
            continue
        if stripped.startswith("Updated:"):
            flush_paragraph()
            flush_quote()
            blocks.append(Block("meta", _collapse(stripped)))
            continue
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_quote()
    return blocks


def _wrap_block(block: Block, style: Style) -> list[str]:
    if not block.text:
        return []
    if block.kind in {"bullet", "numbered"}:
        initial = f"{block.marker} "
        wrapper = textwrap.TextWrapper(
            width=style.max_chars,
            initial_indent=initial,
            subsequent_indent=" " * len(initial),
            break_long_words=False,
            break_on_hyphens=False,
        )
        return wrapper.wrap(block.text)
    return textwrap.wrap(
        block.text,
        width=style.max_chars,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _line_width(text: str, font_size: int) -> float:
    return len(text) * font_size * 0.52


def _text_x(style: Style, line: str, *, center: bool) -> float:
    if center:
        return max(LEFT_MARGIN, (PAGE_WIDTH - _line_width(line, style.size)) / 2)
    return LEFT_MARGIN + style.indent


def _footer_commands(page_number: int) -> list[str]:
    footer = f"ai-knot whitepaper - page {page_number}"
    x = PAGE_WIDTH - RIGHT_MARGIN - _line_width(footer, 9)
    return [
        "0.35 g",
        f"BT /F1 9 Tf {x:.2f} 28 Td ({_escape_pdf_text(footer)}) Tj ET",
        "0 g",
    ]


def _render_pages(blocks: list[Block]) -> list[str]:
    pages: list[str] = []
    commands: list[str] = []
    y = PAGE_HEIGHT - TOP_MARGIN
    page_number = 1

    def new_page() -> None:
        nonlocal commands, y, page_number
        if commands:
            commands.extend(_footer_commands(page_number))
            pages.append("\n".join(commands) + "\n")
            page_number += 1
        commands = []
        y = PAGE_HEIGHT - TOP_MARGIN

    for block in blocks:
        if block.kind == "rule":
            if y - 14 < BOTTOM_MARGIN:
                new_page()
            commands.append("0.75 g")
            commands.append(f"newpath {LEFT_MARGIN} {y:.2f} moveto {PAGE_WIDTH - RIGHT_MARGIN} {y:.2f} lineto 0.8 setlinewidth stroke")
            commands.append("0 g")
            y -= 14
            continue

        style = STYLES[block.kind]
        lines = _wrap_block(block, style)
        needed = style.space_before + style.space_after + max(1, len(lines)) * style.leading
        if y - needed < BOTTOM_MARGIN:
            new_page()
        y -= style.space_before
        center = block.kind in {"title", "subtitle", "meta"}
        for line in lines:
            x = _text_x(style, line, center=center)
            commands.append(
                f"BT /{FONT_KEYS[style.font]} {style.size} Tf {x:.2f} {y:.2f} Td ({_escape_pdf_text(line)}) Tj ET"
            )
            y -= style.leading
        y -= style.space_after

    if commands or not pages:
        commands.extend(_footer_commands(page_number))
        pages.append("\n".join(commands) + "\n")
    return pages


def _pdf_objects(page_streams: list[str]) -> list[bytes]:
    objects: list[bytes] = []

    def add_object(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_ids = {
        "F1": add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
        "F2": add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"),
        "F3": add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>"),
    }

    page_ids: list[int] = []
    pages_id = add_object(b"<< /Type /Pages /Kids [] /Count 0 >>")
    for stream in page_streams:
        data = stream.encode("latin-1", errors="replace")
        content_id = add_object(b"<< /Length %d >>\nstream\n" % len(data) + data + b"endstream")
        page_body = (
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_ids['F1']} 0 R /F2 {font_ids['F2']} 0 R /F3 {font_ids['F3']} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        page_ids.append(add_object(page_body))

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))
    return _serialize_pdf(objects, catalog_id)


def _serialize_pdf(objects: list[bytes], catalog_id: int) -> list[bytes]:
    blob = bytearray()
    blob.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(blob))
        blob.extend(f"{index} 0 obj\n".encode("ascii"))
        blob.extend(body)
        blob.extend(b"\nendobj\n")
    xref_start = len(blob)
    blob.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    blob.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        blob.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    blob.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return [bytes(blob)]


def render_whitepaper_pdf(markdown_text: str, output_path: Path) -> Path:
    blocks = parse_whitepaper_markdown(markdown_text)
    page_streams = _render_pages(blocks)
    pdf_blob = _pdf_objects(page_streams)[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_blob)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE_PATH), help="Source markdown file.")
    parser.add_argument("--output", default=str(_default_output_path()), help="Output PDF path.")
    args = parser.parse_args(argv)

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        print(f"FAIL: source file {source} does not exist", file=sys.stderr)
        return 1

    markdown = source.read_text(encoding="utf-8")
    render_whitepaper_pdf(markdown, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
