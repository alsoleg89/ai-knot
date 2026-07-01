"""Render a repo-native animated hero demo GIF.

The asset is deterministic and based on the real output of
``examples/hero_demo.py``. Run from the repo root with:

    ./.venv/bin/python scripts/render_hero_demo_gif.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "docs" / "assets"
OUTPUT_GIF = ASSETS_DIR / "hero-demo.gif"
OUTPUT_POSTER = ASSETS_DIR / "hero-demo-poster.png"

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 720
WINDOW_X = 64
WINDOW_Y = 48
WINDOW_WIDTH = 1072
WINDOW_HEIGHT = 624

BACKGROUND = "#090f1d"
WINDOW = "#0f172a"
WINDOW_BORDER = "#1f2937"
WINDOW_TOP = "#111827"
TEXT = "#e5edf8"
MUTED = "#94a3b8"
ACCENT = "#7dd3fc"
SUCCESS = "#86efac"
PROMPT = "#34d399"
CURSOR = "#f8fafc"

FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        candidate = Path(path)
        if not candidate.exists():
            continue
        try:
            return ImageFont.truetype(str(candidate), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _capture_demo_output() -> list[str]:
    env = os.environ.copy()
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "hero_demo.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.rstrip().splitlines()


def _draw_window(
    lines: list[str],
    *,
    command_text: str,
    show_cursor: bool,
    hero_caption: str,
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    body_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (WINDOW_X, WINDOW_Y, WINDOW_X + WINDOW_WIDTH, WINDOW_Y + WINDOW_HEIGHT),
        radius=26,
        fill=WINDOW,
        outline=WINDOW_BORDER,
        width=2,
    )
    draw.rounded_rectangle(
        (WINDOW_X, WINDOW_Y, WINDOW_X + WINDOW_WIDTH, WINDOW_Y + 62),
        radius=26,
        fill=WINDOW_TOP,
    )
    draw.rectangle(
        (WINDOW_X, WINDOW_Y + 31, WINDOW_X + WINDOW_WIDTH, WINDOW_Y + 62),
        fill=WINDOW_TOP,
    )

    circles = ["#f87171", "#fbbf24", "#4ade80"]
    for index, color in enumerate(circles):
        cx = WINDOW_X + 28 + index * 20
        draw.ellipse((cx, WINDOW_Y + 22, cx + 12, WINDOW_Y + 34), fill=color)

    draw.text(
        (WINDOW_X + 92, WINDOW_Y + 18),
        "ai-knot hero demo",
        font=small_font,
        fill=MUTED,
    )

    draw.text(
        (WINDOW_X + 36, WINDOW_Y + 92),
        "Store facts once. Recall only what matters.",
        font=title_font,
        fill=TEXT,
    )
    draw.text(
        (WINDOW_X + 36, WINDOW_Y + 128),
        hero_caption,
        font=small_font,
        fill=MUTED,
    )

    prompt_y = WINDOW_Y + 176
    draw.text((WINDOW_X + 36, prompt_y), "$", font=body_font, fill=PROMPT)
    draw.text((WINDOW_X + 62, prompt_y), command_text, font=body_font, fill=ACCENT)
    if show_cursor:
        command_width = draw.textlength(command_text, font=body_font)
        draw.rectangle(
            (
                WINDOW_X + 66 + command_width,
                prompt_y + 6,
                WINDOW_X + 80 + command_width,
                prompt_y + 34,
            ),
            fill=CURSOR,
        )

    text_y = prompt_y + 58
    line_height = 34
    blank_height = 16
    for line in lines:
        fill = SUCCESS if line.startswith("[") else TEXT
        draw.text((WINDOW_X + 36, text_y), line, font=body_font, fill=fill)
        text_y += blank_height if not line else line_height

    footer = (
        "Deterministic output from examples/hero_demo.py  |  "
        "SQLite persistence  |  No LLM on recall"
    )
    draw.text(
        (WINDOW_X + 36, WINDOW_Y + WINDOW_HEIGHT - 34),
        footer,
        font=small_font,
        fill=MUTED,
    )

    return image


def render() -> tuple[Path, Path]:
    demo_lines = _capture_demo_output()
    command = "python examples/hero_demo.py"
    title_font = _load_font(32)
    body_font = _load_font(27)
    small_font = _load_font(19)
    caption = "Fresh process. Same memory on disk. No transcript replay."

    frames: list[Image.Image] = []
    durations: list[int] = []

    for index in range(1, len(command) + 1):
        frames.append(
            _draw_window(
                [],
                command_text=command[:index],
                show_cursor=index != len(command),
                hero_caption=caption,
                title_font=title_font,
                body_font=body_font,
                small_font=small_font,
            )
        )
        durations.append(55)

    frames.append(
        _draw_window(
            [],
            command_text=command,
            show_cursor=False,
            hero_caption=caption,
            title_font=title_font,
            body_font=body_font,
            small_font=small_font,
        )
    )
    durations.append(700)

    visible_lines: list[str] = []
    for line in demo_lines:
        visible_lines.append(line)
        frames.append(
            _draw_window(
                visible_lines,
                command_text=command,
                show_cursor=False,
                hero_caption=caption,
                title_font=title_font,
                body_font=body_font,
                small_font=small_font,
            )
        )
        if line.startswith("["):
            durations.append(1100)
        elif not line:
            durations.append(220)
        elif "Learned 3 facts" in line or "Fresh process" in line or line.startswith("Query:"):
            durations.append(1000)
        else:
            durations.append(760)

    durations[-1] = 2600
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    frames[-1].save(OUTPUT_POSTER)
    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return OUTPUT_GIF, OUTPUT_POSTER


def main() -> None:
    output_gif, output_poster = render()
    print(f"Wrote {output_gif.relative_to(ROOT)}")
    print(f"Wrote {output_poster.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
