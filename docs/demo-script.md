# Demo script

Updated: **June 30, 2026**

Use this page to produce the short terminal capture for the README hero, social
clips, or live demos.

---

## Goal

Show three things in under 20 seconds:

1. ai-knot stores **facts**, not the whole transcript.
2. memory **persists across restarts**.
3. recall returns only the **few relevant facts**.

## Recommended asset

Record [examples/hero_demo.py](../examples/hero_demo.py). It is intentionally:

- deterministic,
- no-API-key,
- short,
- legible in a narrow terminal window.

Run it with:

```bash
python examples/hero_demo.py
```

If you need a repo-native placeholder before recording a real clip, use:

- generate the current README hero with:

```bash
./.venv/bin/python scripts/render_hero_demo_gif.py
```

- [`docs/assets/hero-demo.gif`](assets/hero-demo.gif) for the animated README hero
- [`docs/assets/hero-demo-poster.png`](assets/hero-demo-poster.png) for the last-frame poster / preview
- [`docs/assets/hero-demo.svg`](assets/hero-demo.svg) as the static fallback
- [`docs/assets/social-card.svg`](assets/social-card.svg) for exported PNG social cards

## Recording beats

### Beat 1

Terminal title or caption:

> ai-knot: deterministic memory for AI agents

### Beat 2

Show the script running:

```bash
python examples/hero_demo.py
```

### Beat 3

Let the output land on:

- `Learned 3 facts into SQLite.`
- `Fresh process. Same memory on disk. No transcript replay.`
- the final recalled facts

## Suggested clip length

12-20 seconds.

That is enough for the proof without becoming a product walkthrough.

## Suggested caption

> Store facts once, persist them to disk, and recall only what the next turn needs.
> No LLM on the retrieval path.

## Optional tools

Use whichever recorder you already trust:

- `asciinema`
- `screen.studio`
- QuickTime / OBS
- any GIF workflow that preserves terminal sharpness

The important part is the script and the framing, not the tooling.
