# aiknotbench

Standalone LoCoMo10 benchmark for [ai-knot](../README.md).

Evaluates memory recall quality using an LLM judge (CORRECT/WRONG) across
LoCoMo10 question categories 1–4 (single-hop, multi-hop, temporal, open-ended).

## Setup

```bash
cd aiknotbench
bun install
cp .env.example .env    # fill in OPENAI_API_KEY
```

`ai-knot-mcp` must be on PATH (installed via `pip install "ai-knot[mcp]"`).

## Run

```bash
# Quick smoke test — 2 conversations, categories 1 & 2
bun run bench:quick

# Full benchmark — all 10 conversations, categories 1–4
bun run bench:full

# Custom run
bun run src/index.ts run -r my-run --limit 5 --types 1,2,3,4

# Resume interrupted run
bun run src/index.ts run -r my-run

# Force restart
bun run src/index.ts run -r my-run --force

# List runs
bun run list
```

## Output

Each run writes to `data/runs/{run-id}/`:
- `checkpoint.json` — incremental progress (safe to interrupt)
- `report.json` — final metrics

```json
{
  "summary": { "total": 1990, "correct": 1240, "accuracy": 0.623 },
  "byType": {
    "1": { "total": 520, "correct": 360, "accuracy": 0.692 },
    "2": { "total": 480, "correct": 290, "accuracy": 0.604 }
  },
  "categories1to4": { "total": 1800, "correct": 1120, "accuracy": 0.622 }
}
```

## Tests

```bash
bun run test
```
