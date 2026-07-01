# Competitor bench-pack

`ai-knot` already ships a serious benchmark harness. This guide is the
distribution-facing wrapper around it: how to produce a side-by-side scorecard
you can actually attach to a release, a benchmark discussion, or a launch thread
without hand-editing tables.

The entry point is:

```bash
python scripts/run_competitor_bench_pack.py --profile offline
```

That command writes a directory under `benchmark_results/` with:

- `runner_report.md` — the full benchmark report from the harness
- `runner_raw.json` — the raw schema-v2 data
- `runner_live.jsonl` — streaming scenario-level output
- `scorecard.md` — a curated publish-ready summary
- `summary.json` — a compact machine-readable rollup
- `metadata.json` — profile, command, scenarios, backends, artifact paths

---

## Profiles

| Profile | Best for | Command | What it proves |
|---|---|---|---|
| `offline` | reproducible launch proof | `python scripts/run_competitor_bench_pack.py --profile offline` | deterministic control comparison with zero network |
| `local-llm` | pre-launch apples-to-apples with extraction | `python scripts/run_competitor_bench_pack.py --profile local-llm` | local Ollama extraction + vector controls vs ai-knot |
| `real` | public competitor refresh before posting | `python scripts/run_competitor_bench_pack.py --profile real` | real qdrant + mem0ai surfaces plus ai-knot |

### `offline`

This is the safest profile to run anywhere. It sets:

```bash
AI_KNOT_BENCH_DISABLE_EMBED=1
```

That matters because the qdrant and mem0 emulators otherwise opportunistically
use a local Ollama endpoint if one happens to be running. The offline profile
forces a true no-network control run, even on a machine that already has Ollama.

Use it when you want a benchmark pack that any skeptical reader can reproduce
without services, Docker, or model setup.

### `local-llm`

This is the best "same machine, same local model stack" comparison before a
public launch. It turns extraction on and compares:

- `ai_knot`
- `qdrant_extraction`
- `mem0` emulator
- `baseline`

Use it when the question is not just retrieval, but also learned compression and
latest-state behavior.

### `real`

This is the closest in-repo path to a public competitor refresh. It uses:

- `qdrant_real`
- `mem0_real`
- `ai_knot`
- `baseline`

Use it before publishing fresh numbers in release notes, GitHub discussions, or
benchmark threads.

---

## What to publish

When you post benchmark numbers publicly, publish these three things together:

1. `scorecard.md`
2. the exact command from `metadata.json`
3. `runner_raw.json`

That is the minimum needed for developer trust.

Do **not** post only a screenshot or only a hand-edited summary table. The whole
point of this pack is to make the comparison falsifiable.

---

## Interpretation rules

- The bench-pack is a controlled harness, not a universal leaderboard.
- `offline` is an architecture/control comparison, not a full extraction benchmark.
- `local-llm` is the right profile for talking about compression or latest-state behavior.
- `real` is the right profile for talking about external stacks like Mem0 or Qdrant.
- Always name the profile when quoting numbers.

If you need the lower-level harness details, flags, or scenario definitions, see
[../tests/eval/benchmark/BENCHMARK.md](../tests/eval/benchmark/BENCHMARK.md) and
[benchmarks.md](benchmarks.md).
