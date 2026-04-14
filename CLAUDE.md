# Dev Notes

## Git workflow

**Never push directly to `main`.** No exceptions.

- All changes go to a feature branch
- To merge into `main`: open a pull request, wait for maintainer approval
- Force-pushing `main` is forbidden
- To fix something on `main`: branch → commit → PR → merge

## Commit conventions

**Author**: commits must be attributed to the project maintainer:
`alsoleg89 <155813332+alsoleg89@users.noreply.github.com>`

Verify before pushing:
```bash
git log -1 --format="%an"   # must be: alsoleg89
```

**Subject line** — keep it short and factual. Avoid:
- Tool or service names (editor, AI assistant, CI bot names)
- Internal workflow jargon (audit names, pipeline step names)
- Auto-generated session or trace URLs

**Body** — optional. If present: one blank line after subject, then context.
No auto-generated footers or URLs.

**Merge commits** — subject must not expose internal branch naming conventions.

## Pre-commit checklist (run before every push)

```bash
# 1. Format — CI runs ruff format --check; fix locally first
ruff format src/ tests/

# 2. Lint
ruff check src/ tests/

# 3. Types
mypy src/ai_knot --strict

# 4. Tests
pytest tests/ --ignore=tests/test_performance.py --ignore=tests/test_mcp_e2e.py -q

# 5. Confirm author
git log --format="%an" | sort -u   # must be: alsoleg89

# 6. Confirm no leaked URLs
git log --format="%B" | grep "https://claude\|session_"
```

**Order matters:** format → lint → types → tests. Don't skip format — CI will fail.

## Benchmark commands

### LOCOMO benchmark (aiknotbench — TypeScript)

```bash
# Full run (all 10 conversations, 233 questions)
# Uses new target_query pipeline (raw-episodes + deterministic materialization).
# LLM enrichment mode "dated-learn" is coming in v2.
cd aiknotbench && npx tsx src/index.ts run -r v1 --top-k 60

# Single category (e.g. Cat 2 = multi-hop)
npx tsx src/index.ts run -r v1 --top-k 60 --types 2

# Quick smoke test (1 conversation)
npx tsx src/index.ts run -r smoke --top-k 5 --limit 1

# List previous runs
npx tsx src/index.ts list
```

### Python benchmark (S1–S9 + LOCOMO)

```bash
# All default scenarios (S1–S9 + LOCOMO)
.venv/bin/python -m tests.eval.benchmark.runner

# Specific scenarios
.venv/bin/python -m tests.eval.benchmark.runner --scenarios s1,s4

# Offline (no Ollama judge)
.venv/bin/python -m tests.eval.benchmark.runner --mock-judge
```

### Multi-agent scenarios (S8-MA through S26)

```bash
# All MA scenarios
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent

# By category
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --ma-category protocol   # S10,S11,S13,S17,S20,S25
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --ma-category retrieval  # S8-MA,S9-MA,S12,S14-S16,S18,S19,S21-S24,S26

# Specific MA scenarios
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --scenarios s10,s11

# With specific storage backend
.venv/bin/python -m tests.eval.benchmark.runner --multi-agent --ma-storage sqlite
```
