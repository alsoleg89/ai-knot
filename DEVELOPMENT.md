# Development Guide

## Prerequisites

- Python 3.11 or 3.12
- `pip` ≥ 23
- Git

---

## Installation

```bash
git clone https://github.com/alsoleg89/agentmemo.git
cd Agentmemo

# Install in editable mode with all dev and optional deps
pip install -e ".[dev,openai]"
```

---

## Running tests

```bash
# Full suite with coverage report
pytest

# Run a specific test file
pytest tests/test_knowledge.py -v

# Run a specific test
pytest tests/test_forgetting.py::TestCalculateRetention::test_retention_decreases_over_time -v

# Skip coverage (faster iteration)
pytest --no-cov

# Watch mode (install pytest-watch first)
ptw -- --no-cov
```

Coverage threshold: **80 %** — CI will fail below this.

---

## Linting and type checking

```bash
# Lint
ruff check src/ tests/

# Auto-fix safe lint issues
ruff check --fix src/ tests/

# Format
ruff format src/ tests/

# Check format without writing
ruff format --check src/ tests/

# Type check
mypy src/agentmemo/
```

All three must pass clean before opening a PR.

---

## Project structure

```
src/agentmemo/          # Package source
  types.py              # Fact, MemoryType, ConversationTurn
  storage/              # Storage backends
    base.py             # StorageBackend protocol
    yaml_storage.py
    sqlite_storage.py
  forgetting.py         # Ebbinghaus decay
  retriever.py          # TF-IDF retriever
  extractor.py          # LLM fact extraction
  knowledge.py          # KnowledgeBase (public API)
  cli.py                # Click CLI
  integrations/
    openai.py           # MemoryEnabledOpenAI
tests/
  conftest.py           # Shared fixtures
  test_*.py             # One file per module
examples/
  quickstart.py
  openai_integration.py
skills/                 # Role-based working guides for this project
```

---

## Common workflows

### Try the CLI locally

```bash
# Add facts
agentmemo add mybot "User prefers Python" --importance 0.9
agentmemo add mybot "User deploys on Docker"

# Query
agentmemo recall mybot "how to deploy?"

# Inspect
agentmemo show mybot
agentmemo stats mybot

# Apply forgetting
agentmemo decay mybot
```

### Try the Python API

```bash
python examples/quickstart.py
```

### Run the OpenAI integration example

```bash
# No API key needed for the memory injection demo
python examples/openai_integration.py
```

---

## Adding a new storage backend

See `CONTRIBUTING.md → Adding a storage backend` and `ARCHITECTURE.md → Storage layer`.

Quick checklist:
1. Create file in `src/agentmemo/storage/`
2. Implement `save`, `load`, `delete`, `list_agents`
3. Export from `storage/__init__.py`
4. Add optional dep to `pyproject.toml` if needed
5. Write tests mirroring `test_yaml_storage.py`
6. Add to `test_storage_compat.py`

---

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `OPENAI_API_KEY` | `Extractor`, `MemoryEnabledOpenAI` | OpenAI API key for extraction |
| `ANTHROPIC_API_KEY` | `Extractor` (provider=anthropic) | Anthropic API key |

Never hardcode keys. Pass them to `kb.learn(turns, api_key=os.environ["OPENAI_API_KEY"])`.

---

## Release process

See `skills/principal_devops.md` for the full release checklist.

Short version:
1. Bump `version` in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit + tag: `git tag v0.2.0 && git push --tags`
4. GitHub Actions `publish.yml` will build and publish to PyPI automatically
