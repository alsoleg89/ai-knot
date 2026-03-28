# Contributing to agentmemo

Thank you for your interest in contributing! This document describes how to set up the project, submit changes, and pass code review.

---

## Quick setup

```bash
git clone https://github.com/alsoleg89/Agentmemo.git
cd Agentmemo
pip install -e ".[dev,openai]"
```

Run the test suite:

```bash
pytest
```

Run linters:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/agentmemo/
```

---

## Branching & commits

- Branch from `main`: `git checkout -b feat/my-feature`
- Commit messages: `type: short description` (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`)
- Keep commits focused; one logical change per commit

---

## Pull request process

1. Ensure all tests pass and coverage stays ≥ 80 %
2. Add or update tests for any changed behaviour
3. Update `CHANGELOG.md` under `[Unreleased]`
4. Fill in the PR template (checklist must be ticked)
5. A maintainer will review using the **Principal Python Developer** checklist (see `skills/principal_python_developer.md`)

---

## Code style

| Tool | Config |
|---|---|
| Formatter | `ruff format` (100-char line length) |
| Linter | `ruff check` — rules E, F, I, UP, B, SIM |
| Types | `mypy --strict` |
| Imports | `from __future__ import annotations` in every module |

Key rules:
- All public classes and functions must have docstrings
- No bare `except:` — catch specific exceptions
- No mutable default arguments — use `field(default_factory=...)`
- Storage backends must fully implement `StorageBackend` (see `src/agentmemo/storage/base.py`)

---

## Adding a storage backend

1. Create `src/agentmemo/storage/my_backend.py`
2. Implement the four methods: `save`, `load`, `delete`, `list_agents`
3. Export from `src/agentmemo/storage/__init__.py`
4. Add optional dependency to `pyproject.toml` if needed
5. Add tests in `tests/test_my_backend.py` mirroring `test_yaml_storage.py`
6. Add to the compat test in `test_storage_compat.py`

---

## Adding an integration

1. Create `src/agentmemo/integrations/my_integration.py`
2. Import `KnowledgeBase` from `agentmemo.knowledge`
3. Export from `src/agentmemo/integrations/__init__.py`
4. Add an example in `examples/`
5. Add tests in `tests/test_my_integration.py` — mock all external calls

---

## License

By contributing you agree your work will be released under the MIT License.
