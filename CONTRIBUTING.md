# Contributing to ai-knot

Thank you for your interest in contributing! This document describes how to set up the project, submit changes, and pass code review.

---

## Quick setup

```bash
git clone https://github.com/alsoleg89/ai-knot.git
cd ai-knot
pip install -e ".[dev,openai,mcp,server]"
```

Run the test suite:

```bash
pytest
```

Run linters:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/ai_knot/
```

If you touch the TypeScript client in `npm/`, also run:

```bash
cd npm
npm ci
npm run build
npm run package:audit
npm run typecheck
npm test
```

If you touch release-facing docs, site generators, or public metadata checks,
also run:

```bash
./.venv/bin/python scripts/check_local_launch_ready.py
./.venv/bin/python scripts/check_public_release.py
```

For quick review or docs work without local setup, the repo also includes a
`.devcontainer/devcontainer.json` for GitHub Codespaces.

If you are debugging an install or integration problem before opening an issue,
capture:

```bash
ai-knot doctor --json
```

The output avoids printing secret values and is designed to be pasted into the
install bug template.

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
5. A maintainer will review the PR against the checklist in the PR template

If your change affects the developer journey, update the relevant docs in `README.md`,
`docs/`, `DEVELOPMENT.md`, or `npm/README.md` in the same PR.

If your change affects a publish surface, keep `docs/RELEASE.md` updated in the
same PR too.

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
- Storage backends must fully implement `StorageBackend` (see `src/ai_knot/storage/base.py`)

---

## Adding a storage backend

1. Create `src/ai_knot/storage/my_backend.py`
2. Implement the four methods: `save`, `load`, `delete`, `list_agents`
3. Export from `src/ai_knot/storage/__init__.py`
4. Add optional dependency to `pyproject.toml` if needed
5. Add tests in `tests/test_my_backend.py` mirroring `test_yaml_storage.py`
6. Add to the compat test in `test_storage_compat.py`

---

## Adding an integration

1. Create `src/ai_knot/integrations/my_integration.py`
2. Import `KnowledgeBase` from `ai_knot.knowledge`
3. Export from `src/ai_knot/integrations/__init__.py`
4. Add an example in `examples/`
5. Add tests in `tests/test_my_integration.py` — mock all external calls

---

## License

By contributing you agree your work will be released under the MIT License.
