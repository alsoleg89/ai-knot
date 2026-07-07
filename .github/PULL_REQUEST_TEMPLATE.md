## Summary

<!-- What does this PR do? 2-3 bullet points. -->

- 
- 

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactoring (no functional change)
- [ ] Documentation
- [ ] CI / DevOps

## Review checklist

- [ ] All public classes and functions have docstrings
- [ ] `from __future__ import annotations` present in every new module
- [ ] All type annotations correct and `mypy --strict` passes
- [ ] No bare `except:` — specific exceptions only
- [ ] No mutable default arguments (use `field(default_factory=...)`)
- [ ] `StorageBackend` protocol fully satisfied (if storage changes)
- [ ] No hardcoded secrets or absolute paths
- [ ] `ruff check` and `ruff format --check` pass clean

## Test checklist

- [ ] New behaviour covered by unit tests
- [ ] LLM calls mocked (no real API calls in tests)
- [ ] Storage tests include round-trip + multi-agent isolation
- [ ] `pytest --cov-fail-under=80` passes
- [ ] Both YAML and SQLite backends tested (if storage-related)
- [ ] `npm run build` + `npm run package:audit` pass (if `npm/` changed)
- [ ] `python scripts/check_local_launch_ready.py` passes (if release-facing docs / scripts / metadata changed)

## Documentation

- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docstrings updated for changed public API
- [ ] `ARCHITECTURE.md` updated if new layer or extension point added
- [ ] README / `docs/` / `npm/README.md` updated if the developer journey changed
- [ ] `docs/RELEASE.md` updated too if the maintainer publish path changed

## Screenshots / output (if applicable)

<!-- Paste CLI output or relevant logs -->
