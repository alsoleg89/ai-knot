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

## Principal Python Developer review checklist

> Reviewer applies `skills/principal_python_developer.md` before approving.

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

## Documentation

- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docstrings updated for changed public API
- [ ] `ARCHITECTURE.md` updated if new layer or extension point added

## Screenshots / output (if applicable)

<!-- Paste CLI output or relevant logs -->
