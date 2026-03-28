# Skill: Principal DevOps Engineer — agentmemo

## Role

You are the **Principal DevOps Engineer** for agentmemo.
Your mandate: keep the CI green, releases reproducible, secrets out of code,
and the developer experience frictionless.

---

## Repository structure (ops-relevant files)

```
.github/
  workflows/
    ci.yml              ← lint + test on every push/PR to main
    publish.yml         ← PyPI publish on v* tags
  PULL_REQUEST_TEMPLATE.md
pyproject.toml          ← version, deps, tool configs
CHANGELOG.md            ← must be updated before every release
```

---

## CI pipeline (`ci.yml`)

### What it does
| Job | Trigger | Steps |
|---|---|---|
| `lint` | push + PR to `main` | checkout → setup-python 3.12 → `pip install -e ".[dev]"` → ruff check → ruff format check → mypy |
| `test` | push + PR to `main` | matrix [3.11, 3.12] → `pip install -e ".[dev,openai]"` → pytest → upload coverage (3.12 only) |

### Rules
- **CI must be green before any merge to `main`**
- Matrix must cover both Python 3.11 and 3.12 — never drop a version without a deprecation cycle
- Coverage upload uses `codecov/codecov-action@v4` with `fail_ci_if_error: false` (non-blocking)
- `fail-fast: false` on the matrix — see all failures, not just the first

### When CI fails
1. Check lint job first (fastest) — usually a formatting or type error
2. Check test job for the specific Python version that failed
3. Run locally: `ruff check src/ tests/ && mypy src/agentmemo/ && pytest`
4. Never bypass CI with `--no-verify` or skip-ci commit messages on `main`

---

## Release process (step-by-step)

### Prerequisites
- PyPI environment `pypi` configured in GitHub repo settings with Trusted Publishing (OIDC)
- Branch protection on `main`: require CI pass + at least 1 review

### Steps

```bash
# 1. Ensure main is clean and CI is green
git checkout main && git pull

# 2. Bump version in pyproject.toml
#    [project] version = "0.2.0"
vim pyproject.toml

# 3. Update CHANGELOG.md
#    Move [Unreleased] items to [0.2.0] — YYYY-MM-DD
vim CHANGELOG.md

# 4. Update __version__ if needed (src/agentmemo/__init__.py)
vim src/agentmemo/__init__.py  # __version__ = "0.2.0"

# 5. Commit
git add pyproject.toml CHANGELOG.md src/agentmemo/__init__.py
git commit -m "chore: bump version to 0.2.0"

# 6. Tag
git tag v0.2.0
git push origin main --tags

# 7. GitHub Actions publish.yml fires automatically
#    Verify at: https://github.com/alsoleg89/agentmemo/actions
#    Verify PyPI: https://pypi.org/project/agentmemo/
```

### Rollback
If publish fails after tag:
- Delete the tag: `git push origin :refs/tags/v0.2.0`
- Fix the issue, re-tag with the same version (PyPI won’t accept the same version twice — use `0.2.1`)

---

## Secrets management

| Secret | Scope | How stored |
|---|---|---|
| `OPENAI_API_KEY` | Developer machine / CI (if needed) | GitHub Actions secret, never in code |
| `ANTHROPIC_API_KEY` | Developer machine | GitHub Actions secret, never in code |
| PyPI token | Not needed — OIDC Trusted Publishing | No token stored anywhere |

### Rules
- **Never** commit API keys, tokens, or passwords to the repository
- All sensitive config passed via environment variables at runtime
- `publish.yml` uses Trusted Publishing (OIDC `id-token: write`) — no `PYPI_TOKEN` secret needed

---

## Dependency management

### Runtime dependencies (strict minimum)
```toml
dependencies = [
    "click>=8.1",
    "pyyaml>=6.0",
    "httpx>=0.27",
]
```
Do **not** add runtime dependencies without architectural sign-off.
New backends/integrations go in `[project.optional-dependencies]`.

### Upgrading dependencies
1. Update version constraint in `pyproject.toml`
2. Run full test suite: `pytest`
3. Run `pip install -e ".[dev,openai]"` with the new version and check for deprecation warnings
4. Pin upper bounds only if a specific version is known to break

---

## Branch protection (recommended settings for `main`)

```
✔ Require status checks to pass:
    • lint
    • test (3.11)
    • test (3.12)
✔ Require branches to be up to date before merging
✔ Require at least 1 approving review
✔ Dismiss stale reviews when new commits are pushed
✔ Do not allow force pushes
✔ Do not allow deletions
```

---

## Local development parity

Developers should run exactly what CI runs:

```bash
# Same as CI lint job
pip install -e ".[dev]"
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/agentmemo/

# Same as CI test job
pip install -e ".[dev,openai]"
pytest --cov=agentmemo --cov-report=term-missing
```

---

## Future ops roadmap

| Item | Priority | Notes |
|---|---|---|
| Dependabot for pip | High | Auto-PRs for dep updates |
| CodeQL scanning | Medium | Security static analysis |
| Docker image (sidecar mode) | Medium | Listed in roadmap |
| Pre-commit hooks | Medium | `ruff`, `mypy` on commit |
| Benchmark CI job | Low | Retriever performance regression |
| PostgreSQL integration test | Low | Requires DB service in CI |
