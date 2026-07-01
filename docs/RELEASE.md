# Release runbook

ai-knot ships to **PyPI** (`ai-knot`) and **npm** (`ai-knot`) from a single
version held in three files. This runbook covers a normal release and the one
failure mode that is operator-only: an npm token that has lost publish rights.

## Versioning invariant

One version, three files — a CI test fails the build if they drift:

- `pyproject.toml` → `project.version`
- `src/ai_knot/__init__.py` → `__version__`
- `npm/package.json` → `version`

Bump rule (semver): **MINOR** for a new subsystem, **PATCH** for fixes. See
[CHANGELOG.md](../CHANGELOG.md).

## Cutting a release

The `Create Release` workflow (`workflow_dispatch`, input `version`) is the
single entry point. It:

1. Validates `X.Y.Z` format.
2. Bumps the version in all three files (no-op if already at target).
3. Commits as `alsoleg89` and pushes (skipped if nothing changed).
4. Creates and pushes tag `vX.Y.Z` (skipped if it already exists).
5. Cuts a GitHub Release with generated notes.

Pushing the tag fans out to two independent publish workflows:

- **Publish to PyPI** — OIDC trusted publishing, no secret. Idempotent
  (`skip-existing: true`), so re-running a tag is safe.
- **Publish to npm** — uses the `NPM_TOKEN` secret. Skips automatically if the
  version is already on the registry, so re-running a tag is safe.

Both verify the tag matches `package.json` / `pyproject.toml` before publishing.

## Pre-release gate (run locally first)

```bash
ruff format --check src/ tests/
ruff check src/ tests/
mypy src/ai_knot --strict
pytest tests/ --ignore=tests/test_performance.py --ignore=tests/test_mcp_e2e.py -q
```

Order matters: **format → lint → types → tests**.

## npm token rotation (the one operator-only failure)

**Symptom.** The *Publish to npm* job fails with:

```
npm error code E404
npm error 404 Not Found - PUT https://registry.npmjs.org/ai-knot
npm error 404  'ai-knot@X.Y.Z' is not in this registry.
```

A `404` on the **PUT** (publish) request is npm masking an auth failure — the
package exists, but the token is missing, expired, or lacks publish rights for
`ai-knot`. PyPI will have published fine, leaving the two registries out of sync.

**Fix.**

1. Create a **Granular Access Token** (or Automation token) on npmjs.com with
   *Read and write* permission scoped to the `ai-knot` package.
2. In the GitHub repo: **Settings → Secrets and variables → Actions** → update
   the `NPM_TOKEN` secret with the new value.
3. Re-run the failed *Publish to npm* job, or re-dispatch the workflow. The
   "already published" guard makes a re-run a no-op once the version lands, so
   there is no risk of a double-publish.

**Verify both registries match after a release:**

```bash
curl -s https://pypi.org/pypi/ai-knot/json        | python3 -c "import sys,json;print('PyPI:', json.load(sys.stdin)['info']['version'])"
curl -s https://registry.npmjs.org/ai-knot        | python3 -c "import sys,json;print('npm: ', json.load(sys.stdin)['dist-tags']['latest'])"
```

**Verify the full public launch state (registries + public `main` markers):**

```bash
./.venv/bin/python scripts/check_public_release.py
```
