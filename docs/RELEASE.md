# Release runbook

ai-knot ships to **PyPI** (`ai-knot`), **npm** (`ai-knot`), and the **MCP Registry**
from one repo and one version.

## Versioning invariant

One version must stay in sync across:

- `pyproject.toml`
- `src/ai_knot/__init__.py`
- `npm/package.json`

The repo has tests that fail if these drift.

## Standard release path

Use the `Create Release` workflow in GitHub Actions with a semantic version like
`0.11.0`.

It will:

1. validate the version format,
2. bump the Python and npm versions,
3. refresh `npm/package-lock.json`,
4. commit and push the version bump if needed,
5. create and push `vX.Y.Z`,
6. render GitHub Release notes from `CHANGELOG.md`,
7. create or update the GitHub Release.

Tag push then fans out to:

- PyPI publish
- npm publish
- MCP Registry publish

## Local pre-release checks

Run these before cutting a release:

```bash
cd npm && npm run build && npm run package:audit && npm run typecheck && npm test
cd ..
python scripts/check_local_launch_ready.py
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/ai_knot --strict
pytest tests/ --ignore=tests/test_performance.py --ignore=tests/test_mcp_e2e.py -q
```

If you also want to confirm the public registries / repo state after release:

```bash
python scripts/check_public_release.py
python scripts/check_public_release.py --require-pages
```

## Release notes

GitHub Release notes are rendered from the tagged `CHANGELOG.md` entry:

```bash
python scripts/render_github_release.py --version 0.11.0
```

## Pages artifacts

If the whitepaper or developer article changed, regenerate the checked-in Pages
artifacts before release:

```bash
python scripts/render_site_articles.py
python scripts/render_whitepaper_pdf.py
```

## Repo metadata

If the repo description, topics, or homepage need refresh:

```bash
python scripts/apply_repo_metadata.py
python scripts/apply_repo_metadata.py --apply
```

## npm token failure mode

If the npm publish job fails with an auth-looking `404` on publish, rotate the
`NPM_TOKEN` GitHub Actions secret and rerun the publish job.
