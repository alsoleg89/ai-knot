# Release runbook

ai-knot ships to **PyPI** (`ai-knot`), **npm** (`ai-knot`), and the **MCP Registry**
from one repo and one version.

## Versioning invariant

One version must stay in sync across:

- `pyproject.toml`
- `src/ai_knot/__init__.py`
- `npm/package.json`
- `npm/package-lock.json` (top-level and root package entries)
- `server.json` (manifest and PyPI package entries)

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

## Local publication when Actions is unavailable

Run all checks and publication commands locally when GitHub Actions cannot start.
Keep credentials in the package managers' normal local configuration: `~/.pypirc`
for PyPI and `npm login` for npm. Never commit credentials or include them in
release notes.

1. Update the five version sources above and add a dated `CHANGELOG.md` entry.
2. Run the local gates, including the Python matrix, a real PostgreSQL service,
   MCP subprocess tests, the npm build/tests/package audit, and release preflight.
3. Build wheel and sdist from a clean export of the release commit. Build and pack
   the npm client from that same revision. Inspect package contents and run
   `python -m twine check` before upload.
4. Install those artifacts into clean temporary environments and test the CLI,
   always-on state, and both npm exports before publishing.
5. Publish **PyPI first**, then npm, so the optional Python bridge is available
   when npm's postinstall runs:

   ```bash
   python -m twine upload --non-interactive /path/to/artifacts/ai_knot-X.Y.Z*
   npm publish /path/to/artifacts/ai-knot-X.Y.Z.tgz --access public
   ```

6. Push the release commit and matching `vX.Y.Z` tag, then create the GitHub
   Release using the rendered changelog notes. Registry publication and GitHub
   tag/release state must point to the same tested revision.
7. Install `ai-knot[mcp,server]==X.Y.Z` from PyPI and `ai-knot@X.Y.Z` from npm in
   new environments outside the checkout. Check exact versions, ESM/CJS imports,
   a real MCP round trip, and persistent SQLite intent/replay behavior. Finally
   run `scripts/check_public_release.py` against the public registries.

Do not treat local artifact installation as proof that registry publication
succeeded. Preserve registry checksums and compare them with the uploaded files.

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
