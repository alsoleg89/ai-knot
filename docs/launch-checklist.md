# Launch checklist

Updated: **July 1, 2026**

This is the maintainer-only pre-flight checklist for turning the current branch
into a public launch. Use it together with [RELEASE.md](RELEASE.md),
[channel-playbook.md](channel-playbook.md), and [announce.md](announce.md).

---

## Launch gates

Do not start the main launch until all of these are true:

1. Public GitHub `main` matches this launch-ready branch.
2. npm latest is `0.11.0`, matching PyPI `0.11.0`.
3. The README hero uses the demo asset you want public visitors to see.
4. GitHub release/discussion copy is ready to paste.
5. If you want a cleaner launch URL than raw GitHub markdown, GitHub Pages is enabled for the repo.

---

## Step-by-step

### 1. Re-run the local safety gate

```bash
./.venv/bin/ruff check src tests examples
./.venv/bin/pytest tests/test_version_sync.py -q --no-cov
```

If those fail, fix them before publishing anything.

### 2. Push or merge this branch to public `main`

Reason: every external post sends people to the public repo, not to the local
workspace. Launching before `main` reflects the new README/docs destroys trust.

Minimum outcome:

- README landing page is public,
- docs/launch kit is public,
- framework adapters/examples are public.

### 3. Verify npm / PyPI parity

Expected public versions on **July 1, 2026**:

- PyPI: `0.11.0`
- npm: `0.11.0`

Verification commands from [RELEASE.md](RELEASE.md):

```bash
curl -s https://pypi.org/pypi/ai-knot/json | python3 -c "import sys,json;print('PyPI:', json.load(sys.stdin)['info']['version'])"
curl -s https://registry.npmjs.org/ai-knot | python3 -c "import sys,json;print('npm: ', json.load(sys.stdin)['dist-tags']['latest'])"
```

If npm is still behind:

- update or rotate `NPM_TOKEN` in GitHub Actions secrets if needed,
- re-run the existing npm publish workflow,
- confirm the public npm version before any launch post.

Preferred one-command verification:

```bash
./.venv/bin/python scripts/check_public_release.py
```

That script also checks whether the public `main` branch already exposes the
launch-ready README/docs markers.

If you prefer to run the same verification from GitHub after merge/publish, use
the manual **Public launch audit** workflow.

### 4. Cut or confirm the GitHub release

The repo already has a single-entry `Create Release` workflow in
`.github/workflows/release.yml`.

If `v0.11.0` is not already released publicly:

1. Dispatch `Create Release`
2. Input version: `0.11.0`
3. Let the tag fan out to:
   - PyPI publish
   - npm publish

If the version/tag already exists, confirm the GitHub Release page has notes that
reflect the current branch state.

### 5. Finalize the README hero asset

Current repo-native assets already exist:

- [`docs/assets/hero-demo.gif`](assets/hero-demo.gif)
- [`docs/assets/hero-demo-poster.png`](assets/hero-demo-poster.png)
- [`docs/assets/hero-demo.svg`](assets/hero-demo.svg)
- [`docs/assets/social-card.svg`](assets/social-card.svg)

The current GIF is reproducible from:

```bash
./.venv/bin/python scripts/render_hero_demo_gif.py
```

If you later record a live 12-20s terminal clip, replace the generated GIF only
if the live capture is clearly better. Do not delay launch on perfect motion.

### 6. Set GitHub metadata before posting

Use [submission-pack.md](submission-pack.md):

- repository description,
- topic list,
- short listing blurbs,
- awesome-list snippets.
- issue templates for install bugs, integration requests, and benchmark questions are already in `.github/ISSUE_TEMPLATE/`.

This should be done before GitHub release, Show HN, or Reddit so screenshots and
repo cards already read well.

### 6.5. Enable the Pages landing page if you want a clean share URL

The repo now ships:

- [`docs/site/index.html`](site/index.html)
- [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)

If you want to use GitHub Pages in launch posts, enable Pages after merge and
confirm the site deploys from the workflow before posting externally.

### 7. Use the prepared posting order

Start from:

1. GitHub release + pinned discussion
2. Direct outreach / soft launch
3. r/LocalLLaMA
4. Show HN + X + LinkedIn
5. Prepared CrewAI follow-up
6. Prepared PydanticAI follow-up
7. Prepared Vercel AI SDK follow-up
8. Prepared OpenClaw follow-up
9. Prepared Claude/MCP follow-up

Copy is already in:

- [announce.md](announce.md)
- [crewai-case-study.md](crewai-case-study.md)
- [openclaw-case-study.md](openclaw-case-study.md)
- [claude-mcp-case-study.md](claude-mcp-case-study.md)
- [channel-playbook.md](channel-playbook.md)

---

## If one thing goes wrong

### npm publish fails with a 404 on PUT

Treat it as an auth/publish-rights problem, not as "package missing."

Use the fix sequence from [RELEASE.md](RELEASE.md):

1. rotate `NPM_TOKEN`,
2. re-run the npm publish workflow,
3. verify public npm latest,
4. only then post publicly.

### The demo clip is not ready

Launch with the generated README hero GIF, then replace it later only if a live
terminal capture is clearly better.

### You only have time for one follow-up surface

Use CrewAI first. The repo now has:

- a zero-network proof,
- a full example,
- channel-ready copy.

If you have time for a second follow-up, use PydanticAI next. The repo now has:

- the zero-network `examples/pydanticai_surface_demo.py` proof,
- a real `examples/pydanticai_integration.py` wiring path,
- channel-ready copy in [announce.md](announce.md).

If you have time for a third follow-up, use Vercel AI SDK next. The repo now has:

- the repo-native `npm/examples/vercel-ai-sdk.ts` proof,
- npm-side docs in [../npm/README.md](../npm/README.md),
- channel-ready copy in [announce.md](announce.md).

If you have time for a fourth follow-up, use OpenClaw next. The repo now has:

- the paste-ready `ai-knot setup openclaw` flow,
- a zero-network `examples/openclaw_integration.py` proof,
- channel-ready copy in [openclaw-case-study.md](openclaw-case-study.md).

If you have time for a fifth follow-up, use Claude/MCP next. The repo now has:

- the paste-ready `ai-knot setup claude` flow,
- a zero-network `examples/claude_mcp_setup.py` proof,
- channel-ready copy in [claude-mcp-case-study.md](claude-mcp-case-study.md).
