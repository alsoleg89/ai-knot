# Publish-ready audit

Updated: **July 1, 2026**

This document answers one practical question:

> **Is `ai-knot` ready to publish and promote from the repository side, and what
> exactly still requires maintainer access or public-state verification?**

It is not a strategy doc. It is an evidence map against the actual launch
deliverables requested for this repo.

---

## Verdict

**Repository-side launch packaging is strong and mostly complete.**

The remaining blockers are primarily **external / maintainer-only**:

1. public GitHub `main` still needs to reflect this branch,
2. public npm latest still needs to match PyPI `0.11.0`,
3. prepared follow-up posts still need to be posted publicly.

That means the repo is no longer blocked on missing positioning, missing docs,
missing launch copy, or missing integration surfaces.

### Current public-state snapshot

Verified from official public endpoints on **July 1, 2026**:

- PyPI latest: `0.11.0`
- npm latest: `0.9.3`
- public GitHub repo: `alsoleg89/ai-knot`
- public GitHub default branch: `main`
- public GitHub `updated_at`: `2026-06-24T17:56:03Z`

This is why the launch is still prepared-but-not-executed rather than fully done.

### Current failing public checks

`scripts/check_public_release.py` currently fails on:

1. public npm latest is still `0.9.3` instead of `0.11.0`;
2. the public README is still missing launch-branch markers such as
   `Install by surface`, `What it looks like in your stack`, `skills/README.md`,
   `Browser inspector`, `examples/crewai_surface_demo.py`, and
   `docs/launch-checklist.md`;
3. the public `main` branch still does not expose:
   - `docs/crewai-case-study.md`
   - `docs/openclaw-case-study.md`
   - `docs/claude-mcp-case-study.md`
   - `docs/publish-ready-audit.md`
   - `docs/readme-patterns.md`
   - `docs/site/index.html`
   - `examples/notebook_walkthrough.ipynb`
   - `skills/README.md`

---

## Deliverables audit

| Required deliverable | Current evidence | Status |
|---|---|---|
| Updated positioning | [positioning.md](positioning.md) | ✅ |
| Improved README/docs | [README.md](../README.md), [usage.md](usage.md), [integrations.md](integrations.md), [deployment.md](deployment.md) | ✅ |
| Competitive analysis with conclusions | [competitive-analysis.md](competitive-analysis.md) | ✅ |
| Prioritized gaps | [gap-analysis.md](gap-analysis.md) | ✅ |
| Ready-to-post publication copy | [announce.md](announce.md), [launch-post.md](launch-post.md), [whitepaper.md](whitepaper.md), [developer-article.md](developer-article.md) | ✅ |
| Launch/distribution plan | [launch-plan.md](launch-plan.md), [channel-playbook.md](channel-playbook.md) | ✅ |
| What was already done by hand | [gap-analysis.md](gap-analysis.md#what-this-update-closed-by-hand), [launch-plan.md](launch-plan.md#7-done-vs-remaining) | ✅ |
| Remaining work in priority order | [gap-analysis.md](gap-analysis.md#highest-value-remaining-work), [launch-plan.md](launch-plan.md#7-done-vs-remaining), [launch-checklist.md](launch-checklist.md) | ✅ |

---

## Product and onboarding evidence

### Core product pitch is now clear

- [README.md](../README.md) leads with the problem, the 30-second loop, and the
  deterministic wedge.
- [README.md](../README.md) now uses a repo-native generated GIF hero asset
  (`docs/assets/hero-demo.gif`) instead of a static placeholder.
- [README.md](../README.md) now also shows framework-native and MCP-native
  surface snippets near the top, not only generic API primitives.
- [README.md](../README.md) now also routes readers to the browser inspector as
  a demo/debug surface, not just to SDK and MCP entry points.
- [comparison.md](comparison.md) explains when to pick `ai-knot` versus Mem0,
  Graphiti, Letta, or LangMem.
- [faq.md](faq.md) and [announce.md](announce.md) reduce message drift in public threads.

### Integration surfaces are now first-class

Prepared surfaces in-repo:

- CrewAI: [crewai-case-study.md](crewai-case-study.md), [examples/crewai_surface_demo.py](../examples/crewai_surface_demo.py), [examples/crewai_integration.py](../examples/crewai_integration.py)
- OpenClaw: [openclaw-case-study.md](openclaw-case-study.md), [examples/openclaw_integration.py](../examples/openclaw_integration.py)
- Claude/MCP: [claude-mcp-case-study.md](claude-mcp-case-study.md), [examples/claude_mcp_setup.py](../examples/claude_mcp_setup.py)
- OpenAI Agents SDK: [examples/openai_agents_integration.py](../examples/openai_agents_integration.py)
- PydanticAI: [examples/pydanticai_surface_demo.py](../examples/pydanticai_surface_demo.py), [examples/pydanticai_integration.py](../examples/pydanticai_integration.py)
- AutoGen: [examples/autogen_integration.py](../examples/autogen_integration.py)
- LangChain / LangGraph: [examples/langchain_integration.py](../examples/langchain_integration.py)
- Vercel AI SDK: [../npm/examples/vercel-ai-sdk.ts](../npm/examples/vercel-ai-sdk.ts), [../npm/README.md](../npm/README.md)
- TypeScript / npm: [../npm/README.md](../npm/README.md)
- Assistant skills: [../skills/README.md](../skills/README.md), [../skills/ai-knot/SKILL.md](../skills/ai-knot/SKILL.md)
- HTTP/browser inspection: [deployment.md#browser-inspector](deployment.md#browser-inspector)
- Notebook walkthrough: [../examples/notebook_walkthrough.ipynb](../examples/notebook_walkthrough.ipynb)
- Pages-ready landing page: [site/index.html](site/index.html), [../.github/workflows/pages.yml](../.github/workflows/pages.yml)

### Supportability for first-wave users

- `ai-knot doctor --json` now exists for install/integration triage
- structured issue templates now route install bugs, integration requests, and
  benchmark questions into actionable reports
- [troubleshooting.md](troubleshooting.md) now centralizes first-run, MCP, npm,
  and public-release failure paths
- the HTTP sidecar now includes both JSON inspection (`GET /v1/facts`) and a
  read-only browser inspector (`/inspect`) for first-wave debugging
- the HTTP sidecar now also mirrors the familiar memory loop with
  `POST /v1/facts`, `POST /v1/search`, `GET /v1/facts`, and
  `DELETE /v1/facts/{fact_id}`
- the repo now includes a GitHub Pages-ready landing page so outreach can use a
  cleaner URL after merge

### Install paths are aligned

- Python extras are in [pyproject.toml](../pyproject.toml): `crewai`,
  `autogen`, `agents`, `pydanticai`, `integrations`, `mcp`, `server`, `openai`,
  `postgres`.
- README and integrations docs route by surface instead of forcing one generic install path.

---

## Launch-kit evidence

### Long-form assets exist

- research-style piece: [whitepaper.md](whitepaper.md)
- technical article: [developer-article.md](developer-article.md)
- narrative launch post: [launch-post.md](launch-post.md)

### Channel-specific short copy exists

- Show HN, GitHub release/discussion, X, LinkedIn, Reddit, Dev.to/Medium:
  [announce.md](announce.md)
- directory and metadata blurbs: [submission-pack.md](submission-pack.md)
- structured post-launch intake templates exist in `.github/ISSUE_TEMPLATE/`
  for install bugs, integration requests, and benchmark questions

### Channel strategy and sequencing exist

- strategy: [launch-plan.md](launch-plan.md)
- dated execution: [channel-playbook.md](channel-playbook.md)
- maintainer pre-flight: [launch-checklist.md](launch-checklist.md)

---

## Competitive-research evidence

Current repo-native research assets:

- landscape + takeaways: [competitive-analysis.md](competitive-analysis.md)
- README/integration teardown: [readme-patterns.md](readme-patterns.md)
- buyer-facing comparison: [comparison.md](comparison.md)

Prepared channel wedges now cover:

- framework-first: CrewAI and PydanticAI
- app/MCP-first: OpenClaw
- Claude tool-first: Claude/MCP

That is enough breadth to start distribution without inventing new positioning in the moment.

---

## Verification evidence

Latest targeted checks completed in this workspace:

- `npm run typecheck` in `npm/`: passed
- `npm test` in `npm/`: passed
- `npm run build` in `npm/`: passed
- `ruff check` on the new integration/example paths: passed
- `ruff check scripts/check_public_release.py tests/test_public_release_script.py`: passed
- `ruff check src/ai_knot/server/app.py src/ai_knot/cli.py tests/test_server.py`: passed
- `ruff check src/ai_knot/server/app.py tests/test_server.py`: passed
- `ruff check examples/browser_inspector_demo.py tests/test_examples.py`: passed
- `ruff check src/ai_knot/integrations/pydanticai.py src/ai_knot/_mcp_tools.py src/ai_knot/mcp_server.py tests/test_integrations_pydanticai.py tests/test_mcp_tools.py tests/test_examples.py`: passed
- `./.venv/bin/python scripts/render_hero_demo_gif.py`: passed
- `pytest tests/test_site_artifacts.py -q --no-cov`: passed
- `pytest tests/test_server.py -q --no-cov`: passed
- `pytest tests/test_public_release_script.py -q --no-cov`: passed
- `pytest tests/test_examples.py tests/test_server.py -q --no-cov`: passed
- `pytest tests/test_examples.py tests/test_integrations_crewai.py tests/test_integrations_autogen.py tests/test_integrations_openai_agents.py tests/test_integrations_openclaw.py tests/test_version_sync.py -q --no-cov`: passing in targeted batches during this branch work
- `pytest tests/test_mcp_tools.py tests/test_mcp_server.py tests/test_integrations_pydanticai.py tests/test_examples.py tests/test_version_sync.py -q --no-cov`: passed
- `pytest tests/test_mcp_e2e.py -q --no-cov`: skipped in this workspace because `ai-knot[mcp]` is not installed
- `mypy src/ai_knot/integrations/pydanticai.py src/ai_knot/_mcp_tools.py src/ai_knot/mcp_server.py`: passed

Important practical interpretation:

- the new launch-facing examples are not just prose artifacts,
- the generated README hero GIF is repo-native and reproducible,
- the integration surfaces have regression coverage,
- the new PydanticAI adapter and MCP `search` / `list` / `delete` aliases are verified, not just documented,
- version-sync protections already exist in the repo.
- a one-command public-state verifier now exists:
  [`scripts/check_public_release.py`](../scripts/check_public_release.py)
- the same check is runnable from GitHub Actions via the manual
  `public-launch-audit.yml` workflow.

---

## What is still unproven inside the repo

These are not product gaps; they are external-state gaps:

1. Whether the public GitHub `main` branch already reflects this launch-ready state.
2. Whether public npm latest is already `0.11.0`.
3. Whether the prepared posts have actually been published.
4. Whether the public Codespaces path has been verified after merge.

Until those are true, the launch is **prepared** but not fully **executed**.

---

## Remaining work, in priority order

1. Merge/push this branch to public `main`.
2. Publish or confirm `ai-knot@0.11.0` on npm.
3. Use [launch-checklist.md](launch-checklist.md) to cut/confirm the GitHub release.
4. Post the prepared follow-ups in order:
   CrewAI → OpenClaw → Claude/MCP.
5. Validate the public Codespaces quickstart after `main` is live.
6. Optional post-launch expansion: OpenAI Agents SDK follow-up, competitor bench-pack, more backends.
