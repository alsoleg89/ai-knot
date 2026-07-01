# Gap analysis

Updated: **July 1, 2026**

This document prioritizes product and GTM gaps by **impact on adoption**:

- GitHub stars
- developer trust
- conversion to trial
- clarity of value
- shareability
- integration pull

---

## Summary

ai-knot is no longer "missing the basics." The repo already has strong technical
substance, reproducible benchmarks, multi-agent differentiation, and multiple
integration surfaces. The remaining work is mostly about **public distribution
readiness**, not core product legitimacy.

## P0 — must be true before the main launch

| Gap | Why it matters | Status |
|---|---|---|
| Public GitHub `main` does not reflect the launch-ready branch | people judge the product from the public repo, not the local branch | **Open** |
| npm latest is still `0.9.3` while PyPI is `0.11.0` | broken version parity destroys trust for TS users | **Open** |
| No short demo asset in the README hero | benchmark-heavy products still need a fast visual proof | **Closed in this update** (static SVG hero + social card now in-repo) |

## P1 — lifts trial conversion and shareability

| Gap | Why it matters | Status |
|---|---|---|
| Missing dedicated positioning doc | message drift across README, posts, and replies | **Closed in this update** |
| Missing dedicated competitive analysis | hard to stay honest and sharp about the wedge | **Closed in this update** |
| Missing FAQ / objections handling | launch threads will ask the same questions repeatedly | **Closed in this update** |
| Launch kit was fragmented across a few files | maintainers need one repo-native place to grab assets | **Closed in this update** |
| README onboarding did not route by surface | visitors need an immediate "start here" path | **Closed in this update** |
| No assistant-facing skill surface | coding assistants are now part of the evaluation path for developer tools | **Closed in this update** |
| Integrations were documented but not indexed as a first-class entry point | memory products grow when stack-specific starts are obvious | **Closed in this update** |
| Framework adapters had no repo-native install extras | developers copy the install line before they read the deeper docs | **Closed in this update** |
| Contributor docs used stale repo URL / release notes | small inconsistencies reduce trust | **Closed in this update** |
| No install-free trial path | lowers top-of-funnel conversion from curiosity to first run | **Closed in this update** (repo-ready; public validation still pending) |
| No explicit demo-recording flow for launch assets | makes README/social demo production slower and less repeatable | **Closed in this update** |

## P2 — increases ecosystem pull after launch

| Gap | Why it matters | Status |
|---|---|---|
| No live competitor bench-pack | would reinforce the reproducibility wedge | **Closed in this update** (repo-native offline / local-llm / real profiles plus a publish-ready scorecard generator now ship in-repo) |
| No lightweight web UI knowledge inspector | would make demos and debugging more shareable | **Closed in this update** |
| No notebook-based walkthrough | would help educational and social channels | **Closed in this update** |
| No surface-specific proof asset prepared | concrete proof converts better than generic capability lists | **Closed in this update** (CrewAI + OpenClaw + Claude MCP case-study/post assets now in-repo; public posting still maintainer-only) |

---

## What this update closed by hand

1. Added a formal positioning document
2. Added a competitive analysis with concrete adoption takeaways
3. Added a prioritized gap-analysis document
4. Added a whitepaper and a developer article for publication
5. Added FAQ and objections handling
6. Expanded launch copy to GitHub release/discussion, Reddit, and Dev.to/Medium
7. Expanded the launch plan to a 4-week sequence
8. Improved README onboarding with start-by-surface paths
9. Added a Codespaces/devcontainer path for no-local-install trials
10. Added a deterministic hero demo and a recording script for launch assets
11. Added a buyer-facing comparison guide
12. Fixed stale repo URLs and refreshed contributor / development docs
13. Fixed the npm package repository URL in local metadata for the next publish
14. Surfaced the OpenAI Agents SDK adapter across README, usage docs, and readiness docs
15. Added repo-native visual launch assets: `docs/assets/hero-demo.svg` for the README hero and `docs/assets/social-card.svg` for release/social export
16. Added a submission pack with ready-to-paste directory blurbs, GitHub topics, repo description, and awesome-list snippets
17. Added an AutoGen adapter, runnable example, and framework-focused integration docs/index
18. Added a CrewAI adapter, runnable example, and native docs coverage for `Crew(memory=...)` / `Agent(memory=memory.scope(...))`
19. Added repo-native framework extras and install routes for CrewAI, AutoGen, and the OpenAI Agents SDK
20. Added a zero-network CrewAI surface demo, a CrewAI proof/case-study asset, and a maintainer launch checklist
21. Added an OpenClaw proof/case-study asset and surfaced the zero-network OpenClaw example more directly in docs/launch copy
22. Added a Claude/MCP proof/case-study asset and a zero-network Claude setup demo
23. Added a repo-native `ai-knot` skill plus a skills index for coding assistants that support the skills standard
24. Added a lightweight browser inspector and JSON fact-listing surface on top of the existing HTTP sidecar
25. Added a zero-network browser-inspector demo so the new surface is trialable with one command
26. Added a rendered zero-network notebook walkthrough for educational and social sharing
27. Added a GitHub Pages-ready landing page and deployment workflow for shareable launch links
28. Added a repo-native competitor bench-pack script, guide, and scorecard flow for offline, local-llm, and real comparison profiles
29. Added a named Vercel AI SDK adapter for the npm package, plus repo-native example and docs routing for the mainstream TypeScript app path
30. Added a dependency-light PydanticAI adapter, runnable examples, and docs routing for a framework-native Python agent surface

## Highest-value remaining work

### Maintainer-only blockers

1. Publish `ai-knot@0.11.0` to npm
2. Merge or push the launch-ready branch to public `main`

### High-leverage follow-ups

3. Push the Codespaces path live via public `main` and validate the first-run experience
4. Publish the prepared CrewAI, OpenClaw, and Claude MCP case-study / demo threads
5. Run and publish a fresh competitor bench-pack from merged `main`, ideally with the `real` profile
6. Optionally replace the static README visual with an animated terminal clip after launch

## Non-goals for this launch cycle

Do not delay launch to:

- build a hosted SaaS,
- add every possible backend,
- turn ai-knot into a full agent runtime,
- chase a graph-database story.

Those all dilute the wedge.
