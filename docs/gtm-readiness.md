# GTM readiness & gap register

Updated: **July 5, 2026**

A launch-readiness audit for `ai-knot`: current state, what was just fixed, the prioritized
gap list, and remaining work in order. Pair with [launch-plan.md](launch-plan.md) (the
playbook) and [positioning.md](positioning.md) (the message house).

---

## Status snapshot

| Signal | State | Note |
|---|---|---|
| Product maturity | **Strong** | Deterministic core, bi-temporal recall, multi-agent governance, 1000+ tests, extensive docs |
| Docs / launch kit | **Strong** | Whitepaper, dev article, FAQ, comparison, positioning, benchmarks, case studies, launch copy |
| GitHub stars | **1** | Genuinely pre-launch; no distribution has happened yet |
| PyPI | **0.11.0 published** | Description now matches positioning |
| npm | **0.9.3** ⚠️ | Lags PyPI; the 0.11.0/next build was never published |
| GitHub Release | **v0.9.3 (stale)** ⚠️ | 0.10.0 / 0.11.0 never got a Release; page misrepresents the project |
| `main` branch | **old orphan `41366ca`** ⚠️ | All hardening work sits unreleased on `feat/production-hardening` |
| GitHub Pages | **not enabled (404)** ⚠️ | Landing page, whitepaper, and benchmark history are dark |
| Repo description/topics | **Fixed ✅** | Fixed `antrophic` typo; description now matches positioning; 20 discoverable topics (incl. real providers like `gigachat`) |
| Comparison page | **Fixed ✅** | Was fictional competitor names; now real, checked (Mem0/Zep/Letta/Cognee/LangMem/Memori) |

## Done in this pass (by hand)

1. **Repo metadata cleaned and applied live** (`gh api`): description now
   "Deterministic, self-hosted long-term memory for AI agents…"; topics replaced with 20
   relevant, discoverable tags (removed the `antrophic` typo; kept `gigachat`, which is a
   real provider); widened `RECOMMENDED_GITHUB_TOPICS` in `scripts/check_public_release.py`.
2. **Rewrote `docs/comparison.md`** with real, per-capability-checked competitors and an
   honest maturity/star disclosure. Removed the fictional OpenViking/MemPalace/Engram names.
3. **Rewrote the README "How it compares" section** to match (real names, honest wedge).
4. **Created `docs/launch-post.md`** — the canonical announcement (also fixes a broken
   CHANGELOG reference to a file that didn't exist).
5. **Created `docs/launch-plan.md`** — channel strategy, a dated 4-week calendar, and
   paste-ready copy for Show HN, r/LocalLLaMA, X, LinkedIn, Product Hunt, GitHub Release, and
   the pinned Discussion, all grounded in launch-channel data.
6. **Created this gap register.**
7. **Sharpened `docs/positioning.md`** — the differentiator is now "no LLM on read *or*
   write" (the defensible claim) with an explicit anti-overclaim guardrail.
8. **Fixed the dead GitHub Pages benchmark-history link** in the README.
9. **Reworked the README hero** to lead with the wedge + a pain hook + a one-line
   no-signup try command + the reproducibility callout, instead of the mid-pack LoCoMo
   number. Applied the same sharpening to the **npm package README** and the **GitHub
   Pages landing hero** so all three front doors are consistent.
10. **Added a real GigaChat (Sber) provider** (`ai_knot.providers.gigachat`) with OAuth2
    token exchange, caching, scope, and TLS control — replacing a bearer-token-only shim
    that could not authenticate. Restored `gigachat` as a repo topic (it is a real provider).
11. **Verified DX and link health:** the core `pip install` example entry points and the
    CLI loop run clean offline, and all relative links across 35 markdown files resolve.
12. **Benchmarked the README against real competitors** (Mem0, Letta, Zep/Graphiti, Cognee,
    Memori, LangMem, Supermemory). Finding: ai-knot's README is already top-tier structurally
    (hero, GIF, badges, nav bar, honest comparison). Applied the honest wins — front-loaded the
    reproducible deterministic number, moved the comparison above the benchmark section, removed
    leaked meta-copy, added a non-fake community line — while explicitly avoiding the traps the
    incumbents use (fake social-proof badges, cloud-signup CTA) that would clash with a pure-OSS,
    reproducibility-first positioning.

### Follow-up pass (product-move research → implementation)

Three product moves were each researched with a structured multi-perspective pass, then the
top recommendation of each was implemented:

13. **Surfaced the multi-agent governance moat** — `docs/multi-agent-governance.md` maps the
    governed-shared-memory surface (access control, supersession, provenance, trust, audit) to
    code line-by-line and to the 2026 fleet-memory literature. This closes the sharpest gap
    found: ten framework case studies existed and **zero** for the actual differentiator. The
    doc is wired into the README, the docs index, and the examples index.
14. **Made the governance edge visceral and testable** — `examples/poisoned_pool.py` runs an
    attacker against a shared pool and shows trust collapse, monotonic-CAS stale-replay
    rejection, and poison suppression, all computed at runtime with no LLM. Backed by a
    regression test. This is the un-fakeable, "watch-it-defend-itself" demo an infra tool needs.
15. **Opened the Node/TS audience** — a `Dockerfile` runs the HTTP sidecar and the npm README
    now leads with the HTTP client, so TypeScript users reach the deterministic core with **no
    Python on their machine**. This targets the single biggest reach ceiling (the prior
    Python-on-PATH requirement). A native pure-TS port was deliberately **not** started: it
    would fork the reproducibility wedge into per-SDK behaviour and add a permanent maintenance
    tax; gate it behind real adoption/edge-runtime demand.

## Prioritized gap list

### P0 — blocks a credible launch (maintainer action; needs merge/publish rights)

1. **Merge `feat/production-hardening` → `main` via PR.** Everything below depends on it.
   `main` currently shows an early orphan commit; the real project is on the branch.
2. **Cut a release** (`Create Release` workflow) for the next version so PyPI, npm, and the
   GitHub Release page all reflect the shipped product. See [RELEASE.md](RELEASE.md).
3. **Publish npm** to close the 0.9.3 → current skew (the npm badge and `npm install` path
   currently contradict the docs).
4. **Enable GitHub Pages** (Settings → Pages → GitHub Actions source). Unblocks the landing
   page, the whitepaper PDF, and the benchmark history; then set the repo homepage.

### P1 — high-leverage before/at launch

5. **Warm the Reddit and Hacker News accounts** (2–4 weeks of genuine activity) — a fresh
   account posting a repo link is the top auto-removal trigger. Gate the launch on this.
6. **Add a social preview image** (Settings → Social preview) — it's every shared link's
   thumbnail.
7. **Record a tight 10–20s demo GIF/clip** for Reddit/X (reuse `render_hero_demo_gif.py`).
8. **Publish the Dev.to deep-dive + whitepaper PDF** (canonical URL on your own Pages).
9. **Open `awesome-*` list PRs** (awesome-llm, awesome-ai-agents, awesome-mcp-servers,
   awesome-langchain) — slow but compounding distribution.
10. **Add a benchmark visual** (a simple bar chart of MRR 0.83 vs 0.18 / recall@5 0.26 vs 0.15)
    near the README benchmark section — every high-star competitor shows a benchmark image;
    ai-knot only has text tables. On-brand because it's the reproducible number, not the QA one.
11. **Optional README de-duplication:** "At a glance", "Why teams pick", and "What you can
    build" overlap. Length is fine; redundancy is the only remaining structural nit — a
    subjective cut left to the maintainer, not done here to avoid over-editing.

### P2 — sustained growth (post-launch)

10. **Pursue first-class integration listings** in LangGraph / LlamaIndex / CrewAI / MCP
    directories — being the default memory backend compounds forever.
11. **Ship one real framework integration announcement** to earn a framework reshare (worth
    more than tagging accounts).
12. **Monthly "what shipped + new benchmark" update** to keep the audience warm.
13. **Consider an arXiv paper only if** a genuinely novel method+evaluation emerges (not the
    current engineering write-up — arXiv now rejects those).

## Remaining work, in order

1. PR: `feat/production-hardening` → `main` (P0-1).
2. Release cut + npm publish + GitHub Release notes (P0-2, P0-3).
3. Enable Pages, deploy site, set homepage (P0-4).
4. Warm accounts; produce demo GIF + social preview (P1).
5. Publish Dev.to article + whitepaper PDF; submit awesome-list PRs (P1).
6. Execute the [launch-plan.md](launch-plan.md) calendar (soft launch → Show HN → amplify).
7. Post-launch: integration listings, monthly updates, convert feedback into docs/examples.

## What is explicitly *not* a gap

- **More docs.** The repo already over-documents for its stage; the constraint is
  distribution and release hygiene, not content volume.
- **A bigger headline benchmark number.** Chasing SOTA would undercut the reproducibility
  wedge. The honest, knob-named number plus the deterministic suite is the strategy.
- **arXiv.** Off-norm for a non-novel engineering release (see P2-13).
