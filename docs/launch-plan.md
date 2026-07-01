# ai-knot — go-to-market & launch plan

The single source of truth for *how we ship this to developers*. Positioning
lives in [positioning.md](positioning.md), copy in [announce.md](announce.md),
long-form assets in [whitepaper.md](whitepaper.md) and
[developer-article.md](developer-article.md), and competitive context in
[competitive-analysis.md](competitive-analysis.md). The dated execution checklist
lives in [channel-playbook.md](channel-playbook.md).

---

## 1. Positioning

**One-liner:** Long-term memory for AI agents — structured, self-hosted, and
deterministic, with benchmark numbers you can reproduce.

**ICP (who feels the pain most):**
1. **Indie / startup AI engineers** building chatbots and coding agents who are
   tired of replaying chat history into every prompt and watching the token bill
   grow. They want a `pip install` that just works, self-hosted, no new SaaS bill.
2. **Teams building multi-agent systems** who need shared state with trust,
   provenance, and visibility — not just a common Postgres table.
3. **Regulated / air-gapped deployments** (fintech, health, gov) where "call a
   cloud LLM to decide what to remember" is a non-starter on cost, latency, or
   privacy grounds.

**Main pain solved:** Agents either forget everything between sessions, or you
pay to replay the whole transcript. ai-knot keeps the *knowledge*, not the log,
and retrieves only what the next turn needs — with no LLM on the read path.

**Why now:** (a) Agent frameworks are exploding but treat memory as an append-only
log; (b) the agent-memory benchmark numbers everyone quotes do not reproduce, so
there is an open lane for an *honest* entrant; (c) MCP made "bring your own
memory tool" a first-class integration surface for Claude Desktop/Code.

**Why different:** deterministic retrieval (no LLM on the hot path → reproducible,
cheap, auditable), a human-readable store you can commit to git, and a real
multi-agent governance model (trust + provenance + visibility) — none of which the
incumbents offer together. And the headline benchmark is one a skeptic can re-run
in 30 seconds.

---

## 2. Competitive landscape (takeaways, not a brag sheet)

| Project | Strength we respect | Our wedge against it |
|---|---|---|
| **Mem0** | Mature, hosted offering, large community, broad framework glue | Self-hosted + deterministic + reproducible numbers |
| **Graphiti / Zep** | Genuine temporal knowledge graph; strong relational memory | Deterministic supersession, lower operational complexity |
| **Letta** | Strong brand, product shell, stateful-agent narrative | ai-knot is a library you drop into any agent stack |
| **LangMem** | Path of least resistance inside LangGraph | Framework-agnostic; MCP-native; storage control |

**What made these projects spread** (and what we copy):
- A **dead-simple `pip install` + 5-line snippet** that demonstrates the magic.
- A **benchmark or concept hook** that gives the project a story people can repeat.
- **MCP / framework / app surfaces** that let people try it inside the tools they already use.
- A **clear category label** that says what the project is in one sentence.

The differentiator we lean on hardest: **reproducibility as the marketing**. In a
field with a credibility crisis, the honest number *is* the hook.

---

## 3. Gap analysis (prioritized)

### Closed this cycle ✅
- Developer-first README with a verified, runnable "see it work" example.
- Real, named-reader LLM-judged numbers + a deterministic one-command number.
- String-`type` SQLite crash fixed; embed-fallback noise silenced (clean first run).
- Full API guide split into `docs/usage.md`; release runbook in `docs/RELEASE.md`.
- Dedicated positioning, competitive-analysis, gap-analysis, FAQ, whitepaper, and developer article docs.
- Contributor and development docs updated to the current repo URL and release flow.
- README onboarding upgraded with start-by-surface quick paths (Python, TS, MCP, HTTP, AutoGen, LangChain, shared pool).
- Dedicated integrations index added so framework/runtime entry points are obvious.
- Repo-native install extras added for CrewAI, AutoGen, and the OpenAI Agents SDK so each surface has a direct package path.
- Codespaces/devcontainer path added for install-free trials.
- Deterministic hero demo + recording script prepared for the README/social demo asset.
- Buyer-facing comparison guide added.
- Vercel AI SDK adapter and npm-side example/docs added so the mainstream TypeScript app path is explicit instead of generic.
- PydanticAI adapter and repo-native examples/docs added for a framework-native
  Python agent surface that uses runtime `instructions=...`.
- CrewAI follow-up distribution assets prepared: zero-network demo, case-study copy, and a maintainer launch checklist.
- Vercel AI SDK follow-up distribution assets prepared: named adapter, npm-side example, and channel copy for the TypeScript app path.
- PydanticAI follow-up distribution copy prepared in `docs/announce.md`, tied to
  `examples/pydanticai_surface_demo.py` and `examples/pydanticai_integration.py`.
- OpenClaw follow-up distribution asset prepared: app/MCP case-study copy routed to the zero-network example and setup command.
- Claude/MCP follow-up distribution asset prepared: zero-network setup demo plus channel-ready case-study copy.
- Lightweight browser inspector added on top of the HTTP sidecar for demos and debugging.
- Rendered zero-network notebook walkthrough added for educational and social sharing.
- GitHub Pages-ready landing page plus deploy workflow added for shareable non-README links.

### P0 — blocks a credible launch (do before posting)
1. **npm/PyPI version sync.** As of **June 30, 2026**, PyPI is on `0.11.0` while npm
   is still on `0.9.3`. Rotate `NPM_TOKEN` if needed (see [RELEASE.md](RELEASE.md))
   and publish. A broken or outdated `npm install` on launch day kills trust.
   *(needs maintainer: registry publish rights.)*
2. **Push/merge the launch-ready branch to public `main`.** The public GitHub repo still
   shows the pre-launch packaging and has **1 star**; the distribution push should start
   only after the public landing page matches this branch. *(needs maintainer: merge/push.)*

### P1 — strongly lifts conversion / shareability
3. **A 20-second demo GIF/asciinema** at the top of the README (add → recall →
   "it remembered"). Top dev READMEs all have one; text-only converts worse.
4. **Publish the prepared proof posts.**
   CrewAI, PydanticAI, Vercel AI SDK, OpenClaw, and Claude/MCP now all have repo-native follow-up assets; the remaining
   step is to post them publicly so developers see exact workflows they can copy.
5. **Validate and publish the install-free path.** Codespaces support is now in-repo;
   the maintainer should verify the public `codespaces.new` flow after merging to `main`.
6. **Enable GitHub Pages after merge.** The repo now contains `docs/site/index.html`
   and `.github/workflows/pages.yml`; once `main` is live, enable Pages so launch
   posts can link to a cleaner landing page than raw GitHub markdown.

### P2 — depth, post-launch
7. Run the in-repo competitor bench-pack and export a fresh scorecard (`real` profile if the stack is available).
8. More storage backends (Qdrant/Weaviate/Mongo) — community-PR friendly asks.
9. Public benchmark write-up and competitor scorecard refresh after launch.

---

## 4. Channel strategy

| Channel | Angle | CTA | When |
|---|---|---|---|
| **GitHub release + pinned discussion** | "v0.x — reproducible agent memory" | Star, try the 5-line snippet | Day 0 (soft) |
| **r/LocalLLaMA** | Self-hosted, no-cloud, runs offline | Repo link + snippet | Day 1 (soft) |
| **Show HN** | "deterministic agent memory with reproducible benchmarks" | Repo link, invite scrutiny of the numbers | Week 2 (main) |
| **X/Twitter thread** | The reproducibility-crisis hook | Repo link, thread | Week 2 |
| **LinkedIn** | Same thesis, more trust / infra framing | Repo link | Week 2 |
| **Dev.to / Medium** | Technical article + whitepaper | "Try it" + repo | Week 2 |
| **r/MachineLearning** | Methodology and deterministic benchmark angle | Discussion | Week 3 |
| **Awesome-lists PRs** | Listing | Inclusion | Week 3 |
| **1:1 founder / maintainer outreach** | "Stress-test the benchmarks or adapter flow" | Feedback, first users, quotes | Weeks 1–3 |
| **Follow-up release / adapter post** | "Now works with X stack" | Re-engage observers | Week 4 |

**Soft launch first** (release + Reddit + a few DMs) to shake out install bugs and
collect the first testimonials; **main launch** (Show HN + threads) only after P0
is green and the demo asset is in.

---

## 5. Four-week launch sequence

**Week 0 — pre-flight**
- [ ] Publish npm `0.11.0` so the public registry matches PyPI. *(maintainer)*
- [ ] Merge/push the launch-ready branch so public `main` matches this repo state. *(maintainer)*
- [ ] Record the 20-second demo GIF or terminal capture and add it to the README hero.
- [ ] Cut a tagged GitHub release with notes pulled from `CHANGELOG.md`.

**Week 1 — soft launch / failure-finding**
- Day 0: GitHub release + pinned discussion.
- Day 1: r/LocalLLaMA post with the self-hosted / no-LLM-on-read-path angle.
- Day 2: direct outreach to 10-20 builders who already care about agent memory,
  MCP, or air-gapped deployments. Goal: find install friction and collect the
  first two or three honest quotes.
- Day 3-4: fix anything the soft launch surfaced; tighten docs and FAQ from real questions.

**Week 2 — main launch**
- Tue-Thu, ~9am ET: **Show HN** + X thread + LinkedIn in the same window.
- Same day: publish the developer article on Dev.to or Medium.
- Next day: reply to every benchmark / methodology question with a direct reproduce-it command.

**Week 3 — credibility and ecosystem pull**
- Post the methodology angle to r/MachineLearning.
- Submit awesome-list PRs (agent memory, MCP, AI infra).
- Publish the prepared CrewAI follow-up, then PydanticAI, then Vercel AI SDK,
  then OpenClaw, then Claude/MCP, then rotate to LangChain or HTTP sidecar.

**Week 4 — retention and pull-through**
- Ship the next framework adapter.
- Publish a "what we learned from launch" follow-up with the next roadmap asks.
- Open and label community-friendly issues for backends and adapters.

---

## 6. Success metrics (per step, adoption-weighted)

| Signal | Soft-launch target | Main-launch target |
|---|---|---|
| GitHub stars | first 25–50 | 300+ in launch week |
| `pip`/`npm` installs | install works, zero version-mismatch complaints | steady daily |
| HN | — | front-page /new, >30 points |
| Developer trust | no broken-install reports | benchmark questions answered, not contested |
| Integration pull | — | ≥1 "can you add X backend/adapter" issue |

The north star is **developer trust**: every claim in the repo is reproducible, so
the launch's job is to invite scrutiny, not deflect it.

---

## 7. Done vs remaining

**Done (hands-on this program):** developer-first README, reproducible benchmarks
across both suites, usage/benchmarks/deployment/production-readiness docs, launch
copy, release runbook + idempotent publish workflows, CLI audit ops, LangChain /
LangGraph adapters, a docs-based launch kit (positioning, competitive analysis,
gap analysis, FAQ, whitepaper, developer article), and refreshed contributor docs.
Codespaces/devcontainer support, a buyer-facing comparison guide, a demo-recording
flow, CrewAI / AutoGen / OpenAI Agents / PydanticAI adapters, CrewAI / OpenClaw / Claude MCP
case-study / proof assets, and a submission pack for directory/listing distribution
are also now in the repo. The same is now true for the browser inspector and the
rendered notebook walkthrough.

**Remaining, in priority order:**
1. *(maintainer)* npm publish to `0.11.0`; merge/push the launch-ready branch to public `main`.
2. Demo GIF or terminal capture in README.
3. Enable the GitHub Pages landing page after `main` is updated.
4. Publish the prepared CrewAI, PydanticAI, Vercel AI SDK, OpenClaw, and Claude/MCP surface-specific proof posts.
5. Validate the public Codespaces quickstart after the branch is on `main`.
6. Fresh public competitor scorecard refresh; more backends only if they add a real channel.
