# ai-knot — go-to-market & launch plan

The single source of truth for *how we ship this to developers*. Copy lives in
[announce.md](announce.md) (Show HN, threads) and [launch-post.md](launch-post.md)
(long-form). This file is the strategy that sequences them.

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
log; (b) the agent-memory benchmark numbers everyone quotes don't reproduce, so
there's an open lane for an *honest* entrant; (c) MCP just made "bring your own
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
| **Mem0** | Mature, hosted offering, large community, broad framework glue | Self-hosted + deterministic + reproducible numbers; their 91.6% reproduces at ~58–66% |
| **Zep / Graphiti** | Genuine temporal knowledge graph; strong relational memory | Their bi-temporal reasoning needs an LLM; ours is deterministic supersession. Their 84% LoCoMo → 58% corrected |
| **Letta (MemGPT)** | Strong brand, "agent OS" narrative, self-editing memory | We're a *library* you drop into any agent, not a runtime to adopt; deterministic recall |
| **LangMem** | Path of least resistance inside LangGraph | Framework-agnostic; no lock-in; MCP-native |

**What made these projects spread** (and what we copy):
- A **dead-simple `pip install` + 5-line snippet** that demonstrates the magic.
- A **benchmark claim** that gives the project a number to rally around.
- **MCP / framework integration surfaces** that let people try it inside the tool
  they already use.
- A **clear "why a log isn't enough"** narrative that names the pain.

We now have all four. The differentiator we lean on hardest: **reproducibility as
the marketing** — in a field with a credibility crisis, the honest number *is* the
hook.

---

## 3. Gap analysis (prioritized)

### Closed this cycle ✅
- Developer-first README with a verified, runnable "see it work" example.
- Real, named-reader LLM-judged numbers + a deterministic one-command number.
- String-`type` SQLite crash fixed; embed-fallback noise silenced (clean first run).
- Full API guide split into `docs/usage.md`; launch copy ready in `announce.md`.

### P0 — blocks a credible launch (do before posting)
1. **npm/PyPI version sync.** npm is stuck at 0.9.3 while PyPI is 0.11.0. Rotate
   `NPM_TOKEN` (see [RELEASE.md](RELEASE.md)) and publish. A broken `npm install`
   on launch day kills trust. *(needs maintainer: secret rotation.)*
2. **Merge PR #103 to main** so the repo a visitor lands on is the new one.

### P1 — strongly lifts conversion / shareability
3. **A 20-second demo GIF/asciinema** at the top of the README (add → recall →
   "it remembered"). Top dev READMEs all have one; text-only converts worse.
4. **One framework adapter shipped** (LangGraph *or* OpenAI Agents SDK) — turns
   "interesting library" into "drop-in for my stack." Highest integration-pull.
5. **A hosted/Colab "try without installing" notebook** linked from the README.

### P2 — depth, post-launch
6. Live side-by-side competitor bench-pack (Mem0) in-repo.
7. More storage backends (Qdrant/Weaviate/Mongo) — community-PR friendly asks.
8. Web UI knowledge inspector.

---

## 4. Channel strategy

| Channel | Angle | CTA | When |
|---|---|---|---|
| **GitHub release + pinned discussion** | "v0.x — reproducible agent memory" | Star, try the 5-line snippet | Day 0 (soft) |
| **r/LocalLLaMA** | Self-hosted, no-cloud, runs offline | Repo link + snippet | Day 1 (soft) |
| **Show HN** | "deterministic agent memory with reproducible benchmarks" | Repo link, invite scrutiny of the numbers | Day 3 (main) |
| **X/Twitter thread** | The reproducibility-crisis hook (Zep 84→58, Mem0 91→58–66) | Repo link, thread | Day 3 (with HN) |
| **dev.to / Medium** | Long-form launch-post.md | "Try it" + repo | Day 4 |
| **LinkedIn** | Same thread, professional framing | Repo link | Day 4 |
| **Awesome-lists PRs** (Agent Memory, awesome-mcp) | Listing | Inclusion | Week 2 |
| **r/MachineLearning** | The benchmark methodology piece | Discussion | Week 2 |

**Soft launch first** (release + Reddit + a few DMs) to shake out install bugs and
collect the first ⭐ and testimonials; **main launch** (Show HN + threads) only
after P0 is green and the demo GIF is in.

---

## 5. Two-week launch sequence

**Week 0 — pre-flight (P0 + P1.3)**
- [ ] Rotate `NPM_TOKEN`, publish npm to version parity with PyPI. *(maintainer)*
- [ ] Merge PR #103 → main. *(maintainer)*
- [ ] Record the 20-second demo GIF, add to README hero.
- [ ] Cut a tagged GitHub release with notes drawn from CHANGELOG.

**Week 1 — soft launch**
- Day 0: GitHub release + pinned discussion.
- Day 1: r/LocalLLaMA post (snippet + offline angle). Watch for install issues.
- Day 2: fix anything the soft launch surfaced; gather one or two quotes.

**Week 1–2 — main launch**
- Day 3 (Tue–Thu, ~9am ET): **Show HN** + X thread + LinkedIn, same hour.
  Be present in the HN thread all day — answer every skeptical benchmark question
  with the reproduce-it command.
- Day 4: publish long-form on dev.to / Medium; cross-link from HN if alive.
- Day 5: thank-you + "what we're building next" follow-up; open "good first issue"
  labels (storage backends) to convert interest into contributors.

**Week 2 — sustain**
- Submit awesome-list PRs (Agent Memory, awesome-mcp).
- Post the methodology piece to r/MachineLearning.
- Ship the first framework adapter (P1.4); announce as a small follow-up.

---

## 6. Success metrics (per step, adoption-weighted)

| Signal | Soft-launch target | Main-launch target |
|---|---|---|
| GitHub stars | first 25–50 | 300+ in launch week |
| `pip`/`npm` installs | install works, zero "Invalid Version" reports | steady daily |
| HN | — | front-page /new, >30 points |
| Developer trust | no broken-install reports | benchmark questions answered, not contested |
| Integration pull | — | ≥1 "can you add X backend/adapter" issue |

The north star is **developer trust**: every claim in the repo is reproducible, so
the launch's job is to invite scrutiny, not deflect it.

---

## 7. Done vs remaining

**Done (hands-on this program):** developer-first README, real reproducible
benchmarks across both suites, usage/benchmarks/deployment/production-readiness
docs, launch copy (Show HN + threads + long-form), release runbook + idempotent
publish workflows, CLI audit ops, two first-run DX bug fixes, this GTM plan.

**Remaining, in priority order:**
1. *(maintainer)* npm token rotation → version parity; merge PR #103.
2. Demo GIF in README.
3. One framework adapter (LangGraph / OpenAI Agents).
4. "Try without installing" notebook.
5. Live competitor bench-pack; more backends; web inspector.
