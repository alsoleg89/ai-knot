# ai-knot launch & distribution plan

Updated: **July 5, 2026** · Owner: solo maintainer

This is the operational plan to take `ai-knot` from a 1-star pre-launch repo to first real
adoption. It is built on how comparable OSS agent-memory projects actually grew (MemGPT/Letta,
Mem0, Zep, Cline) and on measured launch-channel data, not folklore. Copy is paste-ready.

If this plan disagrees with [positioning.md](positioning.md), positioning wins.

---

## 1. The strategic thesis (read this first)

**Do not try to win the "biggest LoCoMo number" game. You will lose it, and you shouldn't
want to win it.** Mem0 claims ~92.5%, MemOS ~92.3%, Memori ~82%. ai-knot's honest,
knob-named number is 78%. In a straight number fight ai-knot looks mid-pack.

**Win a different game: reproducibility.** The whole category is in a
[benchmark-credibility crisis](benchmarks.md) — the same system (Zep) has been reported at
84%, 58%, and 75%; published claims span ~58–92%; vendors openly accuse each other of rigged
harnesses. That is the opening. ai-knot's wedge is the contrarian, honest one:

> Everyone's agent-memory benchmark is unreproducible. Here's a memory layer that ships a
> retrieval number **that can't drift** — same fixtures, fixed seeds, no network, no LLM,
> identical on every run — and that needs no LLM on the read path *or* the write path.

This is ideal for a developer audience: it's contrarian, it's testable in one command, and it
leads with the terms Hacker News structurally rewards — **open-source, self-hosted,
local-first, deterministic** — instead of the "AI that does X" framing that
[measurably under-performs](#9-honest-expectations) in 2025–26.

**Three messages, in priority order:**
1. **No LLM on read *or* write** — air-gapped, cheap, reproducible. (The *write* half is the
   rare part; several competitors already skip the LLM on read.)
2. **A benchmark a skeptic can re-run** — the anti-crisis stance.
3. **Multi-agent memory is a governance problem** — trust, provenance, visibility; a real gap
   in most competitors.

## 2. Positioning recap

- **One-liner:** Deterministic, self-hosted long-term memory for AI agents — store facts
  instead of transcripts, recall only what matters, with no LLM on read or write.
- **ICP:** (1) AI engineers building assistants/coding agents who feel the transcript-replay
  tax; (2) teams building multi-agent systems that need shared state with governance; (3)
  regulated / local-first / air-gapped deployments.
- **Primary CTA everywhere:** `pip install ai-knot && ai-knot demo` (or `npx ai-knot-demo`).
- **Proof CTA:** the one-command deterministic benchmark.

## 3. Channel strategy (what, why, in what order)

Ranked by fit for a solo-maintainer OSS memory tool. Rationale is evidence-based; see the
per-channel notes.

| Channel | Role | Why | Realistic yield |
|---|---|---|---|
| **r/LocalLLaMA** | Soft launch | Most tolerant of OSS + self-hostable self-promo; exactly the ICP; safe place to harden the pitch and collect testimonials before the big stage | Highest *conversion* of any channel (~3–8%); first stars + feedback |
| **Show HN** | Main launch | Best stars/traffic yield for OSS; over-indexes on "open-source/self-hosted/deterministic" — ai-knot's exact vocabulary | Front page → hundreds of stars in 48h; but median post dies (see §9) |
| **X / Twitter** | Amplify + build-in-public | Compounding audience; framework communities reshare *useful* tools | Slow burn; warms the launch |
| **Dev.to (+ own blog / Pages canonical)** | Credibility asset | The linkable deep-dive every other channel points to; Dev.to for reach, canonical URL on your domain | Durable SEO + trust |
| **GitHub Release + Discussion** | Home base | Where launch traffic lands and converts; social proof | Conversion surface |
| **Product Hunt** | Secondary awareness | Backlink + badge + maker audience; **not** the centerpiece | Modest spike, ~0.5–2% conversion |
| **awesome-* lists / framework docs** | Sustained distribution | Being the default memory integration compounds forever (the Cline lesson) | The real long-game |
| **LinkedIn** | Founder credibility | Reaches a different, less-technical decision-maker audience | Low but non-zero |

**Deliberately skipped:**
- **arXiv** — as of Nov 2025, arXiv CS rejects non-novel/engineering write-ups and gates
  first-time authors behind endorsement. A product announcement there reads as CV-padding.
  Publish the **whitepaper as a PDF on GitHub Pages** instead (`render_whitepaper_pdf.py`
  already exists). Only revisit arXiv if you write a genuinely novel method+evaluation paper.
- **Medium as primary home** — its paywall/metering suppresses ~44% of search reach and
  developers resent it. Cross-post at most; keep the canonical copy on your own Pages/blog.
- **r/LLMDevs** — self-promotion is prohibited there; don't launch into it.

## 4. Pre-launch groundwork (gate — do all of this before any public post)

Nothing below is optional. A launch spike only converts if the repo can absorb it.

**Product / repo (mostly done — verify):**
- [x] README leads with the pitch, a demo GIF, and a no-signup 30-second try path.
- [x] Deterministic, one-command benchmark in-repo (your credibility defense in threads).
- [x] Clean GitHub description + ~20 topics (done: applied July 2026).
- [x] Comparison page names real competitors honestly ([comparison.md](comparison.md)).
- [ ] **Cut a release.** The whole hardening branch is unreleased and npm still serves
  0.9.3 while PyPI is 0.11.0. Merge `feat/production-hardening` → `main` via PR, then run the
  `Create Release` workflow for the next version (see [RELEASE.md](RELEASE.md)). This also
  un-blocks the Codespaces badge and Pages.
- [ ] **Enable GitHub Pages** (Settings → Pages → source: GitHub Actions), then let
  `pages.yml` deploy on the merge. This makes the landing page, whitepaper, and benchmark
  history live and lets you set the repo homepage (currently empty).
- [ ] **Publish the npm 0.12.0 build** so the badge and `npm install` match the docs.
- [ ] Add a **social preview image** (Settings → Social preview) — it's the thumbnail every
  shared link uses.
- [ ] Open PRs to add ai-knot to relevant lists: `awesome-llm`, `awesome-ai-agents`,
  `awesome-mcp-servers`, `awesome-langchain`. These are slow but compounding.

**Accounts (the #1 avoidable failure):**
- [ ] **Reddit:** the account must be warmed — 2–4 weeks of genuine, diverse comments before
  posting your own repo. A fresh account + a single repo link is the archetypal
  auto-removal/ban trigger. If your account is fresh, add two weeks to the front of this plan.
- [ ] **Hacker News:** you need an aged account; mods restricted Show HN from brand-new
  accounts after the 2025 AI-tool surge. Have some comment history first.

**Assets to have ready before launch day:**
- [ ] A 10–20s **demo GIF/clip** of `ai-knot demo` (silent, autoplay-friendly) — reuse
  `docs/assets/hero-demo.gif` or `render_hero_demo_gif.py`.
- [ ] The Dev.to / blog deep-dive published (see [launch-post.md](launch-post.md) as the spine).
- [ ] The whitepaper PDF live on Pages.

## 5. The 4-week calendar

Concrete dates below assume accounts are already warmed. **If your Reddit/HN accounts are
fresh, insert a 2-week warm-up before Week 1.** The one data-backed timing edge for Show HN
is **Sunday** (lowest competition, ~2× engagement); the weekday-morning-ET advice is folklore.

### Week 1 — Ship & seed (Mon Jul 6 – Sun Jul 12)
| When | Action | Goal |
|---|---|---|
| Mon–Tue | Merge hardening → main (PR); cut release; publish npm; enable Pages | Registries + docs + site all consistent |
| Wed | Add social preview, submit `awesome-*` PRs | Discoverability groundwork |
| Thu–Fri | Publish the Dev.to deep-dive + whitepaper PDF on Pages | The linkable credibility asset exists |
| Daily | Start build-in-public on X (1 post/day: a benchmark, a design choice, a demo clip) | Warm a small audience before launch |

### Week 2 — Soft launch (Mon Jul 13 – Sun Jul 19)
| When | Action | Goal |
|---|---|---|
| **Tue Jul 14** | **r/LocalLLaMA post** (copy in §6). Live in comments all day. | First stars + real feedback; harden the pitch |
| Wed–Thu | Fold r/LocalLLaMA feedback into README/FAQ; fix anything that tripped people | Airtight before Show HN |
| Fri | Post the strongest r/LocalLLaMA takeaway as an X thread | Momentum |

### Week 3 — Main launch (Mon Jul 20 – Sun Jul 26)
| When | Action | Goal |
|---|---|---|
| Mon–Fri | Final polish; pre-write the Show HN maker comment; clear launch-day calendar | Ready |
| **Sun Jul 26, ~12:00 UTC** | **Show HN** (copy in §6). Post the maker comment within 5 min. **Camp the thread for 2+ hours**, reply to everything, link the reproducible benchmark when challenged. | Front page → stars |
| Sun, +1–2h after HN gains traction | **X launch thread** (copy in §6) | Amplify |

*(Weekday alternative if you prefer higher raw volume over lower competition: Tue Jul 21,
~13:00 UTC / 9am ET. Same copy.)*

### Week 4 — Amplify & sustain (Mon Jul 27 – Sun Aug 2)
| When | Action | Goal |
|---|---|---|
| **Tue Jul 28** | **Product Hunt** launch (low effort) + **LinkedIn** post (copy in §6) | Backlink, badge, second audience |
| Wed | **r/AI_Agents** post (reframed for agent builders) | Second Reddit audience |
| Thu–Fri | Reply to every issue/PR/comment fast; ship one visible improvement from feedback | Convert curiosity into trust |
| Ongoing | Weekly build-in-public post; pursue framework-native integration listings | The actual growth engine |

**Two hard rules for a solo maintainer:**
1. **Never fire Show HN + Reddit + Product Hunt the same hour.** Comment presence in the
   first ~2 hours decides each thread, and you can only be in one place. Stagger by days.
2. **Never spend the Show HN before the repo, demo, and benchmark are airtight.** HN punishes
   reposts — you get one real shot. The soft launch exists to harden the pitch first.

## 6. Paste-ready launch copy

All copy leads with the honest wedge and avoids superlatives (which both HN and Reddit
penalize). Fill in the live URL after the release.

### 6a. Show HN

**Title** (≤80 chars, no superlatives, says the two magic words):
```
Show HN: ai-knot – self-hosted agent memory with no LLM on read or write
```
*Alternatives:* `Show HN: ai-knot – deterministic, self-hosted memory for AI agents` ·
`Show HN: ai-knot – agent memory with a benchmark that can't drift`

**Maker comment (post within 5 minutes of submitting):**
```
Author here. I built ai-knot because I got tired of two things: agents that replay the
whole chat transcript into every prompt, and agent-memory benchmarks nobody can reproduce.

What it is: a self-hosted memory layer that stores facts instead of transcripts and recalls
only the few a turn needs. The read path is deterministic — no LLM, no vector-DB-required,
just BM25 + rank fusion — so recall is cheap, auditable, and you can pin it in a regression
test. The part I care most about: it needs no LLM on the WRITE path either. Direct fact
insertion is the default; LLM extraction is opt-in. So the whole pipeline can run air-gapped
with zero model calls.

On benchmarks, I'm deliberately not claiming SOTA. The category is a mess — the same system
has been reported at 84%, 58%, and 75% on LoCoMo, and vendor claims span ~58–92%. So instead
of one more headline number, ai-knot ships a deterministic retrieval number that can't drift
(same fixtures, fixed seeds, no network, no LLM) and reports its LLM-judged QA numbers with
every knob named (78% LoCoMo cat1–4, gpt-4.1 reader / gpt-4o judge). You can re-run the
deterministic one in one command from the repo.

Try it with no signup: `pip install ai-knot && ai-knot demo` (or `npx ai-knot-demo`).

Honest limitation: it's new and far less adopted than Mem0/Zep/Letta, and it deliberately
doesn't build a knowledge graph — if you want LLM-built graph reasoning, Zep/Graphiti or
Cognee are the better fit. Happy to answer anything, and I'd genuinely like to hear where the
determinism trade-off breaks down for your use case.
```

### 6b. r/LocalLLaMA (soft launch)

**Title:** `I built an open-source, self-hostable agent memory layer that runs with no LLM at all (deterministic recall, reproducible benchmark)`

**Body:**
```
I've been building ai-knot, a self-hosted long-term memory layer for AI agents, and it's
now at a point where I'd love feedback from this crowd specifically.

The idea: stop replaying the whole conversation into every prompt. Store facts, recall only
the 3–5 that matter for the next turn. What makes it a bit different from Mem0/Zep/etc.:

- No LLM on the read path (deterministic BM25 + rank fusion — reproducible, testable, cheap).
- No LLM required on the WRITE path either — direct fact insertion is the default, extraction
  is optional. So you can run the entire thing air-gapped with zero model calls.
- Self-hosted on SQLite / Postgres / YAML (YAML is human-readable + git-trackable).
- Multi-agent shared memory with trust/provenance/visibility rules, not just a shared table.
- A deterministic retrieval benchmark you can re-run in one command — because I got tired of
  memory benchmarks nobody can reproduce (the same system has been reported at 84/58/75% on
  LoCoMo).

Try it in 30s, no signup: `pip install ai-knot && ai-knot demo`

Repo + benchmarks: <REPO_URL>

It's early and much smaller than the incumbents — I'm posting here because you all actually
run things locally and will tell me where the no-LLM trade-off falls short. What would you
want it to do that it doesn't?
```
*(On r/LocalLLaMA, an in-post GitHub link on a genuine OSS project is normal. Reply to every
comment. Do not cross-post the same day to other subs.)*

### 6c. X / Twitter launch thread

```
1/ I stopped trusting agent-memory benchmarks.

The same system has been reported at 84%, 58%, and 75% on LoCoMo. Claims span 58–92%.

So I built ai-knot: self-hosted agent memory with a retrieval number that can't drift — and
no LLM on the read path OR the write path.

[demo clip]

2/ The problem: most agents replay the whole transcript into every prompt. Cost grows,
signal drops, and you can't test why a fact showed up.

ai-knot stores facts, recalls only the few a turn needs. Deterministic BM25 + rank fusion.
No model on the hot path.

3/ The rare part is the WRITE path.

Mem0, Zep, Letta, Cognee, LangMem — all use an LLM to populate memory. ai-knot's default
write is direct fact insertion. Extraction is opt-in.

→ the whole pipeline runs air-gapped, zero LLM calls.

4/ On benchmarks I'm not claiming SOTA. I'm claiming reproducible.

Same fixtures, fixed seeds, no network, no LLM → identical numbers every run. Re-run it
yourself in one command. LLM-judged QA numbers are reported with every knob named.

5/ Self-hosted on SQLite / Postgres / YAML. Python + TypeScript + MCP + HTTP + adapters for
CrewAI, LangGraph, LlamaIndex, PydanticAI, OpenAI Agents, AutoGen, Vercel AI SDK.

6/ Try it in 30 seconds, no signup:

pip install ai-knot && ai-knot demo

⭐ the repo if determinism-over-hype is your thing: <REPO_URL>
```
*(Tweet 1 must stand alone with the clip. One CTA, last. Don't tag framework accounts for a
reshare — ship a real integration and let usefulness earn the quote-tweet.)*

### 6d. LinkedIn

```
Agent memory has a trust problem. The same memory system has been publicly reported at 84%,
58%, and 75% on the same benchmark — because the number moves with the reader model, the
judge, the prompts, and which questions you count.

I built ai-knot to take a different position: a self-hosted memory layer for AI agents that
(1) needs no LLM on the read path or the write path, so it runs air-gapped and cheap, and
(2) ships a retrieval benchmark that can't drift — same seeds, no model, identical every run.

It stores facts instead of replaying transcripts, runs on SQLite/Postgres/YAML, and adds real
governance (trust, provenance, visibility) for multi-agent systems.

Open source (MIT), 30-second try, no signup: pip install ai-knot && ai-knot demo

Repo, benchmarks, and an honest comparison vs Mem0/Zep/Letta/Cognee: <REPO_URL>
```

### 6e. Product Hunt

- **Tagline:** `Self-hosted AI agent memory — no LLM, deterministic, reproducible`
- **Description:** `ai-knot stores facts instead of transcripts and recalls only what a turn
  needs — with no LLM on the read or write path. Self-hosted (SQLite/Postgres/YAML), Python +
  TypeScript + MCP, multi-agent governance, and a benchmark you can re-run in one command.`
- **First comment:** reuse the Show HN maker comment, lightly trimmed.

### 6f. GitHub Release notes (seed)

Render from CHANGELOG (`render_github_release.py`), then prepend a 3-line human summary:
```
ai-knot <version> — deterministic, self-hosted memory for AI agents.

Highlights: no LLM required on read or write, bi-temporal point-in-time recall, multi-agent
governance (trust/provenance/visibility), TypeScript + MCP + HTTP surfaces, and a
reproducible retrieval benchmark. Try it: `pip install ai-knot && ai-knot demo`.
```

### 6g. GitHub Discussion (pinned "Show & Tell")

Reuse [launch-post.md](launch-post.md) verbatim as the pinned announcement, with a "what
should the next adapter/backend be?" question at the end to seed contribution.

## 7. Objection handling

Keep [faq.md](faq.md) open in every thread. The four you will get most:

- *"Deterministic = less capable than LLM memory."* → Real trade-off; ai-knot optimizes the
  hot path for reproducibility/cost/auditability and keeps an optional semantic seam for tail
  cases. You don't pay a model tax on every recall to get it.
- *"Why trust your benchmark if the whole field is noisy?"* → Don't trust the headline — run
  the deterministic one yourself; it has zero degrees of freedom. That's the whole point.
- *"Isn't this just Mem0/Zep with fewer features?"* → Different axis. They run an LLM to build
  memory; ai-knot doesn't have to. They're graph/vector; ai-knot is deterministic + graph-free.
  They're bigger; ai-knot is testable and air-gappable. Pick by constraint, not by star count.
- *"Couldn't I build this with a vector DB and prompts?"* → Parts of it. The product is the
  combination: facts + forgetting + bi-temporal recall + shared-pool governance + MCP/TS +
  a reproducible benchmark.

## 8. Metrics to watch

- **Leading:** GitHub stars/day, repo unique visitors (Insights → Traffic), `pip`/`npm`
  installs, Show HN points in first hour, r/LocalLLaMA upvotes + comment sentiment.
- **Converting:** issues opened, Discussions started, adapters requested, first external PR.
- **The one that matters:** does anyone say "I put this in my agent and it worked"? One
  credible testimonial is worth more than 200 drive-by stars.

## 9. Honest expectations

Set these so you don't misread a normal launch as failure:

- **Most Show HN posts die.** Median Show HN earns ~2 points and 0 comments; fewer than 2%
  clear 100. AI-tool submissions structurally under-perform in 2025–26 — which is exactly why
  ai-knot leads with *open-source / self-hosted / deterministic*, not "AI memory."
- **The spike is 48 hours and cannot be the strategy.** ~half of all HN engagement lands in
  the first ~7 hours. Treat launch day as ignition, not the plan.
- **Conversion beats reach.** Reddit and X convert several times better than Product Hunt.
  Hundreds of stars is a good outcome; thousands is an outlier, not a baseline.
- **Sustained presence is the real engine.** Comparable projects (Wasp, Cline) grew over
  months via update posts and framework-native distribution, not one launch day. Plan to keep
  shipping and posting for a quarter.

## 10. After the launch (the compounding work)

- Get listed as an official/first-class memory integration wherever possible (LangGraph,
  LlamaIndex, CrewAI, MCP directories). Being the default is worth more than any spike.
- Turn every good thread question into a docs improvement or a new example.
- Post a monthly "what shipped + new benchmark numbers" update to keep the audience warm.
- Only write an arXiv paper if you develop a genuinely novel method + evaluation — then it
  becomes a legitimate, high-credibility channel, not CV-padding.
