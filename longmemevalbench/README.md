# longmemevalbench — LongMemEval benchmark for ai-knot

A sibling of `aiknotbench/` (the LOCOMO harness). Same conventions: per-run DB
isolation, `checkpoint.json` resume, an LLM judge, and ingest-granularity modes.
Where LOCOMO scores cat1–5, LongMemEval scores six question types plus
abstention, and reports turn-level / session-level memory recall.

LongMemEval (Wu et al., ICLR 2025 — [arXiv 2410.10813](https://arxiv.org/abs/2410.10813),
[repo](https://github.com/xiaowu0162/LongMemEval)) stresses five long-term-memory
abilities — information extraction, multi-session reasoning, temporal reasoning,
knowledge update, abstention — over long user↔assistant chat histories. It maps
onto ai-knot's primitives (bi-temporal `Fact`, `event_time` anchor, `valid_until`
supersession, `Fact.supported`) much better than LOCOMO does.

## Quick start (smoke fixture)

The full 500-question dataset is **not** bundled (see below). A small hand-built
fixture with all six question types + one abstention question ships so the whole
pipeline runs end-to-end:

```bash
cd longmemevalbench
npm install          # links ai-knot from ../npm (build it first — see "ai-knot client")

export OPENAI_API_KEY=sk-...
export AI_KNOT_COMMAND=ai-knot-mcp      # or scripts/ai-knot-mcp-worktree.sh in a worktree
export DEFAULT_JUDGE_MODEL=gpt-4o       # LongMemEval's official judge
export DEFAULT_ANSWER_MODEL=gpt-4.1-nano  # reader = same small model as LOCOMO

node --import tsx src/index.ts run -r smoke \
  --data data/sample_longmemeval.json --force \
  --knot-env AI_KNOT_EMBED_URL=https://api.openai.com \
  --knot-env AI_KNOT_EMBED_MODEL=text-embedding-3-small \
  --knot-env OPENAI_API_KEY=$OPENAI_API_KEY \
  --knot-env AI_KNOT_EMBED_API_KEY=$OPENAI_API_KEY
```

Run `node --import tsx src/index.ts help` for the full option list. Tests
(`npm test`) and typecheck (`npm run typecheck`) need no API keys or MCP server —
they use injected fakes.

## Obtaining the real dataset

The official LongMemEval set is release-gated (HuggingFace / Google Drive linked
from the [repo README](https://github.com/xiaowu0162/LongMemEval)). Download the
variant you want and drop it in `data/`:

| Variant | File you place | Haystack |
|---|---|---|
| LongMemEval_S | `data/longmemeval_s.json` | ~115k tok, ~40–50 sessions (standard) |
| LongMemEval_M | `data/longmemeval_m.json` | ~500 sessions (retrieval variant) |
| LongMemEval_Oracle | `data/longmemeval_oracle.json` | evidence sessions only (reading upper bound) |

Then point the harness at it:

```bash
node --import tsx src/index.ts run -r s-full --data data/longmemeval_s.json
# or: export LONGMEMEVAL_FILE=data/longmemeval_s.json
```

The loader (`src/loader.ts`) parses the official schema directly
(`question_id`/`question_type`/`question`/`answer`/`question_date`/
`haystack_session_ids`/`haystack_dates`/`haystack_sessions`/`answer_session_ids`),
detects the `_abs` suffix as abstention, and reads `has_answer` turn flags. No
schema changes are needed to switch from the fixture to the real set.

> These data files are git-ignored. Do **not** commit the dataset.

## Ingest granularity (the paper's "value granularity" lever)

`--granularity`:
- `window` — sliding 3-turn window per session (the LOCOMO default unit).
- `round` — one (user, assistant) pair per fact (**default**; the paper's
  value-granularity decomposition — finer unit, better multi-session).
- `session` — one fact per whole session (coarse).

Every mode passes the per-session timestamp (`haystack_dates[i]`) as the
**structured `eventTime` anchor** via `kb.add(content, { eventTime })`. The
timestamp is attached to the fact as data; it is **never** prefixed into the
indexed content (the banned `dated` text hack). The core's RRF fuses whatever
granularity is ingested.

## Multi-agent mode (`--multi-agent`)

LongMemEval histories are two-party (user + assistant). With `--multi-agent`,
user turns and assistant turns are ingested under **separate agent namespaces**
(`q-<id>::user` and `q-<id>::assistant`), exercising ai-knot's per-agent
working-memory + shared-layer model (`namespace` + `event_time` + `valid_until`
already map onto this). Recall unions across both namespaces. Without the flag,
both roles share one namespace per question (the default).

## Abstention reader contract (prerequisite C)

Enabled by default (disable with `--no-idk`). It is a **generic** grounded-QA
contract, not a benchmark-tailored rule:

1. **Empty-pool short-circuit** — if recall surfaces nothing, the reader returns
   a deterministic `"I don't know."` with **no LLM call**.
2. **IDK system prompt** — instructs the reader to decline when the context does
   not support an answer or presupposes a false premise, and to use the **most
   recent** value when a fact changed (knowledge-update aware).

Scoring: a `_abs` question is CORRECT iff the reader declined; an *answerable*
question the reader declines is WRONG. The LOCOMO reader (`aiknotbench`) is
untouched — this contract lives only here.

## Recall scoring (turn + session)

LongMemEval reports turn-level recall (`has_answer` turns) and session-level
recall (`answer_session_ids`), excluding the 30 `_abs` questions. The official
scorer matches retrieved *unit ids*. This harness's recall surfaces formatted
**text** (the core returns prompt-ready strings, not fact→session links), so
`src/recall.ts` approximates the id-match with a content-token overlap check
(an evidence unit "made the pool" at ≥60% content-token overlap). This is a
documented, conservative approximation — it can undercount, never leaks the gold
answer — and is a candidate to replace with true id propagation (see "Next
steps").

## ai-knot client (`../npm`)

`package.json` depends on `ai-knot` via `file:../npm`. Build the client once so
its type declarations resolve:

```bash
cd ../npm && npm install && npm run build && cd ../longmemevalbench
```

In an isolated git worktree where `../npm` is unbuilt, either build it there or
symlink `node_modules/ai-knot` to a checkout that has `dist/`.
`scripts/ai-knot-mcp-worktree.sh` launches the MCP server from the worktree's
`src/` (so worktree-local engine changes are exercised) — set
`AI_KNOT_COMMAND` to it.

## What's scored

The report (`data/runs/<id>/report.json`) contains:
- overall accuracy + per-question-type accuracy,
- abstention accuracy (the `_abs` subset),
- turn-level and session-level recall rates.

## Next steps (not yet built)

- **Time-aware query filtering** (paper's +6.8–11.3% temporal lever): derive a
  date window from `question_date` and filter/boost by `event_time`. The
  structured anchor now persists (prerequisite A), so this is unblocked.
- **Knowledge-update ingest via slots**: route KU ingest through
  `kb.learn()` (LLM slot extraction) or `kb.add_resolved()` (pre-extracted
  slots) so `valid_until` supersession fires; then recall with
  `now = question_date`. Plain `kb.add()` windows carry no slot, so supersession
  cannot fire from them (documented limitation).
- **True recall id-propagation** to replace the content-overlap approximation.
- **Frozen-retrieval + reader sweep** (Memoria-style) to attribute gains to
  retrieval vs reader.
- **LongMemEval_M** scale pass.
