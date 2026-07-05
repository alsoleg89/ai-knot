# 🪢 ai-knot

![npm](https://img.shields.io/npm/v/ai-knot)
![License](https://img.shields.io/badge/license-MIT-green)

**Self-hosted memory for AI agents — for Node.js and TypeScript.**

Your agent forgets everything between sessions, and replaying the whole conversation into
every prompt is slow and expensive. ai-knot remembers *facts*, not transcripts: it recalls
only the few your agent needs for the next turn — with **no LLM on the read path or the
write path**, so recall is cheap, deterministic, and testable.

```bash
npm install ai-knot && npx ai-knot-demo   # 30-second proof, no signup
```

TypeScript client for the [ai-knot](https://github.com/alsoleg89/ai-knot) engine — it talks
to the `ai-knot-mcp` subprocess over JSON-RPC. Same deterministic core, and the same
[benchmark you can re-run yourself](https://github.com/alsoleg89/ai-knot/blob/main/docs/benchmarks.md)
(a retrieval number that can't drift, plus LoCoMo 78.0% with every knob named).

---

## Requirements

- Node.js 18+
- Python 3.11+ with `pip`

## Install

```bash
npm install ai-knot
```

The postinstall script automatically runs `pip install "ai-knot[mcp]"`. If pip is not found, a warning is printed — install it manually:

```bash
pip install "ai-knot[mcp]"
npx ai-knot-doctor
```

If you want the fastest installed proof after `npm install ai-knot`, run:

```bash
npx ai-knot-demo
```

If you want the shortest app-level TypeScript path with a mainstream runtime,
pair it with the Vercel AI SDK:

```bash
npm install ai-knot ai @ai-sdk/openai
```

If the install path is unclear or the Python bridge looks suspicious, use:

```bash
npx ai-knot-doctor --json
```

From this repo, the same check is:

```bash
npm run doctor
```

---

## First-run bridge check

The npm client is a thin Node wrapper around the Python `ai-knot-mcp` binary.
Before you debug app code, prove that bridge first:

```bash
npx ai-knot-doctor
```

It checks:

- Node version
- Python `3.11+`
- `pip`
- whether Python can import `ai_knot`
- whether the installed Python package version matches the npm package version
- whether `ai-knot-mcp` is actually on `PATH`
- the Python-side `ai-knot doctor --json` payload underneath

Use `npx ai-knot-doctor --json` when filing an issue or comparing environments.

---

## First-run demo

If you want the shortest installed proof that the npm package and Python bridge
are both alive, run:

```bash
npx ai-knot-demo
```

It delegates to the built-in `ai-knot demo` command through the Python CLI that
the npm package already depends on. By default it uses temporary local storage;
add `--keep-data --data-dir .ai_knot-demo` if you want to inspect the on-disk
store afterward.

---

## HTTP sidecar client

If you already run `ai-knot serve`, you can skip the local MCP subprocess path
entirely and call the same `/v1/*` routes from TypeScript:

```typescript
import { HttpKnowledgeBase } from "ai-knot";

const kb = new HttpKnowledgeBase({
  baseUrl: "http://127.0.0.1:8000",
  token: process.env.AI_KNOT_SERVER_TOKEN,
});

const fact = await kb.add("User deploys APIs with Docker Compose");
console.log(await kb.search("what does the user deploy with?"));
console.log(await kb.list());
console.log(await kb.get(fact.id));
console.log(await kb.lineage(fact.id));
```

Start the sidecar first:

```bash
pip install "ai-knot[server]"
ai-knot --storage sqlite serve assistant --port 8000
```

Repo-native proof after the sidecar is live:

```bash
npm run example:http-sidecar
```

This is the lower-friction path when your app already talks to local or remote
HTTP services and you do not want a Node process to spawn Python itself.

---

## Quickstart

Fastest installed proof: `npx ai-knot-demo`. The code below is the raw
TypeScript API when you want to wire memory into an actual app.

```typescript
import { KnowledgeBase } from 'ai-knot';

const kb = new KnowledgeBase({
  agentId: 'my-agent',
  storage: 'sqlite',
  dbPath: '/absolute/path/to/memory.db',
});

// Add a fact
await kb.add('User prefers TypeScript');

// Search relevant facts for a query — alias: recall(), never calls an LLM
const context = await kb.search('what language does user prefer?');
console.log(context);
// -> "[1] User prefers TypeScript"

// Inspect or remove stored facts when you need to debug or correct memory
console.log(await kb.list());
// -> [{ id: "...", content: "User prefers TypeScript", ... }]
// console.log(await kb.get(factId));

// await kb.delete(factId);    // alias: await kb.forget(factId)

// Use context in your prompt
const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: [
    { role: 'system', content: `You are a helpful assistant.\n\n${context}` },
    { role: 'user', content: 'Write me a script' },
  ],
});

await kb.close();
```

## Basic memory loop

If you only remember one thing, make it this:

`add -> search -> list -> delete`

```typescript
const fact = await kb.add('User deploys APIs with Docker Compose');
console.log(await kb.search('what does the user deploy with?'));
console.log(await kb.list());
console.log(await kb.get(fact.id));
await kb.delete(fact.id);
```

From this repo, the shortest runnable TypeScript proof is:

```bash
npm run example:basic-memory-loop
```

Installed package proof through the same bridge:

```bash
npx ai-knot-demo
```

Most comparable memory quickstarts stop at `add` + `search`. The npm client
keeps `list` + `delete` visible too, because persistent memory needs an easy
inspection/correction loop, not only a write/read loop.

On the MCP-backed npm surface, `list()` / `listFacts()` now return the current
active memories by default. When you want audit history too, ask for it
explicitly:

```typescript
await kb.list({ includeInactive: true });
await kb.listFacts({ includeInactive: true, now: "2026-01-01T00:00:00+00:00" });
```

When the stored value itself changes, keep the same transport and use
`addResolved(...)` with an explicit `op`:

```typescript
await kb.addResolved([
  { content: "User works at Acme", entity: "user", attribute: "employer", valueText: "Acme" },
]);
await kb.addResolved([
  {
    content: "User now works at Globex",
    entity: "user",
    attribute: "employer",
    valueText: "Globex",
    op: "update",
  },
]);
await kb.addResolved([
  {
    content: "User no longer works at Globex",
    entity: "user",
    attribute: "employer",
    slotKey: "user::employer",
    op: "delete",
  },
]);
```

That keeps history queryable instead of overwriting the old fact in place.

When you want the audit trail for one corrected slot, use:

```typescript
await kb.lineage(factId);
await kb.lineage(factId, { now: "2026-01-02T00:00:00+00:00" });
```

If you prefer agent-memory words, the same npm client keeps:

- `recall()` as an alias for `search()`
- `listFacts()` as an alias for `list()`
- `forget()` as an alias for `delete()`

If you already have a `factId`, use `await kb.get(factId)` for targeted
inspection instead of scanning the whole list.

For the same loop mapped across Python, CLI, MCP, and HTTP too, use the repo
guide: [`docs/memory-commands.md`](https://github.com/alsoleg89/ai-knot/blob/main/docs/memory-commands.md).

---

## Vercel AI SDK

`AiKnotAISDKMemory` is the named adapter for AI SDK-style `system` / `messages`
flows. It keeps ai-knot dependency-light: ai-knot does the recall, and your AI
SDK code keeps control of the model call.

If you want the shortest repo-native proof before wiring Python or a model call:

```bash
npm run example:basic-memory-loop
npm run example:vercel-ai-sdk-surface
```

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { AiKnotAISDKMemory, KnowledgeBase } from 'ai-knot';

const dataDir = '/absolute/path/to/tmp-or-app-data';
const kb = new KnowledgeBase({
  agentId: 'assistant',
  dataDir,
});

await kb.add('User prefers TypeScript over JavaScript');
await kb.add('User deploys services with Docker Compose');

const memory = new AiKnotAISDKMemory(kb, { topK: 4 });
const userInput = 'Write a local deployment checklist for my stack.';

const system = await memory.buildSystem(userInput, {
  baseSystem: 'You are a concise staff engineer.',
});

const { text } = await generateText({
  model: openai('gpt-5'),
  system,
  prompt: userInput,
});
```

If you already have an AI SDK `messages` array, use `buildMessages()` instead:

```typescript
const messagesWithMemory = await memory.buildMessages([
  { role: 'system', content: 'You are a concise staff engineer.' },
  { role: 'user', content: 'Write a deployment checklist for my stack.' },
]);
```

Full repo-native example: [`npm/examples/vercel-ai-sdk.ts`](examples/vercel-ai-sdk.ts)

Run it from this repo with:

```bash
OPENAI_API_KEY=... npm run example:vercel-ai-sdk
```

This end-to-end path requires the Python-side `ai-knot-mcp` binary. A normal
`npm install` runs the postinstall hook that tries to install
`ai-knot[mcp]` automatically; if that step was skipped or failed, run:

```bash
pip install "ai-knot[mcp]"
npx ai-knot-doctor
```

If you want to inspect the same memory surface before any model call,
use the repo-native surface proof:

[`npm/examples/vercel-ai-sdk-surface.ts`](examples/vercel-ai-sdk-surface.ts)

If your app already talks to the HTTP sidecar instead of the MCP subprocess,
`AiKnotAISDKMemory` also works with `HttpKnowledgeBase` because it only needs a
`recall()` method.

---

## API

### `new KnowledgeBase(options?)`

```typescript
const kb = new KnowledgeBase({
  agentId?: string,    // default: "default"
  storage?: 'yaml' | 'sqlite',  // default: "yaml"
  dataDir?: string,   // base dir for YAML/SQLite (use absolute path!)
  dbPath?: string,    // full path to SQLite file
  command?: string,   // explicit path to ai-knot-mcp when PATH is not stable
});
```

> Use absolute paths for `dataDir` and `dbPath` — the subprocess may run from a different working directory.
> If the bridge still fails to spawn, run `npx ai-knot-doctor --json` and either
> fix `PATH` or set `command` explicitly.

### Methods

```typescript
await kb.add(content, options?)   // → Fact
await kb.learn(messages, options?) // → { stored, ids, note? }
await kb.search(query, options?)  // → string (alias: recall)
await kb.recall(query, options?)  // → string (formatted facts)
await kb.addResolved(facts)       // → ResolvedResult[]; facts may include op: "update" | "delete" | "noop"
await kb.get(factId)              // → Fact
await kb.lineage(factId, options?) // → LineageFact[]
await kb.list(options?)           // → Fact[] (alias: listFacts)
await kb.delete(factId)           // → void (alias: forget)
await kb.forget(factId)           // → void
await kb.listFacts(options?)      // → Fact[]
await kb.stats()                  // → Stats
await kb.snapshot(name)           // → void
await kb.restore(name)            // → void
await kb.close()                  // → void (shuts down subprocess)
```

If you prefer the familiar memory-tool loop, use `add -> search -> list -> delete`.
If you prefer agent-memory language, keep `recall -> listFacts -> forget`.
For audit/history views on the MCP bridge, pass `includeInactive: true` to
`list(...)` or call `lineage(factId, { now? })` for one slot's version chain.

### `new HttpKnowledgeBase(options)`

```typescript
const kb = new HttpKnowledgeBase({
  baseUrl: "http://127.0.0.1:8000",
  token?: process.env.AI_KNOT_SERVER_TOKEN,
  headers?: { "X-Request-ID": "..." },
});
```

Methods:

```typescript
await kb.add(content, options?)             // → Fact
await kb.learn(messages, options?)          // → { stored, ids, note? }
await kb.search(query, options?)            // → string (alias: recall)
await kb.recall(query, options?)            // → string
await kb.addResolved(facts)                 // → ResolvedResult[]; facts may include op: "update" | "delete" | "noop"
await kb.get(factId)                        // → Fact
await kb.lineage(factId, options?)          // → LineageFact[]
await kb.list(options?)                     // → Fact[] (alias: listFacts)
await kb.listFacts(options?)                // → Fact[]
await kb.delete(factId)                     // → void (alias: forget)
await kb.forget(factId)                     // → void
await kb.stats()                            // → Stats
await kb.health()                           // → { status, version }
await kb.close()                            // → void (no-op, for parity)
```

Use `HttpKnowledgeBase` when the sidecar is already running and you want the
same basic memory loop from Node without the local MCP subprocess. The sidecar
path now also keeps the richer write surfaces: `learn(...)` for extract-on-write
and `addResolved(...)` for structured supersession, including
`op: "update" | "delete" | "noop"` for explicit correction and closure, plus
`lineage(...)` for by-id audit trails over the same transport.

### `new AiKnotAISDKMemory(recallClient, options?)`

```typescript
const memory = new AiKnotAISDKMemory(kb, {
  topK?: number,
  now?: string,
  header?: string,  // default: "Relevant long-term memory (ai-knot):"
});
```

Methods:

```typescript
await memory.buildSystem(input, options?)    // → string | undefined
await memory.buildMessages(messages, options?) // → messages with a prepended system message
```

### `add` options

```typescript
{
  type?: 'semantic' | 'procedural' | 'episodic',  // default: "semantic"
  importance?: number,  // 0.0–1.0, default: 0.8
  tags?: string[],
}
```

### `learn` options

```typescript
{
  provider?: string,   // optional LLM provider override
  apiKey?: string,     // optional provider API key override
  model?: string,      // optional provider model override
  eventTime?: string,  // ISO-8601 anchor for the whole conversation
}
```

### `recall` options

```typescript
{
  topK?: number,  // max facts to return, default: 5
}
```

### `lineage` options

```typescript
{
  now?: string,  // optional ISO-8601 anchor for active/inactive status
}
```

---

## Memory types

| Type | Use for | Example |
|---|---|---|
| `semantic` | Facts about the user/world | "User works at Acme Corp" |
| `procedural` | How the user wants things done | "Always use TypeScript strict mode" |
| `episodic` | Specific past events | "Deploy failed last Tuesday" |

---

## Concurrent calls

All method calls are safe to run concurrently:

```typescript
const [facts, stats] = await Promise.all([
  kb.listFacts(),
  kb.stats(),
]);
```

---

## Snapshots

```typescript
await kb.snapshot('before-refactor');
await kb.restore('before-refactor');
```

Requires SQLite or YAML storage backend.

---

## Lifecycle

Always call `kb.close()` when done:

```typescript
const kb = new KnowledgeBase({ agentId: 'bot' });
try {
  await kb.add('...');
  const ctx = await kb.search('...');
} finally {
  await kb.close();
}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `AI_KNOT_AGENT_ID` | `default` | Agent namespace |
| `AI_KNOT_STORAGE` | `yaml` | `yaml` or `sqlite` |
| `AI_KNOT_DATA_DIR` | `.ai_knot` | Base dir (use absolute path) |
| `AI_KNOT_DB_PATH` | — | Full path to SQLite file |

---

## License

MIT
