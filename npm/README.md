# ai-knot

![npm](https://img.shields.io/npm/v/ai-knot)
![License](https://img.shields.io/badge/license-MIT-green)

**Agent memory for Node.js and TypeScript.** Stores facts, retrieves what's relevant, forgets the rest.

TypeScript client for the [ai-knot](https://github.com/alsoleg89/ai-knot) Python library — communicates with the `ai-knot-mcp` subprocess via JSON-RPC 2.0 over stdio.

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
```

---

## Quickstart

```typescript
import { KnowledgeBase } from 'ai-knot';

const kb = new KnowledgeBase({
  agentId: 'my-agent',
  storage: 'sqlite',
  dbPath: '/absolute/path/to/memory.db',
});

// Add a fact
await kb.add('User prefers TypeScript');

// Recall relevant facts for a query
const context = await kb.recall('what language does user prefer?');
console.log(context);
// -> "[semantic] User prefers TypeScript"

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

---

## API

### `new KnowledgeBase(options?)`

```typescript
const kb = new KnowledgeBase({
  agentId?: string,    // default: "default"
  storage?: 'yaml' | 'sqlite',  // default: "yaml"
  dataDir?: string,   // base dir for YAML/SQLite (use absolute path!)
  dbPath?: string,    // full path to SQLite file
  command?: string,   // path to ai-knot-mcp binary
});
```

> Use absolute paths for `dataDir` and `dbPath` — the subprocess may run from a different working directory.

### Methods

```typescript
await kb.add(content, options?)   // → Fact
await kb.recall(query, options?)  // → string (formatted facts)
await kb.forget(factId)           // → void
await kb.listFacts()              // → Fact[]
await kb.stats()                  // → Stats
await kb.snapshot(name)           // → void
await kb.restore(name)            // → void
await kb.close()                  // → void (shuts down subprocess)
```

### `add` options

```typescript
{
  type?: 'semantic' | 'procedural' | 'episodic',  // default: "semantic"
  importance?: number,  // 0.0–1.0, default: 0.8
  tags?: string[],
}
```

### `recall` options

```typescript
{
  topK?: number,  // max facts to return, default: 5
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
  const ctx = await kb.recall('...');
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
