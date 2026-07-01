/**
 * Unit tests for KnowledgeBase public API.
 * The ai-knot-mcp subprocess is mocked via node:child_process.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { ChildProcess } from "node:child_process";
import type { JsonRpcRequest } from "../types.js";

vi.mock("node:child_process", () => ({ spawn: vi.fn() }));

import { spawn } from "node:child_process";
import { KnowledgeBase } from "../index.js";
import type { Fact } from "../index.js";

// ---------------------------------------------------------------------------
// Shared fake subprocess (same pattern as client.test.ts)
// ---------------------------------------------------------------------------

type Handler = (req: JsonRpcRequest) => object | null;

class FakeProcess extends EventEmitter {
  readonly stdin: PassThrough;
  readonly stdout: PassThrough;

  constructor(handler: Handler) {
    super();
    this.stdin = new PassThrough();
    this.stdout = new PassThrough();
    let buf = "";
    this.stdin.on("data", (chunk: Buffer) => {
      buf += chunk.toString();
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const req = JSON.parse(trimmed) as JsonRpcRequest;
          const resp = handler(req);
          if (resp !== null) this.stdout.write(JSON.stringify(resp) + "\n");
        } catch {}
      }
    });
    this.stdin.on("end", () => setImmediate(() => this.emit("close", 0)));
  }

  kill(): void {
    setImmediate(() => this.emit("close", null));
  }
}

const FAKE_FACT: Fact = {
  id: "abcd1234",
  content: "TypeScript is great",
  type: "semantic",
  importance: 0.8,
  retention_score: 1.0,
  access_count: 0,
  tags: [],
  created_at: "2026-01-01T00:00:00.000Z",
  last_accessed: "2026-01-01T00:00:00.000Z",
};

// Captures the arguments of the most recent tool calls for assertions.
let lastAddArguments: Record<string, unknown> | undefined;
let lastRecallArguments: Record<string, unknown> | undefined;
let lastLearnArguments: Record<string, unknown> | undefined;
let lastAddResolvedArguments: Record<string, unknown> | undefined;

function kbHandler(req: JsonRpcRequest): object | null {
  if (req.method === "initialize") {
    return {
      jsonrpc: "2.0",
      id: req.id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        serverInfo: { name: "fake-mcp", version: "0.0.0" },
      },
    };
  }
  if (req.method === "notifications/initialized") return null;

  if (req.method === "tools/call") {
    const { name, arguments: callArgs } = req.params as {
      name: string;
      arguments: Record<string, unknown>;
    };
    let text = "";

    if (name === "add") {
      lastAddArguments = callArgs;
      text = "Added fact [abcd1234]: TypeScript is great";
    } else if (name === "list_facts") {
      text = JSON.stringify([FAKE_FACT]);
    } else if (name === "recall") {
      lastRecallArguments = callArgs;
      text = "1. [abcd1234] TypeScript is great (relevance: 0.95)";
    } else if (name === "learn") {
      lastLearnArguments = callArgs;
      text = JSON.stringify({ stored: 1, ids: ["abcd1234"] });
    } else if (name === "add_resolved") {
      lastAddResolvedArguments = callArgs;
      text = JSON.stringify([{ id: "abcd1234", slot_key: "alex::salary", version: 1 }]);
    } else if (name === "forget") {
      text = "Fact abcd1234 deleted.";
    } else if (name === "stats") {
      text = JSON.stringify({
        total_facts: 1,
        by_type: { semantic: 1 },
        avg_importance: 0.8,
        avg_retention: 1.0,
      });
    } else if (name === "snapshot") {
      text = "Snapshot 'test-snap' created.";
    } else if (name === "restore") {
      text = "Snapshot 'test-snap' restored.";
    }

    return {
      jsonrpc: "2.0",
      id: req.id,
      result: { content: [{ type: "text", text }] },
    };
  }
  return null;
}

function setup(): void {
  vi.mocked(spawn).mockReturnValue(
    new FakeProcess(kbHandler) as unknown as ChildProcess
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("KnowledgeBase", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("add() returns a Fact with correct shape", async () => {
    setup();
    const kb = new KnowledgeBase();
    const fact = await kb.add("TypeScript is great");
    expect(fact).toMatchObject({
      id: "abcd1234",
      content: "TypeScript is great",
      type: "semantic",
    });
    await kb.close();
  });

  it("add() passes type and importance options", async () => {
    setup();
    const kb = new KnowledgeBase();
    const fact = await kb.add("Do X before Y", { type: "procedural", importance: 0.9 });
    expect(fact.id).toBe("abcd1234");
    await kb.close();
  });

  it("add() forwards eventTime as the event_time MCP argument", async () => {
    setup();
    lastAddArguments = undefined;
    const kb = new KnowledgeBase();
    await kb.add("I joined Acme yesterday", { eventTime: "2023-05-08" });
    expect(lastAddArguments).toMatchObject({
      content: "I joined Acme yesterday",
      event_time: "2023-05-08",
    });
    await kb.close();
  });

  it("add() omits event_time when eventTime is not provided", async () => {
    setup();
    lastAddArguments = undefined;
    const kb = new KnowledgeBase();
    await kb.add("no timestamp here");
    expect(lastAddArguments).toBeDefined();
    expect(lastAddArguments).not.toHaveProperty("event_time");
    await kb.close();
  });

  it("recall() returns a non-empty string", async () => {
    setup();
    const kb = new KnowledgeBase();
    const result = await kb.recall("TypeScript");
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
    await kb.close();
  });

  it("search() matches recall()", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.search("TypeScript")).resolves.toEqual(await kb.recall("TypeScript"));
    await kb.close();
  });

  it("add() forwards tags as the tags MCP argument", async () => {
    setup();
    lastAddArguments = undefined;
    const kb = new KnowledgeBase();
    await kb.add("uses Docker", { tags: ["devops", "tools"] });
    expect(lastAddArguments).toMatchObject({ content: "uses Docker", tags: ["devops", "tools"] });
    await kb.close();
  });

  it("recall() forwards now and topK as MCP arguments", async () => {
    setup();
    lastRecallArguments = undefined;
    const kb = new KnowledgeBase();
    await kb.recall("anything", { now: "2023-05-08T00:00:00+00:00", topK: 7 });
    expect(lastRecallArguments).toMatchObject({
      query: "anything",
      now: "2023-05-08T00:00:00+00:00",
      top_k: 7,
    });
    await kb.close();
  });

  it("recall() omits now when not provided", async () => {
    setup();
    lastRecallArguments = undefined;
    const kb = new KnowledgeBase();
    await kb.recall("anything");
    expect(lastRecallArguments).toBeDefined();
    expect(lastRecallArguments).not.toHaveProperty("now");
    await kb.close();
  });

  it("learn() sends the conversation and parses the result", async () => {
    setup();
    lastLearnArguments = undefined;
    const kb = new KnowledgeBase();
    const res = await kb.learn([
      { role: "user", content: "I prefer dark mode" },
      { role: "assistant", content: "Noted" },
    ]);
    expect(lastLearnArguments).toMatchObject({
      messages: [
        { role: "user", content: "I prefer dark mode" },
        { role: "assistant", content: "Noted" },
      ],
    });
    expect(res).toEqual({ stored: 1, ids: ["abcd1234"] });
    await kb.close();
  });

  it("addResolved() snake-cases fields and parses results", async () => {
    setup();
    lastAddResolvedArguments = undefined;
    const kb = new KnowledgeBase();
    const res = await kb.addResolved([
      {
        content: "Alex earns 120k",
        entity: "Alex",
        attribute: "salary",
        valueText: "120k",
        slotKey: "alex::salary",
        eventTime: "2023-05-08",
      },
    ]);
    expect(lastAddResolvedArguments).toMatchObject({
      facts: [
        {
          content: "Alex earns 120k",
          entity: "Alex",
          attribute: "salary",
          value_text: "120k",
          slot_key: "alex::salary",
          event_time: "2023-05-08",
        },
      ],
    });
    expect(res).toEqual([{ id: "abcd1234", slot_key: "alex::salary", version: 1 }]);
    await kb.close();
  });

  it("listFacts() returns an array of Facts", async () => {
    setup();
    const kb = new KnowledgeBase();
    const facts = await kb.listFacts();
    expect(Array.isArray(facts)).toBe(true);
    expect(facts).toHaveLength(1);
    expect(facts[0]).toMatchObject({ id: "abcd1234" });
    await kb.close();
  });

  it("listFacts() returns empty array when server says no facts", async () => {
    vi.mocked(spawn).mockReturnValue(
      new FakeProcess((req) => {
        if (req.method === "initialize") {
          return {
            jsonrpc: "2.0",
            id: req.id,
            result: { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: {} },
          };
        }
        if (req.method === "notifications/initialized") return null;
        return {
          jsonrpc: "2.0",
          id: req.id,
          result: { content: [{ type: "text", text: "No facts stored." }] },
        };
      }) as unknown as ChildProcess
    );
    const kb = new KnowledgeBase();
    const facts = await kb.listFacts();
    expect(facts).toEqual([]);
    await kb.close();
  });

  it("list() matches listFacts()", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.list()).resolves.toEqual(await kb.listFacts());
    await kb.close();
  });

  it("stats() returns a Stats object", async () => {
    setup();
    const kb = new KnowledgeBase();
    const stats = await kb.stats();
    expect(stats).toMatchObject({ total_facts: 1, by_type: { semantic: 1 } });
    await kb.close();
  });

  it("forget() resolves without error", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.forget("abcd1234")).resolves.toBeUndefined();
    await kb.close();
  });

  it("delete() resolves without error", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.delete("abcd1234")).resolves.toBeUndefined();
    await kb.close();
  });

  it("snapshot() resolves without error", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.snapshot("test-snap")).resolves.toBeUndefined();
    await kb.close();
  });

  it("restore() resolves without error", async () => {
    setup();
    const kb = new KnowledgeBase();
    await expect(kb.restore("test-snap")).resolves.toBeUndefined();
    await kb.close();
  });

  it("snapshot() throws when backend does not support snapshots", async () => {
    vi.mocked(spawn).mockReturnValue(
      new FakeProcess((req) => {
        if (req.method === "initialize") {
          return {
            jsonrpc: "2.0",
            id: req.id,
            result: { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: {} },
          };
        }
        if (req.method === "notifications/initialized") return null;
        return {
          jsonrpc: "2.0",
          id: req.id,
          result: { content: [{ type: "text", text: "Snapshots not supported for yaml storage" }] },
        };
      }) as unknown as ChildProcess
    );
    const kb = new KnowledgeBase();
    await expect(kb.snapshot("s")).rejects.toThrow(/not supported/i);
    await kb.close();
  });

  it("constructor passes agentId and storage as env vars to McpClient", async () => {
    setup();
    const kb = new KnowledgeBase({ agentId: "agent-1", storage: "sqlite" });
    await kb.recall("test");
    expect(spawn).toHaveBeenCalledWith(
      "ai-knot-mcp",
      [],
      expect.objectContaining({
        env: expect.objectContaining({
          AI_KNOT_AGENT_ID: "agent-1",
          AI_KNOT_STORAGE: "sqlite",
        }),
      })
    );
    await kb.close();
  });
});
