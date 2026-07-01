import { beforeEach, describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { ChildProcess } from "node:child_process";
import type { JsonRpcRequest } from "../types.js";

vi.mock("node:child_process", () => ({ spawn: vi.fn() }));

import { spawn } from "node:child_process";
import { AiKnotAISDKMemory, KnowledgeBase } from "../index.js";

type Handler = (req: JsonRpcRequest) => object | null;

class FakeProcess extends EventEmitter {
  readonly stdin: PassThrough;
  readonly stdout: PassThrough;

  constructor(handler: Handler) {
    super();
    this.stdin = new PassThrough();
    this.stdout = new PassThrough();

    let buffer = "";
    this.stdin.on("data", (chunk: Buffer) => {
      buffer += chunk.toString();
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const request = JSON.parse(trimmed) as JsonRpcRequest;
        const response = handler(request);
        if (response !== null) {
          this.stdout.write(JSON.stringify(response) + "\n");
        }
      }
    });

    this.stdin.on("end", () => setImmediate(() => this.emit("close", 0)));
  }

  kill(): void {
    setImmediate(() => this.emit("close", null));
  }
}

let recallText = "1. [abcd1234] User prefers TypeScript";
let lastRecallArguments: Record<string, unknown> | undefined;

function mcpHandler(req: JsonRpcRequest): object | null {
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
    if (name === "recall") {
      lastRecallArguments = callArgs;
      return {
        jsonrpc: "2.0",
        id: req.id,
        result: { content: [{ type: "text", text: recallText }] },
      };
    }
  }

  return {
    jsonrpc: "2.0",
    id: req.id,
    result: { content: [{ type: "text", text: "" }] },
  };
}

function setup(): void {
  vi.mocked(spawn).mockReturnValue(
    new FakeProcess(mcpHandler) as unknown as ChildProcess,
  );
}

describe("AiKnotAISDKMemory", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    recallText = "1. [abcd1234] User prefers TypeScript";
    lastRecallArguments = undefined;
  });

  it("buildSystem() merges base instructions with recalled context", async () => {
    setup();
    const kb = new KnowledgeBase();
    const memory = new AiKnotAISDKMemory(kb, { header: "Memory:" });

    const system = await memory.buildSystem("what language should I use?", {
      baseSystem: "You are a concise engineer.",
      topK: 4,
      now: "2026-07-01T00:00:00+00:00",
    });

    expect(lastRecallArguments).toMatchObject({
      query: "what language should I use?",
      top_k: 4,
      now: "2026-07-01T00:00:00+00:00",
    });
    expect(system).toContain("You are a concise engineer.");
    expect(system).toContain("Memory:");
    expect(system).toContain("User prefers TypeScript");
    await kb.close();
  });

  it("buildSystem() falls back to the base system when no facts are recalled", async () => {
    setup();
    recallText = "No relevant facts found.";
    const kb = new KnowledgeBase();
    const memory = new AiKnotAISDKMemory(kb);

    const system = await memory.buildSystem("what stack do I use?", {
      baseSystem: "Reply in one paragraph.",
    });

    expect(system).toBe("Reply in one paragraph.");
    await kb.close();
  });

  it("buildMessages() replaces a leading system message and uses the latest user turn as query", async () => {
    setup();
    const kb = new KnowledgeBase();
    const memory = new AiKnotAISDKMemory(kb);

    const messages = await memory.buildMessages([
      { role: "system", content: "You are a helpful coding assistant." },
      { role: "assistant", content: "How can I help?" },
      { role: "user", content: "What language do I prefer?" },
    ]);

    expect(lastRecallArguments).toMatchObject({
      query: "What language do I prefer?",
    });
    expect(messages).toHaveLength(3);
    expect(messages[0]).toMatchObject({
      role: "system",
    });
    expect((messages[0] as { content: string }).content).toContain(
      "You are a helpful coding assistant.",
    );
    expect((messages[0] as { content: string }).content).toContain(
      "Relevant long-term memory (ai-knot):",
    );
    await kb.close();
  });

  it("buildMessages() can extract the query from structured content parts", async () => {
    setup();
    const kb = new KnowledgeBase();
    const memory = new AiKnotAISDKMemory(kb);

    await memory.buildMessages([
      {
        role: "user",
        content: [{ type: "text", text: "Summarize my current stack." }],
      },
    ]);

    expect(lastRecallArguments).toMatchObject({
      query: "Summarize my current stack.",
    });
    await kb.close();
  });
});
