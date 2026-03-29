/**
 * Unit tests for McpClient.
 * The subprocess is never actually spawned — node:child_process is mocked.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import type { ChildProcess } from "node:child_process";
import type { JsonRpcRequest } from "../types.js";

// vi.mock is hoisted automatically — must appear before imports that use spawn
vi.mock("node:child_process", () => ({ spawn: vi.fn() }));

import { spawn } from "node:child_process";
import { McpClient } from "../client.js";

// ---------------------------------------------------------------------------
// Fake subprocess
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
          if (resp !== null) {
            this.stdout.write(JSON.stringify(resp) + "\n");
          }
        } catch {
          // ignore non-JSON lines
        }
      }
    });

    // When stdin's writable side is ended (proc.stdin.end() in McpClient.close()), emit close.
    // 'finish' fires on the writable side; 'end' fires on the readable side.
    this.stdin.on("finish", () => setImmediate(() => this.emit("close", 0)));
  }

  kill(_signal?: string): void {
    setImmediate(() => this.emit("close", null));
  }
}

// Default MCP handler: responds to initialize and tools/call
function defaultHandler(req: JsonRpcRequest): object | null {
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
    const p = req.params as { name: string };
    return {
      jsonrpc: "2.0",
      id: req.id,
      result: { content: [{ type: "text", text: `ok:${p.name}` }] },
    };
  }
  return null;
}

function setupSpawn(handler: Handler = defaultHandler): FakeProcess {
  const proc = new FakeProcess(handler);
  vi.mocked(spawn).mockReturnValue(proc as unknown as ChildProcess);
  return proc;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("McpClient", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("spawns the subprocess on the first call", async () => {
    setupSpawn();
    const client = new McpClient();
    await client.call("add", { content: "hello" });
    expect(spawn).toHaveBeenCalledOnce();
    await client.close();
  });

  it("does not spawn again on subsequent calls", async () => {
    setupSpawn();
    const client = new McpClient();
    await client.call("add", { content: "a" });
    await client.call("recall", { query: "a" });
    expect(spawn).toHaveBeenCalledOnce();
    await client.close();
  });

  it("deduplicates concurrent connect() calls", async () => {
    setupSpawn();
    const client = new McpClient();
    const [r1, r2] = await Promise.all([
      client.call("add", { content: "x" }),
      client.call("add", { content: "y" }),
    ]);
    expect(spawn).toHaveBeenCalledOnce();
    expect(r1).toBe("ok:add");
    expect(r2).toBe("ok:add");
    await client.close();
  });

  it("returns the text content from a tool call response", async () => {
    setupSpawn();
    const client = new McpClient();
    const result = await client.call("recall", { query: "python" });
    expect(result).toBe("ok:recall");
    await client.close();
  });

  it("passes custom command and env to spawn", async () => {
    setupSpawn();
    const client = new McpClient({ command: "my-mcp", env: { FOO: "bar" } });
    await client.connect();
    expect(spawn).toHaveBeenCalledWith(
      "my-mcp",
      [],
      expect.objectContaining({ env: expect.objectContaining({ FOO: "bar" }) })
    );
    await client.close();
  });

  it("rejects inflight calls when the process crashes", async () => {
    // Handler that never responds to tools/call — keeps the call inflight
    const proc = setupSpawn((req) => {
      if (req.method === "initialize") return defaultHandler(req);
      if (req.method === "notifications/initialized") return null;
      return null; // tools/call never answered
    });

    const client = new McpClient();
    client.on("error", () => {}); // prevent unhandled EventEmitter error
    await client.connect();

    const callPromise = client.call("add", { content: "hanging" });
    // Yield to let the call register in inflight map
    await new Promise<void>((r) => setImmediate(r));

    proc.emit("close", 1);

    await expect(callPromise).rejects.toThrow(/exited unexpectedly/);
  });

  it("rejects connect() on spawn error", async () => {
    let capturedProc: FakeProcess | null = null;
    vi.mocked(spawn).mockImplementationOnce(() => {
      const proc = new FakeProcess(() => null); // never responds to initialize
      capturedProc = proc;
      return proc as unknown as ChildProcess;
    });

    const client = new McpClient();
    client.on("error", () => {});

    const connectPromise = client.connect();
    // Give _doConnect a tick to register the 'error' handler on proc
    await new Promise<void>((r) => setImmediate(r));
    capturedProc!.emit("error", new Error("ENOENT: agentmemo-mcp not found"));

    await expect(connectPromise).rejects.toThrow(/Failed to spawn/);
  });

  it("close() causes subsequent spawn on next call", async () => {
    setupSpawn();
    const client = new McpClient();
    await client.call("add", { content: "before close" });
    await client.close();

    setupSpawn(); // second proc for after close
    await client.call("recall", { query: "after close" });
    expect(spawn).toHaveBeenCalledTimes(2);
    await client.close();
  });
});
