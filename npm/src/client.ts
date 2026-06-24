import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";
import type { JsonRpcRequest, JsonRpcResponse } from "./types.js";

export interface McpClientOptions {
  command?: string;
  env?: Record<string, string>;
}

type ConnectionState = "idle" | "initializing" | "ready" | "crashed";

type Resolver = {
  resolve: (text: string) => void;
  reject: (err: Error) => void;
};

export class McpClient extends EventEmitter {
  private readonly command: string;
  private readonly extraEnv: Record<string, string>;

  private proc: ChildProcess | null = null;
  private state: ConnectionState = "idle";
  private nextId = 1;
  private readonly inflight = new Map<number, Resolver>();
  private readonly pendingQueue: Array<() => void> = [];
  private connectPromise: Promise<void> | null = null;

  constructor(options: McpClientOptions = {}) {
    super();
    this.command = options.command ?? "ai-knot-mcp";
    this.extraEnv = options.env ?? {};
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  async connect(): Promise<void> {
    if (this.state === "ready") return;
    // Deduplicate: if connect is already running, share the same promise
    if (this.connectPromise !== null) return this.connectPromise;
    this.connectPromise = this._doConnect().finally(() => {
      this.connectPromise = null;
    });
    return this.connectPromise;
  }

  async call(toolName: string, args: Record<string, unknown>): Promise<string> {
    if (this.state === "idle" || this.state === "crashed") {
      await this.connect();
    }

    // If another connect() started in the meantime (race), wait for it
    if (this.state === "initializing") {
      return new Promise<string>((resolve, reject) => {
        this.pendingQueue.push(() => {
          this._sendToolCall(toolName, args).then(resolve, reject);
        });
      });
    }

    return this._sendToolCall(toolName, args);
  }

  async close(): Promise<void> {
    if (this.proc === null) return;
    const proc = this.proc;
    this.proc = null;
    this.state = "idle";

    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        proc.kill("SIGTERM");
        resolve();
      }, 3000);
      proc.once("close", () => {
        clearTimeout(timer);
        resolve();
      });
      proc.stdin?.end();
    });
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private async _doConnect(): Promise<void> {
    this.state = "initializing";

    this.proc = spawn(this.command, [], {
      stdio: ["pipe", "pipe", "inherit"],
      env: { ...process.env, ...this.extraEnv },
    });

    const rl = createInterface({
      input: this.proc.stdout!,
      crlfDelay: Infinity,
    });

    rl.on("line", (line: string) => this._handleLine(line));

    this.proc.on("close", (code) => this._handleClose(code));
    this.proc.on("error", (err) => this._handleSpawnError(err));

    // MCP initialize handshake
    const initId = this.nextId++;
    await new Promise<void>((resolve, reject) => {
      this.inflight.set(initId, {
        resolve: () => resolve(),
        reject,
      });
      this._writeLine({
        jsonrpc: "2.0",
        id: initId,
        method: "initialize",
        params: {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "ai-knot-js", version: "0.2.0" },
        },
      });
    });

    // Notify server that client is ready (no response expected)
    this._writeLine({
      jsonrpc: "2.0",
      method: "notifications/initialized",
      params: {},
    });

    this.state = "ready";
    this.emit("ready");

    // Flush calls that were queued during initialization
    const queued = this.pendingQueue.splice(0);
    for (const fn of queued) fn();
  }

  private _sendToolCall(
    toolName: string,
    args: Record<string, unknown>
  ): Promise<string> {
    const id = this.nextId++;
    return new Promise<string>((resolve, reject) => {
      this.inflight.set(id, { resolve, reject });
      this._writeLine({
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: { name: toolName, arguments: args },
      });
    });
  }

  private _writeLine(obj: JsonRpcRequest): void {
    if (!this.proc?.stdin?.writable) {
      throw new Error("ai-knot-mcp stdin is not writable");
    }
    this.proc.stdin.write(JSON.stringify(obj) + "\n");
  }

  private _handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return;

    let msg: JsonRpcResponse;
    try {
      msg = JSON.parse(trimmed) as JsonRpcResponse;
    } catch {
      return; // ignore non-JSON lines (e.g. startup logs)
    }

    const pending = this.inflight.get(msg.id);
    if (pending === undefined) return;
    this.inflight.delete(msg.id);

    if (msg.error !== undefined) {
      pending.reject(
        new Error(`ai-knot-mcp error ${msg.error.code}: ${msg.error.message}`)
      );
      return;
    }

    const text =
      msg.result?.content
        ?.filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("") ?? "";

    pending.resolve(text);
  }

  private _handleClose(code: number | null): void {
    if (this.state === "idle") return; // intentional close, already cleaned up
    this.state = "crashed";

    const err = new Error(
      `ai-knot-mcp process exited unexpectedly (code ${code ?? "null"})`
    );

    for (const { reject } of this.inflight.values()) reject(err);
    this.inflight.clear();

    // Reject queued pending calls
    const queued = this.pendingQueue.splice(0);
    for (const fn of queued) {
      // Wrap in a try so that individual failures don't block the rest
      try {
        fn();
      } catch {
        // swallow — the call will reject via _sendToolCall's inflight path
      }
    }

    this.emit("error", err);
  }

  private _handleSpawnError(err: Error): void {
    this.state = "crashed";
    const wrapped = new Error(
      `Failed to spawn ai-knot-mcp: ${err.message}. ` +
        `Make sure it is installed: pip install "ai-knot[mcp]"`
    );
    for (const { reject } of this.inflight.values()) reject(wrapped);
    this.inflight.clear();
    this.emit("error", wrapped);
  }
}
