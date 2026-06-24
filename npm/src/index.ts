import { McpClient } from "./client.js";
import type {
  AddOptions,
  Fact,
  KnowledgeBaseOptions,
  LearnMessage,
  LearnResult,
  RecallOptions,
  ResolvedFact,
  ResolvedResult,
  Stats,
} from "./types.js";

export type {
  AddOptions,
  Fact,
  KnowledgeBaseOptions,
  LearnMessage,
  LearnResult,
  MemoryType,
  RecallOptions,
  ResolvedFact,
  ResolvedResult,
  Stats,
} from "./types.js";

/**
 * TypeScript client for ai_knot. Spawns the ai-knot-mcp subprocess on
 * the first method call and communicates via JSON-RPC 2.0 over stdio.
 *
 * @example
 * ```typescript
 * import { KnowledgeBase } from 'ai-knot';
 *
 * const kb = new KnowledgeBase({ agentId: 'my-agent', storage: 'sqlite', dbPath: '/data/mem.db' });
 * await kb.add('User prefers TypeScript');
 * const ctx = await kb.recall('what language?');
 * console.log(ctx);
 * await kb.close();
 * ```
 */
export class KnowledgeBase {
  private readonly client: McpClient;

  constructor(options: KnowledgeBaseOptions = {}) {
    const env: Record<string, string> = { ...options.env };
    if (options.agentId !== undefined) env["AI_KNOT_AGENT_ID"] = options.agentId;
    if (options.storage !== undefined) env["AI_KNOT_STORAGE"] = options.storage;
    if (options.dataDir !== undefined) env["AI_KNOT_DATA_DIR"] = options.dataDir;
    if (options.dbPath !== undefined) env["AI_KNOT_DB_PATH"] = options.dbPath;
    this.client = new McpClient({ command: options.command, env });
  }

  /**
   * Add a fact to the knowledge base.
   * Returns the created Fact with its assigned ID.
   */
  async add(content: string, options: AddOptions = {}): Promise<Fact> {
    const args: Record<string, unknown> = { content };
    if (options.type !== undefined) args["type"] = options.type;
    if (options.importance !== undefined) args["importance"] = options.importance;
    if (options.tags !== undefined) args["tags"] = options.tags;
    if (options.eventTime !== undefined) args["event_time"] = options.eventTime;
    const text = await this.client.call("add", args);
    // Response: "Added fact [<id>]: <content>"
    const match = /\[([a-f0-9]{8})\]/.exec(text);
    const id = match?.[1] ?? "";
    const facts = await this.listFacts();
    const fact = facts.find((f) => f.id === id);
    if (fact !== undefined) return fact;
    // Fallback: construct a minimal Fact from the response text
    return {
      id,
      content,
      type: (options.type ?? "semantic") as Fact["type"],
      importance: options.importance ?? 0.8,
      retention_score: 1.0,
      access_count: 0,
      tags: options.tags ?? [],
      created_at: new Date().toISOString(),
      last_accessed: new Date().toISOString(),
    };
  }

  /**
   * Recall relevant facts for a query. Returns a formatted string ready to
   * inject into a prompt, or "No relevant facts found." if the KB is empty.
   */
  async recall(query: string, options: RecallOptions = {}): Promise<string> {
    const args: Record<string, unknown> = { query };
    if (options.topK !== undefined) args["top_k"] = options.topK;
    if (options.now !== undefined) args["now"] = options.now;
    return this.client.call("recall", args);
  }

  /**
   * Extract and store facts from a conversation (the LLM-backed ingest path).
   * Mirrors the `learn` MCP tool; requires the server to have an LLM provider
   * configured, otherwise it falls back to storing the last user message.
   * Routing ingest through `learn` (rather than `add`) lets slot extraction
   * fire knowledge-update supersession on the server.
   */
  async learn(messages: LearnMessage[]): Promise<LearnResult> {
    const text = await this.client.call("learn", { messages });
    return JSON.parse(text) as LearnResult;
  }

  /**
   * Insert pre-structured facts through the supersession pipeline with no LLM
   * call. A fact addressing an existing active slot with a different value
   * supersedes it (knowledge-update). Mirrors the `add_resolved` MCP tool.
   */
  async addResolved(facts: ResolvedFact[]): Promise<ResolvedResult[]> {
    const wire = facts.map((f) => {
      const o: Record<string, unknown> = { content: f.content };
      if (f.entity !== undefined) o["entity"] = f.entity;
      if (f.attribute !== undefined) o["attribute"] = f.attribute;
      if (f.valueText !== undefined) o["value_text"] = f.valueText;
      if (f.slotKey !== undefined) o["slot_key"] = f.slotKey;
      if (f.eventTime !== undefined) o["event_time"] = f.eventTime;
      return o;
    });
    const text = await this.client.call("add_resolved", { facts: wire });
    return JSON.parse(text) as ResolvedResult[];
  }

  /**
   * Remove a fact by its 8-character hex ID.
   */
  async forget(factId: string): Promise<void> {
    await this.client.call("forget", { fact_id: factId });
  }

  /**
   * List all stored facts as structured objects.
   */
  async listFacts(): Promise<Fact[]> {
    const text = await this.client.call("list_facts", {});
    if (text === "No facts stored.") return [];
    return JSON.parse(text) as Fact[];
  }

  /**
   * Return statistics about the knowledge base.
   */
  async stats(): Promise<Stats> {
    const text = await this.client.call("stats", {});
    return JSON.parse(text) as Stats;
  }

  /**
   * Save the current knowledge base state as a named snapshot.
   * Throws if the storage backend does not support snapshots.
   */
  async snapshot(name: string): Promise<void> {
    const text = await this.client.call("snapshot", { name });
    if (text.toLowerCase().includes("not supported")) {
      throw new Error(text);
    }
  }

  /**
   * Restore the knowledge base from a named snapshot.
   * Throws if the snapshot does not exist or backend does not support snapshots.
   */
  async restore(name: string): Promise<void> {
    const text = await this.client.call("restore", { name });
    if (text.toLowerCase().includes("not supported") || text.toLowerCase().includes("not found")) {
      throw new Error(text);
    }
  }

  /**
   * Diagnostic variant of recall — returns context string plus per-stage trace.
   * Calls the `recall_with_trace` MCP tool. For benchmark diagnostics only.
   */
  async recallWithTrace(
    query: string,
    options: RecallOptions = {},
  ): Promise<{ context: string; packFactIds: string[]; trace: Record<string, unknown> }> {
    const args: Record<string, unknown> = { query };
    if (options.topK !== undefined) args["top_k"] = options.topK;
    if (options.now !== undefined) args["now"] = options.now;
    const text = await this.client.call("recall_with_trace", args);
    const parsed = JSON.parse(text) as {
      context: string;
      pack_fact_ids: string[];
      trace: Record<string, unknown>;
    };
    return {
      context: parsed.context,
      packFactIds: parsed.pack_fact_ids,
      trace: parsed.trace,
    };
  }

  /**
   * Gracefully shut down the ai-knot-mcp subprocess.
   * Call this when you are done using the KnowledgeBase.
   */
  async close(): Promise<void> {
    await this.client.close();
  }
}
