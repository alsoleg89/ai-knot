import { McpClient } from "./client.js";
import { HttpKnowledgeBase } from "./http.js";
import {
  AiKnotAISDKMemory,
  type AISDKMessageLike,
  type AiKnotAISDKBuildMessagesOptions,
  type AiKnotAISDKBuildSystemOptions,
  type AiKnotAISDKMemoryOptions,
} from "./ai_sdk.js";
import type {
  AddOptions,
  Fact,
  HttpKnowledgeBaseListOptions,
  HttpKnowledgeBaseOptions,
  KnowledgeBaseListOptions,
  KnowledgeBaseOptions,
  LearnMessage,
  LearnOptions,
  LearnResult,
  LineageFact,
  LineageOptions,
  MemoryOp,
  RecallOptions,
  ResolvedFact,
  ResolvedResult,
  Stats,
} from "./types.js";

export {
  type AISDKMessageLike,
  type AiKnotAISDKBuildMessagesOptions,
  type AiKnotAISDKBuildSystemOptions,
  type AiKnotAISDKMemoryOptions,
} from "./ai_sdk.js";

export { AiKnotAISDKMemory } from "./ai_sdk.js";
export { HttpKnowledgeBase } from "./http.js";

export type {
  AddOptions,
  Fact,
  HttpKnowledgeBaseListOptions,
  HttpKnowledgeBaseOptions,
  KnowledgeBaseListOptions,
  KnowledgeBaseOptions,
  LearnMessage,
  LearnOptions,
  LearnResult,
  LineageFact,
  LineageOptions,
  MemoryOp,
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
   * Alias for recall() using the market-standard search verb.
   */
  async search(query: string, options: RecallOptions = {}): Promise<string> {
    return this.recall(query, options);
  }

  /**
   * Extract and store facts from a conversation (the LLM-backed ingest path).
   * Mirrors the `learn` MCP tool; requires the server to have an LLM provider
   * configured, otherwise it falls back to storing the last user message.
   * Routing ingest through `learn` (rather than `add`) lets slot extraction
   * fire knowledge-update supersession on the server.
   */
  async learn(messages: LearnMessage[], options: LearnOptions = {}): Promise<LearnResult> {
    const args: Record<string, unknown> = { messages };
    if (options.provider !== undefined) args["provider"] = options.provider;
    if (options.apiKey !== undefined) args["api_key"] = options.apiKey;
    if (options.model !== undefined) args["model"] = options.model;
    if (options.eventTime !== undefined) args["event_time"] = options.eventTime;
    const text = await this.client.call("learn", args);
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
      if (f.op !== undefined) o["op"] = f.op;
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
   * Alias for forget() using the CRUD-style delete verb.
   */
  async delete(factId: string): Promise<void> {
    await this.forget(factId);
  }

  /**
   * List all stored facts as structured objects.
   */
  async listFacts(options: KnowledgeBaseListOptions = {}): Promise<Fact[]> {
    const args: Record<string, unknown> = {};
    if (options.includeInactive !== undefined) args["include_inactive"] = options.includeInactive;
    if (options.now !== undefined) args["now"] = options.now;
    const text = await this.client.call("list_facts", args);
    if (text === "No facts stored.") return [];
    return (JSON.parse(text) as unknown[]).map((item) => normalizeFact(item));
  }

  /**
   * Alias for listFacts() using the familiar list verb.
   */
  async list(options: KnowledgeBaseListOptions = {}): Promise<Fact[]> {
    return this.listFacts(options);
  }

  /**
   * Inspect one stored fact by its ID.
   */
  async get(factId: string): Promise<Fact> {
    const text = await this.client.call("get", { fact_id: factId });
    return normalizeFact(JSON.parse(text) as unknown);
  }

  /**
   * Return the supersession lineage of one fact, newest -> oldest.
   */
  async lineage(factId: string, options: LineageOptions = {}): Promise<LineageFact[]> {
    const args: Record<string, unknown> = { fact_id: factId };
    if (options.now !== undefined) args["now"] = options.now;
    const text = await this.client.call("memory_lineage", args);
    return (JSON.parse(text) as unknown[]).map((item) => normalizeLineageFact(item));
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

function normalizeFact(value: unknown): Fact {
  const fact = (value ?? {}) as Record<string, unknown>;
  const createdAt = asString(fact["created_at"]) ?? new Date().toISOString();
  return {
    id: asString(fact["id"]) ?? "",
    content: asString(fact["content"]) ?? "",
    type: asMemoryType(fact["type"]),
    importance: asNumber(fact["importance"]) ?? 0.8,
    retention_score: asNumber(fact["retention_score"]) ?? asNumber(fact["retention"]) ?? 1.0,
    access_count: asNumber(fact["access_count"]) ?? 0,
    tags: asStringArray(fact["tags"]),
    created_at: createdAt,
    last_accessed: asString(fact["last_accessed"]) ?? createdAt,
    event_time: asNullableString(fact["event_time"]),
    valid_from: asNullableString(fact["valid_from"]),
    valid_until: asNullableString(fact["valid_until"]),
    active: asBoolean(fact["active"]),
  };
}

function normalizeLineageFact(value: unknown): LineageFact {
  const fact = normalizeFact(value);
  const raw = (value ?? {}) as Record<string, unknown>;
  return {
    ...fact,
    slot_key: asNullableString(raw["slot_key"]),
    entity: asNullableString(raw["entity"]),
    attribute: asNullableString(raw["attribute"]),
    value_text: asNullableString(raw["value_text"]),
    version: asNumber(raw["version"]) ?? 0,
    supersedes_id: asNullableString(raw["supersedes_id"]),
    published_by: asNullableString(raw["published_by"]),
  };
}

function asMemoryType(value: unknown): Fact["type"] {
  if (value === "procedural" || value === "episodic") {
    return value;
  }
  return "semantic";
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function asNullableString(value: unknown): string | null | undefined {
  if (value === null) {
    return null;
  }
  return typeof value === "string" ? value : undefined;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}
