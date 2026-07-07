export type MemoryType = "semantic" | "procedural" | "episodic";
export type MemoryOp = "add" | "update" | "delete" | "noop";

export interface Fact {
  id: string;
  content: string;
  type: MemoryType;
  importance: number;
  retention_score: number;
  access_count: number;
  tags: string[];
  created_at: string;
  last_accessed: string;
  /**
   * Optional temporal metadata surfaced by the HTTP sidecar and normalized by
   * the MCP-backed TypeScript client when available.
   */
  event_time?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  active?: boolean;
}

export interface Stats {
  total_facts: number;
  by_type: Record<string, number>;
  avg_importance: number;
  avg_retention: number;
}

export interface KnowledgeBaseOptions {
  /** Agent namespace. Default: "default". */
  agentId?: string;
  /** Storage backend. Default: "yaml". */
  storage?: "yaml" | "sqlite";
  /** Base directory for YAML/SQLite storage. Use an absolute path. */
  dataDir?: string;
  /** Full path to SQLite database file. Overrides dataDir for sqlite. */
  dbPath?: string;
  /** Path to ai-knot-mcp binary. Default: "ai-knot-mcp". */
  command?: string;
  /** Extra environment variables passed to the ai-knot-mcp subprocess. */
  env?: Record<string, string>;
}

export interface HttpKnowledgeBaseListOptions {
  includeInactive?: boolean;
  limit?: number;
  now?: string;
}

export interface KnowledgeBaseListOptions {
  includeInactive?: boolean;
  now?: string;
}

export interface HttpKnowledgeBaseOptions {
  /** Base URL for the ai-knot HTTP sidecar, e.g. http://127.0.0.1:8000 */
  baseUrl: string;
  /** Optional bearer token for protected /v1/* routes. */
  token?: string;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
  /**
   * Optional fetch override for tests, custom runtimes, or explicit wiring.
   * If omitted, the global Node/browser fetch is used.
   */
  fetch?: (url: string, init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
  }) => Promise<{
    ok: boolean;
    status: number;
    statusText: string;
    json(): Promise<unknown>;
    text(): Promise<string>;
  }>;
}

export interface AddOptions {
  type?: MemoryType;
  importance?: number;
  tags?: string[];
  /**
   * ISO-8601 timestamp of when this memory was formed (the real-world anchor
   * used to resolve relative-time expressions like "yesterday" in content).
   * Defaults to server-side now() when omitted.
   */
  eventTime?: string;
}

export interface RecallOptions {
  topK?: number;
  /**
   * Optional ISO-8601 point-in-time anchor. Facts whose validity ended by this
   * instant (superseded knowledge-updates) are excluded and decay is computed
   * as of it. Defaults to the current time when omitted.
   */
  now?: string;
}

export interface LineageOptions {
  /** Optional ISO-8601 point-in-time anchor for active/inactive status. */
  now?: string;
}

/** One conversation turn fed to {@link KnowledgeBase.learn}. */
export interface LearnMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface LearnOptions {
  /**
   * Optional provider override for the LLM-backed extraction path. When
   * omitted, ai-knot falls back to the server-side/default provider wiring.
   */
  provider?: string;
  /** Optional API key override for the extraction provider. */
  apiKey?: string;
  /** Optional model override for the extraction provider. */
  model?: string;
  /**
   * ISO-8601 timestamp of when the conversation happened. Extracted facts are
   * anchored to it for point-in-time recall and supersession correctness.
   */
  eventTime?: string;
}

/** Result of {@link KnowledgeBase.learn}: how many facts were stored. */
export interface LearnResult {
  stored: number;
  ids: string[];
  note?: string;
}

/**
 * A pre-structured fact for {@link KnowledgeBase.addResolved}. A fact addressing
 * an existing active slot with a different value supersedes it (knowledge-update)
 * — no LLM extraction is performed.
 */
export interface ResolvedFact {
  content: string;
  entity?: string;
  attribute?: string;
  valueText?: string;
  slotKey?: string;
  /** Structured write intent: add, update, delete, or noop. */
  op?: MemoryOp;
  /** ISO-8601 real-world anchor for this fact (see AddOptions.eventTime). */
  eventTime?: string;
}

/** One row returned by {@link KnowledgeBase.addResolved}. */
export interface ResolvedResult {
  id: string;
  content?: string;
  slot_key: string;
  version: number;
}

export interface LineageFact extends Fact {
  slot_key?: string | null;
  entity?: string | null;
  attribute?: string | null;
  value_text?: string | null;
  version: number;
  supersedes_id?: string | null;
  published_by?: string | null;
}

// ---- Internal JSON-RPC 2.0 types ----------------------------------------

export interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number;
  method: string;
  params?: unknown;
}

export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: {
    content?: Array<{ type: string; text: string }>;
    [key: string]: unknown;
  };
  error?: { code: number; message: string };
}
