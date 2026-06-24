export type MemoryType = "semantic" | "procedural" | "episodic";

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

/** One conversation turn fed to {@link KnowledgeBase.learn}. */
export interface LearnMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

/** Result of {@link KnowledgeBase.learn}: how many facts were stored. */
export interface LearnResult {
  stored: number;
  ids: string[];
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
  /** ISO-8601 real-world anchor for this fact (see AddOptions.eventTime). */
  eventTime?: string;
}

/** One row returned by {@link KnowledgeBase.addResolved}. */
export interface ResolvedResult {
  id: string;
  slot_key: string;
  version: number;
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
