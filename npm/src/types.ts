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
}

export interface AddOptions {
  type?: MemoryType;
  importance?: number;
  tags?: string[];
}

export interface RecallOptions {
  topK?: number;
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
