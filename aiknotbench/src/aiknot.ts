import { KnowledgeBase } from "ai-knot";

/**
 * Per-conversation adapter around ai-knot KnowledgeBase.
 *
 * Each conversation gets its own agentId (`conv-{idx}`) so that QA pairs
 * cannot bleed across conversation namespaces even though all data is
 * stored in a single SQLite file per run.
 */
export class AiknotAdapter {
  private readonly kb: KnowledgeBase;
  private readonly topK: number;

  constructor(
    /** Absolute path to the run-specific SQLite file. */
    runDbPath: string,
    /** Index of this conversation in the dataset (0-based). */
    convIdx: number,
    /** Path/name of the ai-knot-mcp binary. */
    command = "ai-knot-mcp",
    /** Extra env vars to pass to the ai-knot-mcp subprocess. */
    env: Record<string, string> = {},
    /** Number of facts to recall per query. */
    topK = 5,
  ) {
    this.topK = topK;
    this.kb = new KnowledgeBase({
      agentId: `conv-${convIdx}`,
      storage: "sqlite",
      dbPath: runDbPath,
      command,
      env,
    });
  }

  /** Ingest all turns of a conversation into the knowledge base. */
  async ingest(turns: string[]): Promise<void> {
    for (const turn of turns) {
      await this.kb.add(turn);
    }
  }

  /**
   * Recall relevant context for a question.
   * Returns the raw formatted string from ai-knot (ready to inject into a prompt).
   */
  async recall(question: string): Promise<string> {
    return this.kb.recall(question, { topK: this.topK });
  }

  /** Gracefully shut down the underlying MCP subprocess. */
  async close(): Promise<void> {
    await this.kb.close();
  }
}
