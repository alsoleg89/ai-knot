import { KnowledgeBase } from "ai-knot";
import type { Session } from "./locomo.js";

export type IngestMode = "raw" | "dated" | "session";

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
  private readonly ingestMode: IngestMode;

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
    /** How to ingest conversation data. */
    ingestMode: IngestMode = "raw",
  ) {
    this.topK = topK;
    this.ingestMode = ingestMode;
    this.kb = new KnowledgeBase({
      agentId: `conv-${convIdx}`,
      storage: "sqlite",
      dbPath: runDbPath,
      command,
      env,
    });
  }

  /** Ingest conversation data into the knowledge base. */
  async ingest(turns: string[], sessions?: Session[]): Promise<void> {
    if (this.ingestMode === "dated" && sessions && sessions.length > 0) {
      await this.ingestDated(sessions);
    } else if (this.ingestMode === "session" && sessions && sessions.length > 0) {
      await this.ingestSessions(sessions);
    } else {
      await this.ingestRaw(turns);
    }
  }

  /** Raw mode: one fact per turn (original behavior). */
  private async ingestRaw(turns: string[]): Promise<void> {
    for (const turn of turns) {
      await this.kb.add(turn);
    }
  }

  /**
   * Dated mode: sliding 3-turn window per session, prefixed with `[session.date] `.
   * Mirrors the pf3-runtime ingest pattern: each turn becomes a fact whose context
   * is the surrounding 3 turns and whose date prefix lets enrich_date_tags inject
   * canonical date tags for cat2 (temporal) recall.
   */
  private async ingestDated(sessions: Session[]): Promise<void> {
    const WINDOW = 3;
    for (const session of sessions) {
      const prefix = session.date ? `[${session.date}] ` : "";
      const turns = session.turns;
      for (let i = 0; i < turns.length; i++) {
        const start = Math.max(0, i - Math.floor(WINDOW / 2));
        const end = Math.min(turns.length, start + WINDOW);
        const window = turns.slice(start, end).join(" / ");
        await this.kb.add(`${prefix}${window}`);
      }
    }
  }

  /** Session mode: one fact per session (date-prefixed full text). */
  private async ingestSessions(sessions: Session[]): Promise<void> {
    for (const session of sessions) {
      await this.kb.add(session.text);
    }
  }

  /**
   * Recall relevant context for a question.
   * Returns the raw formatted string from ai-knot (ready to inject into a prompt).
   */
  async recall(question: string): Promise<string> {
    return this.kb.recall(question, { topK: this.topK });
  }

  /**
   * Diagnostic variant of recall — returns context string plus per-stage trace.
   * Only available when AI_KNOT_DIAG=1 is set; throws otherwise.
   */
  async recallWithTrace(question: string): Promise<{
    context: string;
    packFactIds: string[];
    trace: Record<string, unknown>;
  }> {
    return this.kb.recallWithTrace(question, { topK: this.topK });
  }

  /** Gracefully shut down the underlying MCP subprocess. */
  async close(): Promise<void> {
    await this.kb.close();
  }
}
