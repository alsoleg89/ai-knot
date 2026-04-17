import { KnowledgeBase } from "ai-knot";
import type { Session } from "./locomo.js";

export type IngestMode = "raw" | "session" | "dated" | "learn" | "dated-learn" | "raw-episodes";
export type QueryMode = "legacy_recall" | "target_query";

/** Convert "8 May, 2023" or "22 October, 2023" to "2023-05-08". */
function toISODate(raw: string): string | undefined {
  const d = new Date(raw);
  if (isNaN(d.getTime())) return undefined;
  return d.toISOString().slice(0, 10);
}

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
  private readonly queryMode: QueryMode;

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
    /** How to answer questions: legacy kb.recall() or new kb.query(). */
    queryMode: QueryMode = "legacy_recall",
  ) {
    this.topK = topK;
    this.ingestMode = ingestMode;
    this.queryMode = queryMode;
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
    if (this.ingestMode === "raw-episodes" && sessions && sessions.length > 0) {
      await this.ingestRawEpisodes(sessions);
    } else if (this.ingestMode === "dated-learn" && sessions && sessions.length > 0) {
      await this.ingestDated(sessions);
      // Pass dated turns to learn so extraction sees timestamps
      const datedTurns = this.buildDatedTurns(sessions);
      await this.ingestLearn(datedTurns);
    } else if (this.ingestMode === "dated" && sessions && sessions.length > 0) {
      await this.ingestDated(sessions);
    } else if (this.ingestMode === "session" && sessions && sessions.length > 0) {
      await this.ingestSessions(sessions);
    } else if (this.ingestMode === "learn") {
      await this.ingestLearn(turns);
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
   * Raw-episodes mode: one RawEpisode per turn, with full session context.
   *
   * Each turn is ingested as a standalone episode via the new ingest_episode
   * MCP tool. This populates raw_episodes and atomic_claims planes, enabling
   * the contract-first query() path.
   */
  private async ingestRawEpisodes(sessions: Session[]): Promise<void> {
    for (const session of sessions) {
      const sessionId = session.date
        ? `session-${session.date}`
        : `session-${sessions.indexOf(session)}`;
      for (let i = 0; i < session.turns.length; i++) {
        const turnText = session.turns[i] ?? "";
        if (!turnText.trim()) continue;

        // Determine speaker from turn index (alternating user/assistant)
        // LoCoMo alternates: even = person1 (user), odd = person2 (assistant)
        const speaker = i % 2 === 0 ? "user" : "assistant";

        const isoDate = session.date ? toISODate(session.date) : undefined;
        await this.kb.ingestEpisode({
          sessionId,
          turnId: `turn-${i}`,
          rawText: turnText,
          speaker,
          sessionDate: isoDate,
          sourceMeta: { dataset: "locomo", sessionIdx: sessions.indexOf(session) },
        });
      }
    }
  }

  /** Dated mode: sliding window of 3 turns per fact, with session context. */
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

  /** Session mode: one fact per session (grouped, with date prefix). */
  private async ingestSessions(sessions: Session[]): Promise<void> {
    for (const session of sessions) {
      await this.kb.add(session.text);
    }
  }

  /** Build date-prefixed turns from sessions (for dated-learn). */
  private buildDatedTurns(sessions: Session[]): string[] {
    const WINDOW = 3;
    const result: string[] = [];
    for (const session of sessions) {
      const prefix = session.date ? `[${session.date}] ` : "";
      const turns = session.turns;
      for (let i = 0; i < turns.length; i++) {
        const start = Math.max(0, i - Math.floor(WINDOW / 2));
        const end = Math.min(turns.length, start + WINDOW);
        const window = turns.slice(start, end).join(" / ");
        result.push(`${prefix}${window}`);
      }
    }
    return result;
  }

  /** Learn mode: LLM extracts structured facts from conversation. */
  private async ingestLearn(turns: string[]): Promise<void> {
    const messages = turns.map((turn) => ({
      role: "user" as const,
      content: turn,
    }));

    // Batch into groups of 20 (matches kb.learn default batch_size)
    const batchSize = 20;
    for (let i = 0; i < messages.length; i += batchSize) {
      const batch = messages.slice(i, i + batchSize);
      await this.kb.learn(batch);
    }
  }

  /**
   * Answer a question using the configured queryMode.
   *
   * - ``legacy_recall``: calls kb.recall() (original behaviour).
   * - ``target_query``: calls kb.query() via the contract-first pipeline.
   *   Returns answer.text (ready to inject into a prompt).
   */
  async recall(question: string): Promise<string> {
    if (this.queryMode === "target_query") {
      try {
        const answer = await this.kb.query(question, { topK: this.topK });
        return answer.evidence_text || answer.text || "No answer found.";
      } catch {
        // Fall back to legacy recall on error (e.g. no raw episodes ingested).
        return this.kb.recall(question, { topK: this.topK });
      }
    }
    return this.kb.recall(question, { topK: this.topK });
  }

  /** Gracefully shut down the underlying MCP subprocess. */
  async close(): Promise<void> {
    await this.kb.close();
  }
}
