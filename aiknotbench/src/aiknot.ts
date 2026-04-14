import { KnowledgeBase } from "ai-knot";
import type { Session } from "./locomo.js";

/**
 * Current ingest mode.
 *
 * "dated"       — one RawEpisode per turn, feeds the new target_query pipeline
 *                 (raw-episodes + deterministic materialization). This is v1.
 * "dated-learn" — same as "dated" + LLM enrichment pass (coming in v2, requires
 *                 src/ai_knot/enrichment.py / Track B §15 B4).
 */
export type IngestMode = "dated";

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
    ingestMode: IngestMode = "dated",
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

  /**
   * Ingest conversation data into the knowledge base.
   *
   * Dispatches to the mode-specific implementation. Exhaustive switch ensures
   * TypeScript will flag any missing case when "dated-learn" is added in v2.
   */
  async ingest(_turns: string[], sessions?: Session[]): Promise<void> {
    if (!sessions || sessions.length === 0) {
      throw new Error("ingest requires Session[] (LoCoMo sessions)");
    }
    switch (this.ingestMode) {
      case "dated":
        await this.ingestDated(sessions);
        return;
      default: {
        const _exhaustive: never = this.ingestMode;
        throw new Error(`Unknown ingestMode: ${String(_exhaustive)}`);
      }
    }
  }

  /**
   * Dated mode (v1): one RawEpisode per turn, feeds new target_query pipeline.
   *
   * Each turn is ingested as a standalone episode via the ingest_episode MCP
   * tool. This populates raw_episodes and atomic_claims planes, enabling the
   * contract-first query() path. Context is assembled at query time via bundles,
   * not by pre-concatenating turns (no sliding window).
   *
   * In v2 this method is followed by an LLM enrichment pass ("dated-learn").
   */
  private async ingestDated(sessions: Session[]): Promise<void> {
    for (const session of sessions) {
      const sessionId = session.date
        ? `session-${session.date}`
        : `session-${sessions.indexOf(session)}`;
      for (let i = 0; i < session.turns.length; i++) {
        const turnText = session.turns[i] ?? "";
        if (!turnText.trim()) continue;

        // LoCoMo alternates: even index = person1 (user), odd = person2 (assistant)
        const speaker = i % 2 === 0 ? "user" : "assistant";

        await this.kb.ingestEpisode({
          sessionId,
          turnId: `turn-${i}`,
          rawText: turnText,
          speaker,
          sessionDate: session.date ?? undefined,
          sourceMeta: { dataset: "locomo", sessionIdx: sessions.indexOf(session) },
        });
      }
    }
  }

  /**
   * Answer a question using the new contract-first pipeline (kb.query).
   *
   * No fallback to kb.recall() — if v2 planes are empty the error is surfaced
   * explicitly (silent degradation to legacy recall is prohibited per blueprint
   * §8.5 Phase A).
   */
  async recall(question: string): Promise<string> {
    const answer = await this.kb.query(question, { topK: this.topK });
    return answer.text || "No answer found.";
  }

  /** Gracefully shut down the underlying MCP subprocess. */
  async close(): Promise<void> {
    await this.kb.close();
  }
}
