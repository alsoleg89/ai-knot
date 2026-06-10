import { KnowledgeBase } from "ai-knot";
import { parseLmeDate } from "./loader.js";
import type { LmeSession, Turn } from "./loader.js";

/**
 * Ingest granularity for LongMemEval (the paper's "value granularity" lever).
 *
 *   window   : sliding 3-turn window per session (the LOCOMO default unit).
 *   round    : one (user, assistant) pair per fact — the paper's value-granularity
 *              decomposition (finer unit → less reader noise, better multi-session).
 *   session  : one fact per whole session (coarse).
 *
 * All three pass the per-session timestamp as the STRUCTURED ``eventTime`` anchor
 * (P2) — never a date text-prefix. RRF in the core fuses across whatever granularity
 * is ingested; ``round`` is the recommended default for LongMemEval per §3.5 of the
 * research report.
 */
export type Granularity = "window" | "round" | "session";

export interface AdapterOptions {
  /** Absolute path to the run-specific SQLite file (per-run DB isolation). */
  dbPath: string;
  /** Stable namespace for this question's haystack (e.g. "q-<question_id>"). */
  agentId: string;
  /** Path/name of the ai-knot-mcp binary. */
  command: string;
  /** Extra env vars for the ai-knot-mcp subprocess. */
  env: Record<string, string>;
  /** Facts to recall per query. */
  topK: number;
  /** Ingest granularity. */
  granularity: Granularity;
  /**
   * Multi-agent mode (the multi-agent memory angle). When true, user turns and
   * assistant turns are ingested under SEPARATE agent namespaces
   * (``<agentId>::user`` / ``<agentId>::assistant``) so the harness exercises
   * ai-knot's per-agent working-memory + shared-layer model. Recall then unions
   * across both namespaces. Default false (single shared namespace).
   */
  multiAgent: boolean;
}

const WINDOW = 3;

/**
 * Per-question adapter around the ai-knot KnowledgeBase MCP bridge.
 *
 * Each LongMemEval question owns its haystack, so each question gets its own
 * agent namespace — QA cannot bleed across questions even though all data lives
 * in one SQLite file per run (mirrors the LOCOMO harness's per-conversation
 * isolation).
 */
export class AiknotAdapter {
  private readonly opts: AdapterOptions;
  private readonly kbs: Map<string, KnowledgeBase> = new Map();

  constructor(opts: AdapterOptions) {
    this.opts = opts;
  }

  /** Lazily create (and cache) a KnowledgeBase for a namespace. */
  private kb(namespace: string): KnowledgeBase {
    let kb = this.kbs.get(namespace);
    if (!kb) {
      kb = new KnowledgeBase({
        agentId: namespace,
        storage: "sqlite",
        dbPath: this.opts.dbPath,
        command: this.opts.command,
        env: this.opts.env,
      });
      this.kbs.set(namespace, kb);
    }
    return kb;
  }

  /** Namespaces this adapter recalls from (one, or two in multi-agent mode). */
  private namespaces(): string[] {
    if (this.opts.multiAgent) {
      return [`${this.opts.agentId}::user`, `${this.opts.agentId}::assistant`];
    }
    return [this.opts.agentId];
  }

  /** Route a turn to its namespace (single shared, or per-role in multi-agent). */
  private namespaceForRole(role: Turn["role"]): string {
    if (this.opts.multiAgent) return `${this.opts.agentId}::${role}`;
    return this.opts.agentId;
  }

  /** Ingest a haystack (list of sessions) at the configured granularity. */
  async ingest(sessions: LmeSession[]): Promise<void> {
    for (const session of sessions) {
      const eventTime = parseLmeDate(session.date);
      if (this.opts.granularity === "session") {
        await this.ingestSession(session, eventTime);
      } else if (this.opts.granularity === "round") {
        await this.ingestRounds(session, eventTime);
      } else {
        await this.ingestWindows(session, eventTime);
      }
    }
  }

  /** Window mode: sliding 3-turn window; structured eventTime anchor, no prefix. */
  private async ingestWindows(session: LmeSession, eventTime?: string): Promise<void> {
    const lines = session.turns.map((t) => `${t.role}: ${t.content}`);
    for (let i = 0; i < lines.length; i++) {
      const start = Math.max(0, i - Math.floor(WINDOW / 2));
      const end = Math.min(lines.length, start + WINDOW);
      const text = lines.slice(start, end).join(" / ");
      const ns = this.namespaceForRole(session.turns[i]!.role);
      await this.kb(ns).add(text, eventTime ? { eventTime } : {});
    }
  }

  /**
   * Round mode (the paper's value granularity): one (user, assistant) pair per
   * fact. Consecutive user→assistant turns are merged into one round; an
   * unpaired turn becomes its own round. This is the recommended LongMemEval unit.
   */
  private async ingestRounds(session: LmeSession, eventTime?: string): Promise<void> {
    const turns = session.turns;
    let i = 0;
    while (i < turns.length) {
      const cur = turns[i]!;
      const next = turns[i + 1];
      let text: string;
      let role: Turn["role"];
      if (cur.role === "user" && next && next.role === "assistant") {
        text = `user: ${cur.content} / assistant: ${next.content}`;
        role = "user"; // the round is keyed to the user's request in multi-agent mode
        i += 2;
      } else {
        text = `${cur.role}: ${cur.content}`;
        role = cur.role;
        i += 1;
      }
      await this.kb(this.namespaceForRole(role)).add(text, eventTime ? { eventTime } : {});
    }
  }

  /** Session mode: the whole session as one fact (coarse). */
  private async ingestSession(session: LmeSession, eventTime?: string): Promise<void> {
    const text = session.turns.map((t) => `${t.role}: ${t.content}`).join("\n");
    if (!text.trim()) return;
    // A whole session may contain both roles; in multi-agent mode key it to user.
    const ns = this.opts.multiAgent ? `${this.opts.agentId}::user` : this.opts.agentId;
    await this.kb(ns).add(text, eventTime ? { eventTime } : {});
  }

  /**
   * Recall context for a question, unioned across this adapter's namespaces.
   * Returns the concatenated formatted strings (ready for prompt injection).
   */
  async recall(question: string): Promise<string> {
    const parts: string[] = [];
    for (const ns of this.namespaces()) {
      const out = await this.kb(ns).recall(question, { topK: this.opts.topK });
      if (out && out !== "No relevant facts found.") parts.push(out);
    }
    return parts.join("\n");
  }

  /** Gracefully shut down all MCP subprocesses. */
  async close(): Promise<void> {
    for (const kb of this.kbs.values()) {
      await kb.close();
    }
    this.kbs.clear();
  }
}
