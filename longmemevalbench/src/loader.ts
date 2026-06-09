import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * LongMemEval data loader (P1).
 *
 * Parses the LongMemEval JSON schema (Wu et al., ICLR 2025) into a normalised
 * shape the harness can ingest and score. The schema is documented in
 * ``research/longmemeval_research_20260609.md`` §2.2 and mirrors the official
 * repo README (https://github.com/xiaowu0162/LongMemEval).
 *
 * Each evaluation instance is one JSON object:
 *   - question_id          : string. Ends in ``_abs`` => abstention/false-premise.
 *   - question_type        : one of the six types (see QuestionType).
 *   - question             : the query.
 *   - answer               : gold answer (absent / "N/A" for _abs).
 *   - question_date        : the "now" the question is asked at (query-time anchor).
 *   - haystack_session_ids : list[str], parallel to the two lists below.
 *   - haystack_dates       : list[str], per-session timestamps (the event_time source).
 *   - haystack_sessions    : list[session]; a session is a list[turn];
 *                            a turn is {role, content, has_answer?}.
 *   - answer_session_ids   : evidence session ids (session-level recall scoring).
 *
 * This loader does NOT download the dataset (the official set is gated behind a
 * HuggingFace / GitHub release — see README in this folder for how to obtain it).
 * A small hand-built fixture (data/sample_longmemeval.json) ships so the whole
 * pipeline runs end-to-end without the 500-question set.
 */

const DATA_DIR = resolve(fileURLToPath(import.meta.url), "..", "..", "data");
const DEFAULT_SAMPLE = resolve(DATA_DIR, "sample_longmemeval.json");

// ---- The six official question types ----------------------------------------

export type QuestionType =
  | "single-session-user"
  | "single-session-assistant"
  | "single-session-preference"
  | "multi-session"
  | "temporal-reasoning"
  | "knowledge-update";

export const QUESTION_TYPES: readonly QuestionType[] = [
  "single-session-user",
  "single-session-assistant",
  "single-session-preference",
  "multi-session",
  "temporal-reasoning",
  "knowledge-update",
] as const;

// ---- Public normalised types ------------------------------------------------

export interface Turn {
  role: "user" | "assistant";
  content: string;
  hasAnswer: boolean; // from the raw ``has_answer`` flag (turn-level recall)
}

export interface LmeSession {
  id: string; // haystack_session_ids[i]
  date?: string; // haystack_dates[i] — the structured event_time anchor
  turns: Turn[];
}

export interface LmeQuestion {
  id: string; // question_id
  type: QuestionType | string;
  question: string;
  answer: string; // "" for abstention
  questionDate?: string; // query-time anchor
  sessions: LmeSession[]; // haystack
  answerSessionIds: string[]; // evidence (session-level recall)
  isAbstention: boolean; // question_id ends in "_abs"
}

export interface LoadOptions {
  /** Absolute or relative path to a LongMemEval JSON file. */
  dataFile?: string;
  /** Limit to the first N questions (after type/conv filtering elsewhere). */
  limit?: number;
}

// ---- Raw JSON shape (loosely typed; tolerant of optional fields) ------------

interface RawTurn {
  role?: string;
  content?: string;
  has_answer?: boolean;
}

interface RawInstance {
  question_id?: string;
  question_type?: string;
  question?: string;
  answer?: string | number | null;
  question_date?: string;
  haystack_session_ids?: string[];
  haystack_dates?: string[];
  haystack_sessions?: RawTurn[][];
  answer_session_ids?: string[];
}

// ---- Loading ----------------------------------------------------------------

export function loadDataset(opts: LoadOptions = {}): LmeQuestion[] {
  const jsonPath =
    opts.dataFile ?? process.env["LONGMEMEVAL_FILE"] ?? DEFAULT_SAMPLE;

  if (!existsSync(jsonPath)) {
    throw new Error(
      `LongMemEval data file not found: ${jsonPath}\n` +
        `The full 500-question set is NOT auto-downloaded (it is release-gated).\n` +
        `See longmemevalbench/README.md for how to obtain and place it, or pass\n` +
        `--data data/sample_longmemeval.json to run the bundled smoke fixture.`
    );
  }

  const raw = JSON.parse(readFileSync(jsonPath, "utf-8")) as unknown;
  const instances: RawInstance[] = Array.isArray(raw)
    ? (raw as RawInstance[])
    : [];

  const questions = instances.map(normalizeInstance).filter((q): q is LmeQuestion => q !== null);
  return opts.limit !== undefined ? questions.slice(0, opts.limit) : questions;
}

export function normalizeInstance(raw: RawInstance): LmeQuestion | null {
  const id = raw.question_id != null ? String(raw.question_id) : "";
  const question = raw.question != null ? String(raw.question).trim() : "";
  if (!id || !question) return null;

  const isAbstention = id.endsWith("_abs");
  const answer = raw.answer != null ? String(raw.answer).trim() : "";

  const ids = raw.haystack_session_ids ?? [];
  const dates = raw.haystack_dates ?? [];
  const rawSessions = raw.haystack_sessions ?? [];

  const sessions: LmeSession[] = rawSessions.map((rawTurns, i) => {
    const turns: Turn[] = (rawTurns ?? [])
      .filter((t) => t && t.content)
      .map((t) => ({
        role: t.role === "assistant" ? "assistant" : "user",
        content: String(t.content),
        hasAnswer: t.has_answer === true,
      }));
    return {
      id: ids[i] ?? `session_${i}`,
      date: dates[i],
      turns,
    };
  });

  return {
    id,
    type: raw.question_type ?? "unknown",
    question,
    answer,
    questionDate: raw.question_date,
    sessions,
    answerSessionIds: raw.answer_session_ids ?? [],
    isAbstention,
  };
}

// ---- Filtering helpers (used by the runner) ---------------------------------

export function filterQuestions(
  questions: LmeQuestion[],
  types: string[] | undefined,
  sample: number | undefined
): LmeQuestion[] {
  let filtered = types ? questions.filter((q) => types.includes(String(q.type))) : questions;
  if (sample !== undefined && filtered.length > sample) {
    filtered = filtered.slice(0, sample);
  }
  return filtered;
}

/**
 * Parse a LongMemEval timestamp into an ISO date/datetime string suitable for
 * the structured ``eventTime`` anchor. LongMemEval ships ISO-like timestamps
 * (e.g. "2023/05/08 (Mon) 13:56" or "2023-05-08T13:56:00"); this normalises the
 * common forms to ISO-8601. Returns undefined if it cannot parse — a missing
 * anchor is always safe (the fact is simply stored without one).
 *
 * NOTE: this timestamp is used ONLY as a structured anchor passed to
 * ``kb.add(content, { eventTime })``. It is NEVER prefixed into the indexed
 * content (the banned ``dated`` text hack).
 */
export function parseLmeDate(s?: string): string | undefined {
  if (!s) return undefined;
  const trimmed = s.trim();

  // Already ISO-8601 (date or datetime).
  if (/^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?/.test(trimmed)) {
    return trimmed.replace(" ", "T");
  }

  // "2023/05/08 (Mon) 13:56" or "2023/05/08 13:56" or "2023/05/08".
  const m = /(\d{4})\/(\d{1,2})\/(\d{1,2})(?:\D+(\d{1,2}):(\d{2}))?/.exec(trimmed);
  if (m) {
    const [, y, mo, d, hh, mm] = m;
    const date = `${y}-${mo!.padStart(2, "0")}-${d!.padStart(2, "0")}`;
    if (hh && mm) return `${date}T${hh.padStart(2, "0")}:${mm}:00`;
    return date;
  }

  return undefined;
}
