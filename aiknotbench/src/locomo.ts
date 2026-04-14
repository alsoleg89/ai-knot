import { createWriteStream, existsSync, readFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const LOCOMO_URL =
  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";

const DATA_DIR = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "data"
);

const DEFAULT_CACHE_PATH = resolve(DATA_DIR, "locomo10.json");

// ---- Public types -----------------------------------------------------------

export interface QAPair {
  idx: number;
  question: string;
  answer: string;
  category: number;
}

export interface Session {
  key: string;        // "session_1"
  date?: string;      // "8 May, 2023"
  turns: string[];    // speaker-prefixed turns within this session
  text: string;       // concatenated session text (date prefix + turns)
}

export interface Conversation {
  idx: number;
  turns: string[];    // flat (backward compat)
  sessions: Session[];
  qa: QAPair[];
}

export interface LoadOptions {
  locomoFile?: string;
  limit?: number;
}

// ---- Internal LoCoMo JSON schema -------------------------------------------

interface RawTurn {
  text?: string;
  speaker?: string;
  dia_id?: string;
  blip_caption?: string;
  query?: string;
}

interface RawQA {
  question?: string;
  answer?: string;
  category?: number;
  adversarial_answer?: string;
}

interface RawEventSummary {
  date?: string;
  [key: string]: unknown;
}

interface RawConversation {
  conversation?: Record<string, unknown>;
  event_summary?: Record<string, RawEventSummary>;
  qa?: RawQA[];
  [key: string]: unknown;
}

// ---- Dataset loading --------------------------------------------------------

export async function loadDataset(opts: LoadOptions = {}): Promise<Conversation[]> {
  const jsonPath = opts.locomoFile
    ?? process.env["LOCOMO_FILE"]
    ?? DEFAULT_CACHE_PATH;

  if (!existsSync(jsonPath)) {
    await downloadLocomo(DEFAULT_CACHE_PATH);
  }

  const raw: RawConversation[] = JSON.parse(
    readFileSync(jsonPath === DEFAULT_CACHE_PATH ? DEFAULT_CACHE_PATH : jsonPath, "utf-8")
  ) as RawConversation[];

  const slice = opts.limit !== undefined ? raw.slice(0, opts.limit) : raw;
  return slice.map((conv, idx) => normalizeConversation(conv, idx));
}

async function downloadLocomo(dest: string): Promise<void> {
  console.log(`Downloading LoCoMo10 dataset from GitHub…`);
  mkdirSync(dirname(dest), { recursive: true });

  const res = await fetch(LOCOMO_URL);
  if (!res.ok) {
    throw new Error(`Failed to download locomo10.json: HTTP ${res.status}`);
  }
  if (!res.body) throw new Error("Empty response body from locomo download");

  await pipeline(
    Readable.fromWeb(res.body as Parameters<typeof Readable.fromWeb>[0]),
    createWriteStream(dest)
  );
  console.log(`Cached to ${dest}`);
}

// ---- Schema normalisation ---------------------------------------------------

const SESSION_RE = /^session_(\d+)$/;

export function normalizeConversation(raw: RawConversation, idx: number): Conversation {
  const conv = (raw["conversation"] ?? raw) as Record<string, unknown>;
  const eventSummary = raw["event_summary"] ?? {};

  // Collect session_N keys, sort by N
  const numbered: Array<[number, string]> = [];
  for (const key of Object.keys(conv)) {
    const m = SESSION_RE.exec(key);
    if (m) numbered.push([parseInt(m[1]!, 10), key]);
  }
  numbered.sort((a, b) => a[0] - b[0]);

  const allTurns: string[] = [];
  const sessions: Session[] = [];

  for (const [num, key] of numbered) {
    const rawSession = conv[key];
    if (!Array.isArray(rawSession)) continue;

    const sessionTurns: string[] = [];
    for (const turn of rawSession as RawTurn[]) {
      if (turn.text) {
        const speaker = turn.speaker ?? "speaker";
        const line = `${speaker}: ${turn.text}`;
        sessionTurns.push(line);
        allTurns.push(line);
      }
    }

    if (sessionTurns.length === 0) continue;

    // Extract date from event_summary.events_session_N
    const evKey = `events_session_${num}`;
    const ev = (eventSummary as Record<string, RawEventSummary>)[evKey];
    const date = ev?.date;

    const textParts: string[] = [];
    if (date) textParts.push(`[${date}]`);
    textParts.push(sessionTurns.join("\n"));

    sessions.push({
      key,
      date,
      turns: sessionTurns,
      text: textParts.join("\n"),
    });
  }

  const rawQA: RawQA[] = Array.isArray(raw["qa"]) ? (raw["qa"] as RawQA[]) : [];
  const qa: QAPair[] = rawQA
    .map((q, i): QAPair | null => {
      // Coerce to string — real LoCoMo answers can be numbers (years, counts, etc.)
      const question = q.question != null ? String(q.question).trim() : "";
      const category = q.category ?? 0;
      const raw_answer = category === 5
        ? (q.adversarial_answer ?? q.answer)
        : q.answer;
      const answer = raw_answer != null ? String(raw_answer).trim() : "";

      if (!question || !answer) return null;
      return { idx: i, question, answer, category };
    })
    .filter((q): q is QAPair => q !== null);

  return { idx, turns: allTurns, sessions, qa };
}

// ---- Filtering helpers (used by runner) ------------------------------------

export function filterQA(
  qa: QAPair[],
  types: number[] | undefined,
  sample: number | undefined
): QAPair[] {
  let filtered = types ? qa.filter((q) => types.includes(q.category)) : qa;
  if (sample !== undefined && filtered.length > sample) {
    filtered = filtered.slice(0, sample);
  }
  return filtered;
}
