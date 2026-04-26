/**
 * Build a mapping from LOCOMO evidence IDs (D{dialog}:{turn}) to ai-knot fact IDs.
 *
 * In dated ingest mode, one gold turn D{d}:{t} is embedded in up to 3 sliding-window
 * facts (centered at turns t-1, t, t+1). All 3 are marked as gold-bearing.
 *
 * Output: a JSON object keyed by "conv{convIdx}" → evidence-id → fact-id[].
 *
 * Usage:
 *   tsx scripts/build_gold_mapper.ts \
 *     --run <runId> \
 *     --locomo ./data/locomo10.json \
 *     --out /tmp/gold_map.json
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const RUNS_DIR = resolve(BENCH_ROOT, "data", "runs");

// ---- LOCOMO schema (minimal) ------------------------------------------------

interface RawTurn {
  text?: string;
  speaker?: string;
}

interface RawQA {
  evidence?: string[];
}

interface RawConversation {
  conversation?: Record<string, unknown>;
  qa?: RawQA[];
  [key: string]: unknown;
}

// ---- ai-knot Fact (minimal) --------------------------------------------------

interface StoredFact {
  id: string;
  content: string;
}

// ---- Public types -------------------------------------------------------------

/** Maps evidence-id ("D1:3") → list of fact IDs that contain the gold turn. */
export type EvidenceToFactIds = Record<string, string[]>;

/** Maps conv-key ("conv0") → evidence-id → fact-id[]. */
export type GoldMap = Record<string, EvidenceToFactIds>;

// ---- Core logic --------------------------------------------------------------

const SESSION_RE = /^session_(\d+)$/;
const EVIDENCE_RE = /^D(\d+):(\d+)$/;

/**
 * Parse all sessions from a raw LOCOMO conversation, returning
 * a map of (sessionIdx, turnIdx) → turn text.
 */
function extractTurnMap(
  rawConv: RawConversation,
): Map<string, string> {
  const turnMap = new Map<string, string>();
  const conv = (rawConv["conversation"] ?? rawConv) as Record<string, unknown>;

  const numbered: Array<[number, string]> = [];
  for (const key of Object.keys(conv)) {
    const m = SESSION_RE.exec(key);
    if (m) numbered.push([parseInt(m[1]!, 10), key]);
  }
  numbered.sort((a, b) => a[0] - b[0]);

  for (const [sessionNum, key] of numbered) {
    const rawSession = conv[key];
    if (!Array.isArray(rawSession)) continue;
    const turns = rawSession as RawTurn[];
    for (let i = 0; i < turns.length; i++) {
      const turn = turns[i];
      if (turn?.text) {
        const speaker = turn.speaker ?? "speaker";
        const text = `${speaker}: ${turn.text}`;
        // Evidence IDs are 1-based for both session and turn
        turnMap.set(`D${sessionNum}:${i + 1}`, text);
      }
    }
  }

  return turnMap;
}

/**
 * Compute a content fingerprint for matching turns to facts.
 * In dated mode, a fact's content is:
 *   "[date] turn1 / turn2 / turn3"
 * The gold turn text appears somewhere in the window.
 */
function turnAppearsInContent(turnText: string, factContent: string): boolean {
  return factContent.includes(turnText);
}

/**
 * Build gold map for a single conversation.
 * Uses the stored facts (from listFacts via diagnostics_raw or direct DB query) and
 * the LOCOMO turn map to resolve evidence IDs.
 */
export function buildConvGoldMap(
  evidenceIds: string[],
  turnMap: Map<string, string>,
  storedFacts: StoredFact[],
  ingestMode: "raw" | "dated" | "session" = "dated",
): EvidenceToFactIds {
  const result: EvidenceToFactIds = {};

  for (const evidenceId of evidenceIds) {
    const m = EVIDENCE_RE.exec(evidenceId);
    if (!m) continue;

    const turnText = turnMap.get(evidenceId);
    if (!turnText) {
      result[evidenceId] = [];
      continue;
    }

    if (ingestMode === "raw") {
      // Raw mode: one fact per turn; match by exact content equality
      const matches = storedFacts.filter((f) => f.content === turnText);
      result[evidenceId] = matches.map((f) => f.id);
    } else if (ingestMode === "dated") {
      // Dated mode: turn appears in up to 3 sliding-window facts
      const matches = storedFacts.filter((f) =>
        turnAppearsInContent(turnText, f.content),
      );
      result[evidenceId] = matches.map((f) => f.id);
    } else {
      // Session mode: turn is embedded in a session-level fact
      const matches = storedFacts.filter((f) =>
        turnAppearsInContent(turnText, f.content),
      );
      result[evidenceId] = matches.map((f) => f.id);
    }
  }

  return result;
}

/**
 * Collect all unique evidence IDs referenced across all QA pairs of a conv.
 */
export function collectEvidenceIds(qa: RawQA[]): string[] {
  const seen = new Set<string>();
  for (const q of qa) {
    for (const ev of q.evidence ?? []) {
      seen.add(ev);
    }
  }
  return [...seen];
}

/**
 * Validate that the gold mapper produced non-empty results for a given
 * evidence ID (used in tests and acceptance gate check).
 */
export function validateGoldMapCoverage(
  goldMap: EvidenceToFactIds,
): { covered: number; missing: string[] } {
  let covered = 0;
  const missing: string[] = [];
  for (const [evId, factIds] of Object.entries(goldMap)) {
    if (factIds.length > 0) {
      covered++;
    } else {
      missing.push(evId);
    }
  }
  return { covered, missing };
}

// ---- CLI entry point --------------------------------------------------------

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      run: { type: "string" },
      locomo: { type: "string" },
      out: { type: "string" },
      mode: { type: "string", default: "dated" },
    },
    strict: false,
  });

  const runId = values["run"];
  const locomoPath =
    values["locomo"] ?? resolve(BENCH_ROOT, "data", "locomo10.json");
  const outPath = values["out"] ?? `/tmp/gold_map_${runId}.json`;
  const ingestMode = (values["mode"] ?? "dated") as "raw" | "dated" | "session";

  if (!runId) {
    console.error("Usage: tsx scripts/build_gold_mapper.ts --run <runId> [--locomo <path>] [--out <path>] [--mode raw|dated|session]");
    process.exit(1);
  }

  const dbPathForRun = resolve(RUNS_DIR, runId, "knot.db");
  if (!existsSync(dbPathForRun)) {
    console.error(`DB not found: ${dbPathForRun}`);
    process.exit(1);
  }

  if (!existsSync(locomoPath)) {
    console.error(`LOCOMO file not found: ${locomoPath}`);
    process.exit(1);
  }

  const rawConvs: RawConversation[] = JSON.parse(
    readFileSync(locomoPath, "utf-8"),
  );

  // For each conv that has a completed run, load facts from DB and build map
  const { KnowledgeBase } = await import("ai-knot");
  const goldMap: GoldMap = {};

  for (let convIdx = 0; convIdx < rawConvs.length; convIdx++) {
    const rawConv = rawConvs[convIdx]!;
    const qa = (rawConv["qa"] ?? []) as RawQA[];
    const evidenceIds = collectEvidenceIds(qa);

    if (evidenceIds.length === 0) continue;

    const turnMap = extractTurnMap(rawConv);
    const convKey = `conv${convIdx}`;

    // Load facts from the agent namespace for this conv
    const kb = new KnowledgeBase({
      agentId: convKey,
      storage: "sqlite",
      dbPath: dbPathForRun,
    });

    try {
      const storedFacts = await kb.listFacts();
      const convGoldMap = buildConvGoldMap(
        evidenceIds,
        turnMap,
        storedFacts,
        ingestMode,
      );
      goldMap[convKey] = convGoldMap;

      const { covered, missing } = validateGoldMapCoverage(convGoldMap);
      console.log(
        `conv${convIdx}: ${covered}/${evidenceIds.length} evidence IDs resolved` +
          (missing.length > 0 ? ` (missing: ${missing.slice(0, 3).join(", ")})` : ""),
      );
    } finally {
      await kb.close();
    }
  }

  writeFileSync(outPath, JSON.stringify(goldMap, null, 2));
  console.log(`\nGold map written to ${outPath}`);
}

// Only run when invoked directly (not imported by tests)
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
