/**
 * PROC Diagnostics — compute per-question pool/pack/reader metrics from
 * diagnostics_raw.jsonl + gold mapper output.
 *
 * Metrics computed per question:
 *   RawGoldExists       — at least one gold evidence ID has matching facts in DB
 *   PoolGoldRecall@K    — fraction of gold fact IDs in Stage-1 candidate pool
 *   PackGoldRecall@Budget — fraction of gold fact IDs in final evidence pack
 *   GoldPackPosition    — median char position of gold facts in pack (0-indexed)
 *   DistractorDensity   — non-gold / total in pack
 *   ReaderFailDespiteGold — verdict==WRONG and PackGoldRecall==1.0
 *   LexicalExpansionUplift — delta PoolGoldRecall with vs without lexical trace
 *
 * Usage:
 *   tsx scripts/diagnose_recall.ts --run <runId> [--convs 0,1] [--out <path>]
 *   tsx scripts/diagnose_recall.ts --baseline <runA> --candidate <runB> [--out <path>]
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import type { GoldMap } from "./build_gold_mapper.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const RUNS_DIR = resolve(BENCH_ROOT, "data", "runs");

// ---- Schema for diagnostics_raw.jsonl entries --------------------------------

interface DiagRawEntry {
  run_id: string;
  conv_idx: number;
  qa_idx: number;
  category: number;
  query: string;
  stage1_candidates?: {
    from_bm25: string[];
    from_rare_tokens: string[];
    from_entity_hop: string[];
  };
  pack_fact_ids?: string[];
  stage0_lexical_bridge?: Record<string, unknown>;
  full_trace?: Record<string, unknown>;
}

// ---- Checkpoint (for verdict lookup) ----------------------------------------

interface CheckpointResult {
  convIdx: number;
  qaIdx: number;
  category: number;
  verdict: "CORRECT" | "WRONG" | "ABSTAIN";
}

interface Checkpoint {
  results: CheckpointResult[];
}

// ---- Output schema -----------------------------------------------------------

export interface DiagnosticsRecord {
  run_id: string;
  conv_id: string;
  qa_idx: number;
  category: number;
  query: string;
  gold_evidence: string[];
  gold_fact_ids: string[];
  gold_fact_ids_per_evidence: Record<string, string[]>;
  raw_gold_exists: boolean | null;
  pool_gold_recall: number | null;
  pack_gold_recall: number | null;
  gold_pack_position_median: number | null;
  distractor_density: number | null;
  reader_fail_despite_gold: boolean | null;
  lexical_expansion_uplift: number | null;
  answer_verdict: string;
  bucket: "LLM-fail" | "partial-recall" | "low-recall" | "hard-miss" | "correct" | "unknown";
}

export interface DiagnosticsSummary {
  run_id: string;
  total_questions: number;
  buckets: Record<string, number>;
  by_category: Record<string, {
    total_wrong: number;
    buckets: Record<string, number>;
    avg_pool_gold_recall: number;
    avg_pack_gold_recall: number;
  }>;
  avg_distractor_density: number;
  reader_fail_despite_gold_count: number;
}

// ---- LOCOMO schema (minimal) -------------------------------------------------

interface RawQAEntry {
  evidence?: string[];
  category?: number;
}

// ---- Helpers -----------------------------------------------------------------

function loadDiagRaw(runId: string, convFilter?: Set<number>): DiagRawEntry[] {
  const p = resolve(RUNS_DIR, runId, "diagnostics_raw.jsonl");
  if (!existsSync(p)) return [];
  return readFileSync(p, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DiagRawEntry)
    .filter((e) => !convFilter || convFilter.has(e.conv_idx));
}

function loadCheckpoint(runId: string): Checkpoint | null {
  const p = resolve(RUNS_DIR, runId, "checkpoint.json");
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf-8")) as Checkpoint;
}

function loadGoldMap(runId: string): GoldMap | null {
  const p = resolve(RUNS_DIR, runId, "gold_map.json");
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf-8")) as GoldMap;
}

function loadLocomoQA(locomoPath: string): Map<string, string[]> {
  // Returns Map<"conv{idx}:qa{qaIdx}", evidence[]>
  const raw: Array<{ qa?: RawQAEntry[] }> = JSON.parse(
    readFileSync(locomoPath, "utf-8"),
  );
  const out = new Map<string, string[]>();
  for (let i = 0; i < raw.length; i++) {
    const qa = raw[i]?.qa ?? [];
    for (let j = 0; j < qa.length; j++) {
      out.set(`conv${i}:qa${j}`, qa[j]?.evidence ?? []);
    }
  }
  return out;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]!
    : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

function allStage1Ids(
  stage1?: DiagRawEntry["stage1_candidates"],
): Set<string> {
  if (!stage1) return new Set();
  return new Set([
    ...stage1.from_bm25,
    ...stage1.from_rare_tokens,
    ...stage1.from_entity_hop,
  ]);
}

function classifyBucket(
  packGoldRecall: number | null,
  verdict: string,
): DiagnosticsRecord["bucket"] {
  if (verdict === "CORRECT") return "correct";
  if (packGoldRecall === null) return "unknown";
  if (packGoldRecall >= 1.0) return "LLM-fail";
  if (packGoldRecall >= 0.3) return "partial-recall";
  if (packGoldRecall > 0) return "low-recall";
  return "hard-miss";
}

// ---- Core computation --------------------------------------------------------

export function computeDiagnosticsRecord(
  entry: DiagRawEntry,
  goldFactIds: string[],
  goldPerEvidence: Record<string, string[]>,
  goldEvidence: string[],
  verdict: string,
): DiagnosticsRecord {
  const stage1 = allStage1Ids(entry.stage1_candidates);
  const pack = new Set(entry.pack_fact_ids ?? []);
  const goldSet = new Set(goldFactIds);

  const rawGoldExists =
    goldFactIds.length > 0 ? true : goldEvidence.length > 0 ? false : null;

  // PoolGoldRecall@K
  let poolGoldRecall: number | null = null;
  if (goldFactIds.length > 0 && stage1.size > 0) {
    const inPool = goldFactIds.filter((id) => stage1.has(id)).length;
    poolGoldRecall = inPool / goldFactIds.length;
  } else if (goldFactIds.length > 0) {
    poolGoldRecall = 0;
  }

  // PackGoldRecall@Budget
  let packGoldRecall: number | null = null;
  if (goldFactIds.length > 0) {
    const inPack = goldFactIds.filter((id) => pack.has(id)).length;
    packGoldRecall = inPack / goldFactIds.length;
  }

  // GoldPackPosition — approximate: position in ordered pack list (char-based would need content)
  let goldPackPositionMedian: number | null = null;
  const packIds = entry.pack_fact_ids ?? [];
  const goldPositions = packIds
    .map((id, i) => (goldSet.has(id) ? i : -1))
    .filter((i) => i >= 0);
  if (goldPositions.length > 0) {
    goldPackPositionMedian = median(goldPositions);
  }

  // DistractorDensity
  let distractorDensity: number | null = null;
  if (packIds.length > 0) {
    const nonGold = packIds.filter((id) => !goldSet.has(id)).length;
    distractorDensity = nonGold / packIds.length;
  }

  // ReaderFailDespiteGold
  const readerFailDespiteGold =
    packGoldRecall !== null && packGoldRecall >= 1.0 && verdict !== "CORRECT"
      ? true
      : packGoldRecall !== null
        ? false
        : null;

  // LexicalExpansionUplift — computed separately in A0.lexical; null here
  const lexicalExpansionUplift: number | null = null;

  const bucket = classifyBucket(packGoldRecall, verdict);

  return {
    run_id: entry.run_id,
    conv_id: `conv${entry.conv_idx}`,
    qa_idx: entry.qa_idx,
    category: entry.category,
    query: entry.query,
    gold_evidence: goldEvidence,
    gold_fact_ids: goldFactIds,
    gold_fact_ids_per_evidence: goldPerEvidence,
    raw_gold_exists: rawGoldExists,
    pool_gold_recall: poolGoldRecall,
    pack_gold_recall: packGoldRecall,
    gold_pack_position_median: goldPackPositionMedian,
    distractor_density: distractorDensity,
    reader_fail_despite_gold: readerFailDespiteGold,
    lexical_expansion_uplift: lexicalExpansionUplift,
    answer_verdict: verdict,
    bucket,
  };
}

export function computeSummary(
  runId: string,
  records: DiagnosticsRecord[],
): DiagnosticsSummary {
  const buckets: Record<string, number> = {};
  const byCategory: DiagnosticsSummary["by_category"] = {};
  let totalDistractor = 0;
  let distractorCount = 0;
  let readerFailCount = 0;

  for (const r of records) {
    buckets[r.bucket] = (buckets[r.bucket] ?? 0) + 1;

    const catKey = String(r.category);
    if (!byCategory[catKey]) {
      byCategory[catKey] = {
        total_wrong: 0,
        buckets: {},
        avg_pool_gold_recall: 0,
        avg_pack_gold_recall: 0,
      };
    }
    const cat = byCategory[catKey]!;

    if (r.answer_verdict !== "CORRECT") {
      cat.total_wrong++;
      cat.buckets[r.bucket] = (cat.buckets[r.bucket] ?? 0) + 1;
      if (r.pool_gold_recall !== null) {
        cat.avg_pool_gold_recall += r.pool_gold_recall;
      }
      if (r.pack_gold_recall !== null) {
        cat.avg_pack_gold_recall += r.pack_gold_recall;
      }
    }

    if (r.distractor_density !== null) {
      totalDistractor += r.distractor_density;
      distractorCount++;
    }
    if (r.reader_fail_despite_gold === true) readerFailCount++;
  }

  // Normalize averages
  for (const cat of Object.values(byCategory)) {
    if (cat.total_wrong > 0) {
      cat.avg_pool_gold_recall /= cat.total_wrong;
      cat.avg_pack_gold_recall /= cat.total_wrong;
    }
  }

  return {
    run_id: runId,
    total_questions: records.length,
    buckets,
    by_category: byCategory,
    avg_distractor_density:
      distractorCount > 0 ? totalDistractor / distractorCount : 0,
    reader_fail_despite_gold_count: readerFailCount,
  };
}

function renderSummaryMd(summary: DiagnosticsSummary): string {
  const lines: string[] = [
    `# Diagnostics Summary — ${summary.run_id}`,
    "",
    `Total questions: ${summary.total_questions}`,
    "",
    "## Bucket distribution (all categories)",
    "",
    "| Bucket | Count |",
    "| --- | --- |",
  ];
  for (const [bucket, count] of Object.entries(summary.buckets).sort()) {
    lines.push(`| ${bucket} | ${count} |`);
  }

  lines.push("", "## Per-category breakdown", "");
  for (const [catKey, cat] of Object.entries(summary.by_category).sort()) {
    const catNames: Record<string, string> = {
      "1": "single-hop",
      "2": "multi-hop",
      "3": "temporal",
      "4": "open-ended",
      "5": "adversarial",
    };
    lines.push(
      `### Cat ${catKey} (${catNames[catKey] ?? `cat${catKey}`}) — ${cat.total_wrong} wrong`,
    );
    lines.push("", "| Bucket | Count |", "| --- | --- |");
    for (const [b, c] of Object.entries(cat.buckets).sort()) {
      lines.push(`| ${b} | ${c} |`);
    }
    lines.push(
      "",
      `Avg PoolGoldRecall: ${cat.avg_pool_gold_recall.toFixed(3)}`,
      `Avg PackGoldRecall: ${cat.avg_pack_gold_recall.toFixed(3)}`,
      "",
    );
  }

  lines.push(
    `Avg DistractorDensity: ${summary.avg_distractor_density.toFixed(3)}`,
    `ReaderFailDespiteGold: ${summary.reader_fail_despite_gold_count}`,
  );

  return lines.join("\n");
}

// ---- CLI entry point --------------------------------------------------------

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      run: { type: "string" },
      baseline: { type: "string" },
      candidate: { type: "string" },
      convs: { type: "string" },
      out: { type: "string" },
      locomo: { type: "string" },
    },
    strict: false,
  });

  const locomoPath =
    values["locomo"] ?? resolve(BENCH_ROOT, "data", "locomo10.json");

  const convFilter = values["convs"]
    ? new Set(values["convs"].split(",").map(Number))
    : undefined;

  if (values["run"]) {
    // Single-run mode
    const runId = values["run"];
    const outPath = values["out"] ?? resolve(RUNS_DIR, runId, "diagnostics.jsonl");

    const entries = loadDiagRaw(runId, convFilter);
    if (entries.length === 0) {
      console.error(
        `No diagnostics_raw.jsonl found for run '${runId}'. ` +
          "Re-run the benchmark with AI_KNOT_DIAG=1.",
      );
      process.exit(1);
    }

    const goldMap = loadGoldMap(runId);
    const cp = loadCheckpoint(runId);
    const locomoQA = loadLocomoQA(locomoPath);

    const records: DiagnosticsRecord[] = [];

    for (const entry of entries) {
      const convKey = `conv${entry.conv_idx}`;
      const evKey = `${convKey}:qa${entry.qa_idx}`;
      const goldEvidence = locomoQA.get(evKey) ?? [];

      // Resolve gold fact IDs from map (if available)
      const goldPerEvidence: Record<string, string[]> = {};
      const allGoldIds: string[] = [];
      if (goldMap?.[convKey]) {
        for (const evId of goldEvidence) {
          const ids = goldMap[convKey]![evId] ?? [];
          goldPerEvidence[evId] = ids;
          allGoldIds.push(...ids);
        }
      }
      const uniqueGoldIds = [...new Set(allGoldIds)];

      const verdict =
        cp?.results.find(
          (r) => r.convIdx === entry.conv_idx && r.qaIdx === entry.qa_idx,
        )?.verdict ?? "unknown";

      records.push(
        computeDiagnosticsRecord(
          entry,
          uniqueGoldIds,
          goldPerEvidence,
          goldEvidence,
          verdict,
        ),
      );
    }

    // Write JSONL
    const jsonl = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
    writeFileSync(outPath, jsonl);

    // Write summary markdown
    const summary = computeSummary(runId, records);
    const mdPath = resolve(RUNS_DIR, runId, "diagnostics_summary.md");
    writeFileSync(mdPath, renderSummaryMd(summary));

    console.log(`Diagnostics written to ${outPath}`);
    console.log(`Summary written to ${mdPath}`);
    console.log("\nBucket counts:");
    for (const [bucket, count] of Object.entries(summary.buckets).sort()) {
      console.log(`  ${bucket.padEnd(20)} ${count}`);
    }
  } else if (values["baseline"] && values["candidate"]) {
    // Diff mode
    const baselineId = values["baseline"];
    const candidateId = values["candidate"];
    const outPath =
      values["out"] ?? `/tmp/diag_diff_${baselineId}_vs_${candidateId}.md`;

    console.log(
      `Diff mode: comparing ${baselineId} (baseline) vs ${candidateId} (candidate)`,
    );

    // Load records for both
    const loadRecords = (runId: string) => {
      const p = resolve(RUNS_DIR, runId, "diagnostics.jsonl");
      if (!existsSync(p)) return [] as DiagnosticsRecord[];
      return readFileSync(p, "utf-8")
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l) as DiagnosticsRecord);
    };

    const baseRecords = loadRecords(baselineId);
    const candRecords = loadRecords(candidateId);

    // Build lookup by conv_id + qa_idx
    const baseMap = new Map(
      baseRecords.map((r) => [`${r.conv_id}:${r.qa_idx}`, r]),
    );

    const migrations: string[] = [
      `# Bucket Migration Table: ${baselineId} → ${candidateId}`,
      "",
      "| conv | qa | cat | baseline bucket | candidate bucket | verdict change |",
      "| --- | --- | --- | --- | --- | --- |",
    ];

    let totalMoved = 0;
    for (const cand of candRecords) {
      const key = `${cand.conv_id}:${cand.qa_idx}`;
      const base = baseMap.get(key);
      if (!base || base.bucket === cand.bucket) continue;
      migrations.push(
        `| ${cand.conv_id} | ${cand.qa_idx} | ${cand.category} | ${base.bucket} | ${cand.bucket} | ${base.answer_verdict} → ${cand.answer_verdict} |`,
      );
      totalMoved++;
    }
    migrations.push("", `Total questions moved: ${totalMoved}`);

    writeFileSync(outPath, migrations.join("\n") + "\n");
    console.log(`Diff written to ${outPath} (${totalMoved} questions moved)`);
  } else {
    console.error(
      "Usage:\n" +
        "  tsx scripts/diagnose_recall.ts --run <runId> [--convs 0,1] [--out <path>]\n" +
        "  tsx scripts/diagnose_recall.ts --baseline <runA> --candidate <runB> [--out <path>]",
    );
    process.exit(1);
  }
}

// Only run when invoked directly (not imported by tests)
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
