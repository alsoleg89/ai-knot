/**
 * Bucket classifier and summary renderer.
 *
 * Reads diagnostics.jsonl and renders a summary table in Markdown.
 * In diff mode (--baseline / --candidate), shows bucket migration table.
 *
 * Usage:
 *   tsx scripts/eval_recall_buckets.ts --run <runId> [--out <path>]
 *   tsx scripts/eval_recall_buckets.ts --baseline <runA> --candidate <runB> [--out <path>]
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import type { DiagnosticsRecord } from "./diagnose_recall.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const RUNS_DIR = resolve(BENCH_ROOT, "data", "runs");

// ---- Public types -----------------------------------------------------------

export interface BucketTable {
  runId: string;
  /** Per-category bucket counts, keyed by cat number then bucket name. */
  byCategory: Record<
    string,
    Record<"LLM-fail" | "partial-recall" | "low-recall" | "hard-miss", number>
  >;
  /** Aggregate bucket counts across all categories. */
  totals: Record<string, number>;
  avgPoolGoldRecallByBucket: Record<string, number>;
  avgPackGoldRecallByBucket: Record<string, number>;
}

export interface BucketMigration {
  convId: string;
  qaIdx: number;
  category: number;
  fromBucket: string;
  toBucket: string;
  verdictChange: string;
}

// ---- Helpers ----------------------------------------------------------------

function loadRecords(runId: string): DiagnosticsRecord[] {
  const p = resolve(RUNS_DIR, runId, "diagnostics.jsonl");
  if (!existsSync(p)) return [];
  return readFileSync(p, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as DiagnosticsRecord);
}

// ---- Core computation -------------------------------------------------------

export function buildBucketTable(
  runId: string,
  records: DiagnosticsRecord[],
): BucketTable {
  const wrongOnly = records.filter((r) => r.answer_verdict !== "CORRECT");
  const byCategory: BucketTable["byCategory"] = {};
  const totals: Record<string, number> = {};
  const poolByBucket: Record<string, number[]> = {};
  const packByBucket: Record<string, number[]> = {};

  for (const r of wrongOnly) {
    const cat = String(r.category);
    if (!byCategory[cat]) {
      byCategory[cat] = {
        "LLM-fail": 0,
        "partial-recall": 0,
        "low-recall": 0,
        "hard-miss": 0,
      };
    }
    if (r.bucket !== "correct" && r.bucket !== "unknown") {
      byCategory[cat]![r.bucket] = (byCategory[cat]![r.bucket] ?? 0) + 1;
    }
    totals[r.bucket] = (totals[r.bucket] ?? 0) + 1;
    if (r.pool_gold_recall !== null) {
      (poolByBucket[r.bucket] ??= []).push(r.pool_gold_recall);
    }
    if (r.pack_gold_recall !== null) {
      (packByBucket[r.bucket] ??= []).push(r.pack_gold_recall);
    }
  }

  const avg = (arr: number[]) =>
    arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  const avgPoolGoldRecallByBucket: Record<string, number> = {};
  const avgPackGoldRecallByBucket: Record<string, number> = {};
  for (const bucket of Object.keys(totals)) {
    avgPoolGoldRecallByBucket[bucket] = avg(poolByBucket[bucket] ?? []);
    avgPackGoldRecallByBucket[bucket] = avg(packByBucket[bucket] ?? []);
  }

  return {
    runId,
    byCategory,
    totals,
    avgPoolGoldRecallByBucket,
    avgPackGoldRecallByBucket,
  };
}

export function computeBucketMigrations(
  baseRecords: DiagnosticsRecord[],
  candRecords: DiagnosticsRecord[],
): BucketMigration[] {
  const baseMap = new Map(
    baseRecords.map((r) => [`${r.conv_id}:${r.qa_idx}`, r]),
  );

  const migrations: BucketMigration[] = [];
  for (const cand of candRecords) {
    const key = `${cand.conv_id}:${cand.qa_idx}`;
    const base = baseMap.get(key);
    if (!base || base.bucket === cand.bucket) continue;
    migrations.push({
      convId: cand.conv_id,
      qaIdx: cand.qa_idx,
      category: cand.category,
      fromBucket: base.bucket,
      toBucket: cand.bucket,
      verdictChange: `${base.answer_verdict} → ${cand.answer_verdict}`,
    });
  }
  return migrations;
}

// ---- Rendering --------------------------------------------------------------

export function renderBucketTable(table: BucketTable): string {
  const BUCKETS = ["LLM-fail", "partial-recall", "low-recall", "hard-miss"] as const;
  const lines: string[] = [
    `# Bucket Table — ${table.runId}`,
    "",
    "| Bucket | n | AvgPoolRecall | AvgPackRecall | Bottleneck |",
    "| --- | --- | --- | --- | --- |",
  ];

  const bottleneck: Record<string, string> = {
    "LLM-fail": "reader",
    "partial-recall": "pool+pack",
    "low-recall": "pool",
    "hard-miss": "raw/pool/coreference",
  };

  for (const bucket of BUCKETS) {
    const n = table.totals[bucket] ?? 0;
    const pool = (table.avgPoolGoldRecallByBucket[bucket] ?? 0).toFixed(2);
    const pack = (table.avgPackGoldRecallByBucket[bucket] ?? 0).toFixed(2);
    lines.push(
      `| ${bucket.padEnd(16)} | ${n} | ${pool} | ${pack} | ${bottleneck[bucket] ?? ""} |`,
    );
  }

  lines.push("", "## Per-category breakdown", "");

  const catNames: Record<string, string> = {
    "1": "single-hop",
    "2": "multi-hop",
    "3": "temporal",
    "4": "open-ended",
    "5": "adversarial",
  };

  for (const [cat, counts] of Object.entries(table.byCategory).sort()) {
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    lines.push(
      `### Cat ${cat} (${catNames[cat] ?? `cat${cat}`}) — ${total} wrong`,
    );
    lines.push("", "| Bucket | n |", "| --- | --- |");
    for (const bucket of BUCKETS) {
      lines.push(`| ${bucket} | ${counts[bucket] ?? 0} |`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

export function renderMigrationTable(
  baselineId: string,
  candidateId: string,
  migrations: BucketMigration[],
): string {
  const lines: string[] = [
    `# Bucket Migration: ${baselineId} → ${candidateId}`,
    "",
    `Total questions moved: ${migrations.length}`,
    "",
    "| conv | qa | cat | from | to | verdict |",
    "| --- | --- | --- | --- | --- | --- |",
  ];
  for (const m of migrations) {
    lines.push(
      `| ${m.convId} | ${m.qaIdx} | ${m.category} | ${m.fromBucket} | ${m.toBucket} | ${m.verdictChange} |`,
    );
  }
  return lines.join("\n");
}

// ---- CLI entry point --------------------------------------------------------

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      run: { type: "string" },
      baseline: { type: "string" },
      candidate: { type: "string" },
      out: { type: "string" },
    },
    strict: false,
  });

  if (values["run"]) {
    const runId = values["run"];
    const outPath =
      values["out"] ?? resolve(RUNS_DIR, runId, "bucket_table.md");

    const records = loadRecords(runId);
    if (records.length === 0) {
      console.error(
        `No diagnostics.jsonl found for run '${runId}'. Run diagnose_recall.ts first.`,
      );
      process.exit(1);
    }

    const table = buildBucketTable(runId, records);
    const md = renderBucketTable(table);
    writeFileSync(outPath, md);

    console.log(`Bucket table written to ${outPath}`);
    console.log("\nBucket totals:");
    for (const [bucket, count] of Object.entries(table.totals).sort()) {
      console.log(`  ${bucket.padEnd(20)} ${count}`);
    }
  } else if (values["baseline"] && values["candidate"]) {
    const baselineId = values["baseline"];
    const candidateId = values["candidate"];
    const outPath =
      values["out"] ??
      `/tmp/bucket_migration_${baselineId}_vs_${candidateId}.md`;

    const baseRecords = loadRecords(baselineId);
    const candRecords = loadRecords(candidateId);

    if (baseRecords.length === 0) {
      console.error(`No diagnostics.jsonl for baseline '${baselineId}'.`);
      process.exit(1);
    }
    if (candRecords.length === 0) {
      console.error(`No diagnostics.jsonl for candidate '${candidateId}'.`);
      process.exit(1);
    }

    const migrations = computeBucketMigrations(baseRecords, candRecords);
    const md = renderMigrationTable(baselineId, candidateId, migrations);
    writeFileSync(outPath, md);
    console.log(`Migration table written to ${outPath} (${migrations.length} moved)`);
  } else {
    console.error(
      "Usage:\n" +
        "  tsx scripts/eval_recall_buckets.ts --run <runId> [--out <path>]\n" +
        "  tsx scripts/eval_recall_buckets.ts --baseline <runA> --candidate <runB> [--out <path>]",
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
