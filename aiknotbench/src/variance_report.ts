#!/usr/bin/env npx tsx
/**
 * Variance report — per-conversation accuracy breakdown.
 *
 * Usage: npx tsx src/variance_report.ts data/runs/<runId>/
 *
 * Reads log.jsonl (one question result per line) and outputs
 * per_conv_breakdown.json with per-conv, median, deviation stats.
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";

interface LogEntry {
  ts?: string;
  convIdx: number;
  qaIdx?: number;
  category: number; // 1-5 (LOCOMO category)
  question?: string;
  goldAnswer?: string;
  answer?: string;
  verdict: "CORRECT" | "WRONG";
}

interface ConvCatStats {
  total: number;
  correct: number;
  accuracy: number;
}

interface BreakdownOutput {
  runId: string;
  per_conv: Record<number, Record<number, ConvCatStats>>; // convIdx → catN → stats
  median: Record<number, number>; // catN → median accuracy
  max_deviation: Record<number, number>; // catN → max |acc - median|
  outliers: Array<{
    conv: number;
    cat: number;
    accuracy: number;
    deviation: number;
  }>;
  generated_at: string;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

async function readLogEntries(logPath: string): Promise<LogEntry[]> {
  const entries: LogEntry[] = [];
  const stream = fs.createReadStream(logPath, { encoding: "utf8" });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      entries.push(JSON.parse(trimmed) as LogEntry);
    } catch {
      // skip malformed lines
    }
  }
  return entries;
}

async function main(): Promise<void> {
  const runDir = process.argv[2];
  if (!runDir) {
    console.error("Usage: npx tsx src/variance_report.ts data/runs/<runId>/");
    process.exit(1);
  }

  const logPath = path.join(runDir, "log.jsonl");
  if (!fs.existsSync(logPath)) {
    console.error(`log.jsonl not found at ${logPath}`);
    process.exit(1);
  }

  const runId = path.basename(path.resolve(runDir));
  const entries = await readLogEntries(logPath);

  // Aggregate per (convIdx, category)
  const counts: Map<
    number,
    Map<number, { total: number; correct: number }>
  > = new Map();

  for (const entry of entries) {
    const { convIdx, category, verdict } = entry;
    if (!counts.has(convIdx)) counts.set(convIdx, new Map());
    const byCategory = counts.get(convIdx)!;
    if (!byCategory.has(category))
      byCategory.set(category, { total: 0, correct: 0 });
    const stat = byCategory.get(category)!;
    stat.total++;
    if (verdict === "CORRECT") stat.correct++;
  }

  const allCats = new Set<number>();
  for (const byCategory of counts.values()) {
    for (const cat of byCategory.keys()) allCats.add(cat);
  }

  // Build per_conv output
  const per_conv: Record<number, Record<number, ConvCatStats>> = {};
  for (const [convIdx, byCategory] of counts.entries()) {
    per_conv[convIdx] = {};
    for (const cat of allCats) {
      const stat = byCategory.get(cat) ?? { total: 0, correct: 0 };
      per_conv[convIdx][cat] = {
        total: stat.total,
        correct: stat.correct,
        accuracy: stat.total > 0 ? stat.correct / stat.total : 0,
      };
    }
  }

  // Compute median per cat (only from convs that have questions in that cat)
  const medianByCat: Record<number, number> = {};
  for (const cat of allCats) {
    const accs = Object.entries(per_conv)
      .filter(([, byCat]) => (byCat[cat]?.total ?? 0) > 0)
      .map(([, byCat]) => byCat[cat].accuracy);
    medianByCat[cat] = median(accs);
  }

  // Max deviation and outliers
  const maxDevByCat: Record<number, number> = {};
  const outliers: Array<{
    conv: number;
    cat: number;
    accuracy: number;
    deviation: number;
  }> = [];

  for (const cat of allCats) {
    let maxDev = 0;
    for (const [convIdxStr, byCat] of Object.entries(per_conv)) {
      const stats = byCat[cat];
      if (!stats || stats.total === 0) continue;
      const dev = Math.abs(stats.accuracy - medianByCat[cat]);
      if (dev > maxDev) maxDev = dev;
      if (dev > 0.07) {
        outliers.push({
          conv: Number(convIdxStr),
          cat,
          accuracy: stats.accuracy,
          deviation: dev,
        });
      }
    }
    maxDevByCat[cat] = maxDev;
  }

  const output: BreakdownOutput = {
    runId,
    per_conv,
    median: medianByCat,
    max_deviation: maxDevByCat,
    outliers,
    generated_at: new Date().toISOString(),
  };

  const outPath = path.join(runDir, "per_conv_breakdown.json");
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2), "utf8");

  console.log(`\nVariance report written to: ${outPath}`);
  console.log(`\nMedian accuracy per category:`);
  for (const cat of [...allCats].sort((a, b) => a - b)) {
    console.log(`  Cat ${cat}: ${(medianByCat[cat] * 100).toFixed(1)}%`);
  }
  console.log(`\nMax per-conv deviation:`);
  for (const cat of [...allCats].sort((a, b) => a - b)) {
    const flag = maxDevByCat[cat] > 0.07 ? " ⚠ >7pp" : "";
    console.log(
      `  Cat ${cat}: ${(maxDevByCat[cat] * 100).toFixed(1)}pp${flag}`
    );
  }
  if (outliers.length > 0) {
    console.log(`\nOutlier conversations (deviation >7pp):`);
    for (const o of outliers) {
      console.log(
        `  conv${o.conv} cat${o.cat}: accuracy=${(o.accuracy * 100).toFixed(1)}%, dev=${(o.deviation * 100).toFixed(1)}pp`
      );
    }
  } else {
    console.log("\nNo outliers (all deviations ≤7pp).");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
