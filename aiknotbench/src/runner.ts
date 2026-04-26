import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { readdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LanguageModelV1 } from "ai";

import { AiknotAdapter } from "./aiknot.js";
import type { IngestMode } from "./aiknot.js";
import { answerQuestion, judgeAnswer } from "./evaluator.js";
import type { Verdict } from "./evaluator.js";
import { filterQA, loadDataset } from "./locomo.js";
import type { LoadOptions, Session } from "./locomo.js";

// ---- Paths ------------------------------------------------------------------

const RUNS_DIR = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "data",
  "runs"
);

function runDir(runId: string): string {
  return resolve(RUNS_DIR, runId);
}

function checkpointPath(runId: string): string {
  return resolve(runDir(runId), "checkpoint.json");
}

function reportPath(runId: string): string {
  return resolve(runDir(runId), "report.json");
}

function dbPath(runId: string): string {
  return resolve(runDir(runId), "knot.db");
}

// ---- Checkpoint schema ------------------------------------------------------

interface CheckpointResult {
  convIdx: number;
  qaIdx: number;
  category: number;
  verdict: Verdict;
}

interface Checkpoint {
  runId: string;
  judgeModel: string;
  answerModel: string;
  startedAt: string;
  updatedAt: string;
  ingested: number[];
  results: CheckpointResult[];
}

function loadCheckpoint(runId: string): Checkpoint | null {
  const p = checkpointPath(runId);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, "utf-8")) as Checkpoint;
}

function saveCheckpoint(cp: Checkpoint): void {
  cp.updatedAt = new Date().toISOString();
  writeFileSync(checkpointPath(cp.runId), JSON.stringify(cp, null, 2));
}

// ---- Report schema ----------------------------------------------------------

interface TypeStat {
  total: number;
  correct: number;
  accuracy: number;
}

export interface Report {
  runId: string;
  judgeModel: string;
  answerModel: string;
  finishedAt: string;
  summary: TypeStat;
  byType: Record<string, TypeStat>;
  categories1to4: TypeStat;
}

export function computeReport(
  runId: string,
  judgeModel: string,
  answerModel: string,
  results: CheckpointResult[]
): Report {
  const all = results;
  const correct = all.filter((r) => r.verdict === "CORRECT").length;
  const summary: TypeStat = {
    total: all.length,
    correct,
    accuracy: all.length > 0 ? correct / all.length : 0,
  };

  const byType: Record<string, TypeStat> = {};
  for (const r of all) {
    const key = String(r.category);
    if (!byType[key]) byType[key] = { total: 0, correct: 0, accuracy: 0 };
    byType[key]!.total++;
    if (r.verdict === "CORRECT") byType[key]!.correct++;
  }
  for (const stat of Object.values(byType)) {
    stat.accuracy = stat.total > 0 ? stat.correct / stat.total : 0;
  }

  const cat14 = all.filter((r) => r.category >= 1 && r.category <= 4);
  const cat14correct = cat14.filter((r) => r.verdict === "CORRECT").length;
  const categories1to4: TypeStat = {
    total: cat14.length,
    correct: cat14correct,
    accuracy: cat14.length > 0 ? cat14correct / cat14.length : 0,
  };

  return {
    runId,
    judgeModel,
    answerModel,
    finishedAt: new Date().toISOString(),
    summary,
    byType,
    categories1to4,
  };
}

// ---- Injectable evaluator (for testing) -------------------------------------

export interface EvaluatorFns {
  answerFn: (
    model: LanguageModelV1,
    context: string,
    question: string
  ) => Promise<string>;
  judgeFn: (
    model: LanguageModelV1,
    question: string,
    answer: string,
    gold: string
  ) => Promise<Verdict>;
  adapterFactory: (
    runDbPath: string,
    convIdx: number,
    command: string,
    env: Record<string, string>,
    topK: number,
    ingestMode: IngestMode
  ) => AiknotAdapterLike;
}

export interface AiknotAdapterLike {
  ingest(turns: string[], sessions?: Session[]): Promise<void>;
  recall(question: string): Promise<string>;
  close(): Promise<void>;
}

const defaultEvaluatorFns = (
  judgeModel: LanguageModelV1,
  answerModel: LanguageModelV1,
  command: string
): EvaluatorFns => ({
  answerFn: async (_, ctx, q) => (await answerQuestion(answerModel, ctx, q)).text,
  judgeFn: async (_, question, answer, gold) =>
    (await judgeAnswer(judgeModel, question, answer, gold)).verdict,
  adapterFactory: (dbPath, convIdx, _cmd, env, topK, ingestMode) =>
    new AiknotAdapter(dbPath, convIdx, command, env, topK, ingestMode),
});

// ---- RunOptions -------------------------------------------------------------

export interface RunOptions extends LoadOptions {
  runId: string;
  judgeModel: LanguageModelV1;
  answerModel: LanguageModelV1;
  judgeModelName: string;
  answerModelName: string;
  aiKnotCommand: string;
  aiKnotEnv?: Record<string, string>;
  topK?: number;
  types?: number[];
  sample?: number;
  convs?: number[];
  ingestMode?: IngestMode;
  force?: boolean;
  _evaluatorOverride?: Partial<EvaluatorFns>;
}

// ---- Main run logic ---------------------------------------------------------

export async function runBenchmark(opts: RunOptions): Promise<Report> {
  const {
    runId,
    judgeModel,
    answerModel,
    judgeModelName,
    answerModelName,
    aiKnotCommand,
    aiKnotEnv = {},
    topK = 5,
    types,
    sample,
    ingestMode = "raw",
    force,
    _evaluatorOverride,
  } = opts;

  const dir = runDir(runId);

  if (force && existsSync(dir)) {
    rmSync(dir, { recursive: true, force: true });
  }

  mkdirSync(dir, { recursive: true });

  // Load or create checkpoint
  let cp = loadCheckpoint(runId);
  if (!cp) {
    cp = {
      runId,
      judgeModel: judgeModelName,
      answerModel: answerModelName,
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ingested: [],
      results: [],
    };
    saveCheckpoint(cp);
  }

  let dataset = await loadDataset({
    locomoFile: opts.locomoFile,
    limit: opts.limit,
  });
  if (opts.convs && opts.convs.length > 0) {
    const wanted = new Set(opts.convs);
    dataset = dataset.filter((conv) => wanted.has(conv.idx));
  }

  const fns: EvaluatorFns = {
    ...defaultEvaluatorFns(judgeModel, answerModel, aiKnotCommand),
    ..._evaluatorOverride,
  };

  // Compute total work for progress display
  let totalWork = 0;
  for (const conv of dataset) {
    totalWork += filterQA(conv.qa, types, sample).length;
  }

  const pad = String(dataset.length).length;
  console.log(
    `\nrun: ${runId}  convs: ${dataset.length}  ` +
    `judge: ${judgeModelName}  model: ${answerModelName}\n`
  );

  for (const conv of dataset) {
    const convLabel = String(conv.idx + 1).padStart(pad, "0");
    const filteredQA = filterQA(conv.qa, types, sample);
    const pending = filteredQA.filter(
      (qa) =>
        !cp!.results.some(
          (r) => r.convIdx === conv.idx && r.qaIdx === qa.idx
        )
    );

    if (pending.length === 0) {
      console.log(
        `  conv ${convLabel}/${dataset.length} — already complete, skipping`
      );
      continue;
    }

    const adapter = fns.adapterFactory(dbPath(runId), conv.idx, aiKnotCommand, aiKnotEnv, topK, ingestMode);

    try {
      if (!cp!.ingested.includes(conv.idx)) {
        process.stdout.write(
          `  conv ${convLabel}/${dataset.length} — ingesting ${conv.turns.length} turns… `
        );
        await adapter.ingest(conv.turns, conv.sessions);
        cp!.ingested.push(conv.idx);
        saveCheckpoint(cp!);
        process.stdout.write("done\n");
      }

      for (const qa of pending) {
        const context = await adapter.recall(qa.question);
        const answer = await fns.answerFn(answerModel, context, qa.question);
        const verdict = await fns.judgeFn(
          judgeModel,
          qa.question,
          answer,
          qa.answer
        );

        cp!.results.push({
          convIdx: conv.idx,
          qaIdx: qa.idx,
          category: qa.category,
          verdict,
        });
        saveCheckpoint(cp!);

        const done = cp!.results.length;
        const icon = verdict === "CORRECT" ? "✓" : "✗";
        const q = qa.question.length > 55
          ? qa.question.slice(0, 55) + "…"
          : qa.question;
        console.log(
          `  [conv ${convLabel} qa ${String(done).padStart(String(totalWork).length)}/${totalWork}] ` +
          `${icon} ${verdict} (cat ${qa.category}) "${q}"`
        );
      }
    } finally {
      await adapter.close();
    }
  }

  // Write report
  const report = computeReport(
    runId,
    judgeModelName,
    answerModelName,
    cp.results
  );
  writeFileSync(reportPath(runId), JSON.stringify(report, null, 2));

  // Print summary
  const acc = (report.categories1to4.accuracy * 100).toFixed(1);
  const acc14 = `${report.categories1to4.correct}/${report.categories1to4.total}`;
  console.log(`\n${"─".repeat(52)}`);
  console.log(
    `  cat 1–4 accuracy : ${acc}%  (${acc14})`
  );
  console.log(
    `  overall accuracy : ${(report.summary.accuracy * 100).toFixed(1)}%  ` +
    `(${report.summary.correct}/${report.summary.total})`
  );
  for (const [cat, stat] of Object.entries(report.byType).sort()) {
    const catNames: Record<string, string> = {
      "1": "single-hop",
      "2": "multi-hop",
      "3": "temporal",
      "4": "open-ended",
      "5": "adversarial",
    };
    const label = catNames[cat] ?? `cat${cat}`;
    console.log(
      `  cat ${cat} (${label.padEnd(10)}) : ` +
      `${(stat.accuracy * 100).toFixed(1)}%  (${stat.correct}/${stat.total})`
    );
  }
  console.log(`${"─".repeat(52)}`);
  console.log(`  report: data/runs/${runId}/report.json\n`);

  return report;
}

// ---- List runs --------------------------------------------------------------

export interface RunSummary {
  runId: string;
  startedAt: string;
  finishedAt: string | null;
  total: number;
  accuracy: string | null;
}

export function listRuns(opts: { limit?: number } = {}): RunSummary[] {
  if (!existsSync(RUNS_DIR)) return [];

  const dirs = readdirSync(RUNS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);

  const summaries: RunSummary[] = [];

  for (const runId of dirs) {
    const cp = loadCheckpoint(runId);
    const rp = reportPath(runId);
    const report = existsSync(rp)
      ? (JSON.parse(readFileSync(rp, "utf-8")) as Report)
      : null;

    summaries.push({
      runId,
      startedAt: cp?.startedAt ?? "unknown",
      finishedAt: report?.finishedAt ?? null,
      total: cp?.results.length ?? 0,
      accuracy:
        report
          ? `${(report.categories1to4.accuracy * 100).toFixed(1)}%`
          : null,
    });
  }

  // Sort newest first
  summaries.sort((a, b) => b.startedAt.localeCompare(a.startedAt));

  return opts.limit !== undefined ? summaries.slice(0, opts.limit) : summaries;
}
