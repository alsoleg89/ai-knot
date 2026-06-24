import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LanguageModelV1 } from "ai";

import { AiknotAdapter } from "./aiknot.js";
import type { Granularity } from "./aiknot.js";
import {
  answerQuestion,
  isAbstentionAnswer,
  judgeAnswer,
} from "./evaluator.js";
import type { Verdict } from "./evaluator.js";
import { filterQuestions, loadDataset, parseLmeDate } from "./loader.js";
import type { LmeQuestion } from "./loader.js";
import { scoreRecall } from "./recall.js";

// ---- Paths ------------------------------------------------------------------

const RUNS_DIR = resolve(fileURLToPath(import.meta.url), "..", "..", "data", "runs");

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

interface QResult {
  questionId: string;
  type: string;
  isAbstention: boolean;
  verdict: Verdict;
  abstained: boolean; // reader declined
  shortCircuited: boolean; // declined via deterministic empty-pool path
  turnHit: boolean | null;
  sessionHit: boolean | null;
}

interface Checkpoint {
  runId: string;
  judgeModel: string;
  answerModel: string;
  granularity: Granularity;
  multiAgent: boolean;
  idkContract: boolean;
  startedAt: string;
  updatedAt: string;
  results: QResult[];
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

interface Stat {
  total: number;
  correct: number;
  accuracy: number;
}

interface RecallStat {
  scored: number;
  hits: number;
  rate: number;
}

export interface Report {
  runId: string;
  judgeModel: string;
  answerModel: string;
  granularity: Granularity;
  multiAgent: boolean;
  idkContract: boolean;
  finishedAt: string;
  summary: Stat;
  byType: Record<string, Stat>;
  abstention: Stat; // accuracy on the _abs subset (declined == correct)
  turnRecall: RecallStat;
  sessionRecall: RecallStat;
}

export function computeReport(cp: Checkpoint): Report {
  const all = cp.results;
  const correct = all.filter((r) => r.verdict === "CORRECT").length;
  const summary: Stat = {
    total: all.length,
    correct,
    accuracy: all.length > 0 ? correct / all.length : 0,
  };

  const byType: Record<string, Stat> = {};
  for (const r of all) {
    const key = r.type;
    if (!byType[key]) byType[key] = { total: 0, correct: 0, accuracy: 0 };
    byType[key]!.total++;
    if (r.verdict === "CORRECT") byType[key]!.correct++;
  }
  for (const s of Object.values(byType)) {
    s.accuracy = s.total > 0 ? s.correct / s.total : 0;
  }

  const abs = all.filter((r) => r.isAbstention);
  const absCorrect = abs.filter((r) => r.verdict === "CORRECT").length;
  const abstention: Stat = {
    total: abs.length,
    correct: absCorrect,
    accuracy: abs.length > 0 ? absCorrect / abs.length : 0,
  };

  const turnScored = all.filter((r) => r.turnHit !== null);
  const turnHits = turnScored.filter((r) => r.turnHit === true).length;
  const turnRecall: RecallStat = {
    scored: turnScored.length,
    hits: turnHits,
    rate: turnScored.length > 0 ? turnHits / turnScored.length : 0,
  };

  const sessScored = all.filter((r) => r.sessionHit !== null);
  const sessHits = sessScored.filter((r) => r.sessionHit === true).length;
  const sessionRecall: RecallStat = {
    scored: sessScored.length,
    hits: sessHits,
    rate: sessScored.length > 0 ? sessHits / sessScored.length : 0,
  };

  return {
    runId: cp.runId,
    judgeModel: cp.judgeModel,
    answerModel: cp.answerModel,
    granularity: cp.granularity,
    multiAgent: cp.multiAgent,
    idkContract: cp.idkContract,
    finishedAt: new Date().toISOString(),
    summary,
    byType,
    abstention,
    turnRecall,
    sessionRecall,
  };
}

// ---- Injectable evaluator (for testing without LLM/MCP) ---------------------

export interface EvaluatorFns {
  answerFn: (context: string, question: string, idkContract: boolean) => Promise<{ text: string; shortCircuited: boolean }>;
  judgeFn: (question: string, answer: string, gold: string) => Promise<Verdict>;
  adapterFactory: (q: LmeQuestion) => AdapterLike;
}

export interface AdapterLike {
  ingest(sessions: LmeQuestion["sessions"]): Promise<void>;
  recall(question: string, now?: string): Promise<string>;
  close(): Promise<void>;
}

// ---- RunOptions -------------------------------------------------------------

export interface RunOptions {
  runId: string;
  judgeModel: LanguageModelV1;
  answerModel: LanguageModelV1;
  judgeModelName: string;
  answerModelName: string;
  aiKnotCommand: string;
  aiKnotEnv?: Record<string, string>;
  dataFile?: string;
  limit?: number;
  topK?: number;
  types?: string[];
  sample?: number;
  granularity?: Granularity;
  multiAgent?: boolean;
  idkContract?: boolean;
  force?: boolean;
  /** Override evaluator/adapter for tests (no LLM, no MCP). */
  _evaluatorOverride?: Partial<EvaluatorFns>;
}

function defaultEvaluatorFns(opts: RunOptions): EvaluatorFns {
  return {
    answerFn: async (ctx, q, idk) => {
      const r = await answerQuestion(opts.answerModel, ctx, q, { idkContract: idk });
      return { text: r.text, shortCircuited: r.shortCircuited };
    },
    judgeFn: async (question, answer, gold) =>
      (await judgeAnswer(opts.judgeModel, question, answer, gold)).verdict,
    adapterFactory: (q) =>
      new AiknotAdapter({
        dbPath: dbPath(opts.runId),
        agentId: `q-${q.id}`,
        command: opts.aiKnotCommand,
        env: opts.aiKnotEnv ?? {},
        topK: opts.topK ?? 10,
        granularity: opts.granularity ?? "round",
        multiAgent: opts.multiAgent ?? false,
      }),
  };
}

// ---- Main run logic ---------------------------------------------------------

export async function runBenchmark(opts: RunOptions): Promise<Report> {
  const {
    runId,
    judgeModelName,
    answerModelName,
    granularity = "round",
    multiAgent = false,
    idkContract = true,
    force,
  } = opts;

  const dir = runDir(runId);
  if (force && existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  let cp = loadCheckpoint(runId);
  if (!cp) {
    cp = {
      runId,
      judgeModel: judgeModelName,
      answerModel: answerModelName,
      granularity,
      multiAgent,
      idkContract,
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      results: [],
    };
    saveCheckpoint(cp);
  }

  let questions = loadDataset({ dataFile: opts.dataFile, limit: opts.limit });
  questions = filterQuestions(questions, opts.types, opts.sample);

  const fns: EvaluatorFns = { ...defaultEvaluatorFns(opts), ...opts._evaluatorOverride };

  const done = new Set(cp.results.map((r) => r.questionId));
  const pending = questions.filter((q) => !done.has(q.id));

  console.log(
    `\nrun: ${runId}  questions: ${questions.length} (${pending.length} pending)  ` +
      `judge: ${judgeModelName}  reader: ${answerModelName}\n` +
      `  granularity: ${granularity}  multiAgent: ${multiAgent}  idkContract: ${idkContract}\n`
  );

  const total = questions.length;
  for (const q of pending) {
    const adapter = fns.adapterFactory(q);
    try {
      await adapter.ingest(q.sessions);
      // Point-in-time recall: anchor retrieval at the question's date so facts
      // recorded *after* the question was asked are excluded (bi-temporal replay).
      const context = await adapter.recall(q.question, parseLmeDate(q.questionDate));
      const recall = scoreRecall(q, context);

      const { text: answer, shortCircuited } = await fns.answerFn(
        context,
        q.question,
        idkContract
      );
      const abstained = isAbstentionAnswer(answer);

      let verdict: Verdict;
      if (q.isAbstention) {
        // Abstention question: CORRECT iff the reader declined. No judge needed.
        verdict = abstained ? "CORRECT" : "WRONG";
      } else if (abstained) {
        // Answerable question but the reader declined => WRONG (no judge needed).
        verdict = "WRONG";
      } else {
        verdict = await fns.judgeFn(q.question, answer, q.answer);
      }

      cp.results.push({
        questionId: q.id,
        type: String(q.type),
        isAbstention: q.isAbstention,
        verdict,
        abstained,
        shortCircuited,
        turnHit: recall.turnHit,
        sessionHit: recall.sessionHit,
      });
      saveCheckpoint(cp);

      const icon = verdict === "CORRECT" ? "✓" : "✗";
      const n = cp.results.length;
      const qq = q.question.length > 50 ? q.question.slice(0, 50) + "…" : q.question;
      console.log(
        `  [${String(n).padStart(String(total).length)}/${total}] ${icon} ${verdict} ` +
          `(${q.type}${q.isAbstention ? " ABS" : ""}) "${qq}"`
      );
    } finally {
      await adapter.close();
    }
  }

  const report = computeReport(cp);
  writeFileSync(reportPath(runId), JSON.stringify(report, null, 2));
  printSummary(report);
  return report;
}

function printSummary(report: Report): void {
  const pct = (n: number) => (n * 100).toFixed(1);
  console.log(`\n${"─".repeat(60)}`);
  console.log(
    `  overall accuracy : ${pct(report.summary.accuracy)}%  ` +
      `(${report.summary.correct}/${report.summary.total})`
  );
  for (const [type, s] of Object.entries(report.byType).sort()) {
    console.log(
      `  ${type.padEnd(26)} : ${pct(s.accuracy)}%  (${s.correct}/${s.total})`
    );
  }
  if (report.abstention.total > 0) {
    console.log(
      `  abstention (_abs)          : ${pct(report.abstention.accuracy)}%  ` +
        `(${report.abstention.correct}/${report.abstention.total})`
    );
  }
  console.log(
    `  turn-level recall          : ${pct(report.turnRecall.rate)}%  ` +
      `(${report.turnRecall.hits}/${report.turnRecall.scored})`
  );
  console.log(
    `  session-level recall       : ${pct(report.sessionRecall.rate)}%  ` +
      `(${report.sessionRecall.hits}/${report.sessionRecall.scored})`
  );
  console.log(`${"─".repeat(60)}`);
  console.log(`  report: data/runs/${report.runId}/report.json\n`);
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
    const report = existsSync(rp) ? (JSON.parse(readFileSync(rp, "utf-8")) as Report) : null;
    summaries.push({
      runId,
      startedAt: cp?.startedAt ?? "unknown",
      finishedAt: report?.finishedAt ?? null,
      total: cp?.results.length ?? 0,
      accuracy: report ? `${(report.summary.accuracy * 100).toFixed(1)}%` : null,
    });
  }
  summaries.sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  return opts.limit !== undefined ? summaries.slice(0, opts.limit) : summaries;
}
