import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { runBenchmark } from "../src/runner.js";
import type { AdapterLike, EvaluatorFns } from "../src/runner.js";
import type { LmeQuestion } from "../src/loader.js";

const SAMPLE = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "data",
  "sample_longmemeval.json"
);

const RUNS_DIR = resolve(fileURLToPath(import.meta.url), "..", "..", "data", "runs");
const createdRuns: string[] = [];

afterEach(() => {
  for (const runId of createdRuns) {
    const dir = resolve(RUNS_DIR, runId);
    if (existsSync(dir)) rmSync(dir, { recursive: true, force: true });
  }
  createdRuns.length = 0;
});

/**
 * A fake adapter that "recalls" by concatenating every turn whose content
 * overlaps the question — deterministic, no MCP/SQLite. It models a perfect
 * retriever so the runner's scoring/judging logic can be tested in isolation.
 */
function fakeAdapterFactory(q: LmeQuestion): AdapterLike {
  let store = "";
  return {
    async ingest(sessions) {
      store = sessions
        .flatMap((s) => s.turns.map((t) => `${t.role}: ${t.content}`))
        .join("\n");
    },
    async recall(_question: string) {
      // Abstention question s1/s2 do not mention "Japan" → return empty pool so the
      // empty-pool short-circuit fires (proves the abstention path end-to-end).
      if (q.isAbstention) return "No relevant facts found.";
      return store;
    },
    async close() {},
  };
}

// A trivial judge: CORRECT iff the gold answer's first word appears in the answer.
const fakeEval = (): Partial<EvaluatorFns> => ({
  answerFn: async (context, _question, idkContract) => {
    if (idkContract && (context.trim() === "" || context.includes("No relevant facts found."))) {
      return { text: "I don't know.", shortCircuited: true };
    }
    return { text: context, shortCircuited: false }; // echo the context as the "answer"
  },
  judgeFn: async (_question, answer, gold) => {
    const key = gold.split(/\s+/)[0]!.toLowerCase().replace(/[^a-z]/g, "");
    return answer.toLowerCase().includes(key) ? "CORRECT" : "WRONG";
  },
});

// Minimal model placeholders (never invoked because evaluator is overridden).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NULL_MODEL: any = {};

describe("runner end-to-end (fakes, no LLM/MCP)", () => {
  it("runs the full sample, scores abstention, and writes a report", async () => {
    const runId = `test-${Date.now()}`;
    createdRuns.push(runId);

    const report = await runBenchmark({
      runId,
      judgeModel: NULL_MODEL,
      answerModel: NULL_MODEL,
      judgeModelName: "fake-judge",
      answerModelName: "fake-reader",
      aiKnotCommand: "unused",
      dataFile: SAMPLE,
      force: true,
      _evaluatorOverride: { ...fakeEval(), adapterFactory: fakeAdapterFactory },
    });

    // 7 questions total, all evaluated.
    expect(report.summary.total).toBe(7);

    // The abstention question must be scored CORRECT (reader declined via short-circuit).
    expect(report.abstention.total).toBe(1);
    expect(report.abstention.correct).toBe(1);

    // Per-type breakdown is present for all encountered types.
    expect(Object.keys(report.byType).length).toBeGreaterThanOrEqual(5);

    // Recall metrics computed (abstention excluded from recall scoring).
    expect(report.turnRecall.scored).toBeGreaterThan(0);

    // Report file written to the per-run dir.
    const rp = resolve(RUNS_DIR, runId, "report.json");
    expect(existsSync(rp)).toBe(true);
  });

  it("resumes from a checkpoint without re-running completed questions", async () => {
    const runId = `test-resume-${Date.now()}`;
    createdRuns.push(runId);

    // First run on a 2-question subset.
    await runBenchmark({
      runId,
      judgeModel: NULL_MODEL,
      answerModel: NULL_MODEL,
      judgeModelName: "fake-judge",
      answerModelName: "fake-reader",
      aiKnotCommand: "unused",
      dataFile: SAMPLE,
      limit: 2,
      force: true,
      _evaluatorOverride: { ...fakeEval(), adapterFactory: fakeAdapterFactory },
    });

    const cpPath = resolve(RUNS_DIR, runId, "checkpoint.json");
    const firstResults = JSON.parse(readFileSync(cpPath, "utf-8")).results.length;
    expect(firstResults).toBe(2);

    // Resume on the full set (force=false) — should add the remaining 5, not redo 2.
    let ingestCount = 0;
    const countingAdapter = (q: LmeQuestion): AdapterLike => {
      const base = fakeAdapterFactory(q);
      return {
        async ingest(s) {
          ingestCount++;
          await base.ingest(s);
        },
        recall: base.recall,
        close: base.close,
      };
    };

    const report = await runBenchmark({
      runId,
      judgeModel: NULL_MODEL,
      answerModel: NULL_MODEL,
      judgeModelName: "fake-judge",
      answerModelName: "fake-reader",
      aiKnotCommand: "unused",
      dataFile: SAMPLE,
      force: false,
      _evaluatorOverride: { ...fakeEval(), adapterFactory: countingAdapter },
    });

    expect(report.summary.total).toBe(7);
    expect(ingestCount).toBe(5); // only the 5 pending questions were ingested
  });

  it("an answerable question that the reader declines is scored WRONG", async () => {
    const runId = `test-decline-${Date.now()}`;
    createdRuns.push(runId);

    // Build a fixture with one answerable question whose adapter returns empty.
    const tmp = mkdtempSync(resolve(tmpdir(), "lme-"));
    const file = resolve(tmp, "one.json");
    writeFileSync(
      file,
      JSON.stringify([
        {
          question_id: "answerable_001",
          question_type: "single-session-user",
          question: "What is my name?",
          answer: "Alex",
          haystack_session_ids: ["s1"],
          haystack_dates: ["2023-01-01"],
          haystack_sessions: [[{ role: "user", content: "My name is Alex", has_answer: true }]],
          answer_session_ids: ["s1"],
        },
      ])
    );

    const emptyAdapter = (): AdapterLike => ({
      async ingest() {},
      async recall() {
        return "No relevant facts found.";
      },
      async close() {},
    });

    const report = await runBenchmark({
      runId,
      judgeModel: NULL_MODEL,
      answerModel: NULL_MODEL,
      judgeModelName: "fake-judge",
      answerModelName: "fake-reader",
      aiKnotCommand: "unused",
      dataFile: file,
      force: true,
      _evaluatorOverride: { ...fakeEval(), adapterFactory: emptyAdapter },
    });

    rmSync(tmp, { recursive: true, force: true });
    expect(report.summary.total).toBe(1);
    expect(report.summary.correct).toBe(0); // declined on an answerable question => WRONG
  });
});
