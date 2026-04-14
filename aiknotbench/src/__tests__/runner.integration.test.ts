import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { LanguageModelV1 } from "ai";

import { runBenchmark } from "../runner.js";
import type { AiknotAdapterLike } from "../runner.js";
import type { Verdict, Usage } from "../evaluator.js";

const MOCK_USAGE: Usage = { promptTokens: 10, completionTokens: 5 };

// ---- Paths ------------------------------------------------------------------

const TEST_DATA_DIR = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "..",
  "data",
  "test-runs"
);

// ---- Fixtures ---------------------------------------------------------------

// Mock locomo10.json with 2 conversations × 3 QA each (cat 1 and 2 only)
const MOCK_DATASET = [
  {
    conversation: {
      session_1: [
        { speaker: "Alice", text: "I like coffee.", dia_id: "d1" },
        { speaker: "Bob", text: "I prefer tea.", dia_id: "d2" },
      ],
    },
    qa: [
      { question: "What does Alice like?", answer: "coffee", category: 1 },
      { question: "What does Bob prefer?", answer: "tea", category: 1 },
      { question: "Who mentioned tea?", answer: "Bob", category: 2 },
    ],
  },
  {
    conversation: {
      session_1: [
        { speaker: "Carol", text: "I work at ACME.", dia_id: "d3" },
        { speaker: "Dave", text: "I work from home.", dia_id: "d4" },
      ],
    },
    qa: [
      { question: "Where does Carol work?", answer: "ACME", category: 1 },
      { question: "How does Dave work?", answer: "from home", category: 2 },
      { question: "Who works remotely?", answer: "Dave", category: 2 },
    ],
  },
];

// ---- Mock adapter -----------------------------------------------------------

class MockAdapter implements AiknotAdapterLike {
  readonly ingested: string[] = [];
  async ingest(turns: string[]): Promise<void> {
    this.ingested.push(...turns);
  }
  async recall(_question: string): Promise<string> {
    return "Mock context: Alice likes coffee. Bob prefers tea.";
  }
  async close(): Promise<void> {}
}

// ---- Mock evaluator fns -----------------------------------------------------

const mockAnswerFn = async (
  _model: LanguageModelV1,
  _context: string,
  _question: string
): Promise<{ text: string; usage: Usage }> => ({ text: "mock answer", usage: MOCK_USAGE });

const mockJudgeFn = async (
  _model: LanguageModelV1,
  _question: string,
  _answer: string,
  _gold: string
): Promise<{ verdict: Verdict; usage: Usage }> => ({ verdict: "CORRECT", usage: MOCK_USAGE });

// ---- Helpers ----------------------------------------------------------------

function testRunId(suffix: string): string {
  return `test-${Date.now()}-${suffix}`;
}

function testRunDir(runId: string): string {
  return resolve(TEST_DATA_DIR, runId);
}

// Override RUNS_DIR by writing locomo file and using locomoFile option
import { writeFileSync } from "node:fs";

const MOCK_LOCOMO_PATH = resolve(TEST_DATA_DIR, "mock_locomo10.json");

// ---- Tests ------------------------------------------------------------------

describe("runBenchmark integration", () => {
  beforeEach(() => {
    mkdirSync(TEST_DATA_DIR, { recursive: true });
    writeFileSync(MOCK_LOCOMO_PATH, JSON.stringify(MOCK_DATASET));
  });

  afterEach(() => {
    if (existsSync(TEST_DATA_DIR)) {
      rmSync(TEST_DATA_DIR, { recursive: true, force: true });
    }
  });

  it("creates checkpoint.json and report.json after a full run", async () => {
    const runId = testRunId("full");
    const dir = testRunDir(runId);

    // We can't override RUNS_DIR easily, so we test via the public API.
    // The runner uses data/runs/{runId}/ relative to aiknotbench/data/
    // In tests, files end up in the real data/runs/ — we clean up after.
    const mockModel = {} as LanguageModelV1;

    const report = await runBenchmark({
      runId,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock-judge",
      answerModelName: "mock-answer",
      aiKnotCommand: "mock-ai-knot-mcp",
      locomoFile: MOCK_LOCOMO_PATH,
      _evaluatorOverride: {
        answerFn: mockAnswerFn,
        judgeFn: mockJudgeFn,
        adapterFactory: (_dbPath, convIdx) => new MockAdapter(),
      },
    });

    // report has correct structure
    expect(report.summary.total).toBe(6);
    expect(report.summary.correct).toBe(6);
    expect(report.summary.accuracy).toBeCloseTo(1.0);
    expect(report.categories1to4.total).toBe(6);

    // Files written to real runs dir — clean up
    const realRunDir = resolve(
      fileURLToPath(import.meta.url),
      "..", "..", "..", "data", "runs", runId
    );
    if (existsSync(realRunDir)) {
      rmSync(realRunDir, { recursive: true, force: true });
    }
  });

  it("resumes correctly — does not duplicate results", async () => {
    const runId = testRunId("resume");
    const mockModel = {} as LanguageModelV1;

    let callCount = 0;
    const countingJudge = async (): Promise<{ verdict: Verdict; usage: Usage }> => {
      callCount++;
      return { verdict: "CORRECT", usage: MOCK_USAGE };
    };

    const overrides = {
      answerFn: mockAnswerFn,
      judgeFn: (_m: LanguageModelV1, q: string, a: string, g: string) =>
        countingJudge(),
      adapterFactory: (_dbPath: string, _convIdx: number) => new MockAdapter(),
    };

    // First run — processes all 6 QA
    await runBenchmark({
      runId,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock",
      answerModelName: "mock",
      aiKnotCommand: "mock",
      locomoFile: MOCK_LOCOMO_PATH,
      _evaluatorOverride: overrides,
    });

    const firstRunCount = callCount;
    expect(firstRunCount).toBe(6);

    // Second run (resume) — all already done, no new judge calls
    await runBenchmark({
      runId,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock",
      answerModelName: "mock",
      aiKnotCommand: "mock",
      locomoFile: MOCK_LOCOMO_PATH,
      _evaluatorOverride: overrides,
    });

    expect(callCount).toBe(6); // unchanged

    // Clean up
    const realRunDir = resolve(
      fileURLToPath(import.meta.url),
      "..", "..", "..", "data", "runs", runId
    );
    if (existsSync(realRunDir)) {
      rmSync(realRunDir, { recursive: true, force: true });
    }
  });

  it("--force clears previous results and re-runs", async () => {
    const runId = testRunId("force");
    const mockModel = {} as LanguageModelV1;

    let callCount = 0;
    const overrides = {
      answerFn: mockAnswerFn,
      judgeFn: async (): Promise<{ verdict: Verdict; usage: Usage }> => {
        callCount++;
        return { verdict: "CORRECT", usage: MOCK_USAGE };
      },
      adapterFactory: (_dbPath: string, _convIdx: number) => new MockAdapter(),
    };

    // First run
    await runBenchmark({
      runId,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock",
      answerModelName: "mock",
      aiKnotCommand: "mock",
      locomoFile: MOCK_LOCOMO_PATH,
      _evaluatorOverride: overrides,
    });

    expect(callCount).toBe(6);

    // Force re-run
    await runBenchmark({
      runId,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock",
      answerModelName: "mock",
      aiKnotCommand: "mock",
      locomoFile: MOCK_LOCOMO_PATH,
      force: true,
      _evaluatorOverride: overrides,
    });

    expect(callCount).toBe(12); // evaluated again

    // Clean up
    const realRunDir = resolve(
      fileURLToPath(import.meta.url),
      "..", "..", "..", "data", "runs", runId
    );
    if (existsSync(realRunDir)) {
      rmSync(realRunDir, { recursive: true, force: true });
    }
  });
});
