/**
 * Integration test: verifies that the runner writes diagnostics_raw.jsonl
 * when AI_KNOT_DIAG=1 and that the output is valid JSONL.
 *
 * Uses the injectable EvaluatorFns interface so no real MCP subprocess is spawned.
 */

import { describe, it, expect, afterEach } from "vitest";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";
import { runBenchmark, type AiknotAdapterLike } from "../src/runner.js";
import type { Session } from "../src/locomo.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BENCH_ROOT = resolve(__dirname, "..");
const TEST_RUN_ID = "__test_trace_capture__";
const TEST_RUN_DIR = resolve(BENCH_ROOT, "data", "runs", TEST_RUN_ID);

// ---- Minimal adapter mock ---------------------------------------------------

class MockAdapter implements AiknotAdapterLike {
  private callCount = 0;

  async ingest(_turns: string[], _sessions?: Session[]): Promise<void> {}

  async recall(_question: string): Promise<string> {
    return "mock context";
  }

  async recallWithTrace(_question: string): Promise<{
    context: string;
    packFactIds: string[];
    trace: Record<string, unknown>;
  }> {
    this.callCount++;
    return {
      context: "mock context with trace",
      packFactIds: [`fact_${this.callCount}a`, `fact_${this.callCount}b`],
      trace: {
        stage1_candidates: {
          from_bm25: [`fact_${this.callCount}a`, `fact_${this.callCount}b`, `fact_${this.callCount}c`],
          from_rare_tokens: [],
          from_entity_hop: [],
        },
        stage3_rrf: { ranked_ids: [`fact_${this.callCount}a`, `fact_${this.callCount}b`] },
        stage0_lexical_bridge: null,
      },
    };
  }

  async close(): Promise<void> {}
}

// ---- Tests ------------------------------------------------------------------

afterEach(() => {
  if (existsSync(TEST_RUN_DIR)) {
    rmSync(TEST_RUN_DIR, { recursive: true, force: true });
  }
  delete process.env["AI_KNOT_DIAG"];
});

describe("runner trace capture", () => {
  it("writes diagnostics_raw.jsonl when AI_KNOT_DIAG=1", async () => {
    process.env["AI_KNOT_DIAG"] = "1";

    const mockModel = {} as Parameters<typeof runBenchmark>[0]["judgeModel"];

    await runBenchmark({
      runId: TEST_RUN_ID,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock-judge",
      answerModelName: "mock-answer",
      aiKnotCommand: "mock-command",
      locomoFile: resolve(BENCH_ROOT, "data", "locomo10.json"),
      limit: 1,   // only first conversation
      convs: [0],
      types: [1],
      sample: 2,  // at most 2 QA pairs
      ingestMode: "dated",
      force: true,
      _evaluatorOverride: {
        answerFn: async (_m, _ctx, _q) => "mock answer",
        judgeFn: async (_m, _q, _a, _g) => "WRONG" as const,
        adapterFactory: () => new MockAdapter(),
      },
    });

    const diagPath = resolve(TEST_RUN_DIR, "diagnostics_raw.jsonl");
    expect(existsSync(diagPath), "diagnostics_raw.jsonl should exist").toBe(true);

    const lines = readFileSync(diagPath, "utf-8")
      .split("\n")
      .filter(Boolean);

    expect(lines.length).toBeGreaterThan(0);

    // Each line should be valid JSON with required fields
    for (const line of lines) {
      const entry = JSON.parse(line) as Record<string, unknown>;
      expect(entry).toHaveProperty("run_id", TEST_RUN_ID);
      expect(entry).toHaveProperty("conv_idx");
      expect(entry).toHaveProperty("qa_idx");
      expect(entry).toHaveProperty("pack_fact_ids");
      expect(entry).toHaveProperty("stage1_candidates");

      const stage1 = entry["stage1_candidates"] as Record<string, unknown>;
      expect(stage1).toHaveProperty("from_bm25");
      expect(stage1).toHaveProperty("from_rare_tokens");
      expect(stage1).toHaveProperty("from_entity_hop");

      const packIds = entry["pack_fact_ids"] as string[];
      expect(Array.isArray(packIds)).toBe(true);
    }
  });

  it("does NOT write diagnostics_raw.jsonl when AI_KNOT_DIAG is not set", async () => {
    delete process.env["AI_KNOT_DIAG"];

    const mockModel = {} as Parameters<typeof runBenchmark>[0]["judgeModel"];

    await runBenchmark({
      runId: TEST_RUN_ID,
      judgeModel: mockModel,
      answerModel: mockModel,
      judgeModelName: "mock-judge",
      answerModelName: "mock-answer",
      aiKnotCommand: "mock-command",
      locomoFile: resolve(BENCH_ROOT, "data", "locomo10.json"),
      limit: 1,
      convs: [0],
      types: [1],
      sample: 1,
      ingestMode: "dated",
      force: true,
      _evaluatorOverride: {
        answerFn: async (_m, _ctx, _q) => "mock answer",
        judgeFn: async (_m, _q, _a, _g) => "CORRECT" as const,
        adapterFactory: () => new MockAdapter(),
      },
    });

    const diagPath = resolve(TEST_RUN_DIR, "diagnostics_raw.jsonl");
    expect(existsSync(diagPath)).toBe(false);
  });
});
