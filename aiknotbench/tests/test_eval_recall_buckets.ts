import { describe, it, expect } from "vitest";
import {
  buildBucketTable,
  computeBucketMigrations,
  renderBucketTable,
  renderMigrationTable,
  type BucketMigration,
} from "../scripts/eval_recall_buckets.js";
import type { DiagnosticsRecord } from "../scripts/diagnose_recall.js";

// ---- Fixtures ---------------------------------------------------------------

function makeRecord(
  overrides: Partial<DiagnosticsRecord> & {
    verdict?: string;
    bucket?: DiagnosticsRecord["bucket"];
  } = {},
): DiagnosticsRecord {
  return {
    run_id: "test-run",
    conv_id: "conv0",
    qa_idx: 0,
    category: 1,
    query: "q",
    gold_evidence: [],
    gold_fact_ids: ["f1"],
    gold_fact_ids_per_evidence: {},
    raw_gold_exists: true,
    pool_gold_recall: 0.8,
    pack_gold_recall: 1.0,
    gold_pack_position_median: 0,
    distractor_density: 0.5,
    reader_fail_despite_gold: true,
    lexical_expansion_uplift: null,
    answer_verdict: overrides.verdict ?? "WRONG",
    bucket: overrides.bucket ?? "LLM-fail",
    ...overrides,
  };
}

// ---- Tests ------------------------------------------------------------------

describe("buildBucketTable", () => {
  it("counts each bucket-type correctly", () => {
    const records: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, bucket: "LLM-fail", verdict: "WRONG" }),
      makeRecord({ qa_idx: 1, bucket: "hard-miss", verdict: "WRONG" }),
      makeRecord({ qa_idx: 2, bucket: "hard-miss", verdict: "WRONG" }),
      makeRecord({ qa_idx: 3, bucket: "correct", verdict: "CORRECT" }),
    ];

    const table = buildBucketTable("r", records);
    expect(table.totals["LLM-fail"]).toBe(1);
    expect(table.totals["hard-miss"]).toBe(2);
    // buildBucketTable only counts wrong answers; correct records are excluded from totals
    expect(table.totals["correct"]).toBeUndefined();
  });

  it("groups by category correctly", () => {
    const records: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, category: 1, bucket: "LLM-fail", verdict: "WRONG" }),
      makeRecord({ qa_idx: 1, category: 2, bucket: "low-recall", verdict: "WRONG" }),
    ];
    const table = buildBucketTable("r", records);
    expect(table.byCategory["1"]?.["LLM-fail"]).toBe(1);
    expect(table.byCategory["2"]?.["low-recall"]).toBe(1);
    expect(table.byCategory["1"]?.["low-recall"]).toBe(0);
  });

  it("correct answers are excluded from per-category wrong counts", () => {
    const records: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, category: 1, bucket: "correct", answer_verdict: "CORRECT" }),
    ];
    const table = buildBucketTable("r", records);
    // byCategory for cat1 should have zero wrong
    if (table.byCategory["1"]) {
      const total = Object.values(table.byCategory["1"]).reduce((a, b) => a + b, 0);
      expect(total).toBe(0);
    }
  });
});

describe("computeBucketMigrations", () => {
  it("detects bucket changes between baseline and candidate", () => {
    const base: DiagnosticsRecord[] = [
      makeRecord({ conv_id: "conv0", qa_idx: 0, bucket: "hard-miss", answer_verdict: "WRONG" }),
      makeRecord({ conv_id: "conv0", qa_idx: 1, bucket: "LLM-fail", answer_verdict: "WRONG" }),
    ];
    const cand: DiagnosticsRecord[] = [
      makeRecord({ conv_id: "conv0", qa_idx: 0, bucket: "LLM-fail", answer_verdict: "WRONG" }),
      makeRecord({ conv_id: "conv0", qa_idx: 1, bucket: "LLM-fail", answer_verdict: "WRONG" }),
    ];
    const migrations = computeBucketMigrations(base, cand);
    expect(migrations).toHaveLength(1);
    expect(migrations[0]!.fromBucket).toBe("hard-miss");
    expect(migrations[0]!.toBucket).toBe("LLM-fail");
  });

  it("returns empty list when no buckets changed", () => {
    const records: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, bucket: "hard-miss" }),
    ];
    const migrations = computeBucketMigrations(records, records);
    expect(migrations).toHaveLength(0);
  });

  it("classifies regression (CORRECT → WRONG) correctly", () => {
    const base: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, bucket: "correct", answer_verdict: "CORRECT" }),
    ];
    const cand: DiagnosticsRecord[] = [
      makeRecord({ qa_idx: 0, bucket: "LLM-fail", answer_verdict: "WRONG" }),
    ];
    const migrations = computeBucketMigrations(base, cand);
    expect(migrations[0]!.verdictChange).toBe("CORRECT → WRONG");
  });
});

describe("renderBucketTable", () => {
  it("renders markdown with all bucket rows", () => {
    const records: DiagnosticsRecord[] = [
      makeRecord({ bucket: "LLM-fail", answer_verdict: "WRONG" }),
      makeRecord({ qa_idx: 1, bucket: "hard-miss", answer_verdict: "WRONG" }),
    ];
    const table = buildBucketTable("test", records);
    const md = renderBucketTable(table);
    expect(md).toContain("LLM-fail");
    expect(md).toContain("hard-miss");
    expect(md).toContain("partial-recall");
    expect(md).toContain("low-recall");
  });
});

describe("renderMigrationTable", () => {
  it("shows total questions moved", () => {
    const migrations: BucketMigration[] = [
      {
        convId: "conv0",
        qaIdx: 0,
        category: 1,
        fromBucket: "hard-miss",
        toBucket: "LLM-fail",
        verdictChange: "WRONG → WRONG",
      },
    ];
    const md = renderMigrationTable("base", "cand", migrations);
    expect(md).toContain("Total questions moved: 1");
    expect(md).toContain("hard-miss");
    expect(md).toContain("LLM-fail");
  });
});
