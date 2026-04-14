import { describe, it, expect } from "vitest";
import { computeReport } from "../runner.js";

const RESULTS = [
  { convIdx: 0, qaIdx: 0, category: 1, verdict: "CORRECT" as const },
  { convIdx: 0, qaIdx: 1, category: 1, verdict: "WRONG" as const },
  { convIdx: 0, qaIdx: 2, category: 2, verdict: "CORRECT" as const },
  { convIdx: 1, qaIdx: 0, category: 3, verdict: "CORRECT" as const },
  { convIdx: 1, qaIdx: 1, category: 4, verdict: "WRONG" as const },
  { convIdx: 1, qaIdx: 2, category: 5, verdict: "CORRECT" as const },
];

describe("computeReport", () => {
  const report = computeReport("test-run", "gpt-4o-mini", "gpt-4o", RESULTS);

  it("summary totals are correct", () => {
    expect(report.summary.total).toBe(6);
    expect(report.summary.correct).toBe(4);
    expect(report.summary.accuracy).toBeCloseTo(4 / 6);
  });

  it("byType has correct counts", () => {
    expect(report.byType["1"]!.total).toBe(2);
    expect(report.byType["1"]!.correct).toBe(1);
    expect(report.byType["2"]!.total).toBe(1);
    expect(report.byType["2"]!.correct).toBe(1);
    expect(report.byType["5"]!.total).toBe(1);
    expect(report.byType["5"]!.correct).toBe(1);
  });

  it("categories1to4 excludes category 5", () => {
    // cat 1: 1/2, cat 2: 1/1, cat 3: 1/1, cat 4: 0/1 → 3/5
    expect(report.categories1to4.total).toBe(5);
    expect(report.categories1to4.correct).toBe(3);
    expect(report.categories1to4.accuracy).toBeCloseTo(3 / 5);
  });

  it("accuracy is 0 for empty results", () => {
    const empty = computeReport("x", "m", "m", []);
    expect(empty.summary.accuracy).toBe(0);
    expect(empty.categories1to4.accuracy).toBe(0);
  });

  it("embeds run metadata", () => {
    expect(report.runId).toBe("test-run");
    expect(report.judgeModel).toBe("gpt-4o-mini");
    expect(report.answerModel).toBe("gpt-4o");
    expect(report.finishedAt).toBeTruthy();
  });
});
