import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  filterQuestions,
  loadDataset,
  normalizeInstance,
  parseLmeDate,
  QUESTION_TYPES,
} from "../src/loader.js";

const SAMPLE = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "data",
  "sample_longmemeval.json"
);

describe("loader", () => {
  it("loads the bundled sample with all six question types + one abstention", () => {
    const qs = loadDataset({ dataFile: SAMPLE });
    expect(qs.length).toBe(7);
    const types = new Set(qs.map((q) => q.type));
    for (const t of QUESTION_TYPES) expect(types.has(t)).toBe(true);
    expect(qs.filter((q) => q.isAbstention).length).toBe(1);
  });

  it("detects the _abs suffix as abstention", () => {
    const q = normalizeInstance({
      question_id: "x_abs",
      question_type: "knowledge-update",
      question: "did I?",
      answer: "N/A",
      haystack_sessions: [],
    });
    expect(q?.isAbstention).toBe(true);
  });

  it("parses turns with has_answer flags and per-session dates", () => {
    const qs = loadDataset({ dataFile: SAMPLE });
    const ssu = qs.find((q) => q.id === "sample_ssu_001")!;
    expect(ssu.sessions[0]!.date).toBe("2023-05-08T13:56:00");
    const answerTurns = ssu.sessions.flatMap((s) => s.turns.filter((t) => t.hasAnswer));
    expect(answerTurns.length).toBe(1);
    expect(answerTurns[0]!.content).toContain("Python");
  });

  it("filters by question type", () => {
    const qs = loadDataset({ dataFile: SAMPLE });
    const ku = filterQuestions(qs, ["knowledge-update"], undefined);
    expect(ku.every((q) => q.type === "knowledge-update")).toBe(true);
    expect(ku.length).toBe(2); // one KU + one KU _abs
  });

  it("respects limit", () => {
    const qs = loadDataset({ dataFile: SAMPLE, limit: 3 });
    expect(qs.length).toBe(3);
  });

  it("throws a helpful error when the data file is missing", () => {
    expect(() => loadDataset({ dataFile: "/nonexistent/lme.json" })).toThrow(/not found/);
  });
});

describe("parseLmeDate", () => {
  it("passes through ISO-8601 datetimes", () => {
    expect(parseLmeDate("2023-05-08T13:56:00")).toBe("2023-05-08T13:56:00");
  });
  it("passes through ISO dates", () => {
    expect(parseLmeDate("2023-05-08")).toBe("2023-05-08");
  });
  it("normalises slash dates with weekday + time", () => {
    expect(parseLmeDate("2023/05/08 (Mon) 13:56")).toBe("2023-05-08T13:56:00");
  });
  it("normalises bare slash dates", () => {
    expect(parseLmeDate("2023/5/8")).toBe("2023-05-08");
  });
  it("returns undefined for unparseable / empty input", () => {
    expect(parseLmeDate("")).toBeUndefined();
    expect(parseLmeDate(undefined)).toBeUndefined();
    expect(parseLmeDate("sometime last year")).toBeUndefined();
  });
});
