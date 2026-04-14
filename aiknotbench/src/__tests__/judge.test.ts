import { describe, it, expect } from "vitest";
import { parseVerdict } from "../evaluator.js";

describe("parseVerdict", () => {
  it("parses JSON CORRECT", () => {
    expect(parseVerdict('{"verdict": "CORRECT"}')).toBe("CORRECT");
  });

  it("parses JSON WRONG", () => {
    expect(parseVerdict('{"verdict": "WRONG"}')).toBe("WRONG");
  });

  it("fallback: CORRECT in plain text", () => {
    expect(parseVerdict("The answer is CORRECT.")).toBe("CORRECT");
  });

  it("fallback: WRONG in plain text", () => {
    expect(parseVerdict("This is WRONG.")).toBe("WRONG");
  });

  it("case-insensitive fallback", () => {
    expect(parseVerdict("correct")).toBe("CORRECT");
    expect(parseVerdict("wrong")).toBe("WRONG");
  });

  it("unknown text defaults to WRONG", () => {
    expect(parseVerdict("I cannot determine the answer.")).toBe("WRONG");
    expect(parseVerdict("")).toBe("WRONG");
  });

  it("JSON with extra whitespace", () => {
    expect(parseVerdict('  { "verdict" : "CORRECT" }  ')).toBe("CORRECT");
  });

  it("CORRECT takes precedence when both words appear", () => {
    // First match wins — CORRECT appears before WRONG in the string
    expect(parseVerdict("CORRECT not WRONG")).toBe("CORRECT");
  });
});
