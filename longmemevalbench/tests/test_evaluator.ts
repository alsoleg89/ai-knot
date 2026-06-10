import { describe, expect, it } from "vitest";

import {
  IDK,
  answerQuestion,
  isAbstentionAnswer,
  parseVerdict,
} from "../src/evaluator.js";

// A model stub that records the system prompt it received and echoes a canned answer.
function stubModel(captured: { system?: string }, reply = "PostgreSQL") {
  return {
    async doGenerate(opts: { prompt: { role: string; content: unknown }[] }) {
      const sys = opts.prompt.find((m) => m.role === "system");
      if (sys && typeof sys.content === "string") captured.system = sys.content;
      return {
        text: reply,
        usage: { promptTokens: 1, completionTokens: 1 },
        finishReason: "stop" as const,
        rawCall: { rawPrompt: null, rawSettings: {} },
      };
    },
    specificationVersion: "v1" as const,
    provider: "stub",
    modelId: "stub",
    defaultObjectGenerationMode: undefined,
  };
}

describe("abstention reader contract (prerequisite C)", () => {
  it("short-circuits to a deterministic IDK on an empty pool — no LLM call", async () => {
    const captured: { system?: string } = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const res = await answerQuestion(stubModel(captured) as any, "No relevant facts found.", "Q?", {
      idkContract: true,
    });
    expect(res.text).toBe(IDK);
    expect(res.shortCircuited).toBe(true);
    expect(captured.system).toBeUndefined(); // proves the model was never invoked
  });

  it("uses the IDK system prompt when the contract is on and context is non-empty", async () => {
    const captured: { system?: string } = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await answerQuestion(stubModel(captured) as any, "user: I like PostgreSQL", "Q?", {
      idkContract: true,
    });
    expect(captured.system).toContain(IDK);
    expect(captured.system).toContain("MOST RECENT"); // KU-aware
  });

  it("uses the plain (LOCOMO-style) system prompt when the contract is off", async () => {
    const captured: { system?: string } = {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await answerQuestion(stubModel(captured) as any, "user: I like PostgreSQL", "Q?", {
      idkContract: false,
    });
    expect(captured.system).toBe("Answer the question based on the memory context below. Answer concisely.");
  });
});

describe("isAbstentionAnswer", () => {
  it("recognises common refusals", () => {
    expect(isAbstentionAnswer("I don't know.")).toBe(true);
    expect(isAbstentionAnswer("There is no information about that.")).toBe(true);
    expect(isAbstentionAnswer("That is not mentioned in the context.")).toBe(true);
    expect(isAbstentionAnswer("")).toBe(true);
  });
  it("treats a real answer as non-abstention", () => {
    expect(isAbstentionAnswer("You work at Globex.")).toBe(false);
    expect(isAbstentionAnswer("PostgreSQL")).toBe(false);
  });
});

describe("parseVerdict", () => {
  it("parses JSON verdicts", () => {
    expect(parseVerdict('{"verdict": "CORRECT"}')).toBe("CORRECT");
    expect(parseVerdict('{"verdict": "WRONG"}')).toBe("WRONG");
  });
  it("falls back to regex then WRONG", () => {
    expect(parseVerdict("The answer is CORRECT")).toBe("CORRECT");
    expect(parseVerdict("gibberish")).toBe("WRONG");
  });
});
