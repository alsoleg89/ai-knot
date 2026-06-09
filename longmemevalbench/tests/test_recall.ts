import { describe, expect, it } from "vitest";

import type { LmeQuestion } from "../src/loader.js";
import { scoreRecall } from "../src/recall.js";

function q(overrides: Partial<LmeQuestion>): LmeQuestion {
  return {
    id: "x",
    type: "single-session-user",
    question: "?",
    answer: "a",
    sessions: [],
    answerSessionIds: [],
    isAbstention: false,
    ...overrides,
  };
}

describe("scoreRecall", () => {
  it("scores a turn-level hit when the answer turn content appears in context", () => {
    const question = q({
      sessions: [
        {
          id: "s1",
          turns: [
            { role: "user", content: "I adopted a golden retriever puppy named Max", hasAnswer: true },
            { role: "assistant", content: "wonderful", hasAnswer: false },
          ],
        },
      ],
      answerSessionIds: ["s1"],
    });
    const context = "[1] user: I adopted a golden retriever puppy named Max / assistant: wonderful";
    const score = scoreRecall(question, context);
    expect(score.turnHit).toBe(true);
    expect(score.sessionHit).toBe(true);
  });

  it("scores a miss when the evidence is absent from context", () => {
    const question = q({
      sessions: [
        {
          id: "s1",
          turns: [{ role: "user", content: "I adopted a golden retriever puppy", hasAnswer: true }],
        },
      ],
      answerSessionIds: ["s1"],
    });
    const score = scoreRecall(question, "[1] user: the weather is nice today");
    expect(score.turnHit).toBe(false);
    expect(score.sessionHit).toBe(false);
  });

  it("excludes abstention questions from recall scoring", () => {
    const question = q({
      isAbstention: true,
      sessions: [{ id: "s1", turns: [{ role: "user", content: "anything", hasAnswer: true }] }],
      answerSessionIds: ["s1"],
    });
    const score = scoreRecall(question, "[1] user: anything");
    expect(score.turnHit).toBeNull();
    expect(score.sessionHit).toBeNull();
  });

  it("returns null turnHit when no turn carries has_answer", () => {
    const question = q({
      sessions: [{ id: "s1", turns: [{ role: "user", content: "no flag here", hasAnswer: false }] }],
      answerSessionIds: ["s1"],
    });
    const score = scoreRecall(question, "[1] user: no flag here");
    expect(score.turnHit).toBeNull();
    expect(score.sessionHit).toBe(true);
  });
});
