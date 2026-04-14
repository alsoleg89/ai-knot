import { describe, it, expect } from "vitest";
import { normalizeConversation, filterQA } from "../locomo.js";

const FIXTURE = {
  conversation: {
    session_1: [
      { speaker: "Alice", text: "I love hiking.", dia_id: "d1" },
      { speaker: "Bob", text: "Me too.", dia_id: "d2" },
    ],
    session_2: [
      { speaker: "Alice", text: "I went to Yosemite last week.", dia_id: "d3" },
    ],
    session_1_date_time: "2024-01-01",
  },
  qa: [
    { question: "Does Alice like hiking?", answer: "Yes", category: 1 },
    { question: "Where did Alice go?", answer: "Yosemite", category: 2 },
    { question: "What is 2+2?", answer: "4", category: 3 },
    { question: "Describe the conversations.", answer: "Hiking and travel.", category: 4 },
    {
      question: "Does Alice dislike hiking?",
      answer: "No",
      adversarial_answer: "Yes",
      category: 5,
    },
    // pair with no answer — should be filtered out
    { question: "Empty?", answer: "", category: 1 },
  ],
};

describe("normalizeConversation", () => {
  it("flattens sessions in order", () => {
    const conv = normalizeConversation(FIXTURE, 0);
    expect(conv.turns).toEqual([
      "Alice: I love hiking.",
      "Bob: Me too.",
      "Alice: I went to Yosemite last week.",
    ]);
  });

  it("normalizes QA pairs and assigns idx", () => {
    const conv = normalizeConversation(FIXTURE, 0);
    const cat1 = conv.qa.find((q) => q.idx === 0);
    expect(cat1).toBeDefined();
    expect(cat1!.answer).toBe("Yes");
    expect(cat1!.category).toBe(1);
  });

  it("uses adversarial_answer for category 5", () => {
    const conv = normalizeConversation(FIXTURE, 0);
    const cat5 = conv.qa.find((q) => q.category === 5);
    expect(cat5).toBeDefined();
    expect(cat5!.answer).toBe("Yes"); // adversarial_answer
  });

  it("drops QA pairs with empty answer", () => {
    const conv = normalizeConversation(FIXTURE, 0);
    // idx 5 has empty answer and should not appear
    expect(conv.qa.some((q) => q.idx === 5)).toBe(false);
  });

  it("assigns conversation idx", () => {
    const conv = normalizeConversation(FIXTURE, 3);
    expect(conv.idx).toBe(3);
  });
});

describe("filterQA", () => {
  const qa = [
    { idx: 0, question: "q0", answer: "a", category: 1 },
    { idx: 1, question: "q1", answer: "b", category: 2 },
    { idx: 2, question: "q2", answer: "c", category: 3 },
    { idx: 3, question: "q3", answer: "d", category: 4 },
    { idx: 4, question: "q4", answer: "e", category: 1 },
  ];

  it("returns all when no filter", () => {
    expect(filterQA(qa, undefined, undefined)).toHaveLength(5);
  });

  it("filters by types", () => {
    const result = filterQA(qa, [1, 2], undefined);
    expect(result.map((q) => q.idx)).toEqual([0, 1, 4]);
  });

  it("applies sample limit", () => {
    const result = filterQA(qa, [1], 1);
    expect(result).toHaveLength(1);
    expect(result[0]!.idx).toBe(0);
  });

  it("sample larger than available returns all", () => {
    const result = filterQA(qa, [3], 10);
    expect(result).toHaveLength(1);
  });
});
