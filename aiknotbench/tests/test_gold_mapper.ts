import { describe, it, expect } from "vitest";
import {
  buildConvGoldMap,
  collectEvidenceIds,
  validateGoldMapCoverage,
  type EvidenceToFactIds,
} from "../scripts/build_gold_mapper.js";

// ---- Fixtures ---------------------------------------------------------------

const TURN_A = "Alice: I went to the park";
const TURN_B = "Bob: Sounds great";
const TURN_C = "Alice: It was sunny";

const TURN_MAP = new Map([
  ["D1:1", TURN_A],
  ["D1:2", TURN_B],
  ["D1:3", TURN_C],
  ["D2:1", "Carol: Hello"],
]);

const STORED_FACTS_RAW = [
  { id: "fact0001", content: TURN_A },
  { id: "fact0002", content: TURN_B },
  { id: "fact0003", content: TURN_C },
];

// In dated mode, a 3-turn sliding window produces:
// window centered at turn1: turns[0] alone → "Alice: I went to the park"
// window centered at turn2: turns[0..1] → "Alice: I went to the park / Bob: Sounds great"
// window centered at turn3: turns[0..2] or turns[1..2] etc. depending on window position
const DATED_WINDOW_1 = "[2023-05-01] " + TURN_A;
const DATED_WINDOW_2 = "[2023-05-01] " + TURN_A + " / " + TURN_B;
const DATED_WINDOW_3 = "[2023-05-01] " + TURN_B + " / " + TURN_C;

const STORED_FACTS_DATED = [
  { id: "fact_w1", content: DATED_WINDOW_1 },
  { id: "fact_w2", content: DATED_WINDOW_2 },
  { id: "fact_w3", content: DATED_WINDOW_3 },
];

// ---- Tests ------------------------------------------------------------------

describe("collectEvidenceIds", () => {
  it("collects unique IDs from multiple QA pairs", () => {
    const qa = [
      { evidence: ["D1:3"] },
      { evidence: ["D1:9", "D1:11"] },
      { evidence: ["D1:3"] }, // duplicate
    ];
    const ids = collectEvidenceIds(qa);
    expect(ids).toHaveLength(3);
    expect(ids).toContain("D1:3");
    expect(ids).toContain("D1:9");
    expect(ids).toContain("D1:11");
  });

  it("returns empty array for QA with no evidence", () => {
    expect(collectEvidenceIds([{ evidence: [] }, {}])).toHaveLength(0);
  });
});

describe("buildConvGoldMap (raw mode)", () => {
  it("maps D1:1 to exact-match fact in raw mode", () => {
    const result = buildConvGoldMap(
      ["D1:1", "D1:2", "D1:3"],
      TURN_MAP,
      STORED_FACTS_RAW,
      "raw",
    );
    expect(result["D1:1"]).toEqual(["fact0001"]);
    expect(result["D1:2"]).toEqual(["fact0002"]);
    expect(result["D1:3"]).toEqual(["fact0003"]);
  });

  it("returns empty array for evidence with no matching fact", () => {
    const result = buildConvGoldMap(
      ["D2:1"],
      new Map([["D2:1", "Carol: Hello"]]),
      STORED_FACTS_RAW, // doesn't contain Carol's turn
      "raw",
    );
    expect(result["D2:1"]).toEqual([]);
  });

  it("returns empty array for unknown evidence ID", () => {
    const result = buildConvGoldMap(
      ["D9:99"],
      TURN_MAP, // D9:99 not in map
      STORED_FACTS_RAW,
      "raw",
    );
    expect(result["D9:99"]).toEqual([]);
  });
});

describe("buildConvGoldMap (dated mode — sliding window inflation)", () => {
  it("maps one gold turn to up to 3 sliding-window facts", () => {
    // TURN_A appears in fact_w1 and fact_w2 (2 windows)
    const result = buildConvGoldMap(
      ["D1:1"],
      TURN_MAP,
      STORED_FACTS_DATED,
      "dated",
    );
    expect(result["D1:1"]).toContain("fact_w1");
    expect(result["D1:1"]).toContain("fact_w2");
    // turn_a does NOT appear in fact_w3 (which has turn_b + turn_c)
    expect(result["D1:1"]).not.toContain("fact_w3");
  });

  it("TURN_B appears in windows 2 and 3", () => {
    const result = buildConvGoldMap(
      ["D1:2"],
      TURN_MAP,
      STORED_FACTS_DATED,
      "dated",
    );
    expect(result["D1:2"]).toContain("fact_w2");
    expect(result["D1:2"]).toContain("fact_w3");
    expect(result["D1:2"]).not.toContain("fact_w1");
  });

  it("returns empty fact list when no window contains the turn text", () => {
    const result = buildConvGoldMap(
      ["D2:1"],
      new Map([["D2:1", "Carol: Hello"]]),
      STORED_FACTS_DATED,
      "dated",
    );
    expect(result["D2:1"]).toEqual([]);
  });
});

describe("validateGoldMapCoverage", () => {
  it("returns correct covered count and empty missing list", () => {
    const map: EvidenceToFactIds = {
      "D1:1": ["fact0001"],
      "D1:2": ["fact0002"],
    };
    const { covered, missing } = validateGoldMapCoverage(map);
    expect(covered).toBe(2);
    expect(missing).toHaveLength(0);
  });

  it("identifies missing evidence IDs", () => {
    const map: EvidenceToFactIds = {
      "D1:1": ["fact0001"],
      "D1:2": [],
      "D1:3": [],
    };
    const { covered, missing } = validateGoldMapCoverage(map);
    expect(covered).toBe(1);
    expect(missing).toContain("D1:2");
    expect(missing).toContain("D1:3");
  });
});

describe("fingerprint stability", () => {
  it("same turn text maps to same fact IDs on repeated calls (deterministic)", () => {
    const r1 = buildConvGoldMap(
      ["D1:1"],
      TURN_MAP,
      STORED_FACTS_RAW,
      "raw",
    );
    const r2 = buildConvGoldMap(
      ["D1:1"],
      TURN_MAP,
      STORED_FACTS_RAW,
      "raw",
    );
    expect(r1["D1:1"]).toEqual(r2["D1:1"]);
  });
});
