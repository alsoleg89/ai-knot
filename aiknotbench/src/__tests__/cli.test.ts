import { describe, it, expect } from "vitest";
import { parseArgs } from "../index.js";
import { DEFAULT_JUDGE_MODEL, DEFAULT_ANSWER_MODEL } from "../models.js";

describe("parseArgs", () => {
  it("parses run with required -r flag", () => {
    const result = parseArgs(["run", "-r", "my-run"]);
    expect(result).toMatchObject({
      command: "run",
      runId: "my-run",
      judgeModel: DEFAULT_JUDGE_MODEL,
      answerModel: DEFAULT_ANSWER_MODEL,
      force: false,
    });
  });

  it("parses run with all options", () => {
    const result = parseArgs([
      "run", "-r", "full",
      "--judge", "claude-3-5-haiku-20241022",
      "--model", "gpt-4o",
      "--limit", "3",
      "--sample", "10",
      "--types", "1,2,3",
      "--force",
    ]);
    expect(result).toMatchObject({
      command: "run",
      runId: "full",
      judgeModel: "claude-3-5-haiku-20241022",
      answerModel: "gpt-4o",
      limit: 3,
      sample: 10,
      types: [1, 2, 3],
      force: true,
    });
  });

  it("parses list with -l limit", () => {
    const result = parseArgs(["list", "-l", "5"]);
    expect(result).toMatchObject({ command: "list", limit: 5 });
  });

  it("parses list with --limit", () => {
    const result = parseArgs(["list", "--limit", "10"]);
    expect(result).toMatchObject({ command: "list", limit: 10 });
  });

  it("returns help for unknown command", () => {
    const result = parseArgs(["help"]);
    expect(result.command).toBe("help");
  });

  it("returns help when no command given", () => {
    const result = parseArgs([]);
    expect(result.command).toBe("help");
  });

  it("parses --run-id as alias for -r", () => {
    const result = parseArgs(["run", "--run-id", "bench-1"]);
    if (result.command !== "run") throw new Error("expected run");
    expect(result.runId).toBe("bench-1");
  });
});
