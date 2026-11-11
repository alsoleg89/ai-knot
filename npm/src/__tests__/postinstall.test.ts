import { afterEach, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const script = fileURLToPath(new URL("../../scripts/postinstall.js", import.meta.url));
const { version } = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8"));
const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

function run(extraEnv: Record<string, string> = {}): string {
  const directory = mkdtempSync(join(tmpdir(), "ai-knot-postinstall-"));
  directories.push(directory);
  const log = join(directory, "pip-args");
  writeFileSync(join(directory, "pip3"), '#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_PIP_LOG"\n', { mode: 0o755 });
  execFileSync(process.execPath, [script], { env: { PATH: directory, TEST_PIP_LOG: log, ...extraEnv } });
  return log;
}

it("pins the Python bridge to the installed npm version", () => {
  expect(readFileSync(run(), "utf8").trim().split("\n")).toEqual([
    "install", `ai-knot[mcp]==${version}`, "--quiet",
  ]);
});

it.each(["AI_KNOT_SKIP_PYTHON_INSTALL", "CI_SKIP_POSTINSTALL"])("honors %s before invoking pip", (flag) => {
  expect(existsSync(run({ [flag]: "1" }))).toBe(false);
});
