#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

const AI_KNOT_INFO_SCRIPT =
  "import json, sys, ai_knot; print(json.dumps({'executable': sys.executable, 'version': getattr(ai_knot, '__version__', None)}))";

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function safeExecJson(execFile, command, args) {
  try {
    const stdout = execFile(command, args, {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return { ok: true, value: JSON.parse(String(stdout).trim() || "{}") };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
      stdout: String(error?.stdout ?? "").trim(),
      stderr: String(error?.stderr ?? "").trim(),
    };
  }
}

export function findPythonForDemo(options = {}) {
  const execFile = options.execFileSync ?? execFileSync;
  const env = options.env ?? process.env;
  const pythonCommand = options.pythonCommand ?? null;
  const candidates = unique([pythonCommand, env.AI_KNOT_PYTHON, "python3", "python"]);

  for (const candidate of candidates) {
    const result = safeExecJson(execFile, candidate, ["-c", AI_KNOT_INFO_SCRIPT]);
    if (!result.ok) continue;
    return {
      command: candidate,
      executable: String(result.value.executable ?? candidate),
      version: String(result.value.version ?? "unknown"),
    };
  }

  return null;
}

export function buildDemoArgs(options = {}) {
  const agentId = options.agentId ?? "demo";
  const keepData = options.keepData ?? false;
  const storage = options.storage ?? "sqlite";
  const dataDir = options.dataDir ?? null;

  const args = ["-m", "ai_knot.cli", "--storage", storage];
  if (keepData && dataDir) {
    args.push("--data-dir", dataDir);
  }
  args.push("demo", "--agent-id", agentId);
  if (keepData) {
    args.push("--keep-data");
  }
  return args;
}

function usage() {
  return `Usage: ai-knot-demo [--python <path>] [--agent-id <id>] [--keep-data] [--data-dir <dir>] [--storage <yaml|sqlite>]

Runs the built-in ai-knot demo through the Python CLI that powers the npm bridge.

Options:
  --python <path>   use a specific Python executable
  --agent-id <id>   agent namespace for the demo (default: demo)
  --keep-data       keep the demo store instead of using temporary storage
  --data-dir <dir>  storage directory to use together with --keep-data
  --storage <kind>  yaml or sqlite (default: sqlite)
  --help            show this help
`;
}

export function runDemo(options = {}) {
  const execFile = options.execFileSync ?? execFileSync;
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;
  const env = options.env ?? process.env;
  const pythonCommand = options.pythonCommand ?? null;
  const agentId = options.agentId ?? "demo";
  const keepData = options.keepData ?? false;
  const dataDir = options.dataDir ?? null;
  const storage = options.storage ?? "sqlite";

  const python = findPythonForDemo({ execFileSync: execFile, env, pythonCommand });
  if (python === null) {
    stderr.write(
      "Could not find a Python environment with ai-knot installed. " +
        "Run `npx ai-knot-doctor` for bridge diagnostics, or install it manually with " +
        '`pip install "ai-knot[mcp]"`.' +
        "\n",
    );
    return { ok: false, python: null };
  }

  stdout.write(
    `Running ai-knot demo via ${python.command} (${python.executable}, ai-knot ${python.version})\n`,
  );

  try {
    execFile(python.command, buildDemoArgs({ agentId, keepData, dataDir, storage }), {
      stdio: "inherit",
      env,
    });
    return { ok: true, python };
  } catch (error) {
    const detail =
      String(error?.stderr ?? "").trim() ||
      String(error?.stdout ?? "").trim() ||
      (error instanceof Error ? error.message : String(error));
    stderr.write(
      `ai-knot demo failed: ${detail}\nRun \`npx ai-knot-doctor --json\` if the bridge still looks suspicious.\n`,
    );
    return { ok: false, python };
  }
}

export function main(argv = process.argv.slice(2), options = {}) {
  let pythonCommand = null;
  let agentId = "demo";
  let keepData = false;
  let dataDir = null;
  let storage = "sqlite";
  const stderr = options.stderr ?? process.stderr;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      (options.stdout ?? process.stdout).write(usage());
      return 0;
    }
    if (arg === "--python") {
      pythonCommand = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === "--agent-id") {
      agentId = argv[index + 1] ?? "";
      index += 1;
      continue;
    }
    if (arg === "--data-dir") {
      dataDir = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === "--keep-data") {
      keepData = true;
      continue;
    }
    if (arg === "--storage") {
      storage = argv[index + 1] ?? "";
      index += 1;
      continue;
    }
    stderr.write(`Unknown argument: ${arg}\n\n${usage()}`);
    return 2;
  }

  if (argv.includes("--python") && !pythonCommand) {
    stderr.write(`Missing value for --python\n\n${usage()}`);
    return 2;
  }
  if (argv.includes("--agent-id") && !agentId) {
    stderr.write(`Missing value for --agent-id\n\n${usage()}`);
    return 2;
  }
  if (argv.includes("--data-dir") && !dataDir) {
    stderr.write(`Missing value for --data-dir\n\n${usage()}`);
    return 2;
  }
  if (argv.includes("--storage") && !storage) {
    stderr.write(`Missing value for --storage\n\n${usage()}`);
    return 2;
  }
  if (storage !== "yaml" && storage !== "sqlite") {
    stderr.write(`Unsupported storage: ${storage}\n\n${usage()}`);
    return 2;
  }
  if (dataDir !== null && !keepData) {
    stderr.write(`Use --keep-data together with --data-dir\n\n${usage()}`);
    return 2;
  }

  const result = runDemo({
    ...options,
    pythonCommand,
    agentId,
    keepData,
    dataDir,
    storage,
  });
  return result.ok ? 0 : 1;
}

if (
  process.argv[1] &&
  (() => {
    try {
      return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
    } catch {
      return false;
    }
  })()
) {
  process.exitCode = main();
}
