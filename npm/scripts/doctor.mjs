#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { readFileSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DEFAULT_PACKAGE_INFO = JSON.parse(
  readFileSync(join(PACKAGE_ROOT, "package.json"), "utf-8"),
);

const PYTHON_INFO_SCRIPT =
  "import json, sys; print(json.dumps({'executable': sys.executable, 'version': '.'.join(map(str, sys.version_info[:3]))}))";
const AI_KNOT_INFO_SCRIPT =
  "import json, ai_knot; print(json.dumps({'version': getattr(ai_knot, '__version__', None)}))";
const MCP_PATH_SCRIPT =
  "import json, shutil; print(json.dumps({'path': shutil.which('ai-knot-mcp')}))";

function parseSemver(version) {
  const match = /^(\d+)\.(\d+)\.(\d+)/.exec(String(version).trim());
  if (!match) return null;
  return match.slice(1).map((part) => Number.parseInt(part, 10));
}

function isVersionAtLeast(version, minimum) {
  for (let index = 0; index < minimum.length; index += 1) {
    const current = version[index] ?? 0;
    const required = minimum[index] ?? 0;
    if (current > required) return true;
    if (current < required) return false;
  }
  return true;
}

function safeExec(execFile, command, args) {
  try {
    const stdout = execFile(command, args, {
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    return {
      ok: true,
      stdout: String(stdout).trim(),
      stderr: "",
      error: null,
    };
  } catch (error) {
    return {
      ok: false,
      stdout: String(error?.stdout ?? "").trim(),
      stderr: String(error?.stderr ?? "").trim(),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function safeExecJson(execFile, command, args) {
  const result = safeExec(execFile, command, args);
  if (!result.ok) return { ok: false, error: result.error, stdout: result.stdout, stderr: result.stderr };

  try {
    return { ok: true, value: JSON.parse(result.stdout || "{}") };
  } catch (error) {
    return {
      ok: false,
      error: `Expected JSON output from ${command}: ${error instanceof Error ? error.message : String(error)}`,
      stdout: result.stdout,
      stderr: result.stderr,
    };
  }
}

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

function buildPythonCandidates({ env, pythonCommand }) {
  return unique([
    pythonCommand,
    env.AI_KNOT_PYTHON,
    "python3",
    "python",
  ]);
}

function findPython(execFile, { env, pythonCommand }) {
  const candidates = buildPythonCandidates({ env, pythonCommand });
  const matches = [];

  for (const candidate of candidates) {
    const result = safeExecJson(execFile, candidate, ["-c", PYTHON_INFO_SCRIPT]);
    if (!result.ok) continue;

    const version = String(result.value.version ?? "");
    const parsed = parseSemver(version);
    matches.push({
      command: candidate,
      executable: String(result.value.executable ?? candidate),
      version,
      parsedVersion: parsed,
      supported: parsed !== null && isVersionAtLeast(parsed, [3, 11, 0]),
    });
  }

  if (matches.length === 0) {
    return {
      selected: null,
      attempted: candidates,
    };
  }

  return {
    selected: matches.find((match) => match.supported) ?? matches[0],
    attempted: candidates,
  };
}

function addNextAction(list, action) {
  if (!list.includes(action)) list.push(action);
}

export function runDoctor(options = {}) {
  const execFile = options.execFileSync ?? execFileSync;
  const env = options.env ?? process.env;
  const nodeVersion = options.nodeVersion ?? process.versions.node;
  const packageInfo = options.packageInfo ?? DEFAULT_PACKAGE_INFO;
  const pythonCommand = options.pythonCommand ?? null;

  const checks = [];
  const nextActions = [];
  const minimumNode = [18, 0, 0];
  const parsedNode = parseSemver(nodeVersion);
  const nodeOk = parsedNode !== null && isVersionAtLeast(parsedNode, minimumNode);

  checks.push({
    id: "node",
    label: "Node.js",
    ok: nodeOk,
    detail: `v${nodeVersion}`,
  });

  if (!nodeOk) {
    addNextAction(nextActions, "Use Node.js 18+ before running the ai-knot npm package.");
  }

  const python = findPython(execFile, { env, pythonCommand });

  if (python.selected === null) {
    checks.push({
      id: "python",
      label: "Python",
      ok: false,
      detail: `No Python interpreter found on PATH (tried: ${python.attempted.join(", ")})`,
    });
    addNextAction(
      nextActions,
      "Install Python 3.11+ and ensure `python3` or `python` is available on PATH.",
    );

    return {
      packageName: packageInfo.name,
      packageVersion: packageInfo.version,
      ok: false,
      checks,
      nextActions,
      pythonDoctor: null,
    };
  }

  const pythonDetail = `${python.selected.command} -> ${python.selected.executable} (${python.selected.version})`;
  checks.push({
    id: "python",
    label: "Python",
    ok: python.selected.supported,
    detail: pythonDetail,
  });

  if (!python.selected.supported) {
    addNextAction(
      nextActions,
      `Upgrade Python to 3.11+ before using the npm client (current: ${python.selected.version}).`,
    );
  }

  const pip = safeExec(execFile, python.selected.command, ["-m", "pip", "--version"]);
  checks.push({
    id: "pip",
    label: "pip",
    ok: pip.ok,
    detail: pip.ok ? pip.stdout : pip.error,
  });
  if (!pip.ok) {
    addNextAction(
      nextActions,
      `Repair pip for ${python.selected.command} or reinstall Python tooling, then rerun \`npx ai-knot-doctor\`.`,
    );
  }

  const pythonPackage = safeExecJson(execFile, python.selected.command, ["-c", AI_KNOT_INFO_SCRIPT]);
  checks.push({
    id: "python_package",
    label: "Python ai-knot package",
    ok: pythonPackage.ok,
    detail: pythonPackage.ok
      ? `ai-knot ${pythonPackage.value.version ?? "unknown"}`
      : pythonPackage.error,
  });
  if (!pythonPackage.ok) {
    addNextAction(
      nextActions,
      `${python.selected.command} -m pip install "ai-knot[mcp]==${packageInfo.version}"`,
    );
  }

  if (pythonPackage.ok) {
    const pythonVersion = String(pythonPackage.value.version ?? "");
    const versionParityOk = pythonVersion === packageInfo.version;
    checks.push({
      id: "version_parity",
      label: "Version parity",
      ok: versionParityOk,
      detail: `npm ${packageInfo.version} vs Python ${pythonVersion}`,
    });
    if (!versionParityOk) {
      addNextAction(
        nextActions,
        `${python.selected.command} -m pip install --upgrade "ai-knot[mcp]==${packageInfo.version}"`,
      );
    }
  }

  const mcpBinary = safeExecJson(execFile, python.selected.command, ["-c", MCP_PATH_SCRIPT]);
  const mcpPath = mcpBinary.ok ? mcpBinary.value.path : null;
  const mcpBinaryOk = Boolean(mcpPath);
  checks.push({
    id: "mcp_binary",
    label: "ai-knot-mcp on PATH",
    ok: mcpBinaryOk,
    detail: mcpBinaryOk ? String(mcpPath) : (mcpBinary.ok ? "Not found on PATH" : mcpBinary.error),
  });
  if (!mcpBinaryOk) {
    addNextAction(
      nextActions,
      "Ensure the Python environment that owns `ai-knot[mcp]` exposes `ai-knot-mcp` on PATH, or pass `command` explicitly in `new KnowledgeBase({ command: ... })`.",
    );
  }

  let pythonDoctor = null;
  if (pythonPackage.ok) {
    const doctor = safeExecJson(execFile, python.selected.command, [
      "-m",
      "ai_knot.cli",
      "doctor",
      "--json",
    ]);
    const doctorCommandOk = doctor.ok;
    pythonDoctor = doctorCommandOk ? doctor.value : null;

    let doctorBridgeOk = doctorCommandOk;
    let doctorDetail = doctorCommandOk ? "Python-side doctor returned JSON." : doctor.error;
    if (doctorCommandOk) {
      const moduleState = doctor.value.modules?.mcp;
      const pathState = doctor.value.commands?.ai_knot_mcp_on_path;
      doctorDetail = `modules.mcp=${String(moduleState)}, ai_knot_mcp_on_path=${String(pathState)}`;
      doctorBridgeOk = moduleState === true && pathState === true;

      if (moduleState !== true) {
        addNextAction(
          nextActions,
          `${python.selected.command} -m pip install --upgrade "ai-knot[mcp]==${packageInfo.version}"`,
        );
      }
      if (pathState !== true) {
        addNextAction(
          nextActions,
          "Make sure the Python scripts directory is on PATH before starting the Node process, or use `new KnowledgeBase({ command: \"/absolute/path/to/ai-knot-mcp\" })`.",
        );
      }
    }

    checks.push({
      id: "python_cli_doctor",
      label: "Python bridge doctor",
      ok: doctorBridgeOk,
      detail: doctorDetail,
    });

    if (!doctorCommandOk) {
      addNextAction(
        nextActions,
        "Run `python -m ai_knot.cli doctor --json` manually to inspect the Python-side environment in more detail.",
      );
    }
  }

  const ok = checks.every((check) => check.ok);
  return {
    packageName: packageInfo.name,
    packageVersion: packageInfo.version,
    ok,
    checks,
    nextActions,
    pythonDoctor,
  };
}

export function formatDoctorReport(report) {
  const lines = [];
  lines.push(`${report.packageName} npm doctor (${report.packageVersion})`);
  lines.push("");

  for (const check of report.checks) {
    const prefix = check.ok ? "PASS" : "FAIL";
    lines.push(`[${prefix}] ${check.label}: ${check.detail}`);
  }

  lines.push("");
  if (report.ok) {
    lines.push("npm/TypeScript bridge looks healthy.");
  } else if (report.nextActions.length > 0) {
    lines.push("Likely next actions:");
    for (const [index, action] of report.nextActions.entries()) {
      lines.push(`${index + 1}. ${action}`);
    }
  } else {
    lines.push("Doctor found issues, but no automatic next action was derived.");
  }

  return `${lines.join("\n")}\n`;
}

function usage() {
  return `Usage: ai-knot-doctor [--json] [--python <path>]

Checks the Node.js + Python bridge used by the ai-knot npm package.

Options:
  --json           emit machine-readable JSON
  --python <path>  inspect a specific Python executable
  --help           show this help
`;
}

export function main(argv = process.argv.slice(2), options = {}) {
  let jsonOutput = false;
  let pythonCommand = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--json") {
      jsonOutput = true;
      continue;
    }
    if (arg === "--python") {
      pythonCommand = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      process.stdout.write(usage());
      return 0;
    }
    process.stderr.write(`Unknown argument: ${arg}\n\n${usage()}`);
    return 2;
  }

  if (argv.includes("--python") && !pythonCommand) {
    process.stderr.write(`Missing value for --python\n\n${usage()}`);
    return 2;
  }

  const report = runDoctor({ ...options, pythonCommand });
  const output = jsonOutput ? `${JSON.stringify(report, null, 2)}\n` : formatDoctorReport(report);
  process.stdout.write(output);
  return report.ok ? 0 : 1;
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
