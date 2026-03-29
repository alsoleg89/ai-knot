#!/usr/bin/env node
// Runs automatically after `npm install agentmemo`.
// Tries to install the Python agentmemo[mcp] package via pip.
// Always exits 0 — failure is non-fatal (a warning is printed instead).

import { execSync } from "node:child_process";

const PACKAGE = "agentmemo[mcp]";

if (
  process.env["AGENTMEMO_SKIP_PYTHON_INSTALL"] === "1" ||
  process.env["CI_SKIP_POSTINSTALL"] === "1"
) {
  process.exit(0);
}

function tryInstall(cmd) {
  try {
    execSync(`${cmd} install "${PACKAGE}" --quiet`, { stdio: "inherit" });
    return true;
  } catch {
    return false;
  }
}

const candidates = ["pip3", "pip", "python3 -m pip", "python -m pip"];

for (const cmd of candidates) {
  if (tryInstall(cmd)) {
    process.exit(0);
  }
}

console.warn(`
╔══════════════════════════════════════════════════════════╗
║  agentmemo: Python package could not be installed        ║
╠══════════════════════════════════════════════════════════╣
║  The agentmemo npm package requires the Python           ║
║  agentmemo[mcp] package to spawn the MCP subprocess.    ║
║                                                          ║
║  Install it manually:                                    ║
║    pip install "agentmemo[mcp]"                          ║
║                                                          ║
║  Python 3.11+ and pip must be in PATH.                   ║
╚══════════════════════════════════════════════════════════╝
`);

process.exit(0);
