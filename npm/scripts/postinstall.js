#!/usr/bin/env node
// Runs automatically after `npm install ai-knot`.
// Tries to install the Python ai-knot[mcp] package via pip.
// Always exits 0 — failure is non-fatal (a warning is printed instead).

import { execSync } from "node:child_process";

const PACKAGE = "ai-knot[mcp]";

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
║  ai-knot: Python package could not be installed        ║
╠══════════════════════════════════════════════════════════╣
║  The ai-knot npm package requires the Python           ║
║  ai-knot[mcp] package to spawn the MCP subprocess.    ║
║                                                          ║
║  Install it manually:                                    ║
║    pip install "ai-knot[mcp]"                          ║
║                                                          ║
║  Python 3.11+ and pip must be in PATH.                   ║
╚══════════════════════════════════════════════════════════╝
`);

process.exit(0);
