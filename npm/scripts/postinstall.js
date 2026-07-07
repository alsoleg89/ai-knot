#!/usr/bin/env node
// Runs automatically after `npm install ai-knot`.
// Best-effort: tries to install the optional Python `ai-knot[mcp]` bridge, which
// is only needed for the in-process subprocess client (KnowledgeBase). The HTTP
// client (HttpKnowledgeBase) needs no Python on this machine. Always exits 0.

import { execSync } from "node:child_process";

const PACKAGE = "ai-knot[mcp]";

if (
  process.env["AI_KNOT_SKIP_PYTHON_INSTALL"] === "1" ||
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

console.warn(
  [
    "",
    "ai-knot: the optional Python subprocess bridge was not installed.",
    "It is only needed for the in-process client (KnowledgeBase). You have two paths:",
    "",
    "  1) No Python on this machine — run the engine as a server, use the HTTP client:",
    "       docker run -p 8000:8000 -v ai-knot-data:/data ai-knot",
    '       import { HttpKnowledgeBase } from "ai-knot";',
    "",
    "  2) In-process client — install the Python bridge (needs Python 3.11+ and pip):",
    '       pip install "ai-knot[mcp]"',
    "       npx ai-knot-doctor",
    "",
  ].join("\n"),
);

process.exit(0);
