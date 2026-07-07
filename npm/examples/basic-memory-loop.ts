import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { KnowledgeBase } from "../src/index.ts";

async function main(): Promise<void> {
  const dataDir = mkdtempSync(join(tmpdir(), "ai-knot-basic-loop-"));
  const repoCommand = join(process.cwd(), "..", ".venv", "bin", "ai-knot-mcp");
  const kb = new KnowledgeBase({
    agentId: "assistant",
    storage: "sqlite",
    dataDir,
    command: existsSync(repoCommand) ? repoCommand : "ai-knot-mcp",
  });

  const noisyFact = await kb.add("Team standup is at 10am");
  await kb.add("User prefers TypeScript for frontend work");
  await kb.add("User deploys APIs with Docker Compose");

  console.log("=== Search ===");
  console.log(await kb.search("what does the user deploy with?"));
  console.log();

  console.log("=== List ===");
  console.log(await kb.list());
  console.log();

  await kb.delete(noisyFact.id);

  console.log("=== After delete ===");
  console.log(await kb.list());

  await kb.close();
}

void main();
