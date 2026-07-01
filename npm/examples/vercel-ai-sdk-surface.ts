// Repo-native Vercel AI SDK surface proof: builds ai-knot memory into the
// system prompt shape without making any model call.

import { AiKnotAISDKMemory, KnowledgeBase } from "ai-knot";

async function main(): Promise<void> {
  const kb = new KnowledgeBase({
    agentId: "assistant",
    storage: "sqlite",
    dbPath: "/absolute/path/to/ai-knot.db",
  });

  await kb.add("User prefers TypeScript over JavaScript");
  await kb.add("User deploys services with Docker Compose");

  const memory = new AiKnotAISDKMemory(kb, { topK: 4 });
  const userInput = "Write a local deployment checklist for my stack.";

  try {
    const system = await memory.buildSystem(userInput, {
      baseSystem: "You are a concise staff engineer.",
    });

    console.log("=== Vercel AI SDK memory surface (no model call) ===");
    console.log("User prompt:");
    console.log(`  ${userInput}`);
    console.log();
    console.log("System prompt with ai-knot memory:");
    console.log(system);
  } finally {
    await kb.close();
  }
}

void main();
