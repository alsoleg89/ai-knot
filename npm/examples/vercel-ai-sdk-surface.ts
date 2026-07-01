// Repo-native Vercel AI SDK surface proof: shows the exact system/messages
// shape ai-knot builds without requiring Python, MCP, or any model call.

import { AiKnotAISDKMemory } from "ai-knot";

async function main(): Promise<void> {
  const memory = new AiKnotAISDKMemory(
    {
      async recall(): Promise<string> {
        return [
          "[1] User prefers TypeScript over JavaScript",
          "[2] User deploys services with Docker Compose",
        ].join("\n");
      },
    },
    { topK: 4 },
  );
  const userInput = "Write a local deployment checklist for my stack.";
  const system = await memory.buildSystem(userInput, {
    baseSystem: "You are a concise staff engineer.",
  });
  const messages = await memory.buildMessages([
    { role: "system", content: "You are a concise staff engineer." },
    { role: "user", content: userInput },
  ]);

  console.log("=== Vercel AI SDK memory surface (no Python, no model call) ===");
  console.log("User prompt:");
  console.log(`  ${userInput}`);
  console.log();
  console.log("System prompt with ai-knot memory:");
  console.log(system);
  console.log();
  console.log("Normalized AI SDK messages shape:");
  console.log(JSON.stringify(messages, null, 2));
}

void main();
