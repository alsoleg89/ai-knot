import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
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

    const { text } = await generateText({
      model: openai("gpt-5"),
      system,
      prompt: userInput,
    });

    console.log(text);
  } finally {
    await kb.close();
  }
}

void main();
