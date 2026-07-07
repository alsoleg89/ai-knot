import { HttpKnowledgeBase } from "../src/index.ts";

async function main(): Promise<void> {
  const baseUrl = process.env.AI_KNOT_BASE_URL ?? "http://127.0.0.1:8000";
  const token = process.env.AI_KNOT_SERVER_TOKEN;

  const kb = new HttpKnowledgeBase({ baseUrl, token });

  try {
    const health = await kb.health();
    console.log("=== Health ===");
    console.log(health);
    console.log();
  } catch (error) {
    console.error(
      `Could not reach ${baseUrl}. Start the sidecar first, for example:\n` +
        `  ai-knot --storage sqlite serve assistant --port 8000\n`,
    );
    throw error;
  }

  const fact = await kb.add("User deploys APIs with Docker Compose", {
    tags: ["ops"],
  });
  await kb.add("User prefers TypeScript for frontend work");
  const learned = await kb.learn(
    [
      { role: "assistant", content: "I can keep durable notes for future turns." },
      { role: "user", content: "The staging environment runs on Fly.io." },
    ],
    { eventTime: "2026-01-15T00:00:00+00:00" },
  );
  const resolved = await kb.addResolved([
    {
      content: "Alex prefers pnpm",
      entity: "Alex",
      attribute: "package_manager",
      valueText: "pnpm",
      slotKey: "alex::package_manager",
      eventTime: "2026-01-16T00:00:00+00:00",
    },
    {
      content: "Alex works from Berlin",
      entity: "Alex",
      attribute: "office",
      valueText: "Berlin",
      slotKey: "alex::office",
      eventTime: "2026-01-16T00:00:00+00:00",
    },
  ]);
  const corrected = await kb.addResolved([
    {
      content: "Alex now prefers bun",
      entity: "Alex",
      attribute: "package_manager",
      valueText: "bun",
      slotKey: "alex::package_manager",
      op: "update",
      eventTime: "2026-01-17T00:00:00+00:00",
    },
    {
      content: "Alex no longer works from Berlin",
      entity: "Alex",
      attribute: "office",
      slotKey: "alex::office",
      op: "delete",
      eventTime: "2026-01-17T00:00:00+00:00",
    },
  ]);

  console.log("=== Search ===");
  console.log(await kb.search("what does the user deploy with?"));
  console.log();

  console.log("=== Learn ===");
  console.log(learned);
  console.log();

  console.log("=== Structured addResolved ===");
  console.log(resolved);
  console.log();

  console.log("=== Structured correction ===");
  console.log(corrected);
  console.log();

  if (corrected[0]) {
    console.log("=== Structured lineage ===");
    console.log(await kb.lineage(corrected[0].id));
    console.log();
  }

  console.log("=== List (active only) ===");
  console.log(await kb.list());
  console.log();

  console.log("=== List (including history) ===");
  console.log(await kb.list({ includeInactive: true }));
  console.log();

  await kb.delete(fact.id);

  console.log("=== After delete ===");
  console.log(await kb.list());
}

void main();
