/**
 * Debug: show what recall returns vs what the gold answer expects.
 *
 * Usage:
 *   bun run src/_debug_recall_quality.ts [conv_idx] [qa_limit]
 *
 * Examples:
 *   bun run src/_debug_recall_quality.ts 0 5    # conv 0, first 5 QA
 *   bun run src/_debug_recall_quality.ts 1 3    # conv 1, first 3 QA
 */

import { KnowledgeBase } from "ai-knot";
import { loadDataset, filterQA } from "./locomo.js";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, rmSync } from "node:fs";

const convIdx = parseInt(process.argv[2] ?? "0", 10);
const qaLimit = parseInt(process.argv[3] ?? "5", 10);
const topK = parseInt(process.argv[4] ?? "10", 10);

const DB_PATH = resolve(
  fileURLToPath(import.meta.url),
  "..",
  "..",
  "data",
  "runs",
  "_debug_recall",
  "knot.db"
);

const AI_KNOT_CMD =
  process.env["AI_KNOT_COMMAND"] ??
  "/Users/alsoleg/Documents/github/ai-knot/.venv/bin/ai-knot-mcp";

async function main() {
  const dataset = await loadDataset();
  const conv = dataset[convIdx];
  if (!conv) {
    console.error(`No conversation at index ${convIdx}`);
    process.exit(1);
  }

  console.log(
    `\n=== Conv ${convIdx}: ${conv.turns.length} turns, ${conv.qa.length} QA pairs ===\n`
  );

  // Clean previous debug DB
  const dbDir = resolve(DB_PATH, "..");
  if (existsSync(dbDir)) rmSync(dbDir, { recursive: true, force: true });

  const kb = new KnowledgeBase({
    agentId: `conv-${convIdx}`,
    storage: "sqlite",
    dbPath: DB_PATH,
    command: AI_KNOT_CMD,
  });

  // Ingest
  console.log(`Ingesting ${conv.turns.length} turns...`);
  for (const turn of conv.turns) {
    await kb.add(turn);
  }
  console.log("Done ingesting.\n");

  // Test recall for each QA
  const qa = filterQA(conv.qa, [1, 2, 3, 4], undefined).slice(0, qaLimit);

  for (const q of qa) {
    console.log(`${"─".repeat(70)}`);
    console.log(`[cat ${q.category}] Q: ${q.question}`);
    console.log(`GOLD: ${q.answer}`);
    console.log();

    const recall = await kb.recall(q.question, { topK });
    const lines = recall.split("\n").filter((l) => l.trim());

    if (lines.length === 0 || recall === "No relevant facts found.") {
      console.log("RECALL: (empty)\n");
    } else {
      console.log(`RECALL (${lines.length} facts, topK=${topK}):`);
      for (const line of lines.slice(0, 5)) {
        // Truncate long lines
        const display = line.length > 120 ? line.slice(0, 120) + "…" : line;
        console.log(`  ${display}`);
      }
      if (lines.length > 5) {
        console.log(`  ... +${lines.length - 5} more`);
      }

      // Check if gold answer keywords appear anywhere in recall
      const goldWords = q.answer
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 3);
      const recallLower = recall.toLowerCase();
      const found = goldWords.filter((w) => recallLower.includes(w));
      const missed = goldWords.filter((w) => !recallLower.includes(w));

      console.log();
      if (found.length > 0)
        console.log(`  ✓ Gold keywords in recall: ${found.join(", ")}`);
      if (missed.length > 0)
        console.log(`  ✗ Gold keywords MISSING:   ${missed.join(", ")}`);
      if (found.length === 0)
        console.log(`  ✗ NO gold keywords found in recall`);
    }
    console.log();
  }

  await kb.close();
  console.log("Done.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
