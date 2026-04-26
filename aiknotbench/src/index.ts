import type { IngestMode } from "./aiknot.js";
import { loadConfig } from "./config.js";
import {
  DEFAULT_ANSWER_MODEL,
  DEFAULT_JUDGE_MODEL,
  resolveModel,
} from "./models.js";
import { runBenchmark, listRuns } from "./runner.js";

// ---- CLI arg parsing --------------------------------------------------------

export interface RunArgs {
  command: "run";
  runId: string;
  judgeModel: string;
  answerModel: string;
  limit?: number;
  sample?: number;
  types?: number[];
  convs?: number[];
  ingestMode?: IngestMode;
  topK: number;
  aiKnotEnv: Record<string, string>;
  force: boolean;
}

export interface ListArgs {
  command: "list";
  limit?: number;
}

export type CliArgs = RunArgs | ListArgs | { command: "help" };

export function parseArgs(argv: string[]): CliArgs {
  const [cmd, ...rest] = argv;

  if (cmd === "run") return parseRunArgs(rest);
  if (cmd === "list") return parseListArgs(rest);
  return { command: "help" };
}

function parseRunArgs(args: string[]): RunArgs {
  let runId: string | undefined;
  let judgeModel = DEFAULT_JUDGE_MODEL;
  let answerModel = DEFAULT_ANSWER_MODEL;
  let limit: number | undefined;
  let sample: number | undefined;
  let types: number[] | undefined;
  let convs: number[] | undefined;
  let ingestMode: IngestMode | undefined;
  let topK = 5;
  const aiKnotEnv: Record<string, string> = {};
  let force = false;

  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    const next = args[i + 1];

    if ((a === "-r" || a === "--run-id") && next) {
      runId = next;
      i++;
    } else if (a === "--judge" && next) {
      judgeModel = next;
      i++;
    } else if (a === "--model" && next) {
      answerModel = next;
      i++;
    } else if (a === "--limit" && next) {
      limit = parseInt(next, 10);
      i++;
    } else if (a === "--sample" && next) {
      sample = parseInt(next, 10);
      i++;
    } else if (a === "--types" && next) {
      types = next.split(",").map((s) => parseInt(s.trim(), 10));
      i++;
    } else if (a === "--convs" && next) {
      convs = next.split(",").map((s) => parseInt(s.trim(), 10));
      i++;
    } else if (a === "--ingest-mode" && next) {
      if (next === "raw" || next === "dated" || next === "session") {
        ingestMode = next;
      } else {
        console.error(`Error: --ingest-mode must be raw|dated|session (got "${next}")`);
        process.exit(1);
      }
      i++;
    } else if (a === "--top-k" && next) {
      topK = parseInt(next, 10);
      i++;
    } else if (a === "--knot-env" && next) {
      // Format: KEY=VALUE
      const eq = next.indexOf("=");
      if (eq > 0) {
        aiKnotEnv[next.slice(0, eq)] = next.slice(eq + 1);
      }
      i++;
    } else if (a === "--force") {
      force = true;
    }
  }

  if (!runId) {
    console.error("Error: -r <run-id> is required for the run command.");
    process.exit(1);
  }

  return { command: "run", runId, judgeModel, answerModel, limit, sample, types, convs, ingestMode, topK, aiKnotEnv, force };
}

function parseListArgs(args: string[]): ListArgs {
  let limit: number | undefined;

  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    const next = args[i + 1];
    if ((a === "-l" || a === "--limit") && next) {
      limit = parseInt(next, 10);
      i++;
    }
  }

  return { command: "list", limit };
}

// ---- Help -------------------------------------------------------------------

function printHelp(): void {
  console.log(`
aiknotbench — LoCoMo benchmark for ai-knot

Commands:
  run  -r <run-id> [options]   Run or resume a benchmark
  list [-l <n>]                List recent runs
  help                         Show this message

Run options:
  -r, --run-id <id>    Run identifier (required)
  --judge <model>      Judge model  (default: ${DEFAULT_JUDGE_MODEL})
  --model <model>      Answering model (default: ${DEFAULT_ANSWER_MODEL})
  --limit <n>          Limit to first N conversations
  --sample <n>         Max QA pairs per conversation
  --types <1,2,3,4>    Question categories to evaluate (default: all)
  --top-k <n>          Facts to recall per query (default: 5)
  --knot-env K=V       Pass env var to ai-knot-mcp (repeatable)
  --force              Delete existing run data and start fresh

Examples:
  bun run src/index.ts run -r quick --limit 2 --types 1,2
  bun run src/index.ts run -r full --types 1,2,3,4
  bun run src/index.ts run -r tuned --top-k 10 --knot-env AI_KNOT_RRF_WEIGHTS=8,2,2,1,1,0.5
  bun run src/index.ts list -l 5
`);
}

// ---- Entry point ------------------------------------------------------------

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));

  if (parsed.command === "help") {
    printHelp();
    return;
  }

  if (parsed.command === "list") {
    const runs = listRuns({ limit: parsed.limit });
    if (runs.length === 0) {
      console.log("No runs found in data/runs/");
      return;
    }
    const col1 = Math.max(6, ...runs.map((r) => r.runId.length));
    const header =
      `${"Run ID".padEnd(col1)}  ${"Started".padEnd(20)}  ${"Status".padEnd(8)}  ` +
      `${"QA".padStart(6)}  ${"Acc (cat1-4)".padStart(12)}`;
    console.log(`\n${header}`);
    console.log("─".repeat(header.length));
    for (const r of runs) {
      const status = r.finishedAt ? "done" : "partial";
      const started = r.startedAt.slice(0, 16).replace("T", " ");
      console.log(
        `${r.runId.padEnd(col1)}  ${started.padEnd(20)}  ${status.padEnd(8)}  ` +
        `${String(r.total).padStart(6)}  ${(r.accuracy ?? "-").padStart(12)}`
      );
    }
    console.log();
    return;
  }

  // run command
  const config = loadConfig();
  const {
    runId,
    judgeModel: judgeModelName,
    answerModel: answerModelName,
    limit,
    sample,
    types,
    convs,
    ingestMode,
    topK,
    aiKnotEnv,
    force,
  } = parsed;

  await runBenchmark({
    runId,
    judgeModel: resolveModel(judgeModelName),
    answerModel: resolveModel(answerModelName),
    judgeModelName,
    answerModelName,
    aiKnotCommand: config.aiKnotCommand,
    aiKnotEnv,
    topK,
    limit,
    sample,
    types,
    convs,
    ingestMode,
    force,
  });
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
