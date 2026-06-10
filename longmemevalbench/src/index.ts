import type { Granularity } from "./aiknot.js";
import { loadConfig } from "./config.js";
import { QUESTION_TYPES } from "./loader.js";
import { DEFAULT_ANSWER_MODEL, DEFAULT_JUDGE_MODEL, resolveModel } from "./models.js";
import { listRuns, runBenchmark } from "./runner.js";

// ---- CLI arg parsing --------------------------------------------------------

interface RunArgs {
  command: "run";
  runId: string;
  judgeModel: string;
  answerModel: string;
  dataFile?: string;
  limit?: number;
  sample?: number;
  types?: string[];
  topK: number;
  granularity: Granularity;
  multiAgent: boolean;
  idkContract: boolean;
  aiKnotEnv: Record<string, string>;
  force: boolean;
}

interface ListArgs {
  command: "list";
  limit?: number;
}

type CliArgs = RunArgs | ListArgs | { command: "help" };

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
  let dataFile: string | undefined;
  let limit: number | undefined;
  let sample: number | undefined;
  let types: string[] | undefined;
  let topK = 10;
  let granularity: Granularity = "round";
  let multiAgent = false;
  let idkContract = true;
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
    } else if (a === "--data" && next) {
      dataFile = next;
      i++;
    } else if (a === "--limit" && next) {
      limit = parseInt(next, 10);
      i++;
    } else if (a === "--sample" && next) {
      sample = parseInt(next, 10);
      i++;
    } else if (a === "--types" && next) {
      types = next.split(",").map((s) => s.trim());
      i++;
    } else if (a === "--top-k" && next) {
      topK = parseInt(next, 10);
      i++;
    } else if (a === "--granularity" && next) {
      if (next === "window" || next === "round" || next === "session") {
        granularity = next;
      } else {
        console.error(`Error: --granularity must be window|round|session (got "${next}")`);
        process.exit(1);
      }
      i++;
    } else if (a === "--multi-agent") {
      multiAgent = true;
    } else if (a === "--no-idk") {
      idkContract = false;
    } else if (a === "--knot-env" && next) {
      const eq = next.indexOf("=");
      if (eq > 0) aiKnotEnv[next.slice(0, eq)] = next.slice(eq + 1);
      i++;
    } else if (a === "--force") {
      force = true;
    }
  }

  if (!runId) {
    console.error("Error: -r <run-id> is required for the run command.");
    process.exit(1);
  }

  return {
    command: "run",
    runId,
    judgeModel,
    answerModel,
    dataFile,
    limit,
    sample,
    types,
    topK,
    granularity,
    multiAgent,
    idkContract,
    aiKnotEnv,
    force,
  };
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
longmemevalbench — LongMemEval benchmark for ai-knot

Commands:
  run  -r <run-id> [options]   Run or resume a benchmark
  list [-l <n>]                List recent runs
  help                         Show this message

Run options:
  -r, --run-id <id>      Run identifier (required)
  --judge <model>        Judge model    (default: ${DEFAULT_JUDGE_MODEL})
  --model <model>        Reader model   (default: ${DEFAULT_ANSWER_MODEL})
  --data <path>          LongMemEval JSON (default: data/sample_longmemeval.json
                         or $LONGMEMEVAL_FILE)
  --limit <n>            Limit to first N questions
  --sample <n>           Max questions per run after type filtering
  --types <a,b,c>        Question types to evaluate. One or more of:
                         ${QUESTION_TYPES.join(", ")}
  --top-k <n>            Facts to recall per query (default: 10)
  --granularity <g>      Ingest unit: window|round|session (default: round)
  --multi-agent          Split user/assistant turns into per-agent namespaces
  --no-idk               Disable the abstention (IDK) reader contract
  --knot-env K=V         Pass env var to ai-knot-mcp (repeatable)
  --force                Delete existing run data and start fresh

Examples:
  node --import tsx src/index.ts run -r smoke --data data/sample_longmemeval.json
  node --import tsx src/index.ts run -r tr --types temporal-reasoning,knowledge-update
  node --import tsx src/index.ts run -r ma --multi-agent --granularity round
  node --import tsx src/index.ts list -l 5
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
      `${"Q".padStart(5)}  ${"Acc".padStart(8)}`;
    console.log(`\n${header}`);
    console.log("─".repeat(header.length));
    for (const r of runs) {
      const status = r.finishedAt ? "done" : "partial";
      const started = r.startedAt.slice(0, 16).replace("T", " ");
      console.log(
        `${r.runId.padEnd(col1)}  ${started.padEnd(20)}  ${status.padEnd(8)}  ` +
          `${String(r.total).padStart(5)}  ${(r.accuracy ?? "-").padStart(8)}`
      );
    }
    console.log();
    return;
  }

  const config = loadConfig();
  await runBenchmark({
    runId: parsed.runId,
    judgeModel: resolveModel(parsed.judgeModel),
    answerModel: resolveModel(parsed.answerModel),
    judgeModelName: parsed.judgeModel,
    answerModelName: parsed.answerModel,
    aiKnotCommand: config.aiKnotCommand,
    aiKnotEnv: parsed.aiKnotEnv,
    dataFile: parsed.dataFile,
    limit: parsed.limit,
    sample: parsed.sample,
    types: parsed.types,
    topK: parsed.topK,
    granularity: parsed.granularity,
    multiAgent: parsed.multiAgent,
    idkContract: parsed.idkContract,
    force: parsed.force,
  });
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
