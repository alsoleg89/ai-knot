import { describe, expect, it } from "vitest";

type ExecFileSync = (
  command: string,
  args: string[],
  options?: { encoding?: string; env?: NodeJS.ProcessEnv; stdio?: unknown },
) => string;

async function loadDemo(): Promise<{
  buildDemoArgs: (options?: {
    agentId?: string;
    dataDir?: string | null;
    keepData?: boolean;
    storage?: "sqlite" | "yaml";
  }) => string[];
  findPythonForDemo: (options?: {
    env?: NodeJS.ProcessEnv;
    execFileSync?: ExecFileSync;
    pythonCommand?: string | null;
  }) => { command: string; executable: string; version: string } | null;
  main: (
    argv?: string[],
    options?: {
      env?: NodeJS.ProcessEnv;
      execFileSync?: ExecFileSync;
      stderr?: { write: (text: string) => void };
      stdout?: { write: (text: string) => void };
    },
  ) => number;
  runDemo: (options?: {
    agentId?: string;
    dataDir?: string | null;
    env?: NodeJS.ProcessEnv;
    execFileSync?: ExecFileSync;
    keepData?: boolean;
    pythonCommand?: string | null;
    stderr?: { write: (text: string) => void };
    stdout?: { write: (text: string) => void };
    storage?: "sqlite" | "yaml";
  }) => { ok: boolean; python: { command: string; executable: string; version: string } | null };
}> {
  const modulePath = "../../scripts/demo.mjs";
  return (await import(modulePath)) as {
    buildDemoArgs: (options?: {
      agentId?: string;
      dataDir?: string | null;
      keepData?: boolean;
      storage?: "sqlite" | "yaml";
    }) => string[];
    findPythonForDemo: (options?: {
      env?: NodeJS.ProcessEnv;
      execFileSync?: ExecFileSync;
      pythonCommand?: string | null;
    }) => { command: string; executable: string; version: string } | null;
    main: (
      argv?: string[],
      options?: {
        env?: NodeJS.ProcessEnv;
        execFileSync?: ExecFileSync;
        stderr?: { write: (text: string) => void };
        stdout?: { write: (text: string) => void };
      },
    ) => number;
    runDemo: (options?: {
      agentId?: string;
      dataDir?: string | null;
      env?: NodeJS.ProcessEnv;
      execFileSync?: ExecFileSync;
      keepData?: boolean;
      pythonCommand?: string | null;
      stderr?: { write: (text: string) => void };
      stdout?: { write: (text: string) => void };
      storage?: "sqlite" | "yaml";
    }) => { ok: boolean; python: { command: string; executable: string; version: string } | null };
  };
}

function createBuffers() {
  let stdout = "";
  let stderr = "";
  return {
    stderr: { write: (text: string) => void (stderr += text) },
    stdout: { write: (text: string) => void (stdout += text) },
    read: () => ({ stderr, stdout }),
  };
}

describe("ai-knot-demo", () => {
  it("builds the default python demo command", async () => {
    const { buildDemoArgs } = await loadDemo();

    expect(buildDemoArgs()).toEqual([
      "-m",
      "ai_knot.cli",
      "--storage",
      "sqlite",
      "demo",
      "--agent-id",
      "demo",
    ]);
  });

  it("builds keep-data args when requested", async () => {
    const { buildDemoArgs } = await loadDemo();

    expect(
      buildDemoArgs({
        agentId: "proof",
        dataDir: "/tmp/ai-knot-proof",
        keepData: true,
        storage: "yaml",
      }),
    ).toEqual([
      "-m",
      "ai_knot.cli",
      "--storage",
      "yaml",
      "--data-dir",
      "/tmp/ai-knot-proof",
      "demo",
      "--agent-id",
      "proof",
      "--keep-data",
    ]);
  });

  it("finds a Python interpreter that can import ai_knot", async () => {
    const { findPythonForDemo } = await loadDemo();
    const execFileSync: ExecFileSync = (command, args) => {
      if (command === "python3" && args[0] === "-c") {
        return JSON.stringify({
          executable: "/usr/bin/python3",
          version: "0.11.0",
        });
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    expect(findPythonForDemo({ execFileSync })).toEqual({
      command: "python3",
      executable: "/usr/bin/python3",
      version: "0.11.0",
    });
  });

  it("runs the Python demo with the expected arguments", async () => {
    const { runDemo } = await loadDemo();
    const calls: Array<{ command: string; args: string[] }> = [];
    const buffers = createBuffers();
    const execFileSync: ExecFileSync = (command, args) => {
      calls.push({ command, args });
      if (args[0] === "-c") {
        return JSON.stringify({
          executable: "/usr/bin/python3",
          version: "0.11.0",
        });
      }
      if (args[0] === "-m" && args[1] === "ai_knot.cli") {
        return "";
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const result = runDemo({
      agentId: "proof",
      keepData: true,
      dataDir: "/tmp/ai-knot-proof",
      execFileSync,
      stdout: buffers.stdout,
      stderr: buffers.stderr,
    });

    expect(result.ok).toBe(true);
    expect(calls).toEqual([
      {
        command: "python3",
        args: [
          "-c",
          "import json, sys, ai_knot; print(json.dumps({'executable': sys.executable, 'version': getattr(ai_knot, '__version__', None)}))",
        ],
      },
      {
        command: "python3",
        args: [
          "-m",
          "ai_knot.cli",
          "--storage",
          "sqlite",
          "--data-dir",
          "/tmp/ai-knot-proof",
          "demo",
          "--agent-id",
          "proof",
          "--keep-data",
        ],
      },
    ]);

    expect(buffers.read().stdout).toContain("Running ai-knot demo via python3");
  });

  it("returns a helpful failure when python ai-knot is missing", async () => {
    const { main } = await loadDemo();
    const buffers = createBuffers();
    const execFileSync: ExecFileSync = () => {
      throw new Error("missing ai_knot");
    };

    const exitCode = main([], {
      execFileSync,
      stdout: buffers.stdout,
      stderr: buffers.stderr,
    });

    expect(exitCode).toBe(1);
    expect(buffers.read().stderr).toContain("Run `npx ai-knot-doctor`");
  });

  it("rejects --data-dir without --keep-data", async () => {
    const { main } = await loadDemo();
    const buffers = createBuffers();

    const exitCode = main(["--data-dir", "/tmp/demo"], {
      stdout: buffers.stdout,
      stderr: buffers.stderr,
    });

    expect(exitCode).toBe(2);
    expect(buffers.read().stderr).toContain("Use --keep-data together with --data-dir");
  });
});
