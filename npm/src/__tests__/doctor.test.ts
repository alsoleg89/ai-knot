import { describe, expect, it } from "vitest";

type ExecResult = {
  stderr?: string;
  stdout?: string;
  throwMessage?: string;
};

type DoctorCheck = {
  id: string;
  label: string;
  ok: boolean;
  detail: string;
};

type DoctorReport = {
  packageName: string;
  packageVersion: string;
  ok: boolean;
  checks: DoctorCheck[];
  nextActions: string[];
  pythonDoctor: unknown;
};

async function loadDoctor(): Promise<{
  formatDoctorReport: (report: DoctorReport) => string;
  runDoctor: (options: {
    execFileSync: (command: string, args: string[]) => string;
    nodeVersion: string;
    packageInfo: { name: string; version: string };
  }) => DoctorReport;
}> {
  const modulePath = "../../scripts/doctor.mjs";
  return (await import(modulePath)) as {
    formatDoctorReport: (report: DoctorReport) => string;
    runDoctor: (options: {
      execFileSync: (command: string, args: string[]) => string;
      nodeVersion: string;
      packageInfo: { name: string; version: string };
    }) => DoctorReport;
  };
}

function throwingExec(result: ExecResult): never {
  const error = new Error(result.throwMessage ?? "command failed") as Error & {
    stderr?: string;
    stdout?: string;
  };
  error.stderr = result.stderr;
  error.stdout = result.stdout;
  throw error;
}

describe("ai-knot-doctor", () => {
  it("reports a healthy npm/Python bridge", async () => {
    const { runDoctor } = await loadDoctor();
    const execFileSync = (command: string, args: string[]) => {
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("sys.version_info")) {
        return JSON.stringify({
          executable: "/usr/bin/python3",
          version: "3.11.9",
        });
      }
      if (command === "python3" && args[0] === "-m" && args[1] === "pip") {
        return "pip 24.2 from /venv/lib/python3.11/site-packages/pip (python 3.11)";
      }
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("ai_knot")) {
        return JSON.stringify({ version: "0.11.0" });
      }
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("shutil.which")) {
        return JSON.stringify({ path: "/venv/bin/ai-knot-mcp" });
      }
      if (command === "python3" && args[0] === "-m" && args[1] === "ai_knot.cli") {
        return JSON.stringify({
          modules: { mcp: true },
          commands: { ai_knot_mcp_on_path: true },
        });
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const report = runDoctor({
      execFileSync,
      nodeVersion: "22.12.0",
      packageInfo: { name: "ai-knot", version: "0.11.0" },
    });

    expect(report.ok).toBe(true);
    expect(report.nextActions).toEqual([]);
    expect(report.checks.find((check: DoctorCheck) => check.id === "version_parity")?.ok).toBe(
      true,
    );
    expect(report.checks.find((check: DoctorCheck) => check.id === "mcp_binary")?.detail).toContain(
      "ai-knot-mcp",
    );
  });

  it("reports when Python is missing entirely", async () => {
    const { runDoctor } = await loadDoctor();
    const execFileSync = () =>
      throwingExec({
        stderr: "ENOENT",
        throwMessage: "spawn python3 ENOENT",
      });

    const report = runDoctor({
      execFileSync,
      nodeVersion: "22.12.0",
      packageInfo: { name: "ai-knot", version: "0.11.0" },
    });

    expect(report.ok).toBe(false);
    expect(report.checks.find((check: DoctorCheck) => check.id === "python")?.detail).toContain(
      "No Python interpreter found on PATH",
    );
    expect(
      report.nextActions.some((action: string) => action.includes("Install Python 3.11+")),
    ).toBe(true);
  });

  it("reports missing ai-knot[mcp] and suggests the matching install command", async () => {
    const { runDoctor } = await loadDoctor();
    const execFileSync = (command: string, args: string[]) => {
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("sys.version_info")) {
        return JSON.stringify({
          executable: "/usr/bin/python3",
          version: "3.11.9",
        });
      }
      if (command === "python3" && args[0] === "-m" && args[1] === "pip") {
        return "pip 24.2 from /venv/lib/python3.11/site-packages/pip (python 3.11)";
      }
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("ai_knot")) {
        return throwingExec({
          stderr: "ModuleNotFoundError: No module named 'ai_knot'",
          throwMessage: "No module named 'ai_knot'",
        });
      }
      if (command === "python3" && args[0] === "-c" && args[1]?.includes("shutil.which")) {
        return JSON.stringify({ path: null });
      }
      throw new Error(`unexpected command: ${command} ${args.join(" ")}`);
    };

    const report = runDoctor({
      execFileSync,
      nodeVersion: "22.12.0",
      packageInfo: { name: "ai-knot", version: "0.11.0" },
    });

    expect(report.ok).toBe(false);
    expect(report.checks.find((check: DoctorCheck) => check.id === "python_package")?.ok).toBe(
      false,
    );
    expect(
      report.nextActions.some((action: string) =>
        action.includes('python3 -m pip install "ai-knot[mcp]==0.11.0"'),
      ),
    ).toBe(true);
  });

  it("formats human-readable output with derived next actions", async () => {
    const { formatDoctorReport } = await loadDoctor();
    const output = formatDoctorReport({
      packageName: "ai-knot",
      packageVersion: "0.11.0",
      ok: false,
      checks: [
        { id: "node", label: "Node.js", ok: true, detail: "v22.12.0" },
        { id: "python", label: "Python", ok: false, detail: "No Python interpreter found" },
      ],
      nextActions: ["Install Python 3.11+ and ensure `python3` is on PATH."],
      pythonDoctor: null,
    });

    expect(output).toContain("ai-knot npm doctor (0.11.0)");
    expect(output).toContain("[PASS] Node.js: v22.12.0");
    expect(output).toContain("[FAIL] Python: No Python interpreter found");
    expect(output).toContain("Likely next actions:");
    expect(output).toContain("1. Install Python 3.11+");
  });
});
