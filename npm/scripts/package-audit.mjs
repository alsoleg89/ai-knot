import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const cacheDir = mkdtempSync(join(tmpdir(), "ai-knot-npm-pack-"));

const requiredFiles = [
  "README.md",
  "package.json",
  "scripts/demo.mjs",
  "scripts/doctor.mjs",
  "scripts/postinstall.js",
  "dist/esm/index.js",
  "dist/esm/index.d.ts",
  "dist/cjs/index.js",
  "dist/cjs/package.json",
];

class PackageAuditError extends Error {}

function fail(header, paths) {
  console.error(`FAIL: ${header}`);
  for (const path of paths) {
    console.error(` - ${path}`);
  }
  throw new PackageAuditError(header);
}

try {
  const raw = execFileSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: packageRoot,
    encoding: "utf-8",
    env: {
      ...process.env,
      npm_config_cache: cacheDir,
    },
  });

  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.length !== 1 || typeof parsed[0] !== "object") {
    throw new Error("Unexpected npm pack --dry-run JSON shape.");
  }

  const pack = parsed[0];
  const files = Array.isArray(pack.files) ? pack.files : [];
  const paths = files
    .map((entry) => (entry && typeof entry.path === "string" ? entry.path : null))
    .filter((value) => value !== null);

  const compiledTests = paths.filter((path) => path.includes("__tests__/"));
  if (compiledTests.length > 0) {
    fail("compiled test files are included in the npm tarball.", compiledTests);
  }

  const missingRequired = requiredFiles.filter((path) => !paths.includes(path));
  if (missingRequired.length > 0) {
    fail("required files are missing from the npm tarball.", missingRequired);
  }

  if (pack.name !== "ai-knot") {
    throw new Error(`Unexpected tarball package name: ${pack.name}`);
  }

  console.log(`Tarball: ${pack.filename}`);
  console.log(`Entries: ${pack.entryCount}`);
  console.log(`Unpacked size: ${pack.unpackedSize}`);
  console.log("No compiled test files in tarball.");
  console.log("Required runtime files present.");
} catch (error) {
  if (error instanceof PackageAuditError) {
    process.exitCode = 1;
  } else {
    throw error;
  }
} finally {
  rmSync(cacheDir, { recursive: true, force: true });
}
