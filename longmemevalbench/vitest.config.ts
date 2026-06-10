import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests/test_*.ts", "src/**/*.test.ts"],
    environment: "node",
    testTimeout: 30000,
  },
});
