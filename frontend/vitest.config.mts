import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

export default defineConfig({
  resolve: {
    // Mirrors the "@/*" path mapping in tsconfig.json.
    alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
  },
  test: {
    // The modules under test are pure logic over fetch — no DOM required, and
    // a jsdom environment would only slow the suite down.
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
