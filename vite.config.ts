import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/customer-churn-prediction/",
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    clearMocks: true,
  },
});
