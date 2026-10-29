import { describe, expect, it } from "vitest";
import viteConfig from "../vite.config";

describe("GitHub Pages build settings", () => {
  it("builds assets under the repository subpath", () => {
    expect(viteConfig.base).toBe("/customer-churn-prediction/");
  });
});
