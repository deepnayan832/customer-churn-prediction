import { describe, expect, it } from "vitest";
import {
  FEATURE_COLUMNS,
  parseCsvFile,
  parseCsv,
  parseDataset,
  type Customer,
  type CustomerValues,
} from "../src/data";

function row(index: number, churn: "Yes" | "No" = index % 3 === 0 ? "Yes" : "No"): Customer {
  const values = Object.fromEntries(
    FEATURE_COLUMNS.map((column) => [column, "No"]),
  ) as CustomerValues;
  Object.assign(values, {
    gender: index % 2 ? "Male" : "Female",
    SeniorCitizen: String(index % 2),
    Partner: index % 2 ? "Yes" : "No",
    Dependents: "No",
    tenure: String(6 + index),
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "Yes",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "No",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: "72.35",
    TotalCharges: "434.10",
  });
  return { values, churn, customerID: "customer-" + index };
}

function csvCell(value: string): string {
  return /[",\r\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
}

function datasetText(customRows: Customer[], include = [...FEATURE_COLUMNS, "Churn", "customerID"]): string {
  const header = include.join(",");
  const lines = customRows.map((customer) =>
    include.map((column) =>
      column === "Churn"
        ? customer.churn
        : column === "customerID"
          ? customer.customerID
          : customer.values[column as keyof CustomerValues],
    ).map(csvCell).join(","),
  );
  return [header, ...lines].join("\r\n");
}

describe("CSV dataset parsing", () => {
  it("keeps quoted commas and escaped quotes intact while excluding identifiers from features", () => {
    const custom = row(1);
    custom.values.PaymentMethod = 'Bank, "premium" transfer';
    const parsed = parseDataset(datasetText([custom]), "test file");
    expect(parsed.rows).toHaveLength(1);
    expect(parsed.rows[0]?.values.PaymentMethod).toBe('Bank, "premium" transfer');
    expect(parsed.rows[0]?.customerID).toBe("customer-1");
    expect(FEATURE_COLUMNS).not.toContain("customerID" as never);
  });

  it("removes later duplicate IDs and reports blank values", () => {
    const first = row(1);
    const duplicate = row(2, "Yes");
    duplicate.customerID = first.customerID;
    first.values.TotalCharges = "";
    const parsed = parseDataset(datasetText([first, duplicate]), "test file");
    expect(parsed.rows).toHaveLength(1);
    expect(parsed.duplicateIds).toBe(1);
    expect(parsed.missing.TotalCharges).toBe(1);
  });

  it("accepts files with no customerID and validates required columns", () => {
    const parsed = parseDataset(datasetText([row(1)], [...FEATURE_COLUMNS, "Churn"]), "custom");
    expect(parsed.hasCustomerId).toBe(false);
    expect(() =>
      parseDataset(datasetText([row(1)], [...FEATURE_COLUMNS.slice(0, -1), "Churn"]), "bad"),
    ).toThrow(/Missing required columns: TotalCharges/);
  });

  it("rejects malformed values without echoing the customer cell", () => {
    const invalid = row(1);
    invalid.values.MonthlyCharges = "private-cell-content";
    let message = "";
    try {
      parseDataset(datasetText([invalid]), "upload.csv");
    } catch (error) {
      message = error instanceof Error ? error.message : "";
    }
    expect(message).toMatch(/invalid number in MonthlyCharges/);
    expect(message).not.toContain("private-cell-content");
  });

  it("rejects malformed quoting and invalid target labels", () => {
    const header = [...FEATURE_COLUMNS, "Churn", "customerID"].join(",");
    expect(() => parseDataset(header + '\r\n"unterminated', "broken")).toThrow(/ends inside a quoted field/);
    const invalid = row(1);
    invalid.churn = "Yes";
    const text = datasetText([invalid]).replace(",Yes,customer-1", ",Maybe,customer-1");
    expect(() => parseDataset(text, "bad label")).toThrow(/must use Yes or No in Churn/);
  });

  it("reports empty and structurally malformed CSV files", () => {
    expect(() => parseCsv(" \r\n\t")).toThrow(/CSV file is empty/);
    const header = [...FEATURE_COLUMNS, "Churn", "customerID"].join(",");
    const extraFields = Array(FEATURE_COLUMNS.length + 3).fill("extra").join(",");
    expect(() => parseDataset(header + "\r\n" + extraFields, "too many fields")).toThrow(
      /more fields than the header/,
    );
    expect(() => parseDataset(header + "\r\n" + '"broken"suffix', "bad quoting")).toThrow(
      /unexpected text after a quoted field/,
    );
  });

  it("reads a browser File object in memory", async () => {
    const file = new File([datasetText([row(4)])], "customers.csv", { type: "text/csv" });
    const parsed = await parseCsvFile(file);
    expect(parsed.source).toBe("Uploaded locally: customers.csv");
    expect(parsed.rows).toHaveLength(1);
  });

  it("rejects oversized uploads before reading their contents", async () => {
    const file = {
      name: "large.csv",
      size: 10 * 1024 * 1024 + 1,
      text: async () => {
        throw new Error("The file should not be read.");
      },
    };
    await expect(parseCsvFile(file)).rejects.toThrow(/no larger than 10 MB/);
  });

  it("rejects datasets with too many rows", () => {
    const header = [...FEATURE_COLUMNS, "Churn"].join(",");
    const rows = Array.from({ length: 100_001 }, () =>
      [...FEATURE_COLUMNS.map(() => "No"), "No"].join(","),
    );
    expect(() => parseDataset([header, ...rows].join("\n"), "large")).toThrow(
      /at most 100,000 customer rows/,
    );
  });
});
