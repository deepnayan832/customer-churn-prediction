export const FEATURE_COLUMNS = [
  "gender",
  "SeniorCitizen",
  "Partner",
  "Dependents",
  "tenure",
  "PhoneService",
  "MultipleLines",
  "InternetService",
  "OnlineSecurity",
  "OnlineBackup",
  "DeviceProtection",
  "TechSupport",
  "StreamingTV",
  "StreamingMovies",
  "Contract",
  "PaperlessBilling",
  "PaymentMethod",
  "MonthlyCharges",
  "TotalCharges",
] as const;

export type FeatureColumn = (typeof FEATURE_COLUMNS)[number];
export const NUMERIC_COLUMNS: readonly FeatureColumn[] = [
  "SeniorCitizen",
  "tenure",
  "MonthlyCharges",
  "TotalCharges",
];
export const CATEGORICAL_COLUMNS = FEATURE_COLUMNS.filter(
  (column) => !NUMERIC_COLUMNS.includes(column),
);
export const REQUIRED_COLUMNS = [...FEATURE_COLUMNS, "Churn"];

export const FEATURE_LABELS: Record<FeatureColumn, string> = {
  gender: "Gender",
  SeniorCitizen: "Senior citizen",
  Partner: "Partner",
  Dependents: "Dependents",
  tenure: "Tenure (months)",
  PhoneService: "Phone service",
  MultipleLines: "Multiple lines",
  InternetService: "Internet service",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Tech support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
};

export const FEATURE_DESCRIPTIONS: Record<FeatureColumn, string> = {
  gender: "Customer-reported gender category.",
  SeniorCitizen: "Whether the customer is classified as a senior citizen (0 or 1).",
  Partner: "Whether the customer has a partner.",
  Dependents: "Whether the customer has dependents.",
  tenure: "Number of months the customer has been with the service.",
  PhoneService: "Whether phone service is subscribed.",
  MultipleLines: "Phone line subscription status.",
  InternetService: "Internet service type.",
  OnlineSecurity: "Online security add-on status.",
  OnlineBackup: "Online backup add-on status.",
  DeviceProtection: "Device protection add-on status.",
  TechSupport: "Tech support add-on status.",
  StreamingTV: "Streaming TV add-on status.",
  StreamingMovies: "Streaming movies add-on status.",
  Contract: "Billing contract duration.",
  PaperlessBilling: "Whether paperless billing is enabled.",
  PaymentMethod: "Payment method category.",
  MonthlyCharges: "Current monthly amount charged.",
  TotalCharges: "Total amount charged during the relationship.",
};

export type ChurnLabel = "Yes" | "No";
export type CustomerValues = Record<FeatureColumn, string>;

export interface Customer {
  values: CustomerValues;
  churn: ChurnLabel;
  customerID: string;
}

export interface Dataset {
  rows: Customer[];
  source: string;
  headers: string[];
  hasCustomerId: boolean;
  duplicateIds: number;
  missing: Record<string, number>;
}

export function parseCsv(text: string): string[][] {
  if (!text.trim()) throw new Error("The CSV file is empty.");

  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  let afterQuote = false;
  const source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (inQuotes) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else {
          inQuotes = false;
          afterQuote = true;
        }
      } else {
        cell += character;
      }
      continue;
    }

    if (afterQuote) {
      if (character === " " || character === "\t") continue;
      if (character !== "," && character !== "\n" && character !== "\r") {
        throw new Error("The CSV has unexpected text after a quoted field.");
      }
      afterQuote = false;
    }

    if (character === '"') {
      if (cell.length !== 0) {
        throw new Error("The CSV has a quote in an unquoted field.");
      }
      inQuotes = true;
    } else if (character === ",") {
      row.push(cell);
      cell = "";
    } else if (character === "\n" || character === "\r") {
      if (character === "\r" && source[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }

  if (inQuotes) throw new Error("The CSV ends inside a quoted field.");
  if (cell !== "" || row.length > 0) {
    row.push(cell);
    if (row.some((value) => value.trim() !== "")) rows.push(row);
  }
  if (rows.length < 2) throw new Error("The CSV needs a header and at least one customer row.");
  return rows;
}

export function parseDataset(text: string, source: string): Dataset {
  const table = parseCsv(text);
  const rawHeader = table[0];
  if (!rawHeader) throw new Error("The CSV header is missing.");
  const records = table.slice(1);
  const headers = rawHeader.map((name, index) =>
    (index === 0 ? name.replace(/^\uFEFF/, "") : name).trim(),
  );

  if (new Set(headers).size !== headers.length) {
    throw new Error("The CSV contains duplicate column names.");
  }

  const missingRequired = REQUIRED_COLUMNS.filter((column) => !headers.includes(column));
  if (missingRequired.length > 0) {
    throw new Error(
      "Missing required columns: " + missingRequired.join(", ") +
        ". Required fields are listed below the upload control.",
    );
  }

  const hasCustomerId = headers.includes("customerID");
  const idIndex = headers.indexOf("customerID");
  const seenIds = new Set<string>();
  const rows: Customer[] = [];
  let duplicateIds = 0;
  const missing: Record<string, number> = Object.fromEntries(
    headers.map((header) => [header, 0]),
  );

  records.forEach((record, recordIndex) => {
    if (record.length > headers.length) {
      throw new Error("Row " + (recordIndex + 2) + " has more fields than the header.");
    }
    const values = [...record, ...Array(Math.max(0, headers.length - record.length)).fill("")];
    const byHeader = Object.fromEntries(
      headers.map((header, index) => [header, (values[index] ?? "").trim()]),
    ) as Record<string, string>;
    const id = hasCustomerId ? byHeader.customerID ?? "" : "";

    if (id && seenIds.has(id)) {
      duplicateIds += 1;
      return;
    }
    if (id) seenIds.add(id);

    for (const column of headers) {
      if (!byHeader[column]) missing[column] = (missing[column] ?? 0) + 1;
    }

    for (const column of NUMERIC_COLUMNS) {
      const value = byHeader[column];
      if (value && !Number.isFinite(Number(value))) {
        throw new Error(
          "Row " + (recordIndex + 2) + " has an invalid number in " + column + ".",
        );
      }
      if (value) {
        const number = Number(value);
        if (number < 0) {
          throw new Error(
            "Row " + (recordIndex + 2) + " has a negative value in " + column + ".",
          );
        }
        if (
          (column === "SeniorCitizen" && number !== 0 && number !== 1) ||
          (column === "tenure" && !Number.isInteger(number))
        ) {
          throw new Error(
            "Row " + (recordIndex + 2) + " has an unsupported value in " + column + ".",
          );
        }
      }
    }

    const label = (byHeader.Churn ?? "").toLowerCase();
    if (label !== "yes" && label !== "no") {
      throw new Error("Row " + (recordIndex + 2) + " must use Yes or No in Churn.");
    }

    const customerValues = Object.fromEntries(
      FEATURE_COLUMNS.map((column) => [column, byHeader[column] ?? ""]),
    ) as CustomerValues;

    rows.push({
      values: customerValues,
      churn: label === "yes" ? "Yes" : "No",
      customerID: id,
    });
  });

  if (rows.length === 0) throw new Error("No usable customer rows were found.");
  return {
    rows,
    source,
    headers,
    hasCustomerId: idIndex !== -1,
    duplicateIds,
    missing,
  };
}

export async function parseCsvFile(file: Pick<File, "name" | "text">): Promise<Dataset> {
  return parseDataset(await file.text(), "Uploaded locally: " + file.name);
}
