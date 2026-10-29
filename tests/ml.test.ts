import { describe, expect, it } from "vitest";
import {
  FEATURE_COLUMNS,
  type Customer,
  type CustomerValues,
} from "../src/data";
import {
  calculateMetrics,
  featureImportances,
  fitDecisionTree,
  fitLogisticRegression,
  fitPreprocessor,
  predict,
  stratifiedSplit,
  transformRows,
} from "../src/ml";

function makeRows(count = 100): Customer[] {
  return Array.from({ length: count }, (_, index) => {
    const churn = index % 4 === 0 ? "Yes" : "No";
    const values = Object.fromEntries(
      FEATURE_COLUMNS.map((column) => [column, "No"]),
    ) as CustomerValues;
    Object.assign(values, {
      gender: index % 2 ? "Male" : "Female",
      SeniorCitizen: String(index % 2),
      Partner: index % 3 ? "Yes" : "No",
      Dependents: index % 5 ? "No" : "Yes",
      tenure: String(churn === "Yes" ? index % 9 + 1 : 24 + index % 48),
      PhoneService: "Yes",
      MultipleLines: index % 2 ? "Yes" : "No",
      InternetService: "Fiber optic",
      OnlineSecurity: index % 3 ? "No" : "Yes",
      OnlineBackup: "No",
      DeviceProtection: "No",
      TechSupport: "No",
      StreamingTV: "No",
      StreamingMovies: "No",
      Contract: churn === "Yes" ? "Month-to-month" : index % 2 ? "One year" : "Two year",
      PaperlessBilling: "Yes",
      PaymentMethod: index % 2 ? "Electronic check" : "Credit card",
      MonthlyCharges: String(churn === "Yes" ? 80 + index % 20 : 40 + index % 20),
      TotalCharges: String(index * 34),
    });
    return { values, churn, customerID: "id-" + index };
  });
}

describe("training, prediction, and evaluation", () => {
  it("makes a reproducible stratified split with both classes in every partition", () => {
    const rows = makeRows();
    const first = stratifiedSplit(rows, 42);
    const second = stratifiedSplit(rows, 42);
    expect(first.train.map((row) => row.customerID)).toEqual(
      second.train.map((row) => row.customerID),
    );
    for (const partition of [first.train, first.validation, first.test]) {
      expect(partition.some((row) => row.churn === "Yes")).toBe(true);
      expect(partition.some((row) => row.churn === "No")).toBe(true);
    }
    expect(
      new Set([...first.train, ...first.validation, ...first.test].map((row) => row.customerID)).size,
    ).toBe(rows.length);
  });

  it("fits numeric and categorical preprocessing on training rows only", () => {
    const rows = makeRows();
    const split = stratifiedSplit(rows);
    const preprocessing = fitPreprocessor(split.train);
    const expectedTrainMean =
      split.train.reduce((sum, row) => sum + Number(row.values.tenure), 0) / split.train.length;
    expect(preprocessing.numeric.tenure?.mean).toBeCloseTo(expectedTrainMean, 10);
    expect(preprocessing.numeric.tenure?.mean).not.toBeCloseTo(
      rows.reduce((sum, row) => sum + Number(row.values.tenure), 0) / rows.length,
      3,
    );
    expect(preprocessing.features.some((feature) => /Churn|customerID/.test(feature.name))).toBe(false);
    const transformed = transformRows(split.test, preprocessing);
    expect(transformed).toHaveLength(split.test.length);
    const monthToMonthIndex = preprocessing.features.findIndex(
      (feature) => feature.name === "Contract = Month-to-month",
    );
    expect(monthToMonthIndex).toBeGreaterThanOrEqual(0);
    expect(transformed[0]?.[monthToMonthIndex]).toBe(1);
  });

  it("trains both real classifiers and returns bounded probabilities and fitted importance", () => {
    const split = stratifiedSplit(makeRows());
    const preprocessing = fitPreprocessor(split.train);
    const trainFeatures = transformRows(split.train, preprocessing);
    const labels = split.train.map((row) => row.churn === "Yes");
    const logistic = fitLogisticRegression(trainFeatures, labels);
    const tree = fitDecisionTree(trainFeatures, labels, 4, 5);
    const testFeatures = transformRows(split.test, preprocessing);
    for (const model of [logistic, tree]) {
      const probabilities = testFeatures.map((vector) => predict(model, vector));
      expect(probabilities).toHaveLength(split.test.length);
      expect(probabilities.every((value) => value >= 0 && value <= 1)).toBe(true);
      expect(featureImportances(model, preprocessing)).not.toHaveLength(0);
    }
  });

  it("calculates confusion metrics and ROC-AUC from scores", () => {
    const metrics = calculateMetrics(
      [true, false, true, false],
      [0.9, 0.8, 0.4, 0.1],
      0.5,
    );
    expect(metrics).toMatchObject({ tp: 1, fp: 1, tn: 1, fn: 1 });
    expect(metrics.accuracy).toBe(0.5);
    expect(metrics.precision).toBe(0.5);
    expect(metrics.recall).toBe(0.5);
    expect(metrics.f1).toBe(0.5);
    expect(metrics.rocAuc).toBe(0.75);
  });

  it("keeps undefined precision and F1 explicit when no rows are flagged", () => {
    const metrics = calculateMetrics([true, false], [0.1, 0.2], 0.5);
    expect(metrics.precision).toBeNull();
    expect(metrics.recall).toBe(0);
    expect(metrics.f1).toBeNull();
  });

  it("changes classifications when the threshold changes without retraining", () => {
    const atHalf = calculateMetrics([true, false, true], [0.62, 0.51, 0.22], 0.5);
    const atSeventy = calculateMetrics([true, false, true], [0.62, 0.51, 0.22], 0.7);
    expect(atHalf.tp + atHalf.fp).toBe(2);
    expect(atSeventy.tp + atSeventy.fp).toBe(0);
    expect(atHalf.rocAuc).toBe(atSeventy.rocAuc);
  });

  it("reports undefined ROC-AUC when a validation set has one class", () => {
    expect(calculateMetrics([true, true], [0.8, 0.4]).rocAuc).toBeNull();
  });
});
