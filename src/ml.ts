import {
  CATEGORICAL_COLUMNS,
  FEATURE_LABELS,
  NUMERIC_COLUMNS,
  type Customer,
  type FeatureColumn,
} from "./data";

export interface DatasetSplit {
  train: Customer[];
  validation: Customer[];
  test: Customer[];
}

export interface EncodedFeature {
  name: string;
  column: FeatureColumn;
}

interface NumericStats {
  median: number;
  mean: number;
  scale: number;
}

interface CategoryStats {
  mode: string;
  categories: string[];
}

export interface Preprocessor {
  features: EncodedFeature[];
  numeric: Partial<Record<FeatureColumn, NumericStats>>;
  categorical: Partial<Record<FeatureColumn, CategoryStats>>;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface Metrics extends ConfusionMatrix {
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  rocAuc: number | null;
}

export interface LogisticModel {
  kind: "logistic";
  weights: number[];
  intercept: number;
  importances: number[];
}

interface TreeNode {
  probability: number;
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface DecisionTreeModel {
  kind: "tree";
  root: TreeNode;
  importances: number[];
}

export type ChurnModel = LogisticModel | DecisionTreeModel;

export interface ImportanceItem {
  name: string;
  strength: number;
  direction: "increases" | "decreases" | "split";
}

export interface ExplanationItem {
  name: string;
  direction: "increases" | "decreases" | "split";
}

function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex]!, result[index]!];
  }
  return result;
}

export function stratifiedSplit(rows: Customer[], seed = 42): DatasetSplit {
  const positives = rows.filter((row) => row.churn === "Yes");
  const negatives = rows.filter((row) => row.churn === "No");
  if (positives.length < 7 || negatives.length < 7) {
    throw new Error("Use at least seven Churn=Yes and seven Churn=No rows to create a stratified 70/15/15 split.");
  }

  const random = randomGenerator(seed);
  const splitClass = (group: Customer[]) => {
    const shuffled = shuffle(group, random);
    const testSize = Math.max(1, Math.floor(shuffled.length * 0.15));
    const validationSize = Math.max(1, Math.floor(shuffled.length * 0.15));
    return {
      test: shuffled.slice(0, testSize),
      validation: shuffled.slice(testSize, testSize + validationSize),
      train: shuffled.slice(testSize + validationSize),
    };
  };
  const yes = splitClass(positives);
  const no = splitClass(negatives);
  return {
    train: [...yes.train, ...no.train],
    validation: [...yes.validation, ...no.validation],
    test: [...yes.test, ...no.test],
  };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : sorted[middle] ?? 0;
}

function mode(values: string[]): string {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))[0]?.[0] ?? "(missing)";
}

export function fitPreprocessor(rows: Customer[]): Preprocessor {
  if (rows.length === 0) throw new Error("Cannot fit preprocessing without training rows.");

  const numeric: Preprocessor["numeric"] = {};
  const categorical: Preprocessor["categorical"] = {};
  const features: EncodedFeature[] = [];

  for (const column of NUMERIC_COLUMNS) {
    const observed = rows
      .map((row) => row.values[column].trim())
      .filter((value) => value !== "")
      .map(Number);
    const fill = median(observed);
    const complete = rows.map((row) => row.values[column].trim() === "" ? fill : Number(row.values[column]));
    const mean = complete.reduce((sum, value) => sum + value, 0) / complete.length;
    const variance =
      complete.reduce((sum, value) => sum + (value - mean) ** 2, 0) / complete.length;
    numeric[column] = { median: fill, mean, scale: Math.sqrt(variance) || 1 };
    features.push({ name: FEATURE_LABELS[column], column });
  }

  for (const column of CATEGORICAL_COLUMNS) {
    const observed = rows
      .map((row) => row.values[column].trim())
      .filter((value) => value !== "");
    const categories = [...new Set(observed)].sort((left, right) => left.localeCompare(right));
    const fill = mode(observed);
    categorical[column] = { mode: fill, categories };
    for (const category of categories) {
      features.push({ name: FEATURE_LABELS[column] + " = " + category, column });
    }
  }

  return { features, numeric, categorical };
}

export function transformRows(rows: Customer[], preprocessor: Preprocessor): number[][] {
  return rows.map((row) => transformCustomer(row, preprocessor));
}

export function transformCustomer(row: Customer, preprocessor: Preprocessor): number[] {
  const values: number[] = [];
  for (const column of NUMERIC_COLUMNS) {
    const stats = preprocessor.numeric[column];
    if (!stats) throw new Error("Preprocessing is missing " + column + ".");
    const raw = row.values[column].trim();
    const value = raw === "" ? stats.median : Number(raw);
    values.push((value - stats.mean) / stats.scale);
  }

  for (const column of CATEGORICAL_COLUMNS) {
    const stats = preprocessor.categorical[column];
    if (!stats) throw new Error("Preprocessing is missing " + column + ".");
    const raw = row.values[column].trim() || stats.mode;
    for (const category of stats.categories) values.push(raw === category ? 1 : 0);
  }
  return values;
}

function sigmoid(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-Math.min(value, 35)));
  const exponent = Math.exp(Math.max(value, -35));
  return exponent / (1 + exponent);
}

export function fitLogisticRegression(
  features: number[][],
  labels: boolean[],
): LogisticModel {
  if (features.length === 0 || features.length !== labels.length) {
    throw new Error("Logistic Regression needs matching, non-empty features and labels.");
  }
  const featureCount = features[0]?.length ?? 0;
  if (featureCount === 0 || features.some((row) => row.length !== featureCount)) {
    throw new Error("Logistic Regression received an inconsistent feature matrix.");
  }

  const weights = Array(featureCount).fill(0) as number[];
  const positiveRate = labels.filter(Boolean).length / labels.length;
  let intercept = Math.log((positiveRate + 1e-6) / (1 - positiveRate + 1e-6));
  const learningRate = 0.12;
  const l2 = 0.03;
  const epochs = 160;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradient = Array(featureCount).fill(0) as number[];
    let interceptGradient = 0;
    for (let rowIndex = 0; rowIndex < features.length; rowIndex += 1) {
      const vector = features[rowIndex]!;
      let score = intercept;
      for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
        score += weights[featureIndex]! * vector[featureIndex]!;
      }
      const error = sigmoid(score) - (labels[rowIndex] ? 1 : 0);
      interceptGradient += error;
      for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
        gradient[featureIndex] =
          gradient[featureIndex]! + error * vector[featureIndex]!;
      }
    }

    intercept -= learningRate * interceptGradient / features.length;
    for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
      const weight = weights[featureIndex]!;
      weights[featureIndex] =
        weight - learningRate * (gradient[featureIndex]! / features.length + l2 * weight);
    }
  }

  return {
    kind: "logistic",
    weights,
    intercept,
    importances: weights.map(Math.abs),
  };
}

export function predictLogistic(model: LogisticModel, vector: number[]): number {
  let score = model.intercept;
  for (let index = 0; index < model.weights.length; index += 1) {
    score += model.weights[index]! * vector[index]!;
  }
  return sigmoid(score);
}

function gini(positiveCount: number, total: number): number {
  if (total === 0) return 0;
  const probability = positiveCount / total;
  return 2 * probability * (1 - probability);
}

export function fitDecisionTree(
  features: number[][],
  labels: boolean[],
  maxDepth = 4,
  minLeafSize = 30,
): DecisionTreeModel {
  if (features.length === 0 || features.length !== labels.length) {
    throw new Error("Decision Tree needs matching, non-empty features and labels.");
  }
  const featureCount = features[0]?.length ?? 0;
  if (featureCount === 0 || features.some((row) => row.length !== featureCount)) {
    throw new Error("Decision Tree received an inconsistent feature matrix.");
  }

  const importances = Array(featureCount).fill(0) as number[];
  const initialIndices = Array.from({ length: labels.length }, (_, index) => index);
  const build = (indices: number[], depth: number): TreeNode => {
    const positives = indices.reduce((sum, index) => sum + Number(labels[index]), 0);
    const probability = positives / indices.length;
    const node: TreeNode = { probability };
    if (
      depth >= maxDepth ||
      indices.length < minLeafSize * 2 ||
      positives === 0 ||
      positives === indices.length
    ) {
      return node;
    }

    const parentGini = gini(positives, indices.length);
    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;
    for (let featureIndex = 0; featureIndex < featureCount; featureIndex += 1) {
      const ordered = indices
        .map((rowIndex) => ({
          value: features[rowIndex]![featureIndex]!,
          positive: labels[rowIndex]!,
        }))
        .sort((left, right) => left.value - right.value);
      let leftPositives = 0;
      for (let position = 1; position < ordered.length; position += 1) {
        leftPositives += Number(ordered[position - 1]!.positive);
        const leftCount = position;
        const rightCount = ordered.length - position;
        if (
          leftCount < minLeafSize ||
          rightCount < minLeafSize ||
          ordered[position]!.value === ordered[position - 1]!.value
        ) {
          continue;
        }
        const rightPositives = positives - leftPositives;
        const impurity =
          (leftCount * gini(leftPositives, leftCount) +
            rightCount * gini(rightPositives, rightCount)) /
          indices.length;
        const gain = parentGini - impurity;
        if (gain > bestGain + 1e-12) {
          bestGain = gain;
          bestFeature = featureIndex;
          bestThreshold =
            ordered[position - 1]!.value +
            (ordered[position]!.value - ordered[position - 1]!.value) / 2;
        }
      }
    }

    if (bestFeature < 0) return node;
    const left = indices.filter(
      (index) => features[index]![bestFeature]! <= bestThreshold,
    );
    const right = indices.filter(
      (index) => features[index]![bestFeature]! > bestThreshold,
    );
    if (left.length < minLeafSize || right.length < minLeafSize) return node;
    importances[bestFeature] =
      importances[bestFeature]! + bestGain * indices.length;
    node.featureIndex = bestFeature;
    node.threshold = bestThreshold;
    node.left = build(left, depth + 1);
    node.right = build(right, depth + 1);
    return node;
  };

  return {
    kind: "tree",
    root: build(initialIndices, 0),
    importances,
  };
}

export function predictTree(model: DecisionTreeModel, vector: number[]): number {
  let node = model.root;
  while (node.featureIndex !== undefined) {
    const next =
      vector[node.featureIndex]! <= node.threshold! ? node.left : node.right;
    if (!next) break;
    node = next;
  }
  return node.probability;
}

export function predict(model: ChurnModel, vector: number[]): number {
  return model.kind === "logistic"
    ? predictLogistic(model, vector)
    : predictTree(model, vector);
}

export function confusionMatrix(
  labels: boolean[],
  probabilities: number[],
  threshold = 0.5,
): ConfusionMatrix {
  if (labels.length !== probabilities.length) {
    throw new Error("Metric labels and probabilities must have equal length.");
  }
  return labels.reduce<ConfusionMatrix>(
    (matrix, actual, index) => {
      const predicted = probabilities[index]! >= threshold;
      if (actual && predicted) matrix.tp += 1;
      else if (!actual && predicted) matrix.fp += 1;
      else if (!actual && !predicted) matrix.tn += 1;
      else matrix.fn += 1;
      return matrix;
    },
    { tp: 0, fp: 0, tn: 0, fn: 0 },
  );
}

function rocAuc(labels: boolean[], probabilities: number[]): number | null {
  const positives = labels.filter(Boolean).length;
  const negatives = labels.length - positives;
  if (positives === 0 || negatives === 0 || labels.length !== probabilities.length) {
    return null;
  }

  const ordered = probabilities
    .map((probability, index) => ({ probability, positive: labels[index]! }))
    .sort((left, right) => left.probability - right.probability);
  let positiveRankSum = 0;
  for (let start = 0; start < ordered.length; ) {
    let end = start + 1;
    while (
      end < ordered.length &&
      ordered[end]!.probability === ordered[start]!.probability
    ) {
      end += 1;
    }
    const averageRank = (start + 1 + end) / 2;
    for (let index = start; index < end; index += 1) {
      if (ordered[index]!.positive) positiveRankSum += averageRank;
    }
    start = end;
  }
  return (
    (positiveRankSum - (positives * (positives + 1)) / 2) /
    (positives * negatives)
  );
}

export function calculateMetrics(
  labels: boolean[],
  probabilities: number[],
  threshold = 0.5,
): Metrics {
  if (labels.length === 0 || labels.length !== probabilities.length) {
    throw new Error("Metrics need matching, non-empty labels and probabilities.");
  }
  if (probabilities.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("Model probabilities must be finite values between zero and one.");
  }
  const confusion = confusionMatrix(labels, probabilities, threshold);
  const precisionDenominator = confusion.tp + confusion.fp;
  const recallDenominator = confusion.tp + confusion.fn;
  const precision = precisionDenominator === 0 ? null : confusion.tp / precisionDenominator;
  const recall = recallDenominator === 0 ? null : confusion.tp / recallDenominator;
  const f1 =
    precision === null || recall === null || precision + recall === 0
      ? null
      : (2 * precision * recall) / (precision + recall);
  return {
    ...confusion,
    accuracy: (confusion.tp + confusion.tn) / labels.length,
    precision,
    recall,
    f1,
    rocAuc: rocAuc(labels, probabilities),
  };
}

export function featureImportances(
  model: ChurnModel,
  preprocessor: Preprocessor,
  limit = 8,
): ImportanceItem[] {
  const total = model.importances.reduce((sum, value) => sum + value, 0);
  if (total === 0) return [];
  return model.importances
    .map((strength, index) => ({
      name: preprocessor.features[index]?.name ?? "Feature",
      strength: strength / total,
      direction:
        model.kind === "tree"
          ? "split" as const
          : model.weights[index]! >= 0
            ? "increases" as const
            : "decreases" as const,
    }))
    .filter((item) => item.strength > 0)
    .sort((left, right) => right.strength - left.strength)
    .slice(0, limit);
}

export function explainPrediction(
  model: ChurnModel,
  preprocessor: Preprocessor,
  vector: number[],
  limit = 3,
): ExplanationItem[] {
  if (model.kind === "logistic") {
    return model.weights
      .map((weight, index) => ({
        name: preprocessor.features[index]?.name ?? "Feature",
        contribution: weight * (vector[index] ?? 0),
        direction: weight * (vector[index] ?? 0) >= 0
          ? "increases" as const
          : "decreases" as const,
      }))
      .filter((item) => Math.abs(item.contribution) > 0.025)
      .sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution))
      .slice(0, limit)
      .map(({ name, direction }) => ({ name, direction }));
  }

  const path: ExplanationItem[] = [];
  let node = model.root;
  while (node.featureIndex !== undefined && path.length < limit) {
    const featureIndex = node.featureIndex;
    const goesLeft = vector[featureIndex]! <= node.threshold!;
    path.push({
      name: preprocessor.features[featureIndex]?.name ?? "Feature",
      direction: "split",
    });
    const next = goesLeft ? node.left : node.right;
    if (!next) break;
    node = next;
  }
  return path;
}

export function churnTargets(rows: Customer[]): boolean[] {
  return rows.map((row) => row.churn === "Yes");
}
