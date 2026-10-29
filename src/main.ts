import "./styles.css";
import {
  FEATURE_COLUMNS,
  NUMERIC_COLUMNS,
  parseCsvFile,
  parseDataset,
  type Customer,
  type CustomerValues,
  type Dataset,
} from "./data";
import {
  calculateMetrics,
  churnTargets,
  fitDecisionTree,
  fitLogisticRegression,
  fitPreprocessor,
  predict,
  stratifiedSplit,
  transformRows,
} from "./ml";
import {
  renderDashboard,
  updateThresholdView,
  type CandidateResult,
  type DashboardState,
  type ModelKey,
  type TrainedResults,
} from "./ui";

const root = (() => {
  const element = document.querySelector<HTMLElement>("#app");
  if (!element) throw new Error("Application root element is missing.");
  return element;
})();

const state: DashboardState = {
  dataset: { rows: [], source: "Loading sample", headers: [], hasCustomerId: false, duplicateIds: 0, missing: {} },
  defaultDataset: { rows: [], source: "Loading sample", headers: [], hasCustomerId: false, duplicateIds: 0, missing: {} },
  trained: null,
  selected: "logistic",
  threshold: 0.5,
  loading: true,
  error: "",
  message: "",
  predictionRow: null,
  formError: "",
};

function labelsFor(rows: Customer[]): boolean[] {
  return churnTargets(rows);
}

function trainCandidates(dataset: Dataset): TrainedResults {
  const split = stratifiedSplit(dataset.rows, 42);
  const validationLabels = labelsFor(split.validation);
  const testLabels = labelsFor(split.test);

  const selectionPreprocessor = fitPreprocessor(split.train);
  const selectionTrain = transformRows(split.train, selectionPreprocessor);
  const selectionValidation = transformRows(split.validation, selectionPreprocessor);
  const selectionLabels = labelsFor(split.train);
  const logisticForSelection = fitLogisticRegression(selectionTrain, selectionLabels);
  const treeForSelection = fitDecisionTree(selectionTrain, selectionLabels, 4, 30);
  const logisticValidationProbabilities = selectionValidation.map((vector) =>
    predict(logisticForSelection, vector),
  );
  const treeValidationProbabilities = selectionValidation.map((vector) =>
    predict(treeForSelection, vector),
  );
  const logisticValidationF1 =
    calculateMetrics(validationLabels, logisticValidationProbabilities).f1 ?? -1;
  const treeValidationF1 =
    calculateMetrics(validationLabels, treeValidationProbabilities).f1 ?? -1;
  const selected: ModelKey =
    logisticValidationF1 >= treeValidationF1 ? "logistic" : "tree";

  const finalPreprocessor = selectionPreprocessor;
  const testFeatures = transformRows(split.test, finalPreprocessor);
  const logisticModel = logisticForSelection;
  const treeModel = treeForSelection;
  const logisticTestProbabilities = testFeatures.map((vector) => predict(logisticModel, vector));
  const treeTestProbabilities = testFeatures.map((vector) => predict(treeModel, vector));

  const candidate = (
    model: CandidateResult["model"],
    testProbabilities: number[],
  ): CandidateResult => ({
    model,
    preprocessor: finalPreprocessor,
    validationLabels,
    validationProbabilities:
      model.kind === "logistic"
        ? logisticValidationProbabilities
        : treeValidationProbabilities,
    testMetrics: calculateMetrics(testLabels, testProbabilities),
  });

  return {
    logistic: candidate(logisticModel, logisticTestProbabilities),
    tree: candidate(treeModel, treeTestProbabilities),
    selected,
    trainCount: split.train.length,
    validationCount: split.validation.length,
    testCount: split.test.length,
    runDate: new Date().toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }),
  };
}

async function nextPaint(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

async function train(): Promise<void> {
  state.loading = true;
  state.error = "";
  state.predictionRow = null;
  renderDashboard(root, state);
  await nextPaint();
  try {
    state.trained = trainCandidates(state.dataset);
    state.selected = state.trained.selected;
    state.threshold = 0.5;
    state.error = "";
  } catch (error) {
    state.trained = null;
    state.error = error instanceof Error ? error.message : "Model training failed.";
  } finally {
    state.loading = false;
    renderDashboard(root, state);
  }
}

function showPredictionError(message: string): void {
  state.formError = message;
  const error = root.querySelector<HTMLElement>("#form-error");
  if (error) error.textContent = message;
}

function handlePrediction(form: HTMLFormElement): void {
  if (!state.trained) return;
  const data = new FormData(form);
  const values = Object.fromEntries(
    FEATURE_COLUMNS.map((column) => [column, String(data.get(column) ?? "").trim()]),
  ) as CustomerValues;

  for (const column of FEATURE_COLUMNS) {
    const value = values[column];
    if (!value) {
      showPredictionError("Complete every customer field before estimating risk.");
      return;
    }
    if (NUMERIC_COLUMNS.includes(column)) {
      const number = Number(value);
      if (!Number.isFinite(number) || number < 0) {
        showPredictionError("Enter a valid, non-negative number for " + column + ".");
        return;
      }
      if ((column === "SeniorCitizen" && number !== 0 && number !== 1) ||
          (column === "tenure" && !Number.isInteger(number))) {
        showPredictionError(column + " must be a whole-number value in its supported range.");
        return;
      }
    }
  }

  state.formError = "";
  state.predictionRow = { values, churn: "No", customerID: "" };
  renderDashboard(root, state);
}

async function handleFile(file: File): Promise<void> {
  try {
    const dataset = await parseCsvFile(file);
    state.dataset = dataset;
    state.trained = null;
    state.selected = "logistic";
    state.threshold = 0.5;
    state.error = "";
    state.message =
      "Loaded locally: " + dataset.rows.length + " rows" +
      (dataset.duplicateIds ? "; removed " + dataset.duplicateIds + " duplicate IDs." : ".");
    await train();
  } catch (error) {
    state.message =
      "CSV was not loaded: " +
      (error instanceof Error ? error.message : "The file could not be read.");
    renderDashboard(root, state);
  }
}

root.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const button = target.closest<HTMLButtonElement>("[data-action]");
  if (!button || button.disabled) return;

  if (button.dataset.action === "upload") {
    root.querySelector<HTMLInputElement>("#csv-file")?.click();
  } else if (button.dataset.action === "train") {
    void train();
  } else if (button.dataset.action === "reset") {
    state.dataset = state.defaultDataset;
    state.trained = null;
    state.selected = "logistic";
    state.threshold = 0.5;
    state.error = "";
    state.message = "Reset complete. The local IBM sample is active; uploaded rows and their model results were cleared.";
    state.predictionRow = null;
    state.formError = "";
    void train();
  }
});

root.addEventListener("change", (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement && target.id === "csv-file") {
    const file = target.files?.[0];
    if (file) void handleFile(file);
  } else if (target instanceof HTMLSelectElement && target.id === "model-choice") {
    state.selected = target.value as ModelKey;
    renderDashboard(root, state);
  }
});

root.addEventListener("input", (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement && target.id === "threshold") {
    state.threshold = Number(target.value);
    updateThresholdView(root, state);
  }
});

root.addEventListener("submit", (event) => {
  const target = event.target;
  if (target instanceof HTMLFormElement && target.id === "customer-form") {
    event.preventDefault();
    handlePrediction(target);
  }
});

async function initialize(): Promise<void> {
  renderDashboard(root, state);
  try {
    const response = await fetch(import.meta.env.BASE_URL + "data/telco-customer-churn.csv");
    if (!response.ok) throw new Error("The bundled sample dataset could not be loaded.");
    const text = await response.text();
    const dataset = parseDataset(text, "IBM Telco sample");
    state.dataset = dataset;
    state.defaultDataset = dataset;
    state.message =
      "Source: IBM Telco Customer Churn sample · " + dataset.rows.length +
      " rows · " + dataset.duplicateIds + " duplicate IDs removed.";
    await train();
  } catch (error) {
    state.loading = false;
    state.error =
      error instanceof Error
        ? error.message
        : "The sample dataset could not be loaded. Import a compatible CSV to continue.";
    state.message = "Sample data did not load. Import a compatible CSV file to begin.";
    renderDashboard(root, state);
  }
}

void initialize();
