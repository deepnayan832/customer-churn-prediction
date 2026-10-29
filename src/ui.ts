import {
  CATEGORICAL_COLUMNS,
  FEATURE_COLUMNS,
  FEATURE_DESCRIPTIONS,
  FEATURE_LABELS,
  type Customer,
  type Dataset,
  type FeatureColumn,
} from "./data";
import {
  calculateMetrics,
  explainPrediction,
  featureImportances,
  predict,
  transformCustomer,
  type ChurnModel,
  type Metrics,
  type Preprocessor,
} from "./ml";

export type ModelKey = "logistic" | "tree";

export interface CandidateResult {
  model: ChurnModel;
  preprocessor: Preprocessor;
  validationLabels: boolean[];
  validationProbabilities: number[];
  testMetrics: Metrics;
}

export interface TrainedResults {
  logistic: CandidateResult;
  tree: CandidateResult;
  selected: ModelKey;
  trainCount: number;
  validationCount: number;
  testCount: number;
  runDate: string;
}

export interface DashboardState {
  dataset: Dataset;
  defaultDataset: Dataset;
  trained: TrainedResults | null;
  selected: ModelKey;
  threshold: number;
  loading: boolean;
  error: string;
  message: string;
  predictionRow: Customer | null;
  formError: string;
}

const numberFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
const integerFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
const percent = (value: number | null): string =>
  value === null ? "Not defined" : (value * 100).toFixed(1) + "%";
const number = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);

export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => {
    const escapes: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return escapes[character]!;
  });
}

function modelName(key: ModelKey): string {
  return key === "logistic" ? "Logistic Regression" : "Decision Tree";
}

function countClass(rows: Customer[], label: "Yes" | "No"): number {
  return rows.filter((row) => row.churn === label).length;
}

function mean(rows: Customer[], column: "tenure" | "MonthlyCharges"): number | null {
  const values = rows
    .map((row) => row.values[column].trim())
    .filter((value) => value !== "")
    .map(Number);
  return values.length === 0
    ? null
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function statCard(label: string, value: string, detail: string, tone = ""): string {
  return (
    '<article class="stat-card ' + tone + '">' +
    '<span class="stat-label">' + escapeHtml(label) + "</span>" +
    '<strong class="stat-value">' + escapeHtml(value) + "</strong>" +
    '<span class="stat-detail">' + escapeHtml(detail) + "</span>" +
    "</article>"
  );
}

function churnRing(rows: Customer[]): string {
  const churned = countClass(rows, "Yes");
  const rate = rows.length ? (churned / rows.length) * 100 : 0;
  const dash = ((rate / 100) * 276).toFixed(2);
  return (
    '<div class="ring-wrap">' +
    '<svg viewBox="0 0 120 120" role="img" aria-label="' +
    escapeHtml(rate.toFixed(1) + "% churn among " + integerFormat.format(rows.length) + " loaded customers") +
    '">' +
    '<circle class="ring-track" cx="60" cy="60" r="44"></circle>' +
    '<circle class="ring-value" cx="60" cy="60" r="44" stroke-dasharray="' +
    dash + " 276\"></circle>" +
    "</svg>" +
    '<div class="ring-center"><strong>' + rate.toFixed(1) + "%</strong><span>churn rate</span></div>" +
    "</div>"
  );
}

function contractChart(rows: Customer[]): string {
  const groups = new Map<string, Customer[]>();
  for (const row of rows) {
    const contract = row.values.Contract.trim() || "(missing)";
    groups.set(contract, [...(groups.get(contract) ?? []), row]);
  }
  const bars = [...groups.entries()]
    .map(([label, members]) => ({
      label,
      members,
      rate: members.filter((row) => row.churn === "Yes").length / members.length,
    }))
    .sort((left, right) => right.rate - left.rate);
  if (!bars.length) return '<p class="empty-note">No contract values to chart.</p>';
  return (
    '<div class="bar-chart" aria-label="Churn rate by contract type">' +
    bars.map(({ label, members, rate }) =>
      '<div class="bar-row"><div class="bar-caption"><span>' +
      escapeHtml(label) + '</span><span>' + percent(rate) + " <small>· " +
      integerFormat.format(members.length) + " customers</small></span></div>" +
      '<div class="bar-track"><div class="bar-fill" style="width:' +
      Math.min(100, Math.max(0, rate * 100)).toFixed(2) + '%"></div></div></div>',
    ).join("") +
    "</div>"
  );
}

function tenureChart(rows: Customer[]): string {
  const buckets = [
    { label: "0–11 mo", min: 0, max: 12 },
    { label: "12–23 mo", min: 12, max: 24 },
    { label: "24–47 mo", min: 24, max: 48 },
    { label: "48+ mo", min: 48, max: Infinity },
  ].map((bucket) => {
    const members = rows.filter((row) => {
      const value = row.values.tenure.trim();
      return value !== "" && Number(value) >= bucket.min && Number(value) < bucket.max;
    });
    return {
      ...bucket,
      members,
      rate: members.length
        ? members.filter((row) => row.churn === "Yes").length / members.length
        : 0,
    };
  });
  return (
    '<div class="tenure-chart" role="img" aria-label="Churn rate by customer tenure">' +
    buckets.map(({ label, members, rate }) =>
      '<div class="tenure-column"><div class="tenure-bar-area"><span class="tenure-rate">' +
      (members.length ? percent(rate) : "—") + '</span><div class="tenure-bar" style="height:' +
      (members.length ? Math.max(5, rate * 100) : 0).toFixed(2) +
      '%"></div></div><strong>' + escapeHtml(label) + '</strong><small>' +
      integerFormat.format(members.length) + " customers</small></div>",
    ).join("") +
    "</div>"
  );
}

function chargesChart(rows: Customer[]): string {
  const groups = [
    { label: "No churn", churn: "No" as const, color: "green" },
    { label: "Churn", churn: "Yes" as const, color: "coral" },
  ].map((group) => {
    const values = rows
      .filter((row) => row.churn === group.churn && row.values.MonthlyCharges.trim() !== "")
      .map((row) => Number(row.values.MonthlyCharges));
    return {
      ...group,
      average: values.length
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : null,
    };
  });
  const max = Math.max(...groups.map((group) => group.average ?? 0), 1);
  return (
    '<div class="charges-chart" aria-label="Average monthly charges by churn label">' +
    groups.map((group) =>
      '<div class="charges-row"><div class="bar-caption"><span>' +
      group.label + '</span><strong>' + (group.average === null ? "—" : number(group.average)) +
      '</strong></div><div class="bar-track"><div class="bar-fill ' + group.color +
      '" style="width:' + (((group.average ?? 0) / max) * 100).toFixed(2) +
      '%"></div></div></div>',
    ).join("") +
    '<p class="chart-footnote">Mean charge among customers with a recorded MonthlyCharges value.</p>' +
    "</div>"
  );
}

function missingReport(dataset: Dataset): string {
  const missing = Object.entries(dataset.missing).filter(([, count]) => count > 0);
  return (
    '<div class="quality-summary">' +
    '<span class="quality-pill ' + (dataset.duplicateIds ? "warning" : "good") + '">' +
    (dataset.hasCustomerId
      ? integerFormat.format(dataset.duplicateIds) + " duplicate IDs removed"
      : "No customerID column") + "</span>" +
    (missing.length
      ? '<ul class="missing-list">' + missing.map(([column, count]) =>
          "<li><span>" + escapeHtml(column) + "</span><strong>" +
          integerFormat.format(count) + " blank</strong></li>",
        ).join("") + "</ul>"
      : '<span class="quality-pill good">No blank cells in loaded columns</span>') +
    '<p class="small-note">Blank numeric values are imputed with medians and blank categories with the training-set mode. These values are learned from the training partition only.</p>' +
    "</div>"
  );
}

function previewTable(rows: Customer[]): string {
  const columns: FeatureColumn[] = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "PaymentMethod",
  ];
  return (
    '<div class="table-scroll"><table class="data-table"><thead><tr>' +
    columns.map((column) => "<th>" + escapeHtml(FEATURE_LABELS[column]) + "</th>").join("") +
    "<th>Churn</th></tr></thead><tbody>" +
    rows.slice(0, 6).map((row) =>
      "<tr>" + columns.map((column) =>
        "<td>" + escapeHtml(row.values[column]) + "</td>",
      ).join("") +
      '<td><span class="label-chip ' + (row.churn === "Yes" ? "churn" : "stable") +
      '">' + row.churn + "</span></td></tr>",
    ).join("") +
    "</tbody></table></div>"
  );
}

function schemaTable(): string {
  return (
    '<details class="schema-details"><summary>View supported columns and definitions</summary>' +
    '<div class="table-scroll"><table class="data-table schema-table"><thead><tr><th>Column</th><th>Role</th><th>Meaning</th></tr></thead><tbody>' +
    '<tr><td>customerID</td><td>Identifier</td><td>Optional duplicate check only; excluded from prediction features.</td></tr>' +
    FEATURE_COLUMNS.map((column) =>
      "<tr><td>" + escapeHtml(column) + "</td><td>Feature</td><td>" +
      escapeHtml(FEATURE_DESCRIPTIONS[column]) + "</td></tr>",
    ).join("") +
    '<tr><td>Churn</td><td>Target</td><td>Original Yes or No outcome label.</td></tr>' +
    "</tbody></table></div></details>"
  );
}

function predictionField(column: FeatureColumn, dataset: Dataset): string {
  const label = FEATURE_LABELS[column];
  const id = "customer-" + column;
  if (CATEGORICAL_COLUMNS.includes(column)) {
    const options = [...new Set(dataset.rows
      .map((row) => row.values[column].trim())
      .filter((value) => value !== ""))]
      .sort((left, right) => left.localeCompare(right));
    return (
      '<label class="field" for="' + id + '"><span>' + escapeHtml(label) +
      '</span><select id="' + id + '" name="' + column + '" required>' +
      '<option value="">Choose a value</option>' +
      options.map((option) =>
        '<option value="' + escapeHtml(option) + '">' + escapeHtml(option) + "</option>",
      ).join("") +
      "</select></label>"
    );
  }
  const step = column === "SeniorCitizen" || column === "tenure" ? "1" : "any";
  const note = column === "SeniorCitizen" ? "Enter 0 or 1" : "Non-negative number";
  return (
    '<label class="field" for="' + id + '"><span>' + escapeHtml(label) +
    '<small>' + note + '</small></span><input id="' + id + '" name="' + column +
    '" type="number" min="0" ' +
    (column === "SeniorCitizen" ? 'max="1" ' : "") +
    'step="' + step + '" required></label>'
  );
}

function metricValue(value: number | null): string {
  return value === null ? "Not defined" : percent(value);
}

function confusionTable(metrics: Metrics): string {
  return (
    '<div class="matrix-scroll"><table class="matrix-table"><caption>Held-out test confusion matrix</caption><thead><tr><th></th><th>Predicted no</th><th>Predicted churn</th></tr></thead><tbody>' +
    "<tr><th>Actual no</th><td>" + integerFormat.format(metrics.tn) +
    "</td><td>" + integerFormat.format(metrics.fp) + "</td></tr>" +
    "<tr><th>Actual churn</th><td>" + integerFormat.format(metrics.fn) +
    "</td><td>" + integerFormat.format(metrics.tp) + "</td></tr>" +
    "</tbody></table></div>"
  );
}

function candidateCard(key: ModelKey, candidate: CandidateResult): string {
  const metrics = candidate.testMetrics;
  const values = [
    ["Accuracy", metricValue(metrics.accuracy)],
    ["Precision", metricValue(metrics.precision)],
    ["Recall", metricValue(metrics.recall)],
    ["F1 score", metricValue(metrics.f1)],
    ["ROC-AUC", metricValue(metrics.rocAuc)],
  ];
  return (
    '<article class="model-card"><div class="model-card-heading"><div><span class="eyebrow">HELD-OUT TEST</span><h3>' +
    modelName(key) + '</h3></div><span class="model-tag">' +
    (key === "logistic" ? "L2 · 160 steps" : "Gini · depth 4") +
    '</span></div><div class="metric-grid">' +
    values.map(([label, value]) =>
      '<div class="metric-cell"><span>' + label + '</span><strong>' + value + "</strong></div>",
    ).join("") +
    "</div>" + confusionTable(metrics) +
    '<p class="small-note">Test rows were not used to select the model or fit preprocessing. Precision, recall, and F1 are shown as undefined when their denominators are zero.</p>' +
    "</article>"
  );
}

function thresholdMetrics(trained: TrainedResults, selected: ModelKey, threshold: number): string {
  const candidate = trained[selected];
  const metrics = calculateMetrics(
    candidate.validationLabels,
    candidate.validationProbabilities,
    threshold,
  );
  const values = [
    ["Precision", metricValue(metrics.precision)],
    ["Recall", metricValue(metrics.recall)],
    ["F1 score", metricValue(metrics.f1)],
    ["Predicted churn", integerFormat.format(metrics.tp + metrics.fp)],
  ];
  return (
    '<div class="threshold-metrics">' + values.map(([label, value]) =>
      '<div><span>' + label + '</span><strong>' + value + "</strong></div>",
    ).join("") + "</div>" +
    '<p class="small-note">Validation counts: TP ' + metrics.tp +
    " · FP " + metrics.fp + " · TN " + metrics.tn + " · FN " + metrics.fn +
    ". These validation rows are used for threshold exploration; the held-out test results above stay fixed.</p>"
  );
}

function importanceList(candidate: CandidateResult): string {
  const items = featureImportances(candidate.model, candidate.preprocessor, 7);
  if (!items.length) return '<p class="empty-note">This fitted model did not use a feature split with measurable importance.</p>';
  return (
    '<ol class="importance-list">' + items.map((item) =>
      '<li><div class="importance-label"><span>' + escapeHtml(item.name) +
      '</span><strong>' + percent(item.strength) + "</strong></div>" +
      '<div class="importance-track"><span style="width:' +
      Math.max(1, item.strength * 100).toFixed(2) + '%"></span></div>' +
      '<small>' + (item.direction === "split"
        ? "impurity decrease"
        : item.direction === "increases"
          ? "positive fitted coefficient"
          : "negative fitted coefficient") +
      "</small></li>",
    ).join("") + "</ol>"
  );
}

function predictionResult(state: DashboardState): string {
  if (!state.predictionRow || !state.trained) return "";
  const candidate = state.trained[state.selected];
  const vector = transformCustomer(state.predictionRow, candidate.preprocessor);
  const actualProbability = predict(candidate.model, vector);
  const predicted = actualProbability >= state.threshold;
  const explanations = explainPrediction(candidate.model, candidate.preprocessor, vector);
  return (
    '<section class="prediction-result ' + (predicted ? "risk" : "steady") +
    '" aria-live="polite"><div class="result-heading"><div><span class="eyebrow">MODEL ESTIMATE</span><h3>' +
    (predicted ? "Churn risk flagged" : "No churn flag at this threshold") +
    '</h3></div><strong>' + percent(actualProbability) +
    '</strong></div><p>Estimated probability from ' + modelName(state.selected) +
    " at a " + percent(state.threshold) + " threshold. This estimate can be wrong and is not a certainty.</p>" +
    (explanations.length
      ? '<div class="factor-list"><strong>' +
        (state.selected === "logistic" ? "Largest fitted contributions" : "Decision path") +
        '</strong><ul>' + explanations.map((item) =>
          "<li>" + escapeHtml(item.name) +
          (item.direction === "split"
            ? " · split used"
            : item.direction === "increases"
              ? " · raises fitted churn score"
              : " · lowers fitted churn score") +
          "</li>",
        ).join("") + "</ul></div>"
      : '<p class="small-note">No single feature had a large individual fitted contribution for this prediction.</p>') +
    '<p class="small-note">Model explanations describe learned associations and paths, not causes. Review domain context before acting on any score.</p></section>'
  );
}

function modelLab(state: DashboardState): string {
  if (!state.trained) {
    return (
      '<div class="empty-state"><span class="empty-mark">↗</span><h3>' +
      (state.loading ? "Training two classifiers…" : "Train the models") +
      '</h3><p>' + (state.error
        ? escapeHtml(state.error)
        : "The app creates a stratified 70/15/15 split, then compares Logistic Regression and a shallow Decision Tree.") +
      '</p><button class="button primary" type="button" data-action="train" ' +
      (state.loading ? "disabled" : "") + ">" +
      (state.loading ? "Working…" : "Train and compare") + "</button></div>"
    );
  }

  const trained = state.trained;
  const selectedName = modelName(trained.selected);
  const size = trained.trainCount + trained.validationCount + trained.testCount;
  return (
    '<div class="model-overview"><div><span class="eyebrow">REPRODUCIBLE RUN</span><h3>70 / 15 / 15 stratified split</h3><p>Fixed seed: 42 · ' +
    integerFormat.format(trained.trainCount) + " train · " +
    integerFormat.format(trained.validationCount) + " validation · " +
    integerFormat.format(trained.testCount) + " test rows</p></div><div class=\"run-meta\"><span>Run date</span><strong>" +
    escapeHtml(trained.runDate) + "</strong><span>" + integerFormat.format(size) +
    " unique customer rows</span></div></div>" +
    '<div class="selection-note"><span class="selection-check">✓</span><p><strong>Default customer demo model: ' +
    selectedName + ".</strong> The rule is highest validation F1 at the default 0.50 threshold; exact ties select Logistic Regression. You can switch the demo model below. Both models are fitted on training rows only and compared on the untouched test split.</p></div>" +
    '<div class="model-grid">' +
    candidateCard("logistic", trained.logistic) +
    candidateCard("tree", trained.tree) +
    "</div>" +
    '<div class="lab-lower-grid"><section class="subpanel"><div class="section-head compact"><div><span class="eyebrow">THRESHOLD EXPLORER</span><h3>Adjust the decision cutoff</h3></div><span class="live-tag">Validation set</span></div>' +
    '<div class="threshold-control"><label for="threshold">Churn threshold <output id="threshold-output" for="threshold">' +
    percent(state.threshold) + '</output></label><input id="threshold" type="range" min="0.10" max="0.90" step="0.05" value="' +
    state.threshold.toFixed(2) + '"><div class="range-labels"><span>More customers flagged</span><span>Fewer customers flagged</span></div></div>' +
    '<label class="field model-choice" for="model-choice"><span>Model for customer demos<select id="model-choice">' +
    '<option value="logistic" ' + (state.selected === "logistic" ? "selected" : "") + ">Logistic Regression</option>" +
    '<option value="tree" ' + (state.selected === "tree" ? "selected" : "") + ">Decision Tree</option>" +
    "</select></span></label>" +
    '<div id="threshold-metrics">' + thresholdMetrics(trained, state.selected, state.threshold) + "</div>" +
    "</section><section class=\"subpanel\"><div class=\"section-head compact\"><div><span class=\"eyebrow\">LEARNED SIGNALS</span><h3>" +
    escapeHtml(modelName(state.selected)) + " importance</h3></div></div>" +
    importanceList(trained[state.selected]) +
    '<p class="small-note">' +
    (state.selected === "logistic"
      ? "Absolute standardized coefficient size ranks model influence; coefficient sign gives direction in log-odds. Correlated and one-hot features share effects, so this is not a causal claim."
      : "Importance is the normalized reduction in Gini impurity across fitted splits. It can favor features with many possible split points and is not causal.") +
    "</p></section></div>"
  );
}

function methodCards(): string {
  return (
    '<div class="method-grid">' +
    '<article><span class="method-index">01</span><h3>Prepare without leakage</h3><p>Identifiers and the Churn target stay out of model inputs. Numeric medians, scaling, category modes, and one-hot columns are fitted on training rows only.</p></article>' +
    '<article><span class="method-index">02</span><h3>Compare on a holdout</h3><p>A seeded stratified split sets aside validation rows for selection and threshold exploration, then a separate test set for reported final metrics. This snapshot is not a time-based forecast.</p></article>' +
    '<article><span class="method-index">03</span><h3>Use scores carefully</h3><p>These models estimate patterns in a fictional telecom sample. Association is not causation, and a score alone should never drive consequential customer decisions.</p></article>' +
    "</div>"
  );
}

export function renderDashboard(root: HTMLElement, state: DashboardState): void {
  const rows = state.dataset.rows;
  const churnCount = countClass(rows, "Yes");
  const churnRate = rows.length ? churnCount / rows.length : 0;
  const averageTenure = mean(rows, "tenure");
  const averageCharges = mean(rows, "MonthlyCharges");
  const missingCells = Object.values(state.dataset.missing).reduce((sum, value) => sum + value, 0);
  root.innerHTML =
    '<header class="topbar"><a class="brand" href="#top" aria-label="Customer Churn Prediction home"><span class="brand-mark">C</span><span>Churn<span class="brand-light">Lab</span><small>MODEL OBSERVATORY</small></span></a><nav aria-label="Main navigation"><a href="#overview">Overview</a><a href="#dataset">Dataset</a><a href="#models">Model lab</a><a href="#predict">Predict</a></nav><div class="topbar-actions"><span class="privacy-chip"><span></span>Runs in your browser</span><button class="button quiet" type="button" data-action="reset">Reset demo</button></div></header>' +
    '<main id="main" class="page-shell"><section id="top" class="hero"><div class="hero-copy"><span class="eyebrow">CUSTOMER CHURN PREDICTION</span><h1>See the signals.<br><span>Test the models.</span></h1><p>Explore churn patterns, compare two explainable classifiers, and score a customer profile. Your uploaded CSV stays in this browser.</p><div class="hero-actions"><a class="button primary" href="#models">Start analysis <span aria-hidden="true">↘</span></a><button class="button secondary" type="button" data-action="upload">Import CSV</button><input id="csv-file" type="file" accept=".csv,text/csv" hidden><span class="format-note">CSV · no server upload</span></div></div><aside class="source-card"><div class="source-icon">▤</div><div><span class="eyebrow">CURRENT DATA</span><h2>' + escapeHtml(state.dataset.source) + '</h2><p>Fictional telecom sample · ' + integerFormat.format(rows.length) + ' unique rows</p><a href="#dataset">Inspect source and data quality <span aria-hidden="true">→</span></a></div><span class="source-dot"></span></aside></section>' +
    (state.message ? '<p class="status-banner" role="status">' + escapeHtml(state.message) + "</p>" : "") +
    '<section id="overview" class="section-block"><div class="section-head"><div><span class="eyebrow">AT A GLANCE</span><h2>Customer overview</h2></div><span class="population-note">Population: currently loaded rows</span></div><div class="stats-grid">' +
    statCard("Customers analyzed", integerFormat.format(rows.length), state.dataset.hasCustomerId ? "After duplicate ID checks" : "Imported rows") +
    statCard("Churn rate", percent(churnRate), integerFormat.format(churnCount) + " labeled Churn = Yes", "accent") +
    statCard("Average tenure", averageTenure === null ? "—" : numberFormat.format(averageTenure) + " mo", "Months with service") +
    statCard("Average monthly charge", averageCharges === null ? "—" : numberFormat.format(averageCharges), "Dataset charge units") +
    "</div><div class=\"analytics-grid\"><article class=\"panel distribution-panel\"><div class=\"panel-heading\"><div><span class=\"eyebrow\">LABEL MIX</span><h3>Churn distribution</h3><p>Observed outcomes in the loaded dataset</p></div></div><div class=\"distribution-content\">" +
    churnRing(rows) + '<div class="legend"><div><i class="legend-dot churn-dot"></i><span>Churn</span><strong>' +
    integerFormat.format(churnCount) + '</strong></div><div><i class="legend-dot stable-dot"></i><span>No churn</span><strong>' +
    integerFormat.format(rows.length - churnCount) + "</strong></div></div></div></article>" +
    '<article class="panel"><div class="panel-heading"><div><span class="eyebrow">OBSERVED ASSOCIATION</span><h3>Churn by contract</h3><p>Share labeled Churn within each contract group</p></div></div>' +
    contractChart(rows) + "</article>" +
    '<article class="panel"><div class="panel-heading"><div><span class="eyebrow">TENURE SHAPE</span><h3>Churn across tenure</h3><p>Observed churn rate within each month band</p></div></div>' +
    tenureChart(rows) + "</article>" +
    '<article class="panel"><div class="panel-heading"><div><span class="eyebrow">BILLING SNAPSHOT</span><h3>Monthly charges</h3><p>Average by observed Churn label</p></div></div>' +
    chargesChart(rows) + "</article></div><p class=\"data-caveat\">Charts describe associations in the currently loaded cross-sectional sample; they do not establish why customers leave.</p></section>" +
    '<section id="dataset" class="section-block"><div class="section-head"><div><span class="eyebrow">DATA EXPLORER</span><h2>Inspect the dataset</h2></div><button class="button secondary" type="button" data-action="upload">Choose a CSV file</button></div><div class="data-grid"><article class="panel"><div class="panel-heading"><div><span class="eyebrow">DATA QUALITY</span><h3>What was loaded</h3><p>' +
    integerFormat.format(rows.length) + " rows · " + integerFormat.format(state.dataset.headers.length) +
    " columns · " + integerFormat.format(missingCells) + " blank cells</p></div></div>" +
    missingReport(state.dataset) +
    '<div class="upload-guidance"><strong>Compatible CSV format</strong><p>Include every feature below and a Churn column containing Yes or No. customerID is optional; when present, later duplicate IDs are removed. Blank numeric cells are accepted and imputed from training data; invalid numbers and malformed files are rejected.</p><p class="required-columns">Required: ' +
    escapeHtml(FEATURE_COLUMNS.join(", ") + ", Churn") +
    '</p><span class="privacy-note">File contents are parsed in memory in this page and are not sent to an API.</span></div></article><article class="panel preview-panel"><div class="panel-heading"><div><span class="eyebrow">SAMPLE ROWS</span><h3>Preview</h3><p>Identifier column hidden; Churn labels remain visible</p></div><span class="preview-count">First ' + Math.min(6, rows.length) + "</span></div>" +
    previewTable(rows) + "</article></div>" + schemaTable() + "</section>" +
    '<section id="models" class="section-block"><div class="section-head"><div><span class="eyebrow">MODEL LABORATORY</span><h2>Train and compare</h2><p class="section-description">All metrics below come from predictions on rows held out from final fitting.</p></div>' +
    (state.trained
      ? '<button class="button secondary" type="button" data-action="train" ' + (state.loading ? "disabled" : "") + ">" +
        (state.loading ? "Training…" : "Retrain") + "</button>"
      : "") + "</div>" +
    (state.error && state.trained
      ? '<p class="error-banner" role="alert">' + escapeHtml(state.error) + "</p>"
      : "") +
    modelLab(state) + "</section>" +
    '<section id="predict" class="section-block"><div class="section-head"><div><span class="eyebrow">CUSTOMER-LEVEL DEMO</span><h2>Estimate a churn probability</h2><p class="section-description">Enter all supported feature values; customerID is never used as a model input.</p></div></div><div class="prediction-layout"><form id="customer-form" class="panel prediction-form"><div class="panel-heading"><div><span class="eyebrow">MODEL INPUTS</span><h3>Customer profile</h3><p>Values must match the loaded dataset schema.</p></div></div><div class="form-grid">' +
    FEATURE_COLUMNS.map((column) => predictionField(column, state.dataset)).join("") +
    '</div><p id="form-error" class="error-banner" role="alert">' + escapeHtml(state.formError) +
    '</p><button class="button primary full-width" type="submit" ' +
    (!state.trained || state.loading ? "disabled" : "") + ">Estimate churn risk</button><p class=\"small-note\">A score is an estimate, not proof of future behavior. Do not make consequential decisions solely from a model score.</p></form>" +
    '<div id="prediction-output" class="prediction-output">' +
    (state.predictionRow
      ? predictionResult(state)
      : '<div class="prediction-placeholder"><span class="placeholder-icon">◎</span><h3>Prediction results appear here</h3><p>Train a model, complete the profile, then submit to see the estimated class, probability, threshold, and model factors.</p></div>') +
    "</div></div></section>" +
    '<section id="methodology" class="section-block methodology"><div class="section-head"><div><span class="eyebrow">METHOD AND LIMITS</span><h2>How to read this demo</h2><p class="section-description">A transparent teaching workflow, designed for browser-only use.</p></div></div>' +
    methodCards() +
    '<div class="responsible-use"><span>!</span><p><strong>Responsible use:</strong> the IBM sample describes fictional telecom customers. Predictions can be wrong; review them with domain knowledge and never use a model score alone for consequential customer decisions. Uploaded rows remain in this page’s memory until reset or reload.</p></div></section></main>' +
    '<footer class="footer"><a class="brand footer-brand" href="#top"><span class="brand-mark">C</span><span>Churn<span class="brand-light">Lab</span></span></a><p>Educational model lab · no external prediction service</p><a href="https://github.com/IBM/telco-customer-churn-on-icp4d" target="_blank" rel="noreferrer">IBM dataset source ↗</a></footer>';
}

export function updateThresholdView(root: HTMLElement, state: DashboardState): void {
  if (!state.trained) return;
  const output = root.querySelector<HTMLOutputElement>("#threshold-output");
  const metrics = root.querySelector<HTMLElement>("#threshold-metrics");
  const prediction = root.querySelector<HTMLElement>("#prediction-output");
  if (output) output.textContent = percent(state.threshold);
  if (metrics) {
    metrics.innerHTML = thresholdMetrics(state.trained, state.selected, state.threshold);
  }
  if (prediction) {
    prediction.innerHTML = state.predictionRow
      ? predictionResult(state)
      : '<div class="prediction-placeholder"><span class="placeholder-icon">◎</span><h3>Prediction results appear here</h3><p>Train a model, complete the profile, then submit to see the estimated class, probability, threshold, and model factors.</p></div>';
  }
}
