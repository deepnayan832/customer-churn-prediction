# Customer Churn Prediction

**Live GitHub Pages site:** <https://deepnayan832.github.io/customer-churn-prediction/>

Published from `main` through GitHub Actions.

An interactive, browser-only model lab for exploring churn patterns in a fictional telecom sample. The app trains two classifiers in the page, evaluates them on held-out rows, shows fitted feature signals, and estimates churn probabilities for customer profiles. It has no backend, account, API key, or paid prediction service.

## Preview

Open the live GitHub Pages site above in any modern browser. Locally, run the steps in Development and open the Vite URL. The bundled sample and all app assets are included in the static build, so the demo does not need a runtime data download.

## Features

- Overview charts and summary statistics recomputed from the currently loaded CSV.
- Dataset preview, column guide, missing-value report, duplicate ID handling, and local CSV import.
- Logistic Regression and CART Decision Tree training and comparison.
- Stratified 70/15/15 train, validation, and test partitions with a fixed seed.
- Validation-based model selection and threshold exploration; final metrics on an untouched test set.
- Accuracy, precision, recall, F1, ROC-AUC, and per-model confusion matrices.
- Model-specific fitted feature importance and a customer-level prediction form.
- GitHub Pages deployment under /customer-churn-prediction/.

## Dataset and attribution

The bundled file is IBM's [Telco Customer Churn sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv), downloaded from the [source repository](https://github.com/IBM/telco-customer-churn-on-icp4d). The source repository declares the Apache License 2.0; a copy is included at public/data/IBM-LICENSE. The sample represents a fictional telecom business and is not live customer data. The bundled CSV contains 7,043 records with 1,869 Churn=Yes labels; 11 blank TotalCharges values are imputed during training.

The required model fields are gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, and Churn.

Churn is the original binary outcome (Yes or No). customerID is optional for imported files and is used only to drop later duplicate IDs. It is never a model input. Blank numeric cells are accepted and imputed; non-numeric values, negative numeric values, unsupported senior-citizen values, and fractional tenure values are rejected with row and column context. Blank categories are imputed using the training partition's most frequent value. The original target labels remain available in charts and previews.

Imported CSVs must include all 19 feature columns and Churn. The file is parsed in browser memory only; rows are not posted to a server, analytics endpoint, or external prediction API. Uploaded data and model results live only in the current page state and are cleared on reset or reload.

## Models and evaluation

The preprocessing pipeline is fitted on training rows: numeric values are median-imputed and standardized, while categories are mode-imputed and one-hot encoded. The target and customer identifier are excluded from the feature matrix. Validation and test rows are transformed with the fitted training statistics.

- **Logistic Regression:** binary logistic classifier, L2 regularization, stable sigmoid, and gradient-descent fitting. Displayed coefficient magnitudes rank fitted signals; coefficient signs describe direction in log-odds, not cause.
- **Decision Tree:** Gini-based CART splitting with maximum depth 4 and minimum leaf size 30. Leaf probabilities are the training churn share. Importance is the normalized Gini decrease and can favor features with more split points.

The seeded stratified split uses 70% training, 15% validation, and 15% test rows (rounded within each class). Both models are fit on training rows only; the default model is the one with the higher validation F1 at threshold 0.50, with exact ties resolved in favor of Logistic Regression. The same fitted models are evaluated once on the untouched test set. Test metrics use a 0.50 threshold; changing the threshold updates validation metrics only, so the test scores do not become tuning data. ROC-AUC uses test probabilities and is shown only when both classes are present.

Accuracy is the share classified correctly. Precision is the share of flagged customers that churned; recall is the share of churners the model flagged. F1 balances precision and recall. ROC-AUC summarizes ranking across thresholds. Precision, recall, F1, or AUC are shown as undefined when their required denominator or class is absent rather than being replaced with a misleading zero.

This data is a cross-sectional snapshot. A random holdout is not a chronological forecast, and model performance does not establish that retention actions will prevent churn. Predictions are estimates, can be wrong, and should never be the sole basis for consequential customer decisions.

## Development

Requirements: Node.js 24 and npm.

    npm ci
    npm test
    npm run build
    npm run dev

The built site is written to dist/. Vite uses the repository path /customer-churn-prediction/ for both local development and GitHub Pages builds.

## Deployment

The workflow in .github/workflows/pages.yml runs on pushes to main and manual dispatch. It installs Node.js 24 from the lockfile, runs the tests and production build, then deploys dist/ with the official GitHub Pages actions. It requests only contents read, Pages write, and the OIDC token permission.

For a new repository, set **Repository → Settings → Pages → Build and deployment → Source → GitHub Actions**. This repository is already configured and deployed at the live URL above.

## Project license

The application source code is licensed under the MIT License in LICENSE. The bundled IBM dataset remains under its upstream Apache License 2.0; see public/data/IBM-LICENSE and the attribution above.

## Portfolio summary

Built a private-by-design, client-side churn analytics dashboard that compares an L2 Logistic Regression model and a Gini Decision Tree, validates preprocessing and model selection against held-out data, explains fitted signals, and supports local CSV analysis and customer-level probability estimates.

## Known limitations

- The two simple classifiers are educational implementations, not replacements for a production model review.
- The benchmark is a fictional, cross-sectional telecom sample and can encode sample-specific associations.
- Threshold exploration uses one validation split; repeated experimentation can overfit that validation set.
- Probability calibration, temporal validation, drift monitoring, and intervention-effect estimation are outside this demo.
