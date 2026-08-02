# XGBoost

This directory contains the XGBoost workflow (baseline and tuned) for the
fraud-detection model comparison. It uses the original training distribution
with no resampling and no class weights. Class imbalance is handled at the
decision threshold using the q* prevalence rule, keeping predicted
probabilities on the true-prevalence scale for comparability with RFQC.

## Baseline protocol
- 144 leakage-approved predictors.
- Native XGBoost categorical handling (`enable_categorical=True`) for the 18
  categorical predictors, matching the factor treatment used by RFQC.
- Grouped five-fold cross-validation using the shared `fold_assignments.parquet`;
  one fold held out for early-stopping validation.
- Binary gradient boosting with `learning_rate=0.05`, `max_depth=6`, and an
  upper limit of 2,000 trees.
- Early stopping after 50 rounds without validation PR-AUC improvement.
- Operating threshold: q*, the training fraud prevalence (0.00418849), locked
  before the sealed test set is scored.
- No SMOTE, undersampling, `scale_pos_weight`, or class weighting, so that the
  baseline stays directly comparable across the three model tracks.

## Tuning protocol
- Random search over 40 configurations (tree depth, learning rate, number of
  estimators, minimum child weight, subsampling, column sampling, regularization).
- Each configuration scored by PR-AUC under grouped five-fold cross-validation
  on the training data only.
- Selected configuration scored once on the sealed test set after tuning.

## Interpretation
- TreeSHAP applied to the 500 highest-scoring final-test observations, using
  the tuned model.
- LIME applied to the same 500 highest-scoring final-test observations, using
  the same tuned model, for a second, independent explanation method.
- Both methods' figures use human-readable feature labels (e.g. "Policy
  tenure (days)" rather than `antiguedad_poliza`) rather than raw variable
  names, per supervisor feedback that public-facing figures should not
  expose internal variable naming. The mapping is maintained in
  `LABEL_MAP` inside each script and checked against
  `variables_ddbb_fraud_original.xlsx`.
- **Open item:** the feature `scoring` (`risk_scoring_group` in the data
  dictionary) ranks in the top 10 for both SHAP and LIME. Its exact
  definition, whether it reflects underwriting/pricing risk or a
  fraud-investigation output, is not yet confirmed with the data sponsor.
  Do not treat SHAP/LIME rankings involving this feature as final for the
  report until confirmed.

## Files

Baseline stage:
- `xgboost_model.py` — trains the baseline model and saves test-set fraud probabilities.
- `xgboost_metrics.py` — final-test metrics for the baseline model at the q* threshold.
- `xgboost_figures.py` — PR curve, ROC curve, and confusion matrix for the baseline model.

Tuning stage:
- `xgboost_tuning.py` — random search over hyperparameters.
- `xgboost_tuned_model.py` — fits the selected configuration and saves test-set probabilities.

Current versions (baseline and tuned):
- `xgboost_metrics_v2.py` — final-test metrics for both the baseline and tuned models.
- `xgboost_figures_v2.py` — PR curve, ROC curve, and confusion matrix for both models.
- `xgboost_report_figures_v2.py` — composite figures, combined metrics table, and
  baseline-vs-tuned comparison table used in the report. Uses a consistent
  visual identity (color, font) to distinguish XGBoost's figures from RFQC's
  and LightGBM's in the final manuscript, and human-readable class/axis
  labels throughout.

Interpretation:
- `xgboost_shap.py` — TreeSHAP interpretation and feature-importance plots
  (summary/beeswarm and bar chart), with human-readable labels.
- `xgboost_lime.py` — LIME interpretation and feature-importance bar chart
  over the same top-500 alerts, for comparison against the SHAP results.
  Requires `pip install lime`.

## Data
The confidential training and test Parquet files and all model outputs (fold
assignments, predictions, fitted models, figures) are excluded from Git and
must be provided locally. Only reviewed aggregate metrics and figures are
copied into `outputs/xgboost/`.

`variables_ddbb_fraud_original.xlsx` (the data dictionary) is used to build
the human-readable label maps in `xgboost_shap.py` and `xgboost_lime.py`.
If either script prints an "unmapped feature" warning after a run, check
this file and add the missing entry to `LABEL_MAP` before using the
figure in the report.

## Reproducibility note
Early stopping uses a single held-out grouped fold rather than a full
out-of-fold protocol. The fold assignments themselves are shared across all
three model tracks, so the train/test split and the operating threshold
remain directly comparable; the difference is confined to how the stopping
iteration is selected.

LIME's local surrogate fitting involves random perturbation sampling, so
exact feature weights and their relative order can shift slightly between
runs even with identical data and model. The set of features appearing in
the top 10 has been stable across repeated runs; individual rank order
among closely-weighted features should not be over-interpreted.
