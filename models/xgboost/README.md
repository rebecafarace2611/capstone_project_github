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
- TreeSHAP applied to the 500 highest-scoring final-test observations, using the
  tuned model.

## Files
- `xgboost_model.py` — trains the baseline model and saves test-set fraud probabilities.
- `xgboost_tuning.py` — random search over hyperparameters.
- `xgboost_tuned_model.py` — fits the selected configuration and saves test-set probabilities.
- `xgboost_metrics.py` — computes the final-test metric table at the q* threshold.
- `xgboost_figures.py` — generates the PR curve, ROC curve, and confusion matrix.
- `xgboost_report_figures.py` — generates the composite figures and metrics tables used in the report.
- `xgboost_shap.py` — TreeSHAP interpretation and feature-importance plots.

## Data
The confidential training and test Parquet files and all model outputs (fold
assignments, predictions, fitted models, figures) are excluded from Git and must
be provided locally. Only reviewed aggregate metrics and figures are copied into
`outputs/xgboost/`.

## Reproducibility note
Early stopping uses a single held-out grouped fold rather than a full
out-of-fold protocol. The fold assignments themselves are shared across all
three model tracks, so the train/test split and the operating threshold remain
directly comparable; the difference is confined to how the stopping iteration
is selected.
