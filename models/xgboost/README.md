# XGBoost Baseline

This directory contains the baseline XGBoost workflow for the fraud-detection model comparison. It uses the original training distribution with no resampling and no class weights. Class imbalance is handled at the decision threshold using the q* prevalence rule, keeping predicted probabilities on the true-prevalence scale for comparability with RFQC.

## Baseline protocol
- 144 leakage-approved predictors.
- Native XGBoost categorical handling (`enable_categorical=True`) for the 18 categorical predictors, matching the factor treatment used by RFQC.
- Grouped five-fold cross-validation using the shared `fold_assignments.parquet`; one fold held out for early-stopping validation.
- Binary gradient boosting with `learning_rate=0.05`, `max_depth=6`, and an upper limit of 2,000 trees.
- Early stopping after 50 rounds without validation PR-AUC improvement.
- Operating threshold: q*, the training fraud prevalence (0.00418849), locked before the sealed test set is scored.
- No SMOTE, undersampling, `scale_pos_weight`, or class weighting; these are reserved for the Phase 2 improvement stage.

## Files
- `xgboost_model.py` — trains the baseline model and saves test-set fraud probabilities.
- `xgboost_metrics.py` — computes the final-test metric table at the q* threshold.
- `xgboost_figures.py` — generates the PR curve, ROC curve, and confusion matrix.

## Data
The confidential training and test Parquet files and all model outputs (fold assignments, predictions, fitted models, figures) are excluded from Git and must be provided locally. Only reviewed aggregate metrics and figures are copied into `outputs/xgboost/`.

## Reproducibility note
The current baseline uses a single held-out grouped fold for early stopping. A full out-of-fold protocol matching the LightGBM workflow is planned for the tuned model in Phase 2.
