import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# PART 4: Evaluation metrics for baseline XGBoost
# Threshold rule matches RFQC: q* = training fraud prevalence
# ----------------------------------------------------------------------

# 1. Load saved test predictions from Part 2
preds = pd.read_parquet("xgboost_grouped_test_predictions.parquet")
y_true = preds["y_true"].values
y_proba = preds["y_proba"].values

# 2. Threshold = training fraud prevalence (same rule RFQC locked on)
train = pd.read_parquet("train_model_dataset.parquet")
q_star = train["respuesta_dicot_c"].mean()
print(f"Locked threshold q* (training prevalence) = {q_star:.8f}")

y_pred = (y_proba >= q_star).astype(int)

# 3. Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"\nConfusion matrix:")
print(f"  TN={tn:,}  FP={fp:,}")
print(f"  FN={fn:,}  TP={tp:,}")

# 4. Metrics — the same nine RFQC reported
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred)          # = sensitivity = TPR
specificity = tn / (tn + fp)                       # = TNR
sensitivity = tp / (tp + fn)                       # same as recall, shown explicitly
fpr = fp / (fp + tn)
f1  = f1_score(y_true, y_pred)
gmean = np.sqrt(sensitivity * specificity)
roc_auc = roc_auc_score(y_true, y_proba)           # threshold-independent
pr_auc  = average_precision_score(y_true, y_proba) # threshold-independent, matches RFQC's PR-AUC

# 5. Print table
print("\n" + "="*45)
print("BASELINE XGBOOST — FINAL TEST METRICS")
print("="*45)
rows = [
    ("Threshold (q*)", q_star),
    ("Precision",      precision),
    ("Recall",         recall),
    ("Sensitivity",    sensitivity),
    ("Specificity",    specificity),
    ("F1-score",       f1),
    ("G-mean",         gmean),
    ("False Positive Rate", fpr),
    ("ROC-AUC",        roc_auc),
    ("PR-AUC",         pr_auc),
]
for name, val in rows:
    print(f"  {name:<22} {val:.4f}")

# 6. Save the table for the report + comparison with RFQC
pd.DataFrame(rows, columns=["metric", "xgboost_baseline"]).to_csv(
    "xgboost_baseline_metrics.csv", index=False
)
print("\nSaved: xgboost_baseline_metrics.csv")
