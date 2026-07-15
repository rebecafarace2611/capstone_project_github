import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# PART 5: Baseline XGBoost figures — PR curve, ROC curve, confusion matrix
# Reads saved test predictions; independent of the fold protocol.
# ----------------------------------------------------------------------

preds = pd.read_parquet("xgboost_tuned_test_predictions.parquet")
y_true = preds["y_true"].values
y_proba = preds["y_proba"].values

# Threshold = training prevalence (q*), same rule as Part 4
q_star = pd.read_parquet("train_model_dataset.parquet")["respuesta_dicot_c"].mean()
y_pred = (y_proba >= q_star).astype(int)

# ---- Figure 1: Precision-Recall curve ----
prec, rec, _ = precision_recall_curve(y_true, y_proba)
pr_auc = average_precision_score(y_true, y_proba)
prevalence = y_true.mean()

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, color="#1f77b4", lw=2, label=f"XGBoost (PR-AUC = {pr_auc:.4f})")
plt.axhline(prevalence, color="grey", ls="--", lw=1,
            label=f"Prevalence = {prevalence:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Tuned XGBoost — Precision-Recall Curve (Final Test)")
plt.ylim(0, 0.12)  # zoom: precision is low under 0.42% prevalence
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("xgb_pr_curve_tuned.png", dpi=200)
plt.close()

# ---- Figure 2: ROC curve ----
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = roc_auc_score(y_true, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"XGBoost (ROC-AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Tuned XGBoost — ROC Curve (Final Test)")
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig("xgb_roc_curve_tuned.png", dpi=200)
plt.close()

# ---- Figure 3: Confusion matrix at q* ----
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["No Fraud", "Fraud"])
ax.set_yticklabels(["No Fraud", "Fraud"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Observed")
ax.set_title(f"Tuned XGBoost — Confusion Matrix (q* = {q_star:.5f})")

# annotate each cell; white text on dark cells for readability
labels = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i][j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("xgb_confusion_matrix_tuned.png", dpi=200)
plt.close()

print("Saved 3 figures:")
print("  xgb_pr_curve_tuned.png")
print("  xgb_roc_curve_tuned.png")
print("  xgb_confusion_matrix_tuned.png")