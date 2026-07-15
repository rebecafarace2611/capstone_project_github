import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report,
)

q_star = pd.read_parquet("train_model_dataset.parquet")["respuesta_dicot_c"].mean()

# EDIT THESE two computation-time values (match Jenny's unit; total time per model)
COMP_TIME = {
    "baseline": "0.85 min",   # baseline final training time
    "tuned":    "301.5 min",  # tuned = full random search + final fit (total)
}

MODELS = {
    "baseline": "xgboost_grouped_test_predictions.parquet",
    "tuned":    "xgboost_tuned_test_predictions.parquet",
}


def compute_metrics(name, pred_file):
    """Compute all metrics + generate the composite figure. Returns a dict of values."""
    preds = pd.read_parquet(pred_file)
    y_true = preds["y_true"].values
    y_proba = preds["y_proba"].values
    y_pred = (y_proba >= q_star).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    roc_auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    prevalence = y_true.mean()

    recall = tp / (tp + fn)
    fpr_v = fp / (fp + tn)
    spec = tn / (tn + fp)
    gmean = np.sqrt(recall * spec)
    prec_v = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * prec_v * recall / (prec_v + recall) if (prec_v + recall) else 0

    # ---- Composite 2x2 : CM | ROC / PR | classification report ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # top-left: confusion matrix
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(f"{name.capitalize()} XGBoost confusion matrix (test dataset)")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # top-right: ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="orange")
    ax.plot(fp / (fp + tn), tp / (tp + fn), "o", label="q-star operating point")
    ax.set_title("ROC curve (test dataset)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")

    # bottom-left: precision-recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ax = axes[1, 0]
    ax.plot(rec, prec, label=f"PR curve (AP = {ap:.3f})")
    ax.axhline(prevalence, ls="--", color="orange",
               label=f"Fraud prevalence = {prevalence:.4f}")
    ax.plot(tp / (tp + fn), tp / (tp + fp), "o", label="q-star operating point")
    ax.set_title("Precision-recall curve (test dataset)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right")

    # bottom-right: classification report (monospace, larger, aligned to top-left)
    ax = axes[1, 1]
    ax.axis("off")
    rep = classification_report(y_true, y_pred,
                                target_names=["nonfraud", "fraud"], digits=4)
    ax.text(0.02, 0.98, rep, fontfamily="monospace", fontsize=11,
            va="top", ha="left", transform=ax.transAxes)
    ax.set_title("Classification report (test dataset)")

    plt.tight_layout()
    plt.savefig(f"xgb_composite_{name}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: xgb_composite_{name}.png")

    return dict(recall=recall, spec=spec, fpr=fpr_v, gmean=gmean,
                prec=prec_v, f1=f1, roc=roc_auc, pr=ap,
                tp=tp, fp=fp, fn=fn, tn=tn)


def styled_table(ax, cell_text, col_labels, col_widths=None):
    """Jenny-style grey-alternating table."""
    t = ax.table(cellText=cell_text, colLabels=col_labels,
                 cellLoc="left", colLoc="left", loc="center",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9d9d9")
        else:
            cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")
    return t


results = {}
for name, f in MODELS.items():
    results[name] = compute_metrics(name, f)

b, t = results["baseline"], results["tuned"]


def fmt_pct(x):
    return f"{x*100:.2f}%"


# ---- Combined metrics table (baseline + tuned in one) ----
combined_rows = [
    ("Test observations",        "111,020",      "111,020"),
    ("Fraud cases",              "465 (0.42%)",  "465 (0.42%)"),
    ("q-star threshold",         f"{q_star:.4f}", f"{q_star:.4f}"),
    ("Fraud detected (of 465)",  f"{b['tp']:,}", f"{t['tp']:,}"),
    ("Recall / sensitivity",     fmt_pct(b['recall']), fmt_pct(t['recall'])),
    ("Specificity",              fmt_pct(b['spec']),   fmt_pct(t['spec'])),
    ("False positive rate",      fmt_pct(b['fpr']),    fmt_pct(t['fpr'])),
    ("G-Mean",                   f"{b['gmean']:.4f}",  f"{t['gmean']:.4f}"),
    ("Precision",                fmt_pct(b['prec']),   fmt_pct(t['prec'])),
    ("F1-score",                 f"{b['f1']:.4f}",     f"{t['f1']:.4f}"),
    ("ROC-AUC",                  f"{b['roc']:.4f}",    f"{t['roc']:.4f}"),
    ("PR-AUC",                   f"{b['pr']:.4f}",     f"{t['pr']:.4f}"),
    ("True positives",           f"{b['tp']:,}",       f"{t['tp']:,}"),
    ("False positives",          f"{b['fp']:,}",       f"{t['fp']:,}"),
    ("False negatives",          f"{b['fn']:,}",       f"{t['fn']:,}"),
    ("True negatives",           f"{b['tn']:,}",       f"{t['tn']:,}"),
    ("Computation time",         COMP_TIME["baseline"], COMP_TIME["tuned"]),
]
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis("off")
styled_table(ax, combined_rows,
             col_labels=["Metric", "Baseline", "Tuned"],
             col_widths=[0.42, 0.29, 0.29])
plt.tight_layout()
plt.savefig("xgb_metrics_table_combined.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: xgb_metrics_table_combined.png")


# ---- Figure 3: baseline vs tuned comparison table (label column widened) ----
comp_labels = ["Evaluation", "G-Mean", "Recall", "Specificity", "FPR", "ROC-AUC", "PR-AUC"]
comp_rows = [
    ("Baseline (grouped CV)", f"{b['gmean']:.4f}", f"{b['recall']:.4f}",
     f"{b['spec']:.4f}", f"{b['fpr']:.4f}", f"{b['roc']:.4f}", f"{b['pr']:.4f}"),
    ("Tuned (random search)", f"{t['gmean']:.4f}", f"{t['recall']:.4f}",
     f"{t['spec']:.4f}", f"{t['fpr']:.4f}", f"{t['roc']:.4f}", f"{t['pr']:.4f}"),
]
fig, ax = plt.subplots(figsize=(11, 2))
ax.axis("off")
# wide first column so the row labels aren't truncated
widths = [0.26] + [0.123] * 6
t2 = ax.table(cellText=comp_rows, colLabels=comp_labels,
              cellLoc="center", colLoc="center", loc="center",
              colWidths=widths)
t2.auto_set_font_size(False)
t2.set_fontsize(10)
t2.scale(1, 1.6)
for (r, c), cell in t2.get_celld().items():
    cell.set_edgecolor("white")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")  # left-align the label column
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d9d9d9")
    else:
        cell.set_facecolor("#f2f2f2" if r % 2 == 0 else "white")
plt.tight_layout()
plt.savefig("xgb_baseline_vs_tuned.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: xgb_baseline_vs_tuned.png")