"""Generate polished LightGBM-only figures and a three-line results table.

Row-level predictions are read directly from the external private backup and
are never copied into the repository. Only figures, aggregate plotting data,
and a provenance manifest are written here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import tarfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_BACKUP = Path(r"D:\Download\lightgbm_final_private_backup.tar.gz")

LGBM_TEST_MEMBER = "runs/lightgbm/final_test/final_test_predictions.parquet"
LGBM_OOF_MEMBER = "runs/lightgbm/fixed_oof_lock/oof_ensemble_predictions.parquet"

EXPECTED_BACKUP_SHA256 = (
    "e311b28aff4789dc4d44e6e6723504a903e5a5a028a09045e486dd5255763df5"
)

# Modern high-saturation blue palette with a sparse coral accent.
NAVY = "#155EEF"
BLUE = "#4B8DFF"
TEAL = "#4B8DFF"
GOLD = "#F06473"
INK = "#153B7A"
SLATE = "#4B73B8"
MIST = "#EDF4FF"
PALE_BLUE = "#A8C5FF"
GRID = "#D9E6FF"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lightgbm-backup",
        type=Path,
        default=DEFAULT_BACKUP,
        help="External private LightGBM backup tar.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORT_DIR,
        help="Report directory containing figures/, tables/, and aggregate data.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Segoe UI",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.52,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.36,
            "grid.alpha": 0.72,
            "axes.axisbelow": True,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tar_parquet(archive: tarfile.TarFile, member: str) -> pd.DataFrame:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Missing archive member: {member}")
    return pd.read_parquet(io.BytesIO(extracted.read()))


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12) -> None:
    ax.text(
        x,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NAVY,
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)


def figure_1_workflow(figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.70, 2.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.125, "DATA DESIGN", "Training data", "Grouped 5-fold CV"),
        (0.375, "OPTIMISATION", "RUS strategy screen", "Optuna search +\nlocal refinement"),
        (0.625, "MODEL LOCK", "Five-seed\nOOF ensemble", "Threshold = 0.2367"),
        (0.875, "FINAL EVALUATION", "One-time\nsealed test", "PR-AUC · ROC-AUC\nConfusion matrix · SHAP"),
    ]
    node_w, node_h = 0.188, 0.175
    top_y, bottom_y = 0.585, 0.245

    for index, (x, heading, top_text, bottom_text) in enumerate(stages):
        ax.text(
            x, 0.90, heading, ha="center", va="center",
            fontsize=6.2, color=NAVY, fontweight="bold",
        )

        top_edge = NAVY if index < 3 else BLUE
        ax.add_patch(
            FancyBboxPatch(
                (x - node_w / 2, top_y), node_w, node_h,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor=MIST, edgecolor=top_edge, linewidth=0.65,
            )
        )
        ax.text(
            x, top_y + node_h / 2, top_text, ha="center", va="center",
            fontsize=5.9, color=INK, fontweight="bold",
        )
        ax.add_patch(
            FancyBboxPatch(
                (x - node_w / 2, bottom_y), node_w, node_h,
                boxstyle="round,pad=0.005,rounding_size=0.012",
                facecolor=WHITE, edgecolor=PALE_BLUE, linewidth=0.62,
            )
        )
        bottom_colour = GOLD if index == 2 else INK
        ax.text(
            x, bottom_y + node_h / 2, bottom_text, ha="center", va="center",
            fontsize=5.65, color=bottom_colour,
        )
        ax.add_patch(
            FancyArrowPatch(
                (x, top_y - 0.004), (x, bottom_y + node_h + 0.004),
                arrowstyle="-|>", mutation_scale=6.5,
                linewidth=0.55, color=SLATE,
            )
        )

        if index < len(stages) - 1:
            next_x = stages[index + 1][0]
            ax.add_patch(
                FancyArrowPatch(
                    (x + node_w / 2 + 0.006, top_y + node_h / 2),
                    (next_x - node_w / 2 - 0.006, top_y + node_h / 2),
                    arrowstyle="-|>", mutation_scale=6.5,
                    linewidth=0.58, color=SLATE,
                )
            )

    ax.plot([0.75, 0.75], [0.14, 0.84], color=GOLD, linewidth=0.52, linestyle=(0, (3, 3)))
    ax.text(
        0.75,
        0.90,
        "LOCKED",
        ha="center",
        va="center",
        fontsize=5.4,
        color=GOLD,
        bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5},
    )
    save_figure(fig, figures_dir, "figure_1_modelling_workflow")


def figure_2_selection(
    trials: pd.DataFrame,
    seed_summary: pd.DataFrame,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    trial_plot = trials.copy()
    trial_plot["fpr_percent"] = trial_plot["mean_fpr"] * 100
    trial_plot["pr_auc_percent"] = trial_plot["mean_pr_auc"] * 100
    trial_plot.to_csv(data_dir / "figure_1_optuna_trials.csv", index=False)
    seed_summary.to_csv(data_dir / "figure_1_seed_stability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(5.70, 2.72), constrained_layout=True)
    ax = axes[0]
    selected = trial_plot["trial"].astype(int) == 43
    ax.scatter(
        trial_plot.loc[~selected, "fpr_percent"],
        trial_plot.loc[~selected, "pr_auc_percent"],
        s=20,
        color=PALE_BLUE,
        edgecolor=WHITE,
        linewidth=0.25,
        alpha=0.62,
        zorder=2,
    )
    ax.scatter(
        trial_plot.loc[selected, "fpr_percent"],
        trial_plot.loc[selected, "pr_auc_percent"],
        s=58,
        color=TEAL,
        edgecolor=NAVY,
        linewidth=0.65,
        zorder=4,
    )
    chosen = trial_plot.loc[selected].iloc[0]
    ax.annotate(
        "Selected: trial 43\nRUS 1:5",
        xy=(chosen["fpr_percent"], chosen["pr_auc_percent"]),
        xytext=(chosen["fpr_percent"] + 0.45, chosen["pr_auc_percent"] - 0.18),
        fontsize=6.7,
        color=NAVY,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.55},
        bbox={"boxstyle": "round,pad=0.3", "fc": WHITE, "ec": GRID, "lw": 0.45},
    )
    ax.set_xlabel("Mean validation false-positive rate (%)")
    ax.set_ylabel("Mean validation PR-AUC (%)")
    style_axis(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    order = ["A", "B", "C"]
    x_positions = np.arange(len(order))
    seed_values = [
        (
            seed_summary.loc[seed_summary["candidate_id"] == candidate, "mean_fpr"]
            * 100
        ).to_numpy()
        for candidate in order
    ]
    box = ax.boxplot(
        seed_values,
        positions=x_positions,
        widths=0.48,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": GOLD,
            "markeredgecolor": NAVY,
            "markersize": 4.2,
        },
        medianprops={"color": NAVY, "linewidth": 0.72},
        whiskerprops={"color": SLATE, "linewidth": 0.55},
        capprops={"color": SLATE, "linewidth": 0.55},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": WHITE,
            "markeredgecolor": SLATE,
        },
    )
    for patch, color in zip(box["boxes"], [MIST, PALE_BLUE, BLUE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
        patch.set_edgecolor(BLUE)
        patch.set_linewidth(0.52)
    ax.set_xticks(x_positions, ["A\nRUS 1:7", "B\nRUS 1:10", "C\nRUS 1:5"])
    ax.set_ylabel("Seed-level mean false-positive rate (%)")
    ax.get_xticklabels()[-1].set_color(TEAL)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    style_axis(ax, grid_axis="y")
    panel_label(ax, "(b)")
    save_figure(fig, figures_dir, "figure_1_model_selection_and_stability")


def threshold_grid(
    y_true: np.ndarray, scores: np.ndarray, selected_threshold: float
) -> pd.DataFrame:
    thresholds = np.unique(
        np.concatenate([np.linspace(0.0, 0.60, 601), [selected_threshold]])
    )
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    cumulative_positive = np.concatenate([[0], np.cumsum(sorted_y == 1)])
    cumulative_negative = np.concatenate([[0], np.cumsum(sorted_y == 0)])
    total_positive = int((y_true == 1).sum())
    total_negative = int((y_true == 0).sum())
    rows = []
    for threshold in thresholds:
        index = int(np.searchsorted(sorted_scores, threshold, side="left"))
        fn = int(cumulative_positive[index])
        tn = int(cumulative_negative[index])
        tp = total_positive - fn
        fp = total_negative - tn
        rows.append(
            {
                "threshold": threshold,
                "recall": tp / total_positive,
                "specificity": tn / total_negative,
                "fpr": fp / total_negative,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "is_locked_threshold": bool(np.isclose(threshold, selected_threshold)),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold")


def figure_3_threshold(
    oof_predictions: pd.DataFrame,
    selected_threshold: float,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    curve = threshold_grid(
        oof_predictions["target"].to_numpy(dtype=int),
        oof_predictions["score"].to_numpy(dtype=float),
        selected_threshold,
    )
    curve.to_csv(data_dir / "figure_2_threshold_curve.csv", index=False)
    selected = curve.loc[curve["is_locked_threshold"]].iloc[0]
    tradeoff = curve.loc[curve["recall"].between(0.60, 0.92)].sort_values("fpr")

    fig, ax = plt.subplots(figsize=(5.70, 3.05), constrained_layout=True)
    ax.plot(
        tradeoff["fpr"] * 100,
        tradeoff["recall"] * 100,
        color=BLUE,
        linewidth=1.2,
        solid_capstyle="round",
    )
    sx, sy = selected["fpr"] * 100, selected["recall"] * 100
    ax.axhline(sy, xmax=(sx - 9.0) / 20.5, color=GOLD, lw=0.55, ls=(0, (3, 3)))
    ax.axvline(sx, ymax=(sy - 59.0) / 34.0, color=GOLD, lw=0.55, ls=(0, (3, 3)))
    ax.scatter(
        sx,
        sy,
        s=62,
        marker="o",
        color=GOLD,
        edgecolor=NAVY,
        linewidth=0.65,
        zorder=4,
    )
    ax.annotate(
        f"Locked threshold  {selected_threshold:.3f}\nRecall  {sy:.1f}%   ·   FPR  {sx:.1f}%",
        xy=(sx, sy),
        xytext=(sx + 1.1, sy - 3.0),
        fontsize=7.1,
        color=NAVY,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.55},
        bbox={"boxstyle": "round,pad=0.38", "fc": WHITE, "ec": GRID, "lw": 0.45},
    )
    ax.text(
        0.015,
        0.97,
        "OUT-OF-FOLD ENSEMBLE",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.3,
        fontweight="bold",
        color=SLATE,
    )
    ax.set_xlim(tradeoff["fpr"].min() * 100 - 0.8, tradeoff["fpr"].max() * 100 + 0.8)
    ax.set_ylim(59, 93)
    ax.set_xlabel("False-positive rate (%)")
    ax.set_ylabel("Recall (%)")
    style_axis(ax)
    save_figure(fig, figures_dir, "figure_2_threshold_selection")


def downsample_curve(frame: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, max_rows, dtype=int)
    return frame.iloc[np.unique(indices)].copy()


def classification_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
    prediction: np.ndarray | None = None,
) -> dict[str, int | float]:
    y_true = y_true.astype(int)
    score = score.astype(float)
    pred = (score >= threshold).astype(int) if prediction is None else prediction.astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    positives = tp + fn
    negatives = tn + fp
    n = positives + negatives
    recall = tp / positives
    specificity = tn / negatives
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": threshold,
        "n": n,
        "fraud_cases": positives,
        "prevalence": positives / n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": recall,
        "specificity": specificity,
        "fpr": fp / negatives,
        "precision": precision,
        "f1": f1,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / mcc_denominator,
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "predicted_positive": tp + fp,
        "alert_rate": (tp + fp) / n,
        "false_positives_per_10000_legitimate": fp / negatives * 10000,
    }


def figure_4_final_performance(
    lgbm_test: pd.DataFrame,
    selected_threshold: float,
    figures_dir: Path,
    data_dir: Path,
) -> dict[str, int | float]:
    y_true = lgbm_test["target"].to_numpy(dtype=int)
    score = lgbm_test["fraud_probability"].to_numpy(dtype=float)
    prediction = lgbm_test["primary_prediction"].to_numpy(dtype=int)
    metrics = classification_metrics(y_true, score, selected_threshold, prediction)

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    curve = pd.DataFrame({"recall": recall, "precision": precision}).sort_values("recall")
    downsample_curve(curve).to_csv(
        data_dir / "figure_3_precision_recall_curve.csv", index=False
    )
    pd.DataFrame([metrics]).to_csv(
        data_dir / "figure_4_confusion_matrix_metrics.csv", index=False
    )

    pr_fig, ax = plt.subplots(figsize=(5.70, 3.20), constrained_layout=True)
    ax.plot(
        curve["recall"],
        curve["precision"] * 100,
        color=TEAL,
        linewidth=1.2,
        solid_capstyle="round",
    )
    ax.axhline(
        metrics["prevalence"] * 100,
        color=SLATE,
        linestyle=(0, (4, 3)),
        linewidth=0.6,
    )
    ax.scatter(
        metrics["recall"],
        metrics["precision"] * 100,
        s=48,
        color=GOLD,
        edgecolor=NAVY,
        linewidth=0.65,
        zorder=4,
    )
    ax.annotate(
        f"Locked operating point\nRecall {metrics['recall']:.1%}  ·  Precision {metrics['precision']:.1%}",
        xy=(metrics["recall"], metrics["precision"] * 100),
        xytext=(0.47, 4.8),
        fontsize=6.55,
        color=NAVY,
        arrowprops={"arrowstyle": "-", "color": GOLD, "lw": 0.55},
        bbox={"boxstyle": "round,pad=0.34", "fc": WHITE, "ec": GRID, "lw": 0.45},
    )
    ax.text(
        0.97,
        0.96,
        f"PR-AUC  {metrics['pr_auc']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        0.97,
        0.885,
        f"Prevalence  {metrics['prevalence']:.2%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=SLATE,
    )
    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(0, 16)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision (%)")
    style_axis(ax)
    save_figure(pr_fig, figures_dir, "figure_3_precision_recall_curve")

    cm_fig, ax = plt.subplots(figsize=(4.80, 3.55), constrained_layout=True)
    counts = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    cmap = LinearSegmentedColormap.from_list(
        "cobalt_counts", ["#F7FAFF", "#A8C5FF", "#4B8DFF", "#155EEF", "#0B2A6B"]
    )
    heatmap = ax.imshow(counts, cmap=cmap, vmin=0, vmax=float(counts.max()), aspect="equal")
    for row in range(2):
        for col in range(2):
            value = int(counts[row, col])
            text_color = WHITE if value > counts.max() * 0.52 else INK
            ax.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=text_color,
            )
    ax.set_xticks([0, 1], ["Non-fraud", "Fraud"])
    ax.set_yticks([0, 1], ["Non-fraud", "Fraud"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.4)
    ax.grid(False, which="major")
    colour_bar = cm_fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035)
    colour_bar.outline.set_visible(False)
    colour_bar.ax.tick_params(labelsize=6.2, width=0.45, colors=SLATE)
    colour_bar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value):,}")
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(cm_fig, figures_dir, "figure_4_confusion_matrix")
    return metrics


def display_feature(name: str, width: int = 24) -> str:
    display_names = {
        "garantia_agrupada": "garantia agrupada",
        "comarcaid": "comarca ID",
        "tomadornivel": "tomador nivel",
        "antiguedad_poliza": "antiguedad poliza",
        "aceptoculpasinantecedentes": "acepto culpa sin antecedentes",
        "dias_notificacion": "dias notificacion",
        "tomadorcomarcaid": "tomador comarca ID",
        "tomadormunicipioid": "tomador municipio ID",
        "edad_conductor1": "edad conductor 1",
        "formapago": "forma de pago",
    }
    return textwrap.fill(display_names.get(name, name.replace("_", " ")), width=width)


def figure_5_shap(
    shap_summary: pd.DataFrame,
    figures_dir: Path,
    data_dir: Path,
) -> None:
    importance = shap_summary.sort_values("rank").head(12).copy()
    importance["display_feature"] = importance["feature"].map(display_feature)
    importance.to_csv(data_dir / "figure_5_shap_importance.csv", index=False)

    plot_data = importance.iloc[::-1].copy()
    colors = [PALE_BLUE] * len(plot_data)
    for index in range(max(0, len(colors) - 5), len(colors)):
        colors[index] = BLUE
    colors[-2:] = [NAVY, NAVY]

    fig, ax = plt.subplots(figsize=(5.70, 3.75), constrained_layout=True)
    bars = ax.barh(
        plot_data["display_feature"],
        plot_data["mean_abs_ensemble_shap"],
        color=colors,
        height=0.58,
        edgecolor="none",
    )
    maximum = float(plot_data["mean_abs_ensemble_shap"].max())
    for bar, value in zip(bars, plot_data["mean_abs_ensemble_shap"]):
        ax.text(
            value + maximum * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=SLATE,
        )
    ax.text(
        0.995,
        0.985,
        "TOP 500 HIGHEST-RISK ALERTS",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=SLATE,
        fontweight="bold",
    )
    ax.set_xlim(0, maximum * 1.12)
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    style_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    save_figure(fig, figures_dir, "figure_5_shap_high_risk_alerts")


def format_metric(metric: str, value: int | float) -> str:
    integers = {"n", "fraud_cases", "tp", "fp", "fn", "tn", "predicted_positive"}
    percentages = {
        "prevalence",
        "recall",
        "specificity",
        "fpr",
        "precision",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "alert_rate",
    }
    if metric in integers:
        return f"{int(value):,}"
    if metric in percentages:
        return f"{float(value) * 100:.2f}%"
    if metric == "threshold":
        return f"{float(value):.4f}"
    if metric == "false_positives_per_10000_legitimate":
        return f"{float(value):,.1f}"
    return f"{float(value):.4f}"


def table_1_final_metrics(
    oof_metrics: dict[str, int | float],
    test_metrics: dict[str, int | float],
    tables_dir: Path,
    data_dir: Path,
) -> None:
    rows = [
        ("prevalence", "Fraud prevalence"),
        ("threshold", "Decision threshold"),
        ("predicted_positive", "Predicted alerts"),
        ("alert_rate", "Alert rate"),
        ("recall", "Recall / sensitivity"),
        ("specificity", "Specificity"),
        ("fpr", "False-positive rate"),
        ("precision", "Precision"),
        ("f1", "F1-score"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "Matthews correlation coefficient"),
        ("false_positives_per_10000_legitimate", "False positives per 10,000 legitimate"),
        ("pr_auc", "PR-AUC (average precision)"),
        ("roc_auc", "ROC-AUC"),
    ]
    records = []
    for key, label in rows:
        records.append(
            {
                "metric": label,
                "oof_development": oof_metrics[key],
                "sealed_final_test": test_metrics[key],
            }
        )
    table_data = pd.DataFrame(records)
    table_data.to_csv(data_dir / "table_1_final_performance.csv", index=False)

    display_rows = []
    for key, label in rows:
        display_rows.append(
            (
                label,
                format_metric(key, oof_metrics[key]),
                format_metric(key, test_metrics[key]),
            )
        )

    fig, ax = plt.subplots(figsize=(5.70, 4.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    left_x, oof_x, test_x = 0.030, 0.715, 0.970
    top_y = 0.950
    ax.plot([0.020, 0.980], [top_y, top_y], color="black", lw=0.85)
    ax.text(left_x, 0.910, "Metric", ha="left", va="center", fontsize=9.0, fontweight="bold", fontfamily="Arial", color="black")
    ax.text(oof_x, 0.910, "OOF development", ha="right", va="center", fontsize=9.0, fontweight="bold", fontfamily="Arial", color="black")
    ax.text(test_x, 0.910, "Sealed final test", ha="right", va="center", fontsize=9.0, fontweight="bold", fontfamily="Arial", color="black")
    ax.plot([0.020, 0.980], [0.870, 0.870], color="black", lw=0.55)

    start_y, end_y = 0.825, 0.090
    step = (start_y - end_y) / max(len(display_rows) - 1, 1)
    y = start_y
    for label, oof_value, test_value in display_rows:
        ax.text(left_x, y, label, ha="left", va="center", fontsize=9.0, fontfamily="Arial", color="black")
        ax.text(oof_x, y, oof_value, ha="right", va="center", fontsize=9.0, fontfamily="Arial", color="black")
        ax.text(test_x, y, test_value, ha="right", va="center", fontsize=9.0, fontfamily="Arial", color="black")
        y -= step
    bottom_y = max(0.022, y + step * 0.42)
    ax.plot([0.020, 0.980], [bottom_y, bottom_y], color="black", lw=0.85)
    save_figure(fig, tables_dir, "table_1_final_performance")


def main() -> None:
    args = parse_args()
    configure_style()
    backup = args.lightgbm_backup.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    data_dir = output_dir / "figure_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not backup.is_file():
        raise FileNotFoundError(f"LightGBM private backup not found: {backup}")
    backup_hash = sha256_file(backup)
    if backup_hash.lower() != EXPECTED_BACKUP_SHA256:
        raise ValueError(
            "Unexpected LightGBM private backup SHA-256: "
            f"{backup_hash}; expected {EXPECTED_BACKUP_SHA256}"
        )

    with tarfile.open(backup, "r:gz") as archive:
        lgbm_test = read_tar_parquet(archive, LGBM_TEST_MEMBER)
        lgbm_oof = read_tar_parquet(archive, LGBM_OOF_MEMBER)

    trials = pd.read_csv(
        REPO_ROOT / "outputs/lightgbm/experiment_archive/optuna_global/trial_ranking.csv"
    )
    seed_summary = pd.read_csv(
        REPO_ROOT / "outputs/lightgbm/experiment_archive/seed_confirmation/seed_summary.csv"
    )
    shap_summary = pd.read_csv(
        REPO_ROOT / "outputs/lightgbm/experiment_archive/final_shap/shap_summary.csv"
    )
    final_summary = json.loads(
        (REPO_ROOT / "outputs/lightgbm/final_results_summary.json").read_text(encoding="utf-8")
    )
    selected_threshold = float(final_summary["locked_model"]["threshold"])

    oof_metrics = classification_metrics(
        lgbm_oof["target"].to_numpy(dtype=int),
        lgbm_oof["score"].to_numpy(dtype=float),
        selected_threshold,
    )

    figure_2_selection(trials, seed_summary, figures_dir, data_dir)
    figure_3_threshold(lgbm_oof, selected_threshold, figures_dir, data_dir)
    test_metrics = figure_4_final_performance(
        lgbm_test, selected_threshold, figures_dir, data_dir
    )
    figure_5_shap(shap_summary, figures_dir, data_dir)
    table_1_final_metrics(oof_metrics, test_metrics, tables_dir, data_dir)

    expected_oof = final_summary["fixed_oof"]
    expected_test = final_summary["final_test"]
    checks = {
        "oof_tp_matches": oof_metrics["tp"] == expected_oof["tp"],
        "oof_fp_matches": oof_metrics["fp"] == expected_oof["fp"],
        "oof_pr_auc_matches": bool(np.isclose(oof_metrics["pr_auc"], expected_oof["pr_auc"], atol=1e-12)),
        "test_tp_matches": test_metrics["tp"] == expected_test["tp"],
        "test_fp_matches": test_metrics["fp"] == expected_test["fp"],
        "test_pr_auc_matches": bool(np.isclose(test_metrics["pr_auc"], expected_test["pr_auc"], atol=1e-12)),
    }
    if not all(checks.values()):
        raise AssertionError(f"Figure metric validation failed: {checks}")

    manifest = {
        "schema_version": 2,
        "generator": Path(__file__).name,
        "scope": "LightGBM only",
        "style": "Vivid cobalt academic white-background format",
        "figure_width_inches": 5.70,
        "png_dpi": 300,
        "private_input_policy": (
            "Row-level predictions are read in memory only and are not copied into the report archive."
        ),
        "inputs": {
            "lightgbm_private_backup_name": backup.name,
            "lightgbm_private_backup_sha256": backup_hash,
        },
        "validation": checks,
        "computed_metrics": {
            "oof_development": oof_metrics,
            "sealed_final_test": test_metrics,
        },
        "figures": [
            "figure_1_model_selection_and_stability",
            "figure_2_threshold_selection",
            "figure_3_precision_recall_curve",
            "figure_4_confusion_matrix",
            "figure_5_shap_high_risk_alerts",
        ],
        "tables": ["table_1_final_performance"],
    }
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
