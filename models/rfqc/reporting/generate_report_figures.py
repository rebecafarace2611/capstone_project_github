from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "outputs" / "rfqc" / "experiment_archive"
FINAL = ARCHIVE / "final" / "final_local_qstar_3000"
REPORT_DIR = ROOT / "reports" / "rfqc_stage_report"
FIGURE_DIR = REPORT_DIR / "figures"
DATA_DIR = REPORT_DIR / "figure_data"

BLUE = "#3569A8"
DARK = "#222222"
GREY = "#777777"
LIGHT_GREY = "#C8C8C8"
PALE_GREY = "#E6E6E6"
REPORT_FIGURE_WIDTH = 5.70


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "Arial"

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "lines.linewidth": 1.15,
            "lines.markersize": 4,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def prepare_output() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    obsolete = [
        FIGURE_DIR / "figure_1_hyperparameter_tuning.png",
        FIGURE_DIR / "figure_1_hyperparameter_tuning.pdf",
        FIGURE_DIR / "figure_2_tree_convergence.png",
        FIGURE_DIR / "figure_2_tree_convergence.pdf",
        FIGURE_DIR / "figure_3_threshold_selection.png",
        FIGURE_DIR / "figure_3_threshold_selection.pdf",
        FIGURE_DIR / "figure_4_roc_pr.png",
        FIGURE_DIR / "figure_4_roc_pr.pdf",
        DATA_DIR / "figure_1_quick_search.csv",
        DATA_DIR / "figure_1_local_refinement.csv",
        DATA_DIR / "figure_2_tree_convergence.csv",
        DATA_DIR / "figure_3_threshold_points.csv",
    ]
    for path in obsolete:
        path.unlink(missing_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tuning_matrix(
    data: pd.DataFrame,
    *,
    splitrule: str,
    mtry_values: list[int],
    nodesize_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    subset = data[
        (data["splitrule"] == splitrule)
        & (data["threshold_rule"] == "q_star_prevalence")
    ]
    means = np.full((len(nodesize_values), len(mtry_values)), np.nan)
    sds = np.full_like(means, np.nan)
    for row_index, nodesize in enumerate(nodesize_values):
        for column_index, mtry in enumerate(mtry_values):
            row = subset[
                (subset["nodesize"] == nodesize) & (subset["mtry"] == mtry)
            ].iloc[0]
            means[row_index, column_index] = row["mean_validation_gmean"]
            sds[row_index, column_index] = row["std_validation_gmean"]
    return means, sds


def draw_tuning_heatmap(
    ax: plt.Axes,
    means: np.ndarray,
    sds: np.ndarray,
    *,
    mtry_values: list[int],
    nodesize_values: list[int],
    panel: str,
    selected: tuple[int, int] | None = None,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "rfqc_blue",
        ["#F5F5F5", "#D8E4F0", "#7FA3C9", BLUE],
    )
    image = ax.imshow(means, cmap=cmap, vmin=0.715, vmax=0.790, aspect="equal")
    ax.set_xticks(np.arange(len(mtry_values)), [str(value) for value in mtry_values])
    ax.set_yticks(
        np.arange(len(nodesize_values)),
        [str(value) for value in nodesize_values],
    )
    ax.set_xlabel("mtry")
    ax.set_ylabel("Terminal node size")
    ax.set_title(panel, loc="left", fontweight="bold", pad=5)
    ax.tick_params(length=0)

    for row_index in range(means.shape[0]):
        for column_index in range(means.shape[1]):
            value = means[row_index, column_index]
            sd = sds[row_index, column_index]
            text_color = "white" if value >= 0.768 else DARK
            ax.text(
                column_index,
                row_index - 0.08,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                column_index,
                row_index + 0.19,
                f"SD {sd:.3f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=text_color,
            )

    if selected is not None:
        selected_mtry, selected_nodesize = selected
        column_index = mtry_values.index(selected_mtry)
        row_index = nodesize_values.index(selected_nodesize)
        ax.add_patch(
            Rectangle(
                (column_index - 0.47, row_index - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor=DARK,
                linewidth=1.5,
            )
        )

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_model_selection() -> None:
    quick = pd.read_csv(
        ARCHIVE / "tuning" / "01_quick_search_500" / "cv_ranking.csv"
    )
    local = pd.read_csv(
        ARCHIVE / "tuning" / "02_local_refine_gini_500" / "cv_ranking.csv"
    )

    quick_gini = tuning_matrix(
        quick,
        splitrule="gini",
        mtry_values=[12, 24],
        nodesize_values=[1, 10],
    )
    quick_auc = tuning_matrix(
        quick,
        splitrule="auc",
        mtry_values=[12, 24],
        nodesize_values=[1, 10],
    )
    local_gini = tuning_matrix(
        local,
        splitrule="gini",
        mtry_values=[24, 48],
        nodesize_values=[10, 20],
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(REPORT_FIGURE_WIDTH, 2.15),
        gridspec_kw={"wspace": 0.48},
    )
    image = draw_tuning_heatmap(
        axes[0],
        *quick_gini,
        mtry_values=[12, 24],
        nodesize_values=[1, 10],
        panel="(a) Quick search: Gini",
    )
    draw_tuning_heatmap(
        axes[1],
        *quick_auc,
        mtry_values=[12, 24],
        nodesize_values=[1, 10],
        panel="(b) Quick search: AUC",
    )
    draw_tuning_heatmap(
        axes[2],
        *local_gini,
        mtry_values=[24, 48],
        nodesize_values=[10, 20],
        panel="(c) Local refinement: Gini",
        selected=(24, 20),
    )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.07,
        pad=0.22,
        aspect=35,
    )
    colorbar.set_label("Five-fold validation G-mean", labelpad=3)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2.5, width=0.5)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.30, top=0.91)
    save(fig, "figure_1_model_selection")

    combined = pd.concat(
        [
            quick.assign(search_stage="quick"),
            local.assign(search_stage="local"),
        ],
        ignore_index=True,
    )
    combined[
        combined["threshold_rule"] == "q_star_prevalence"
    ].to_csv(DATA_DIR / "figure_1_model_selection.csv", index=False)


def plot_threshold_selection() -> None:
    curve = pd.read_csv(
        FINAL / "final_threshold_curve.csv",
        usecols=[
            "threshold",
            "sensitivity",
            "specificity",
            "gmean",
            "balance_gap",
            "is_selected_threshold",
        ],
    )
    curve = curve[
        curve["threshold"].between(0.0025, 0.0070, inclusive="both")
    ].sort_values("threshold")
    indices = np.linspace(0, len(curve) - 1, min(2500, len(curve))).astype(int)
    plotted = curve.iloc[np.unique(indices)]

    selected = curve[curve["is_selected_threshold"]].iloc[0]
    best = curve.loc[curve["gmean"].idxmax()]

    fig, ax = plt.subplots(figsize=(REPORT_FIGURE_WIDTH, 2.75))
    ax.plot(
        plotted["threshold"],
        plotted["sensitivity"],
        color=DARK,
        linewidth=1.25,
    )
    ax.plot(
        plotted["threshold"],
        plotted["specificity"],
        color=GREY,
        linestyle=(0, (4, 2)),
        linewidth=1.25,
    )
    ax.plot(
        plotted["threshold"],
        plotted["gmean"],
        color=BLUE,
        linewidth=1.55,
    )

    ax.axvline(
        best["threshold"],
        color=GREY,
        linestyle=(0, (2, 2)),
        linewidth=0.8,
    )
    ax.axvline(
        selected["threshold"],
        color=BLUE,
        linestyle=(0, (2, 2)),
        linewidth=0.9,
    )
    ax.scatter(
        best["threshold"],
        best["gmean"],
        s=25,
        facecolor="white",
        edgecolor=DARK,
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        selected["threshold"],
        selected["gmean"],
        marker="s",
        s=25,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )

    ax.text(
        best["threshold"] - 0.00004,
        0.894,
        f"Maximum G-mean\n{best['threshold']:.5f}",
        ha="right",
        va="top",
        fontsize=6.8,
        color=GREY,
    )
    ax.text(
        selected["threshold"] + 0.00004,
        0.894,
        f"Locked q*\n{selected['threshold']:.5f}",
        ha="left",
        va="top",
        fontsize=6.8,
        color=BLUE,
    )

    label_x = 0.00707
    ax.text(
        label_x,
        plotted["sensitivity"].iloc[-1],
        "Sensitivity",
        color=DARK,
        va="center",
        fontsize=7,
    )
    ax.text(
        label_x,
        plotted["specificity"].iloc[-1],
        "Specificity",
        color=GREY,
        va="center",
        fontsize=7,
    )
    ax.text(
        label_x,
        plotted["gmean"].iloc[-1],
        "G-mean",
        color=BLUE,
        va="center",
        fontsize=7,
    )

    ax.set_xlim(0.0025, 0.00745)
    ax.set_ylim(0.70, 0.90)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Training out-of-bag metric")
    ax.set_xticks([0.003, 0.004, 0.005, 0.006, 0.007])
    ax.yaxis.grid(True, color=PALE_GREY, linewidth=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.18, top=0.96)
    save(fig, "figure_2_threshold_selection")

    points = pd.DataFrame(
        [
            {
                "point": "maximum_oob_gmean",
                "threshold": best["threshold"],
                "sensitivity": best["sensitivity"],
                "specificity": best["specificity"],
                "gmean": best["gmean"],
                "balance_gap": best["balance_gap"],
            },
            {
                "point": "locked_q_star",
                "threshold": selected["threshold"],
                "sensitivity": selected["sensitivity"],
                "specificity": selected["specificity"],
                "gmean": selected["gmean"],
                "balance_gap": selected["balance_gap"],
            },
        ]
    )
    points.to_csv(DATA_DIR / "figure_2_threshold_points.csv", index=False)


def load_final_metrics() -> dict:
    with (FINAL / "final_test_metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)["primary_test_metrics"]


def load_test_predictions() -> tuple[np.ndarray, np.ndarray]:
    predictions = pd.read_parquet(
        FINAL / "final_test_predictions.parquet",
        columns=["row_index", "fraud_probability"],
    ).sort_values("row_index")
    target = pd.read_parquet(
        ROOT / "data" / "test_model_dataset.parquet",
        columns=["respuesta_dicot_c"],
    )["respuesta_dicot_c"].to_numpy(dtype=int)

    expected_index = np.arange(len(predictions))
    if len(predictions) != len(target):
        raise ValueError("Prediction and target row counts differ.")
    if not np.array_equal(predictions["row_index"].to_numpy(), expected_index):
        raise ValueError("Final predictions are not in test-row order.")
    return target, predictions["fraud_probability"].to_numpy(dtype=float)


def plot_test_discrimination() -> None:
    target, scores = load_test_predictions()
    metrics = load_final_metrics()
    fpr, tpr, _ = roc_curve(target, scores)
    precision, recall, _ = precision_recall_curve(target, scores)
    prevalence = target.mean()

    calculated_roc_auc = np.trapezoid(tpr, fpr)
    calculated_ap = np.sum((recall[:-1] - recall[1:]) * precision[:-1])
    if not np.isclose(calculated_roc_auc, metrics["roc_auc"], atol=1e-10):
        raise ValueError("Calculated ROC-AUC does not match the archive.")
    if not np.isclose(calculated_ap, metrics["pr_auc"], atol=1e-9):
        raise ValueError("Calculated average precision does not match the archive.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(REPORT_FIGURE_WIDTH, 2.65),
        gridspec_kw={"wspace": 0.34},
    )

    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=1.4)
    ax.plot([0, 1], [0, 1], color=LIGHT_GREY, linestyle="--", linewidth=0.8)
    ax.scatter(
        metrics["fpr"],
        metrics["sensitivity"],
        marker="s",
        s=24,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.annotate(
        "q*",
        (metrics["fpr"], metrics["sensitivity"]),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=6.8,
        color=BLUE,
    )
    ax.text(
        0.96,
        0.07,
        f"ROC-AUC  {metrics['roc_auc']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(a)", loc="left", fontweight="bold", pad=4)
    ax.set_aspect("equal", adjustable="box")
    clean_axis(ax)

    ax = axes[1]
    ax.plot(recall, precision, color=BLUE, linewidth=1.4)
    ax.axhline(prevalence, color=LIGHT_GREY, linestyle="--", linewidth=0.8)
    ax.scatter(
        metrics["sensitivity"],
        metrics["precision"],
        marker="s",
        s=24,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.annotate(
        "q*",
        (metrics["sensitivity"], metrics["precision"]),
        xytext=(-13, 7),
        textcoords="offset points",
        fontsize=6.8,
        color=BLUE,
    )
    ax.text(
        0.96,
        0.92,
        f"PR-AUC  {metrics['pr_auc']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    ax.text(
        0.98,
        prevalence + 0.002,
        f"Prevalence  {prevalence:.4f}",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=GREY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(b)", loc="left", fontweight="bold", pad=4)
    ax.yaxis.grid(True, color=PALE_GREY, linewidth=0.45)
    ax.set_axisbelow(True)
    clean_axis(ax)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.95)
    save(fig, "figure_3_test_discrimination")


def write_captions() -> None:
    captions = """# RFQC Stage Report Figure Captions

**Figure 1. Structural model selection during five-fold cross-validation.**
Mean validation G-mean is shown for the q* prevalence threshold during (a, b) the
quick search over split rule, `mtry`, and terminal node size and (c) local refinement
under Gini splitting. Cell text reports the fold mean and standard deviation. The
outlined cell marks the structure locked for final fitting (`mtry = 24`, terminal node
size = 20).

**Figure 2. Training out-of-bag performance across candidate classification thresholds.**
The filled square and blue vertical line identify the locked q* prevalence threshold.
The open circle and grey vertical line identify the threshold with the maximum
training OOB G-mean. The q* threshold retained near-maximal G-mean while providing a
smaller sensitivity-specificity imbalance and was fixed before test evaluation.

**Figure 3. Discrimination performance on the untouched final test set.**
(a) Receiver operating characteristic curve and (b) precision-recall curve. Filled
squares show the operating point produced by the pre-locked q* threshold. The dashed
horizontal line in panel (b) is the test-set fraud prevalence. The precision axis is
restricted to 0-0.12 to display the operationally relevant portion of the curve.
"""
    (REPORT_DIR / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def main() -> None:
    configure_style()
    prepare_output()
    plot_model_selection()
    plot_threshold_selection()
    plot_test_discrimination()
    write_captions()
    print(f"Generated core RFQC report figures in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
