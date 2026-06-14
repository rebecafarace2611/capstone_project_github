"""Combine baseline, cross-validation, and final-test RFQC results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "stage",
    "result_role",
    "candidate",
    "ntree",
    "mtry",
    "nodesize",
    "nsplit",
    "splitrule",
    "threshold_rule",
    "threshold",
    "gmean",
    "sensitivity",
    "specificity",
    "fpr",
    "precision",
    "f1",
    "p4",
    "roc_auc",
    "pr_auc",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def baseline_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(path):
        common = {
            "stage": "baseline",
            "candidate": "",
            "ntree": row["ntree"],
            "mtry": 12,
            "nodesize": 1,
            "nsplit": 10,
            "splitrule": "gini",
            "precision": "",
            "f1": "",
            "p4": "",
        }
        output.append(
            {
                **common,
                "result_role": "training_oob",
                "threshold_rule": "gmean_optimized",
                "threshold": "",
                "gmean": row["gmean_optimized_gmean"],
                "sensitivity": row["gmean_optimized_sensitivity"],
                "specificity": row["gmean_optimized_specificity"],
                "fpr": row["gmean_optimized_fpr"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
            }
        )
        output.append(
            {
                **common,
                "result_role": "training_oob",
                "threshold_rule": "q_star_prevalence",
                "threshold": "",
                "gmean": row["q_star_gmean"],
                "sensitivity": row["q_star_sensitivity"],
                "specificity": row["q_star_specificity"],
                "fpr": row["q_star_fpr"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
            }
        )
    return output


def cv_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in read_csv(path):
        if int(row["candidate"]) != 3:
            continue
        output.append(
            {
                "stage": "cross_validation",
                "result_role": "mean_validation_5fold",
                "candidate": row["candidate"],
                "ntree": row["ntree"],
                "mtry": row["mtry"],
                "nodesize": row["nodesize"],
                "nsplit": row["nsplit"],
                "splitrule": row["splitrule"],
                "threshold_rule": row["threshold_rule"],
                "threshold": row["mean_threshold"],
                "gmean": row["mean_validation_gmean"],
                "sensitivity": row["mean_sensitivity"],
                "specificity": row["mean_specificity"],
                "fpr": 1 - float(row["mean_specificity"]),
                "precision": row["mean_precision"],
                "f1": row["mean_f1"],
                "p4": row["mean_p4"],
                "roc_auc": row["mean_roc_auc"],
                "pr_auc": row["mean_pr_auc"],
            }
        )
    if len(output) != 2:
        raise ValueError("Expected both candidate-3 threshold rows in cv_ranking.csv.")
    return output


def final_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parameters = payload["parameters"]
    metrics = payload["primary_test_metrics"]
    return {
        "stage": "final_test",
        "result_role": "single_untouched_test",
        "candidate": payload["locked_candidate"],
        "ntree": parameters["ntree"],
        "mtry": parameters["mtry"],
        "nodesize": parameters["nodesize"],
        "nsplit": parameters["nsplit"],
        "splitrule": parameters["splitrule"],
        "threshold_rule": payload["threshold_rule"],
        "threshold": payload["selected_threshold"],
        "gmean": metrics["gmean"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "p4": metrics["p4"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


def write_summary(
    baseline_summary: Path,
    cv_ranking: Path,
    final_metrics: Path,
    output: Path,
) -> None:
    rows = [
        *baseline_rows(baseline_summary),
        *cv_rows(cv_ranking),
        final_row(final_metrics),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RFQC baseline, CV, and final test results."
    )
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--cv-ranking", type=Path, required=True)
    parser.add_argument("--final-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.baseline_summary, args.cv_ranking, args.final_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)
    write_summary(
        args.baseline_summary,
        args.cv_ranking,
        args.final_metrics,
        args.output,
    )
    print(f"RFQC result summary written to {args.output}")


if __name__ == "__main__":
    main()
