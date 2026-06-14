"""Lock the selected RFQC configuration before final testing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCKED_PARAMETERS = {
    "candidate": 3,
    "mtry": 24,
    "nodesize": 20,
    "nsplit": 10,
    "splitrule": "gini",
    "threshold_rule": "q_star_prevalence",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_number(value: str) -> int | float | str:
    try:
        integer = int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
    return integer


def read_locked_cv_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["candidate"]) == LOCKED_PARAMETERS["candidate"]
            and row["threshold_rule"] == LOCKED_PARAMETERS["threshold_rule"]
        ]
    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one candidate=3, "
            "threshold_rule=q_star_prevalence row."
        )
    row = {key: parse_number(value) for key, value in rows[0].items()}
    for key in ("candidate", "mtry", "nodesize", "nsplit"):
        if int(row[key]) != LOCKED_PARAMETERS[key]:
            raise ValueError(f"Locked {key} does not match cv_ranking.csv.")
    for key in ("splitrule", "threshold_rule"):
        if row[key] != LOCKED_PARAMETERS[key]:
            raise ValueError(f"Locked {key} does not match cv_ranking.csv.")
    return row


def build_configuration(
    cv_ranking: Path,
    run_context: Path,
    baseline_summary: Path,
    final_trees: int,
    training_rows: int,
    training_fraud: int,
    test_rows: int = 111020,
    test_fraud: int = 465,
) -> dict[str, Any]:
    cv_row = read_locked_cv_row(cv_ranking)
    context = json.loads(run_context.read_text(encoding="utf-8"))
    threshold = training_fraud / training_rows
    cv_metrics = {
        key: value
        for key, value in cv_row.items()
        if key
        not in {
            "candidate",
            "ntree",
            "mtry",
            "nodesize",
            "nsplit",
            "splitrule",
            "threshold_rule",
        }
    }
    return {
        "schema_version": 1,
        "status": "locked_for_final_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "native_randomForestSRC_RFQ",
        "candidate": LOCKED_PARAMETERS["candidate"],
        "source_cv_rank": int(cv_row["rank"]),
        "source_cv_ntree": int(cv_row["ntree"]),
        "ntree": final_trees,
        "mtry": LOCKED_PARAMETERS["mtry"],
        "nodesize": LOCKED_PARAMETERS["nodesize"],
        "nsplit": LOCKED_PARAMETERS["nsplit"],
        "splitrule": LOCKED_PARAMETERS["splitrule"],
        "threshold_rule": LOCKED_PARAMETERS["threshold_rule"],
        "locked_threshold": threshold,
        "threshold_definition": (
            "Full-training fraud prevalence: training_fraud / training_rows."
        ),
        "training_rows": training_rows,
        "training_fraud": training_fraud,
        "expected_test_rows": test_rows,
        "expected_test_fraud": test_fraud,
        "random_state": int(context["random_state"]),
        "selection_reason": (
            "Candidate 3 retained the best structural CV result. "
            "q_star_prevalence was locked before final testing because its "
            "mean G-mean was effectively tied with gmean_optimized while "
            "reducing mean validation FPR and improving class balance."
        ),
        "cv_metrics": cv_metrics,
        "source_files": {
            "cv_ranking": str(cv_ranking.resolve()),
            "run_context": str(run_context.resolve()),
            "baseline_summary": str(baseline_summary.resolve()),
        },
        "source_sha256": {
            "cv_ranking": sha256_file(cv_ranking),
            "run_context": sha256_file(run_context),
            "baseline_summary": sha256_file(baseline_summary),
        },
        "expected_input_sha256": context.get("input_sha256", {}),
        "test_policy": (
            "The test Parquet is read once only after the model and q* "
            "threshold are fixed. Test results cannot change this configuration."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the locked RFQC candidate-3 q* final configuration."
    )
    parser.add_argument("--cv-ranking", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-trees", type=int, default=3000)
    parser.add_argument("--training-rows", type=int, default=444074)
    parser.add_argument("--training-fraud", type=int, default=1860)
    parser.add_argument("--test-rows", type=int, default=111020)
    parser.add_argument("--test-fraud", type=int, default=465)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.cv_ranking, args.run_context, args.baseline_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.final_trees != 3000:
        raise ValueError("The locked final model must use exactly 3000 trees.")
    if args.training_rows <= 0:
        raise ValueError("training_rows must be positive.")
    if not 0 < args.training_fraud < args.training_rows:
        raise ValueError("training_fraud must be between zero and training_rows.")
    if args.test_rows <= 0:
        raise ValueError("test_rows must be positive.")
    if not 0 < args.test_fraud < args.test_rows:
        raise ValueError("test_fraud must be between zero and test_rows.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = build_configuration(
        args.cv_ranking,
        args.run_context,
        args.baseline_summary,
        args.final_trees,
        args.training_rows,
        args.training_fraud,
        args.test_rows,
        args.test_fraud,
    )
    output_path = args.output_dir / "best_configuration.json"
    output_path.write_text(
        json.dumps(configuration, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Locked final configuration written to {output_path}")


if __name__ == "__main__":
    main()
