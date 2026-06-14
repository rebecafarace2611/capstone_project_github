from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.rfqc.data import (
    TARGET_COLUMN,
    feature_groups,
    load_approved_features,
    load_model_frame,
)


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic stratified grouped folds for native RFQ tuning."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_model_dataset.parquet",
    )
    parser.add_argument(
        "--approved-features",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "leakage_analysis"
        / "approved_features.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rfqc" / "folds",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    output_dir = args.output_dir.resolve()

    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")

    approved = load_approved_features(approved_path)
    train = load_model_frame(train_path, approved, include_target=True)
    groups = feature_groups(train, approved)
    target = train[TARGET_COLUMN]

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )
    fold = np.full(len(train), -1, dtype=np.int8)
    for fold_index, (_, validation_index) in enumerate(
        splitter.split(np.zeros(len(train), dtype=np.int8), target, groups)
    ):
        fold[validation_index] = fold_index

    if (fold < 0).any():
        raise AssertionError("At least one training row was not assigned to a fold.")

    assignment = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            "fold": fold,
        }
    )
    group_fold_counts = (
        pd.DataFrame({"group": groups, "fold": fold})
        .groupby("group", sort=False)["fold"]
        .nunique()
    )
    if int(group_fold_counts.max()) != 1:
        raise AssertionError("Identical approved feature vectors cross CV folds.")

    fold_summary = (
        pd.DataFrame({"fold": fold, "target": target.to_numpy()})
        .groupby("fold", sort=True)["target"]
        .agg(rows="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    if len(fold_summary) != args.n_splits:
        raise AssertionError("Unexpected number of generated folds.")
    if (fold_summary["fraud_cases"] == 0).any():
        raise AssertionError("At least one validation fold contains no fraud cases.")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "fold_assignments.parquet"
    summary_path = output_dir / "fold_summary.csv"
    context_path = output_dir / "fold_context.json"
    assignment.to_parquet(
        assignment_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    fold_summary.to_csv(summary_path, index=False)

    context = {
        "method": "StratifiedGroupKFold",
        "grouping_rule": "pandas_hash_of_all_approved_features",
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "rows": len(train),
        "fraud_cases": int(target.sum()),
        "fraud_rate": float(target.mean()),
        "unique_groups": int(groups.nunique()),
        "max_folds_per_group": int(group_fold_counts.max()),
        "train_path": project_path(train_path),
        "approved_features_path": project_path(approved_path),
        "train_sha256": sha256_file(train_path),
        "approved_features_sha256": sha256_file(approved_path),
        "fold_summary": fold_summary.to_dict(orient="records"),
    }
    context_path.write_text(
        json.dumps(context, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("Native RFQ grouped folds created successfully.")
    print(f"Assignments: {assignment_path}")
    print(f"Rows: {len(train):,}; groups: {groups.nunique():,}")
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
