from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


TARGET_COLUMN = "respuesta_dicot_c"
RANDOM_STATE = 42
N_SPLITS = 5
TEST_FOLD = 0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a leakage-resistant 80/20 split by keeping identical approved "
            "feature vectors in the same partition."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "ddbb_fraud.csv",
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
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--leakage-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "leakage_analysis",
    )
    return parser.parse_args()


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        target = pd.to_numeric(series, errors="coerce")
    else:
        target = (
            series.astype("string")
            .str.strip()
            .str.upper()
            .map({"NO FRAUDE": 0, "FRAUDE": 1, "0": 0, "1": 1})
        )

    if target.isna().any():
        unexpected = series[target.isna()].drop_duplicates().tolist()
        raise ValueError(f"Unexpected target values: {unexpected}")

    return target.astype("int8")


def load_approved_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("approved_features")
    if not isinstance(features, list) or not features:
        raise ValueError("approved_features.json does not contain a feature list.")
    if TARGET_COLUMN in features:
        raise ValueError("Target column must not appear in approved features.")
    if len(features) != len(set(features)):
        raise ValueError("Approved feature list contains duplicate names.")
    return features


def create_groups(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[features], index=False)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def duplicate_rows_participating(frame: pd.DataFrame) -> int:
    hashes = pd.util.hash_pandas_object(frame, index=False)
    return int(hashes.duplicated(keep=False).sum())


def sync_leakage_outputs(
    leakage_output_dir: Path,
    approved_path: Path,
    source_path: Path,
    train_path: Path,
    test_path: Path,
    summary: dict[str, object],
) -> None:
    audit_path = leakage_output_dir / "leakage_audit.csv"
    report_path = leakage_output_dir / "leakage_analysis_report.md"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Leakage audit table not found for synchronization: {audit_path}"
        )

    audit = pd.read_csv(audit_path)
    excluded = audit.loc[audit["decision"] == "excluded", ["variable", "reason"]]
    excluded_lines = "\n".join(
        f"- `{row.variable}`: {row.reason}" for row in excluded.itertuples()
    )

    approved_payload = json.loads(approved_path.read_text(encoding="utf-8"))
    approved_payload["final_split"] = {
        "format": "parquet",
        "method": "stratified_group_5_fold_with_one_test_fold",
        "grouping_rule": summary["grouping_rule"],
        "random_state": summary["random_state"],
        "train_path": project_path(train_path),
        "test_path": project_path(test_path),
        "train_rows": summary["train_rows"],
        "test_rows": summary["test_rows"],
        "train_fraud": summary["train_fraud"],
        "test_fraud": summary["test_fraud"],
        "cross_split_group_overlap": summary["cross_split_group_overlap"],
    }
    approved_path.write_text(
        json.dumps(approved_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    source_hash = sha256_file(source_path)
    train_hash = sha256_file(train_path)
    test_hash = sha256_file(test_path)
    dictionary_path = source_path.parent / "variables_ddbb_fraud.xlsx"
    dictionary_hash = sha256_file(dictionary_path)

    report = f"""# Leakage Analysis Report

## Decision context

- Prediction point: immediately after first claim notification and before fraud investigation.
- Target: `{TARGET_COLUMN}` (`0` non-fraud, `1` fraud).
- Approved model features: **{summary["approved_features"]}**.
- Excluded features: **{len(excluded)}**.
- Final modelling files use **Parquet** format.

## Confirmed exclusions

{excluded_lines}

## Final conclusion

The feature audit approves **{summary["approved_features"]}** predictors. The original
random split contained 712 duplicate feature vectors across train and test. This issue
has been corrected by grouping rows on all approved features and assigning every group
entirely to one partition.

The final grouped split is approved for model development and evaluation:

| Check | Result |
|---|---:|
| Source rows | {summary["source_rows"]} |
| Approved features | {summary["approved_features"]} |
| Train rows | {summary["train_rows"]} |
| Test rows | {summary["test_rows"]} |
| Train fraud cases | {summary["train_fraud"]} |
| Test fraud cases | {summary["test_fraud"]} |
| Train fraud rate | {summary["train_fraud_rate"]:.6%} |
| Test fraud rate | {summary["test_fraud_rate"]:.6%} |
| Unique feature groups | {summary["unique_groups"]} |
| Cross-split feature-group overlap | {summary["cross_split_group_overlap"]} |
| Cross-split full-row overlap | {summary["cross_split_full_row_overlap"]} |
| Train duplicate rows retained within train | {summary["train_duplicate_rows_participating"]} |
| Test duplicate rows retained within test | {summary["test_duplicate_rows_participating"]} |
| Feature groups with conflicting labels | {summary["groups_with_conflicting_targets"]} |
| Missing cells in final split | {summary["final_missing_cells"]} |

Duplicate observations are retained within a single partition rather than deleted.
The two identical-feature groups with conflicting labels are also kept together in one
partition, preventing leakage while preserving the supplied outcomes.

## Statistical proxy screen

Every non-target field was screened using the training data available during the
initial leakage audit. Numeric fields received Pearson correlation and orientation-free
univariate ROC-AUC. Categorical fields received Cramer's V, class purity, and
deterministic-mapping checks. Strong association alone was not treated as leakage.

## Limitations

- No claim, policy, customer, or vehicle identifier is present, so entity-level overlap beyond identical approved feature vectors cannot be tested.
- No claim date or external-data vintage is present, so temporal alignment of area-level statistics cannot be verified directly.
- The supplied dataset already has no missing values, so the historical fitting scope of any earlier imputation cannot be reconstructed.

## Final files and reproducibility

- Train: `{project_path(train_path)}`
- Test: `{project_path(test_path)}`
- Split method: stratified five-fold group split, fold 0 used as the 20% test set.
- Grouping key: hash of all approved features.
- Random seed: `{summary["random_state"]}`
- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- SciPy: `{scipy.__version__}`
- scikit-learn: `{sklearn.__version__}`
- SHA-256:
  - `{source_path.name}`: `{source_hash}`
  - `{train_path.name}`: `{train_hash}`
  - `{test_path.name}`: `{test_hash}`
  - `{dictionary_path.name}`: `{dictionary_hash}`
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    approved_path = args.approved_features.resolve()
    output_dir = args.output_dir.resolve()
    leakage_output_dir = args.leakage_output_dir.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if not approved_path.exists():
        raise FileNotFoundError(f"Approved feature list not found: {approved_path}")

    features = load_approved_features(approved_path)
    model_columns = features + [TARGET_COLUMN]
    frame = pd.read_csv(source, usecols=model_columns, low_memory=False)

    missing_columns = sorted(set(model_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Source dataset is missing columns: {missing_columns}")

    frame = frame[model_columns].copy()
    frame[TARGET_COLUMN] = normalize_target(frame[TARGET_COLUMN])
    if frame.isna().any().any():
        raise ValueError("Model dataset contains missing values.")

    groups = create_groups(frame, features)
    conflicting_groups = (
        pd.DataFrame({"group": groups, "target": frame[TARGET_COLUMN]})
        .groupby("group")["target"]
        .nunique()
        .gt(1)
        .sum()
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    folds = list(
        splitter.split(
            np.zeros(len(frame), dtype=np.int8),
            frame[TARGET_COLUMN],
            groups,
        )
    )
    train_indices, test_indices = folds[TEST_FOLD]

    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)

    train_groups = set(groups.iloc[train_indices].astype("uint64"))
    test_groups = set(groups.iloc[test_indices].astype("uint64"))
    group_overlap = train_groups & test_groups
    if group_overlap:
        raise AssertionError(
            f"Grouped split failed: {len(group_overlap)} groups occur in both sets."
        )

    if len(train) + len(test) != len(frame):
        raise AssertionError("Split row counts do not reconstruct the source dataset.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_model_dataset.parquet"
    test_path = output_dir / "test_model_dataset.parquet"

    train.to_parquet(
        train_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    test.to_parquet(
        test_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    summary = {
        "source_rows": len(frame),
        "approved_features": len(features),
        "output_columns_including_target": len(model_columns),
        "grouping_rule": "hash_of_all_approved_features",
        "unique_groups": int(groups.nunique()),
        "groups_with_conflicting_targets": int(conflicting_groups),
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "test_fold": TEST_FOLD,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_fraud": int(train[TARGET_COLUMN].sum()),
        "test_fraud": int(test[TARGET_COLUMN].sum()),
        "train_fraud_rate": float(train[TARGET_COLUMN].mean()),
        "test_fraud_rate": float(test[TARGET_COLUMN].mean()),
        "cross_split_group_overlap": len(group_overlap),
        "cross_split_full_row_overlap": len(
            set(pd.util.hash_pandas_object(train, index=False))
            & set(pd.util.hash_pandas_object(test, index=False))
        ),
        "train_duplicate_rows_participating": duplicate_rows_participating(train),
        "test_duplicate_rows_participating": duplicate_rows_participating(test),
        "final_missing_cells": int(
            train.isna().sum().sum() + test.isna().sum().sum()
        ),
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
    }
    sync_leakage_outputs(
        leakage_output_dir=leakage_output_dir,
        approved_path=approved_path,
        source_path=source,
        train_path=train_path.resolve(),
        test_path=test_path.resolve(),
        summary=summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
