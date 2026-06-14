from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_COLUMN = "respuesta_dicot_c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptive summaries from the grouped training data."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_model_dataset.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "descriptive_analysis",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--high-correlation-threshold", type=float, default=0.80)
    return parser.parse_args()


def classify_variable(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= 10:
            return "numeric_categorical_or_discrete"
        return "numeric_continuous"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date_time"
    return "categorical_or_text"


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
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("The training target must contain only fraud labels 0 and 1.")
    return target.astype("int8")


def create_correlation_outputs(
    frame: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    threshold: float,
) -> None:
    numeric = frame.select_dtypes(include=["number"])
    correlation = numeric.corr(method="pearson")
    target_correlation = (
        correlation[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .rename_axis("variable_name")
        .reset_index(name="correlation_with_target")
    )
    target_correlation["absolute_correlation_with_target"] = target_correlation[
        "correlation_with_target"
    ].abs()
    target_correlation = target_correlation.sort_values(
        "absolute_correlation_with_target",
        ascending=False,
    )

    feature_correlation = correlation.drop(
        index=TARGET_COLUMN,
        columns=TARGET_COLUMN,
    )
    upper = feature_correlation.where(
        np.triu(np.ones(feature_correlation.shape), k=1).astype(bool)
    )
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["absolute_correlation"] = pairs["correlation"].abs()
    pairs = pairs.loc[pairs["absolute_correlation"] >= threshold].sort_values(
        "absolute_correlation",
        ascending=False,
    )

    correlation.to_csv(output_dir / "correlation_matrix_numeric.csv")
    target_correlation.to_csv(
        output_dir / "target_correlation_all_numeric_features.csv",
        index=False,
    )
    target_correlation.head(top_n).to_csv(
        output_dir / "target_correlation_top_features.csv",
        index=False,
    )
    pairs.to_csv(output_dir / "highly_correlated_feature_pairs.csv", index=False)


def main() -> None:
    args = parse_args()
    train_path = args.train.resolve()
    output_dir = args.output_dir.resolve()
    if not train_path.exists():
        raise FileNotFoundError(
            f"Grouped training data not found: {train_path}\n"
            "Run scripts/leakage/run_grouped_train_test_split.py first."
        )

    frame = pd.read_parquet(train_path, engine="pyarrow")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    frame[TARGET_COLUMN] = normalize_target(frame[TARGET_COLUMN])
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = frame.isna().sum()
    data_dictionary = pd.DataFrame(
        {
            "variable_name": frame.columns,
            "storage_dtype": frame.dtypes.astype(str).values,
            "interpreted_type": [
                classify_variable(frame[column]) for column in frame.columns
            ],
            "non_missing_count": frame.notna().sum().values,
            "missing_count": missing.values,
            "missing_percentage": missing.values / len(frame) * 100,
            "unique_values": frame.nunique(dropna=True).values,
        }
    )
    target_distribution = (
        frame[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("target_value")
        .reset_index(name="count")
    )
    target_distribution["percentage"] = (
        target_distribution["count"] / len(frame) * 100
    )
    summary = pd.DataFrame(
        {
            "item": [
                "rows",
                "columns_including_target",
                "features",
                "fraud_cases",
                "fraud_prevalence",
                "missing_cells",
                "training_file",
            ],
            "value": [
                len(frame),
                len(frame.columns),
                len(frame.columns) - 1,
                int(frame[TARGET_COLUMN].sum()),
                float(frame[TARGET_COLUMN].mean()),
                int(frame.isna().sum().sum()),
                str(train_path),
            ],
        }
    )

    data_dictionary.to_csv(output_dir / "data_dictionary.csv", index=False)
    target_distribution.to_csv(output_dir / "target_distribution.csv", index=False)
    summary.to_csv(output_dir / "descriptive_summary.csv", index=False)
    create_correlation_outputs(
        frame,
        output_dir,
        top_n=args.top_n,
        threshold=args.high_correlation_threshold,
    )

    print(f"Descriptive outputs written to {output_dir}")
    print(
        f"Rows: {len(frame):,}; features: {len(frame.columns) - 1}; "
        f"fraud cases: {int(frame[TARGET_COLUMN].sum()):,}"
    )


if __name__ == "__main__":
    main()
