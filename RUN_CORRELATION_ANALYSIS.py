# ============================================================
# RUN CORRELATION ANALYSIS ON CLEAN MODEL DATASET
# ============================================================
# This script is intentionally separate from the preprocessing script.
# It uses the cleaned Parquet dataset already created by the previous step
# and saves the correlation outputs into the inspection_outputs folder.

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data"
)

OUTPUT_DIR = DATA_DIR / "inspection_outputs"
CLEAN_DATASET_PATH = OUTPUT_DIR / "clean_model_dataset.parquet"
CORRELATION_OUTPUT_DIR = OUTPUT_DIR / "correlation_analysis"

TARGET_COLUMN = "respuesta_dicot_c"
TOP_N = 30
HIGH_CORRELATION_THRESHOLD = 0.80


# ------------------------------------------------------------
# 2. LOAD CLEAN DATASET
# ------------------------------------------------------------

def load_clean_dataset(path: Path) -> pd.DataFrame:
    """
    Loads the cleaned modelling dataset created by the preprocessing script.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Run INSPECT_INSURANCE_DATASET_clean_with_parquet_and_correlation.py first."
        )

    return pd.read_parquet(path, engine="pyarrow")


# ------------------------------------------------------------
# 3. CREATE CORRELATION ANALYSIS
# ------------------------------------------------------------

def create_correlation_analysis(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    top_n: int = 30,
    high_correlation_threshold: float = 0.80
) -> dict[str, pd.DataFrame]:
    """
    Creates and saves correlation analysis outputs.

    The analysis uses Pearson correlation on numeric variables only.
    Since the target variable is binary, the correlation between each
    numeric feature and the target can be interpreted as an initial
    point-biserial-style association with fraud.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include=["number"])

    if target_column not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_column}' must be numeric before correlation analysis."
        )

    # 1. Full numeric correlation matrix
    correlation_matrix = numeric_df.corr(method="pearson")

    # 2. Correlation of each numeric feature with the fraud target
    target_correlation = (
        correlation_matrix[target_column]
        .drop(labels=[target_column])
        .reset_index()
    )

    target_correlation.columns = [
        "variable_name",
        "correlation_with_target"
    ]

    target_correlation["absolute_correlation_with_target"] = (
        target_correlation["correlation_with_target"].abs()
    )

    target_correlation = target_correlation.sort_values(
        by="absolute_correlation_with_target",
        ascending=False
    )

    top_target_correlation = target_correlation.head(top_n)

    # 3. Strong feature-to-feature correlations
    feature_correlation_matrix = correlation_matrix.drop(
        index=target_column,
        columns=target_column
    )

    upper_triangle = feature_correlation_matrix.where(
        np.triu(
            np.ones(feature_correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    highly_correlated_pairs = upper_triangle.stack().reset_index()

    highly_correlated_pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation"
    ]

    highly_correlated_pairs["absolute_correlation"] = (
        highly_correlated_pairs["correlation"].abs()
    )

    highly_correlated_pairs = highly_correlated_pairs.sort_values(
        by="absolute_correlation",
        ascending=False
    )

    highly_correlated_pairs = highly_correlated_pairs[
        highly_correlated_pairs["absolute_correlation"] >= high_correlation_threshold
    ]

    # 4. Save files
    correlation_matrix.to_csv(
        output_dir / "correlation_matrix_numeric.csv"
    )

    target_correlation.to_csv(
        output_dir / "target_correlation_all_numeric_features.csv",
        index=False
    )

    top_target_correlation.to_csv(
        output_dir / "target_correlation_top_features.csv",
        index=False
    )

    highly_correlated_pairs.to_csv(
        output_dir / "highly_correlated_feature_pairs.csv",
        index=False
    )

    summary = pd.DataFrame({
        "item": [
            "rows_used",
            "numeric_columns_used_in_correlation",
            "target_column",
            "top_n_target_correlations",
            "high_correlation_threshold",
            "output_folder"
        ],
        "value": [
            df.shape[0],
            numeric_df.shape[1],
            target_column,
            top_n,
            high_correlation_threshold,
            str(output_dir)
        ]
    })

    summary.to_csv(
        output_dir / "correlation_analysis_summary.csv",
        index=False
    )

    return {
        "correlation_matrix": correlation_matrix,
        "target_correlation": target_correlation,
        "top_target_correlation": top_target_correlation,
        "highly_correlated_pairs": highly_correlated_pairs,
        "summary": summary
    }


# ------------------------------------------------------------
# 4. MAIN WORKFLOW
# ------------------------------------------------------------

def main() -> None:
    clean_model_df = load_clean_dataset(CLEAN_DATASET_PATH)

    create_correlation_analysis(
        df=clean_model_df,
        target_column=TARGET_COLUMN,
        output_dir=CORRELATION_OUTPUT_DIR,
        top_n=TOP_N,
        high_correlation_threshold=HIGH_CORRELATION_THRESHOLD
    )


if __name__ == "__main__":
    main()
