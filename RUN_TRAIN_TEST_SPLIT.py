# ============================================================
# RUN TRAIN/TEST SPLIT ON CLEAN MODEL DATASET
# ============================================================
# This script is intentionally separate from the preprocessing script.
# It uses the cleaned Parquet dataset already created by the previous step
# and saves a stratified train/test split for modelling.
#
# Run after:
# INSPECT_INSURANCE_DATASET_clean_with_parquet_and_correlation.py
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data"
)

OUTPUT_DIR = DATA_DIR / "inspection_outputs"
CLEAN_DATASET_PATH = OUTPUT_DIR / "clean_model_dataset.parquet"
SPLIT_OUTPUT_DIR = OUTPUT_DIR / "train_test_split"

TARGET_COLUMN = "respuesta_dicot_c"
TEST_SIZE = 0.20
RANDOM_STATE = 42


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
            "Run the preprocessing script first."
        )

    return pd.read_parquet(path, engine="pyarrow")


# ------------------------------------------------------------
# 3. TARGET DISTRIBUTION
# ------------------------------------------------------------

def create_target_distribution(y: pd.Series) -> pd.DataFrame:
    """
    Creates a target distribution table with counts and percentages.
    """
    counts = y.value_counts(dropna=False).sort_index()
    percentages = y.value_counts(normalize=True, dropna=False).sort_index() * 100

    return pd.DataFrame({
        "target_value": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    })


# ------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ------------------------------------------------------------

def create_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    test_size: float = 0.20,
    random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Creates a reproducible stratified train/test split.

    A stratified split is used because fraud is a rare class. Stratification
    preserves the fraud/non-fraud proportion in both train and test datasets.

    Outputs saved:
    1. train_model_dataset.parquet
    2. test_model_dataset.parquet
    3. train_test_split_summary.csv
    4. train_test_target_distribution.csv
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if df[target_column].isnull().sum() > 0:
        raise ValueError("Target column contains missing values. Split cannot be created.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_column]
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_parquet(
        output_dir / "train_model_dataset.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    test_df.to_parquet(
        output_dir / "test_model_dataset.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False
    )

    split_summary = pd.DataFrame({
        "item": [
            "total_rows",
            "train_rows",
            "test_rows",
            "total_columns",
            "target_column",
            "test_size",
            "random_state",
            "split_type",
            "output_folder"
        ],
        "value": [
            df.shape[0],
            train_df.shape[0],
            test_df.shape[0],
            df.shape[1],
            target_column,
            test_size,
            random_state,
            "stratified",
            str(output_dir)
        ]
    })

    train_distribution = create_target_distribution(train_df[target_column])
    train_distribution["split"] = "train"

    test_distribution = create_target_distribution(test_df[target_column])
    test_distribution["split"] = "test"

    target_distribution_by_split = pd.concat(
        [train_distribution, test_distribution],
        ignore_index=True
    )

    target_distribution_by_split = target_distribution_by_split[
        ["split", "target_value", "count", "percentage"]
    ]

    split_summary.to_csv(
        output_dir / "train_test_split_summary.csv",
        index=False
    )

    target_distribution_by_split.to_csv(
        output_dir / "train_test_target_distribution.csv",
        index=False
    )

    return {
        "train_df": train_df,
        "test_df": test_df,
        "split_summary": split_summary,
        "target_distribution_by_split": target_distribution_by_split
    }


# ------------------------------------------------------------
# 5. MAIN WORKFLOW
# ------------------------------------------------------------

def main() -> None:
    clean_model_df = load_clean_dataset(CLEAN_DATASET_PATH)

    create_train_test_split(
        df=clean_model_df,
        target_column=TARGET_COLUMN,
        output_dir=SPLIT_OUTPUT_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


if __name__ == "__main__":
    main()
