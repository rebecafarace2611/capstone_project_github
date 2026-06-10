# ============================================================
# INSPECT INSURANCE DATASET
# Clean inspection + one-time CSV to Parquet conversion
#
# Purpose:
# 1. Convert the raw CSV dataset to Parquet once.
# 2. Load the dataset from Parquet in future runs.
# 3. Create inspection reports without unnecessary print statements.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\rebec\OneDrive\Documentos\UCD Third Trimester\Capstone Project\data"
)

CSV_PATH = DATA_DIR / "ddbb_fraud.csv"
PARQUET_PATH = DATA_DIR / "ddbb_fraud.parquet"
OUTPUT_DIR = DATA_DIR / "inspection_outputs"

TARGET_COLUMN = "respuesta_dicot_c"


# ------------------------------------------------------------
# 2. CSV TO PARQUET CONVERSION
# ------------------------------------------------------------

def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    compression: str = "snappy"
) -> None:
    """
    Converts the original CSV file to Parquet format.

    This is a one-time operation. If the Parquet file already exists,
    the conversion is skipped.
    """
    if parquet_path.exists():
        return

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)

        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression=compression,
            index=False
        )

    except ImportError as exc:
        raise ImportError(
            "The 'pyarrow' package is required to write/read Parquet files. "
            "Install it in the VS Code terminal using: "
            "python -m pip install pyarrow"
        ) from exc


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

def load_dataset(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """
    Loads the dataset from Parquet.

    If the Parquet file does not exist yet, it is created first from
    the original CSV file.
    """
    convert_csv_to_parquet(csv_path, parquet_path)

    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")

    except ImportError as exc:
        raise ImportError(
            "The 'pyarrow' package is required to read Parquet files. "
            "Install it in the VS Code terminal using: "
            "python -m pip install pyarrow"
        ) from exc


# ------------------------------------------------------------
# 4. VARIABLE CLASSIFICATION
# ------------------------------------------------------------

def classify_variable(series: pd.Series) -> str:
    """
    Provides a simple automatic interpretation of each variable type.
    """
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= 10:
            return "Numeric categorical / discrete"
        return "Numeric continuous"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Date/time"

    return "Categorical / text"


# ------------------------------------------------------------
# 5. CREATE INSPECTION REPORTS
# ------------------------------------------------------------

def create_variable_types_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a report with the Python/pandas data type of each variable.
    """
    return pd.DataFrame({
        "variable_name": df.columns,
        "data_type": df.dtypes.astype(str).values
    })


def create_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a missing value report for all variables.
    """
    missing_count = df.isnull().sum()

    missing_report = pd.DataFrame({
        "variable_name": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": (missing_count.values / len(df)) * 100
    })

    return missing_report.sort_values(
        by="missing_percentage",
        ascending=False
    )


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates an initial data dictionary using automatic dataset inspection.
    """
    missing_count = df.isnull().sum()

    data_dictionary = pd.DataFrame({
        "variable_name": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "non_missing_count": df.notnull().sum().values,
        "missing_count": missing_count.values,
        "missing_percentage": (missing_count.values / len(df)) * 100,
        "unique_values": df.nunique(dropna=True).values
    })

    data_dictionary["interpreted_type"] = [
        classify_variable(df[column]) for column in df.columns
    ]

    return data_dictionary


# ------------------------------------------------------------
# 6. CLEAN TARGET VARIABLE
# ------------------------------------------------------------

def clean_target_variable(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Converts the target variable into binary format.

    Mapping:
    - NO FRAUDE -> 0
    - FRAUDE    -> 1
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    cleaned_df = df.copy()

    original_target = (
        cleaned_df[target_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned_df[target_column] = original_target.map({
        "NO FRAUDE": 0,
        "FRAUDE": 1
    })

    if cleaned_df[target_column].isnull().sum() > 0:
        unexpected_values = original_target[
            cleaned_df[target_column].isnull()
        ].unique()

        raise ValueError(
            f"Target conversion failed. Unexpected values: {unexpected_values}"
        )

    return cleaned_df


# ------------------------------------------------------------
# 7. IDENTIFY FEATURE TYPES
# ------------------------------------------------------------

def identify_feature_types(
    df: pd.DataFrame,
    target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Separates features and target, then identifies numeric and categorical columns.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    return X, y, numeric_columns, categorical_columns


def create_target_distribution(y: pd.Series) -> pd.DataFrame:
    """
    Creates a target distribution table.
    """
    counts = y.value_counts(dropna=False).sort_index()
    percentages = y.value_counts(normalize=True, dropna=False).sort_index() * 100

    return pd.DataFrame({
        "target_value": counts.index,
        "count": counts.values,
        "percentage": percentages.values
    })


def create_feature_types_report(
    numeric_columns: list[str],
    categorical_columns: list[str]
) -> pd.DataFrame:
    """
    Creates a compact summary of numeric and categorical feature counts.
    """
    return pd.DataFrame({
        "feature_type": ["numeric", "categorical"],
        "number_of_columns": [len(numeric_columns), len(categorical_columns)],
        "columns": [numeric_columns, categorical_columns]
    })


# ------------------------------------------------------------
# 8. CORRELATION ANALYSIS
# ------------------------------------------------------------

def create_correlation_analysis(
    df: pd.DataFrame,
    target_column: str,
    output_dir: Path,
    top_n: int = 30,
    high_correlation_threshold: float = 0.80
) -> dict[str, pd.DataFrame]:
    """
    Creates correlation analysis outputs for numeric variables.

    This function uses Pearson correlation, which is appropriate for measuring
    linear relationships between numeric variables. Since the target variable is
    converted to 0/1 before this function is called, the correlation between each
    numeric feature and the target can be interpreted as an early exploratory
    signal of association with fraud.

    Outputs saved:
    1. correlation_matrix_numeric.csv
       Full Pearson correlation matrix for all numeric columns.

    2. target_correlation_all_numeric_features.csv
       Every numeric feature ranked by absolute correlation with the target.

    3. target_correlation_top_features.csv
       The strongest target correlations only, controlled by top_n.

    4. highly_correlated_feature_pairs.csv
       Numeric feature pairs with absolute correlation above the selected
       threshold. These may indicate redundant predictors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include=["number"])

    if target_column not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_column}' must be numeric before correlation analysis."
        )

    correlation_matrix = numeric_df.corr(method="pearson")

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

    return {
        "correlation_matrix": correlation_matrix,
        "target_correlation": target_correlation,
        "top_target_correlation": top_target_correlation,
        "highly_correlated_pairs": highly_correlated_pairs
    }


# ------------------------------------------------------------
# 9. SAVE OUTPUTS
# ------------------------------------------------------------

def save_reports(
    output_dir: Path,
    variable_types: pd.DataFrame,
    missing_report: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    target_distribution: pd.DataFrame,
    feature_types: pd.DataFrame
) -> None:
    """
    Saves inspection outputs as CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_types.to_csv(output_dir / "variable_types.csv", index=False)
    missing_report.to_csv(output_dir / "missing_report.csv", index=False)
    data_dictionary.to_csv(output_dir / "data_dictionary.csv", index=False)
    target_distribution.to_csv(output_dir / "target_distribution.csv", index=False)
    feature_types.to_csv(output_dir / "feature_types.csv", index=False)


def save_clean_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    file_name: str = "clean_model_dataset.parquet"
) -> None:
    """
    Saves the cleaned modelling dataset as Parquet for later modelling steps.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        output_dir / file_name,
        engine="pyarrow",
        compression="snappy",
        index=False
    )


# ------------------------------------------------------------
# 10. MAIN WORKFLOW
# ------------------------------------------------------------

def run_preprocessing() -> dict[str, pd.DataFrame]:
    """
    Runs the inspection and basic preprocessing workflow.

    Returns the key outputs as a dictionary, which is useful if this file
    is imported into another notebook or script.
    """
    raw_df = load_dataset(CSV_PATH, PARQUET_PATH)

    variable_types = create_variable_types_report(raw_df)
    missing_report = create_missing_report(raw_df)
    data_dictionary = create_data_dictionary(raw_df)

    clean_model_df = clean_target_variable(raw_df, TARGET_COLUMN)

    correlation_outputs = create_correlation_analysis(
        df=clean_model_df,
        target_column=TARGET_COLUMN,
        output_dir=OUTPUT_DIR,
        top_n=30,
        high_correlation_threshold=0.80
    )

    X, y, numeric_columns, categorical_columns = identify_feature_types(
        clean_model_df,
        TARGET_COLUMN
    )

    target_distribution = create_target_distribution(y)

    feature_types = create_feature_types_report(
        numeric_columns,
        categorical_columns
    )

    preprocessing_summary = pd.DataFrame({
        "item": [
            "rows",
            "columns",
            "features",
            "numeric_features",
            "categorical_features",
            "target_column",
            "parquet_file"
        ],
        "value": [
            clean_model_df.shape[0],
            clean_model_df.shape[1],
            X.shape[1],
            len(numeric_columns),
            len(categorical_columns),
            TARGET_COLUMN,
            str(PARQUET_PATH)
        ]
    })

    save_reports(
        OUTPUT_DIR,
        variable_types,
        missing_report,
        data_dictionary,
        target_distribution,
        feature_types
    )

    preprocessing_summary.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False
    )

    save_clean_dataset(clean_model_df, OUTPUT_DIR)

    return {
        "variable_types": variable_types,
        "missing_report": missing_report,
        "data_dictionary": data_dictionary,
        "target_distribution": target_distribution,
        "feature_types": feature_types,
        "preprocessing_summary": preprocessing_summary,
        "correlation_matrix": correlation_outputs["correlation_matrix"],
        "target_correlation": correlation_outputs["target_correlation"],
        "top_target_correlation": correlation_outputs["top_target_correlation"],
        "highly_correlated_pairs": correlation_outputs["highly_correlated_pairs"],
        "clean_model_df": clean_model_df
    }


if __name__ == "__main__":
    run_preprocessing()
