from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


TARGET_COLUMN = "respuesta_dicot_c"

# These fields are stored numerically but represent unordered business codes.
# Binary indicators remain numeric because a binary split is equivalent to encoding.
NUMERIC_CATEGORICAL_FEATURES = {
    "comarcaid",
    "tomadorcodigopostal",
    "tomadormunicipioid",
    "tomadorprovinciaid",
    "tomadorcomarcaid",
    "tomadornacionalidadid",
}


@dataclass(frozen=True)
class FeatureSchema:
    approved_features: list[str]
    categorical_features: list[str]
    numeric_features: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "approved_features": self.approved_features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
        }


def load_approved_features(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("approved_features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"No approved feature list found in {path}.")
    if TARGET_COLUMN in features:
        raise ValueError("The target must not appear in approved_features.")
    if len(features) != len(set(features)):
        raise ValueError("The approved feature list contains duplicates.")
    return features


def load_training_frame(path: Path, approved_features: list[str]) -> pd.DataFrame:
    columns = approved_features + [TARGET_COLUMN]
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")

    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected column order in {path}.")
    if frame.isna().any().any():
        raise ValueError(
            f"{path} contains missing values. The supplied model dataset is "
            "expected to be fully preprocessed."
        )

    target_values = set(frame[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Unexpected target values in {path}: {target_values}")
    if len(target_values) != 2:
        raise ValueError(f"Both target classes must be present in {path}.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
    return frame


def infer_feature_schema(
    frame: pd.DataFrame,
    approved_features: list[str],
) -> FeatureSchema:
    missing = sorted(set(approved_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Approved features missing from model frame: {missing}")

    text_features = set(
        frame[approved_features]
        .select_dtypes(include=["object", "string", "category"])
        .columns
    )
    categorical_set = text_features | (
        NUMERIC_CATEGORICAL_FEATURES & set(approved_features)
    )
    categorical = [
        feature for feature in approved_features if feature in categorical_set
    ]
    numeric = [
        feature for feature in approved_features if feature not in categorical_set
    ]

    if set(categorical) & set(numeric):
        raise AssertionError("Feature schema contains overlapping roles.")
    if set(categorical) | set(numeric) != set(approved_features):
        raise AssertionError("Feature schema does not cover every approved feature.")

    return FeatureSchema(
        approved_features=list(approved_features),
        categorical_features=categorical,
        numeric_features=numeric,
    )


def compact_for_lightgbm(
    frame: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Compact a frame while preserving stable native categorical mappings."""
    for feature in schema.categorical_features:
        frame[feature] = frame[feature].astype("category")
    for feature in schema.numeric_features:
        frame[feature] = frame[feature].astype("float32")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype("int8")
    return frame


def feature_groups(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[features], index=False).astype("uint64")


def create_grouped_fold_assignments(
    frame: pd.DataFrame,
    approved_features: list[str],
    *,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    target = frame[TARGET_COLUMN]
    groups = feature_groups(frame, approved_features)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold = np.full(len(frame), -1, dtype=np.int16)
    for fold_index, (_, validation_index) in enumerate(
        splitter.split(np.zeros(len(frame), dtype=np.int8), target, groups)
    ):
        fold[validation_index] = fold_index

    assignments = pd.DataFrame(
        {
            "row_index": np.arange(len(frame), dtype=np.int64),
            "fold": fold,
        }
    )
    validate_fold_assignments(
        assignments,
        rows=len(frame),
        target=target,
        groups=groups,
        n_splits=n_splits,
    )
    return assignments


def load_fold_assignments(path: Path) -> pd.DataFrame:
    assignments = pd.read_parquet(
        path,
        columns=["row_index", "fold"],
        engine="pyarrow",
    )
    return assignments.astype({"row_index": "int64", "fold": "int16"})


def validate_fold_assignments(
    assignments: pd.DataFrame,
    *,
    rows: int,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> None:
    if list(assignments.columns) != ["row_index", "fold"]:
        raise ValueError("Fold assignments must contain row_index and fold columns.")
    if len(assignments) != rows:
        raise ValueError(
            f"Fold assignments contain {len(assignments):,} rows; expected {rows:,}."
        )
    if assignments["row_index"].duplicated().any():
        raise ValueError("Fold assignments contain duplicate row_index values.")

    ordered = assignments.sort_values("row_index", kind="stable").reset_index(drop=True)
    expected_index = np.arange(rows, dtype=np.int64)
    if not np.array_equal(ordered["row_index"].to_numpy(), expected_index):
        raise ValueError("Fold assignments do not cover every training row exactly once.")

    folds = ordered["fold"].to_numpy()
    expected_folds = set(range(n_splits))
    if set(np.unique(folds)) != expected_folds:
        raise ValueError(
            f"Unexpected fold labels: {sorted(set(folds))}; expected "
            f"{sorted(expected_folds)}."
        )

    group_fold_counts = (
        pd.DataFrame({"group": groups.to_numpy(), "fold": folds})
        .groupby("group", sort=False)["fold"]
        .nunique()
    )
    if int(group_fold_counts.max()) != 1:
        raise ValueError("Identical approved feature vectors cross CV folds.")

    fold_targets = pd.DataFrame(
        {"fold": folds, "target": target.to_numpy(dtype=np.int8)}
    )
    fraud_by_fold = fold_targets.groupby("fold", sort=True)["target"].sum()
    if (fraud_by_fold == 0).any():
        raise ValueError("At least one validation fold contains no fraud cases.")


def ordered_fold_vector(assignments: pd.DataFrame) -> np.ndarray:
    return (
        assignments.sort_values("row_index", kind="stable")["fold"]
        .to_numpy(dtype=np.int16)
    )

