from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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


def load_model_frame(
    path: Path,
    approved_features: list[str],
    *,
    include_target: bool,
) -> pd.DataFrame:
    columns = approved_features + ([TARGET_COLUMN] if include_target else [])
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")

    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected column order in {path}.")
    if frame.isna().any().any():
        raise ValueError(
            f"{path} contains missing values. The RFQC workflow expects the "
            "supplied, fully preprocessed model datasets."
        )

    if include_target:
        target_values = set(frame[TARGET_COLUMN].unique())
        if not target_values.issubset({0, 1}):
            raise ValueError(f"Unexpected target values in {path}: {target_values}")
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
    numeric_code_features = NUMERIC_CATEGORICAL_FEATURES & set(approved_features)
    categorical = [
        feature
        for feature in approved_features
        if feature in text_features or feature in numeric_code_features
    ]
    numeric = [
        feature for feature in approved_features if feature not in categorical
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


def compact_feature_dtypes(
    frame: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    for feature in schema.categorical_features:
        frame[feature] = frame[feature].astype("category")
    for feature in schema.numeric_features:
        frame[feature] = frame[feature].astype("float32")
    return frame


def feature_groups(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[features], index=False).astype("uint64")
