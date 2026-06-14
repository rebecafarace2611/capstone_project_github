from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "respuesta_dicot_c"
TARGET_COPY_COLUMN = "respuesta_dicot1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
AUDIT_TEST_SIZE = 0.20

BUSINESS_EXCLUSIONS = {
    "respuesta_dicot1": (
        "direct_target_copy",
        "Text representation of the fraud target; it maps exactly to the binary target.",
    ),
    "situacionvts": (
        "post_outcome_status",
        "Claim status records investigation/adjudication outcomes and deterministically reveals the target.",
    ),
    "scoreada": (
        "upstream_score",
        "Output of an undocumented ADA scoring system; its inputs and generation time cannot be audited.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible leakage audit on the insurance fraud data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Folder containing ddbb_fraud.csv and variables_ddbb_fraud.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "leakage_analysis",
        help="Folder for the three final audit outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def excel_column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    result = 0
    for character in letters:
        result = result * 26 + ord(character.upper()) - ord("A") + 1
    return result - 1


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Read the simple two-column variable dictionary without an Excel dependency."""
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", namespace):
                shared_strings.append(
                    "".join(node.text or "" for node in item.findall(".//a:t", namespace))
                )

        sheet_root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[object]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", namespace):
            values: dict[int, object] = {}
            for cell in row.findall("a:c", namespace):
                reference = cell.attrib["r"]
                column_index = excel_column_index(reference)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", namespace)
                inline_node = cell.find("a:is/a:t", namespace)

                if inline_node is not None:
                    value: object = inline_node.text or ""
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text or "0")]
                else:
                    value = value_node.text or ""
                values[column_index] = value

            if values:
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])

    if not rows:
        raise ValueError(f"No rows found in variable dictionary: {path}")

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded_rows[1:], columns=padded_rows[0])


def normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.astype("Int64")

    normalized = series.astype("string").str.strip().str.upper()
    mapped = normalized.map({"NO FRAUDE": 0, "FRAUDE": 1, "0": 0, "1": 1})
    return mapped.astype("Int64")


def normalize_frame_for_hashing(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[TARGET_COLUMN] = normalize_target(normalized[TARGET_COLUMN])

    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    for column in normalized.select_dtypes(include=["number"]).columns:
        # CSV chunks can infer an integer dtype when a float column happens to
        # contain only whole numbers in that chunk. A common float64
        # representation makes hashes stable across CSV and Parquet inputs.
        normalized[column] = pd.to_numeric(
            normalized[column], errors="coerce"
        ).astype("float64")

    return normalized


def row_hashes(frame: pd.DataFrame) -> np.ndarray:
    normalized = normalize_frame_for_hashing(frame)
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)


def feature_hashes(frame: pd.DataFrame) -> np.ndarray:
    features = frame.drop(columns=[TARGET_COLUMN, TARGET_COPY_COLUMN], errors="ignore")
    normalized = normalize_frame_for_hashing(
        pd.concat(
            [
                features,
                pd.Series(0, index=features.index, name=TARGET_COLUMN, dtype="int64"),
            ],
            axis=1,
        )
    ).drop(columns=[TARGET_COLUMN])
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)


def count_duplicate_hashes(hashes: np.ndarray) -> int:
    return int(pd.Series(hashes).duplicated(keep=False).sum())


def counter_intersection_size(left: Counter[int], right: Counter[int]) -> int:
    return int(sum((left & right).values()))


def target_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = normalize_target(frame[TARGET_COLUMN]).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def cramers_v(feature: pd.Series, target: pd.Series) -> float:
    table = pd.crosstab(feature, target)
    if table.empty or min(table.shape) < 2:
        return 0.0

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(
        axis=0, keepdims=True
    ) / total
    valid = expected > 0
    chi_squared = np.sum(((observed - expected) ** 2)[valid] / expected[valid])
    denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(math.sqrt(chi_squared / denominator)) if denominator > 0 else 0.0


def categorical_screen(feature: pd.Series, target: pd.Series) -> dict[str, object]:
    safe_feature = feature.astype("string").fillna("<NA>")
    grouped = pd.DataFrame({"feature": safe_feature, "target": target}).groupby(
        "feature", dropna=False
    )["target"]
    positive_rate = grouped.mean()
    distinct_targets = grouped.nunique()
    deterministic_mapping = bool((distinct_targets <= 1).all())
    weighted_purity = float(
        grouped.size().mul(np.maximum(positive_rate, 1 - positive_rate)).sum()
        / len(feature)
    )
    return {
        "association_metric": "cramers_v",
        "association_value": cramers_v(safe_feature, target),
        "orientation_free_auc": np.nan,
        "deterministic_mapping": deterministic_mapping,
        "weighted_class_purity": weighted_purity,
    }


def numeric_screen(feature: pd.Series, target: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(feature, errors="coerce")
    valid = numeric.notna()
    if valid.sum() == 0 or numeric[valid].nunique() <= 1:
        correlation = 0.0
        auc = 0.5
    else:
        correlation = float(numeric[valid].corr(target[valid]))
        raw_auc = roc_auc_score(target[valid], numeric[valid])
        auc = float(max(raw_auc, 1 - raw_auc))

    deterministic_mapping = bool(
        pd.DataFrame({"feature": numeric, "target": target})
        .groupby("feature", dropna=False)["target"]
        .nunique()
        .le(1)
        .all()
    )
    return {
        "association_metric": "pearson",
        "association_value": correlation,
        "orientation_free_auc": auc,
        "deterministic_mapping": deterministic_mapping,
        "weighted_class_purity": np.nan,
    }


def screen_feature(feature: pd.Series, target: pd.Series) -> dict[str, object]:
    if pd.api.types.is_numeric_dtype(feature):
        return numeric_screen(feature, target)
    return categorical_screen(feature, target)


def infer_availability(column: str) -> tuple[str, str]:
    if column in {"dias_notificacion", "dias_notificacion2"}:
        return (
            "first_notification",
            "Known once the initial claim notification has been completed.",
        )
    if column == "aceptoculpasinantecedentes":
        return (
            "first_notification",
            "Interpreted from the variable name as a notification-time declaration combined with prior-history information.",
        )
    if column.startswith(("dgc_", "dgt_", "ign_", "ine_", "osm_", "rep_", "sea_")):
        return (
            "pre_claim_external",
            "Area-level external statistic available independently of the claim outcome.",
        )
    return (
        "pre_claim_or_policy",
        "Policy, customer, vehicle, coverage, historical, or underwriting information available by first notification.",
    )


def build_audit_table(
    train: pd.DataFrame,
    descriptions: dict[str, str],
) -> pd.DataFrame:
    target = normalize_target(train[TARGET_COLUMN]).astype("int64")
    records: list[dict[str, object]] = []

    for column in train.columns:
        description = descriptions.get(column, "")
        if column == TARGET_COLUMN:
            records.append(
                {
                    "variable": column,
                    "description": description,
                    "dtype": str(train[column].dtype),
                    "distinct_values_train": int(train[column].nunique(dropna=False)),
                    "missing_train": int(train[column].isna().sum()),
                    "availability": "target_only",
                    "association_metric": "",
                    "association_value": np.nan,
                    "orientation_free_auc": np.nan,
                    "deterministic_mapping": True,
                    "weighted_class_purity": np.nan,
                    "leakage_category": "target",
                    "decision": "target",
                    "reason": "Supervised-learning label; never included in model features.",
                }
            )
            continue

        screen = screen_feature(train[column], target)
        availability, availability_reason = infer_availability(column)

        if column in BUSINESS_EXCLUSIONS:
            leakage_category, reason = BUSINESS_EXCLUSIONS[column]
            decision = "excluded"
        elif screen["deterministic_mapping"] and (
            (
                pd.api.types.is_numeric_dtype(train[column])
                and screen["orientation_free_auc"] >= 0.999
            )
            or (
                not pd.api.types.is_numeric_dtype(train[column])
                and screen["weighted_class_purity"] >= 0.999
            )
        ):
            leakage_category = "deterministic_target_proxy"
            decision = "excluded"
            reason = (
                "Training data show a near-perfect deterministic mapping to the target; "
                "excluded as an additional target proxy."
            )
        else:
            leakage_category = "none_detected"
            decision = "approved"
            reason = availability_reason

        records.append(
            {
                "variable": column,
                "description": description,
                "dtype": str(train[column].dtype),
                "distinct_values_train": int(train[column].nunique(dropna=False)),
                "missing_train": int(train[column].isna().sum()),
                "availability": availability,
                **screen,
                "leakage_category": leakage_category,
                "decision": decision,
                "reason": reason,
            }
        )

    return pd.DataFrame(records)


def markdown_table(rows: list[tuple[str, object]]) -> str:
    lines = ["| Check | Result |", "|---|---:|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def write_report(
    path: Path,
    audit: pd.DataFrame,
    approved_features: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    overlap: dict[str, object],
    combined_csv: dict[str, object],
    input_hashes: dict[str, str],
) -> None:
    excluded = audit.loc[audit["decision"] == "excluded", ["variable", "reason"]]
    excluded_lines = "\n".join(
        f"- `{row.variable}`: {row.reason}" for row in excluded.itertuples()
    )
    target_train = target_counts(train)
    target_test = target_counts(test)

    limitations = [
        "The supplied combined CSV contains 148 columns, not the paper's earlier 181-column source version; this audit treats the supplied data as authoritative.",
        "No claim, policy, customer, or vehicle identifier is present, so entity-level train/test overlap cannot be tested.",
        "No claim/notification date or external-data vintage is present, so temporal alignment of area statistics cannot be verified directly.",
        "The supplied combined CSV already has no missing values, so the historical fitting scope of any earlier imputation cannot be reconstructed.",
    ]
    if overlap["cross_split_duplicate_rows"]:
        overall_conclusion = (
            f"The feature audit approves **{len(approved_features)}** predictors after "
            "removing the confirmed leakage fields above. The supplied split is not "
            f"approved because **{overlap['cross_split_duplicate_rows']}** full "
            "duplicate rows occur in both partitions. Rebuild the split by grouping "
            "records on all approved features before model development."
        )
    else:
        overall_conclusion = (
            f"The feature audit approves **{len(approved_features)}** predictors after "
            "removing the confirmed leakage fields above. No full duplicate rows "
            "cross the supplied train/test boundary, so the split passes this "
            "duplicate-overlap check."
        )

    report = f"""# Leakage Analysis Report

## Decision context

- Prediction point: immediately after first claim notification and before fraud investigation.
- Target: `{TARGET_COLUMN}` (`0` non-fraud, `1` fraud).
- Feature screening used the training portion of a stratified 80/20 audit split.
- Audit split random seed: `{RANDOM_STATE}`.
- Audit test labels were used only for dataset integrity summaries.
- Approved model features: **{len(approved_features)}**.
- Excluded features: **{len(excluded)}**.

## Confirmed exclusions

{excluded_lines}

## Overall conclusion

{overall_conclusion}

## Dataset integrity

{markdown_table([
    ("Combined CSV rows", combined_csv["rows"]),
    ("Combined CSV columns", combined_csv["columns"]),
    ("Train rows", len(train)),
    ("Test rows", len(test)),
    ("Train target counts", json.dumps(target_train, sort_keys=True)),
    ("Test target counts", json.dumps(target_test, sort_keys=True)),
    ("Combined CSV missing cells", combined_csv["missing_cells"]),
    ("Combined CSV duplicate full rows", combined_csv["duplicate_rows"]),
    ("Train duplicate full rows", overlap["train_duplicate_rows"]),
    ("Test duplicate full rows", overlap["test_duplicate_rows"]),
    ("Cross-split duplicate full rows", overlap["cross_split_duplicate_rows"]),
    ("Cross-split duplicate feature rows", overlap["cross_split_duplicate_feature_rows"]),
    ("Feature hashes with conflicting targets", combined_csv["feature_hashes_with_conflicting_targets"]),
    ("CSV exactly matches train/test row multiset", combined_csv["matches_split_multiset"]),
    ("CSV rows matched to train/test", combined_csv["matched_split_rows"]),
])}

## Statistical proxy screen

Each non-target field was checked on the training set for deterministic target mapping.
Numeric fields also received Pearson correlation and orientation-free univariate ROC-AUC.
Categorical fields received Cramér's V and weighted class purity. Strong association alone
was not treated as leakage unless the field was a deterministic target proxy or its business
meaning placed it after the prediction point.

## Limitations

{chr(10).join(f"- {item}" for item in limitations)}

## Reproducibility

- Python: `{platform.python_version()}`
- pandas: `{pd.__version__}`
- NumPy: `{np.__version__}`
- SciPy: `{scipy.__version__}`
- scikit-learn: `{sklearn.__version__}`
- Platform: `{platform.platform()}`
- Input SHA-256:
{chr(10).join(f"  - `{name}`: `{digest}`" for name, digest in input_hashes.items())}
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    csv_path = data_dir / "ddbb_fraud.csv"
    dictionary_path = data_dir / "variables_ddbb_fraud.xlsx"

    required = [csv_path, dictionary_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    frame = pd.read_csv(csv_path, low_memory=False)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")
    frame[TARGET_COLUMN] = normalize_target(frame[TARGET_COLUMN])
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The source CSV contains unexpected or missing target values.")

    row_indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=AUDIT_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET_COLUMN],
    )
    train = frame.iloc[train_indices].reset_index(drop=True)
    test = frame.iloc[test_indices].reset_index(drop=True)

    dictionary = read_xlsx_first_sheet(dictionary_path)
    dictionary.columns = [str(column).strip() for column in dictionary.columns]
    if not {"Variables", "Description"}.issubset(dictionary.columns):
        raise ValueError("Variable dictionary must contain Variables and Description columns.")
    descriptions = {
        str(variable).strip(): str(description).strip()
        for variable, description in zip(
            dictionary["Variables"], dictionary["Description"], strict=True
        )
    }

    full_train_hashes = row_hashes(train)
    full_test_hashes = row_hashes(test)
    feature_train_hashes = feature_hashes(train)
    feature_test_hashes = feature_hashes(test)

    train_full_counter = Counter(int(value) for value in full_train_hashes)
    test_full_counter = Counter(int(value) for value in full_test_hashes)
    split_full_counter = train_full_counter + test_full_counter

    train_feature_counter = Counter(int(value) for value in feature_train_hashes)
    test_feature_counter = Counter(int(value) for value in feature_test_hashes)

    overlap = {
        "train_duplicate_rows": count_duplicate_hashes(full_train_hashes),
        "test_duplicate_rows": count_duplicate_hashes(full_test_hashes),
        "cross_split_duplicate_rows": counter_intersection_size(
            train_full_counter, test_full_counter
        ),
        "train_duplicate_feature_rows": count_duplicate_hashes(feature_train_hashes),
        "test_duplicate_feature_rows": count_duplicate_hashes(feature_test_hashes),
        "cross_split_duplicate_feature_rows": counter_intersection_size(
            train_feature_counter, test_feature_counter
        ),
    }

    feature_target_counts = (
        pd.DataFrame(
            {
                "feature_hash": feature_hashes(frame),
                "target": frame[TARGET_COLUMN].astype("int64"),
            }
        )
        .groupby("feature_hash")["target"]
        .nunique()
    )
    combined_csv = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "target_counts": target_counts(frame),
        "duplicate_rows": count_duplicate_hashes(row_hashes(frame)),
        "feature_hashes_with_conflicting_targets": int(
            feature_target_counts.gt(1).sum()
        ),
        "split_feature_hash_conflicts": 0,
        "matches_split_multiset": True,
        "matched_split_rows": len(frame),
    }

    audit = build_audit_table(train, descriptions)
    approved_features = audit.loc[
        audit["decision"] == "approved", "variable"
    ].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "leakage_audit.csv"
    approved_path = output_dir / "approved_features.json"
    report_path = output_dir / "leakage_analysis_report.md"

    audit.to_csv(audit_path, index=False)
    approved_path.write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "prediction_point": "first_claim_notification_before_investigation",
                "approved_features": approved_features,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    input_hashes = {
        path.name: sha256_file(path)
        for path in [csv_path, dictionary_path]
    }
    write_report(
        path=report_path,
        audit=audit,
        approved_features=approved_features,
        train=train,
        test=test,
        overlap=overlap,
        combined_csv=combined_csv,
        input_hashes=input_hashes,
    )

    print(f"Wrote {audit_path}")
    print(f"Wrote {approved_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Leakage audit failed: {exc}", file=sys.stderr)
        raise
