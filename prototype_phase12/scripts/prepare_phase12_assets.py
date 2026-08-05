from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "respuesta_dicot_c"
EXPECTED_TRAIN_ROWS = 444_074
EXPECTED_TEST_ROWS = 111_020
EXPECTED_TEST_FRAUD = 465
DEFAULT_THRESHOLD = 0.0041884911

DISPLAY_FIELDS = [
    "garantia",
    "garantia_agrupada",
    "antiguedad_poliza",
    "dias_notificacion",
    "comarcaid",
    "tomadornivel",
    "edad_conductor1",
    "formapago",
    "tipo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Phase 1 prototype assets: fitted model files, q-star config, "
            "LABEL_MAP, and a user-safe claim pool."
        )
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="One fitted model file. Repeat for an ensemble.",
    )
    parser.add_argument(
        "--model-family",
        choices=["xgboost", "lightgbm"],
        default="xgboost",
        help="Chosen prototype model family.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Optional row-level predictions for claim-pool selection. Must contain "
            "row_index and fraud_probability."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--claim-pool-size", type=int, default=250)
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help="Allow draft asset creation when supplied split counts do not match the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    approved = json.loads((out / "approved_features.json").read_text(encoding="utf-8"))[
        "approved_features"
    ]
    labels = json.loads((out / "label_map.json").read_text(encoding="utf-8"))["labels"]

    train_frame = pd.read_parquet(args.train, columns=approved + [TARGET_COLUMN])
    train_rows, train_fraud = len(train_frame), int(train_frame[TARGET_COLUMN].sum())
    test = pd.read_parquet(args.test)
    test_rows = len(test)
    test_fraud = int(test[TARGET_COLUMN].sum())
    if (
        not args.allow_split_mismatch
        and (
            train_rows != EXPECTED_TRAIN_ROWS
            or test_rows != EXPECTED_TEST_ROWS
            or test_fraud != EXPECTED_TEST_FRAUD
        )
    ):
        raise SystemExit(
            "Input split does not match the current report: "
            f"train_rows={train_rows}, test_rows={test_rows}, test_fraud={test_fraud}. "
            "Provide the current split or rerun with --allow-split-mismatch for a draft only."
        )

    missing_features = [f for f in approved if f not in test.columns]
    if missing_features:
        raise SystemExit(f"Test file is missing approved features: {missing_features[:10]}")

    model_paths = _copy_models(args.model, out)
    categorical_features = _categorical_features_from_data(train_frame, approved)
    _write_category_map(out, train_frame, categorical_features)
    _write_config(out, args.threshold, model_paths, categorical_features, args.model_family)
    _write_claim_pool(
        out=out,
        test=test,
        approved_features=approved,
        labels=labels,
        predictions_path=args.predictions,
        threshold=args.threshold,
        claim_pool_size=args.claim_pool_size,
    )
    print(f"Prepared Phase 1/2 assets in {out}")


def _copy_models(model_paths: list[Path], out: Path) -> list[str]:
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    relative_paths: list[str] = []
    for i, source in enumerate(model_paths, start=1):
        if not source.exists():
            raise SystemExit(f"Model file does not exist: {source}")
        dest = model_dir / f"member_{i:02d}_{source.name}"
        shutil.copy2(source, dest)
        relative_paths.append(dest.relative_to(out).as_posix())
    return relative_paths


def _categorical_features_from_data(
    frame: pd.DataFrame,
    approved_features: list[str],
) -> list[str]:
    numeric_categorical = {
        "comarcaid",
        "tomadorcodigopostal",
        "tomadormunicipioid",
        "tomadorprovinciaid",
        "tomadorcomarcaid",
        "tomadornacionalidadid",
    }
    text_features = set(
        frame[approved_features].select_dtypes(include=["object", "string", "category"]).columns
    )
    categorical = text_features | (numeric_categorical & set(approved_features))
    return [feature for feature in approved_features if feature in categorical]


def _write_config(
    out: Path,
    threshold: float,
    model_paths: list[str],
    categorical_features: list[str],
    model_family: str,
) -> None:
    config = {
        "schema_version": 1,
        "model_family": model_family,
        "model_track": f"{model_family}_chosen_track",
        "model_type": (
            "ensemble" if len(model_paths) > 1 else "single fitted model"
        ),
        "q_star_threshold": threshold,
        "threshold_role": "fixed operating threshold, shown as context and not editable",
        "target_column": TARGET_COLUMN,
        "model_paths": model_paths,
        "approved_features_path": "approved_features.json",
        "label_map_path": "label_map.json",
        "category_map_path": "category_map.json",
        "claim_pool_path": "claim_pool_features.parquet",
        "claim_pool_ui_path": "claim_pool_ui.json",
        "private_audit_path": "claim_pool_audit_private.csv",
        "categorical_features": categorical_features,
        "explanation_method": (
            "XGBoost native TreeSHAP via pred_contribs=True"
            if model_family == "xgboost"
            else "LightGBM native TreeSHAP via Booster.predict(pred_contrib=True)"
        ),
    }
    (out / "model_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_category_map(
    out: Path,
    train_frame: pd.DataFrame,
    categorical_features: list[str],
) -> None:
    categories: dict[str, list[Any]] = {}
    for feature in categorical_features:
        values = train_frame[feature].astype("category").cat.categories.tolist()
        categories[feature] = [_json_safe(value) for value in values]
    payload = {
        "schema_version": 1,
        "source": "training data categorical levels",
        "categories": categories,
    }
    (out / "category_map.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_claim_pool(
    *,
    out: Path,
    test: pd.DataFrame,
    approved_features: list[str],
    labels: dict[str, str],
    predictions_path: Path | None,
    threshold: float,
    claim_pool_size: int,
) -> None:
    frame = test.reset_index(drop=True).copy()
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)

    predictions = None
    if predictions_path is not None:
        predictions = pd.read_parquet(predictions_path)
        required = {"row_index", "fraud_probability"}
        if not required.issubset(predictions.columns):
            raise SystemExit(f"Predictions file must contain {sorted(required)}")
        predictions = predictions[["row_index", "fraud_probability"]].copy()
        predictions["row_index"] = predictions["row_index"].astype("int64")
        frame = frame.merge(predictions, on="row_index", how="left", validate="one_to_one")
        if frame["fraud_probability"].isna().any():
            raise SystemExit("Predictions do not cover every test row.")
        selected_indices = _select_with_predictions(frame, threshold, claim_pool_size)
    else:
        selected_indices = np.linspace(
            0, len(frame) - 1, num=min(claim_pool_size, len(frame)), dtype=np.int64
        )

    pool = frame.iloc[selected_indices].copy()
    pool["claim_id"] = [f"CLM-{int(i):06d}" for i in pool["row_index"]]
    pool_features = pool[["claim_id", "row_index"] + approved_features]
    pool_features.to_parquet(out / "claim_pool_features.parquet", index=False)

    ui_claims = []
    for _, row in pool.iterrows():
        summary = {
            labels.get(feature, feature): _json_safe(row[feature])
            for feature in DISPLAY_FIELDS
            if feature in row.index
        }
        ui_claims.append(
            {
                "claim_id": row["claim_id"],
                "selector_label": row["claim_id"],
                "summary": summary,
            }
        )
    (out / "claim_pool_ui.json").write_text(
        json.dumps({"claims": ui_claims}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if predictions is not None:
        audit = pool[["claim_id", "row_index", TARGET_COLUMN, "fraud_probability"]].copy()
        audit["flagged"] = audit["fraud_probability"] >= threshold
        audit["confusion_bucket"] = np.select(
            [
                (audit[TARGET_COLUMN] == 1) & audit["flagged"],
                (audit[TARGET_COLUMN] == 0) & audit["flagged"],
                (audit[TARGET_COLUMN] == 1) & ~audit["flagged"],
                (audit[TARGET_COLUMN] == 0) & ~audit["flagged"],
            ],
            ["TP", "FP", "FN", "TN"],
            default="unknown",
        )
        audit.to_csv(out / "claim_pool_audit_private.csv", index=False)

    manifest = {
        "schema_version": 1,
        "claim_pool_rows": int(len(pool_features)),
        "target_column_removed_from_user_visible_pool": True,
        "prediction_file_used_for_selection": str(predictions_path) if predictions_path else None,
    }
    (out / "claim_pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _select_with_predictions(
    frame: pd.DataFrame,
    threshold: float,
    claim_pool_size: int,
) -> np.ndarray:
    flagged = frame[frame["fraud_probability"] >= threshold]
    not_flagged = frame[frame["fraud_probability"] < threshold]
    groups = [
        flagged.nlargest(claim_pool_size // 3, "fraud_probability"),
        flagged.assign(distance=(flagged["fraud_probability"] - threshold).abs())
        .nsmallest(claim_pool_size // 3, "distance")
        .drop(columns=["distance"]),
        not_flagged.assign(distance=(not_flagged["fraud_probability"] - threshold).abs())
        .nsmallest(claim_pool_size // 3, "distance")
        .drop(columns=["distance"]),
        not_flagged.nsmallest(max(1, claim_pool_size // 6), "fraud_probability"),
    ]
    selected = pd.concat(groups, ignore_index=False)
    selected = selected[~selected.index.duplicated(keep="first")]
    if len(selected) < claim_pool_size:
        filler = frame.drop(index=selected.index, errors="ignore").sample(
            n=min(claim_pool_size - len(selected), len(frame) - len(selected)),
            random_state=42,
        )
        selected = pd.concat([selected, filler], ignore_index=False)
    return selected.head(claim_pool_size).index.to_numpy(dtype=np.int64)


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
