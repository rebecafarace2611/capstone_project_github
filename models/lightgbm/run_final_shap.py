from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lightgbm.data import (
    TARGET_COLUMN,
    compact_for_lightgbm,
    infer_feature_schema,
    load_approved_features,
    load_training_frame,
)
from models.lightgbm.run_baseline import (
    import_lightgbm,
    package_version,
    project_path,
    sha256_file,
    utc_now,
    write_csv_atomic,
    write_json,
    write_parquet_atomic,
)
from models.lightgbm.run_final_test import (
    EXPECTED_TEST_SHA256,
    align_test_categories,
)
from models.lightgbm.run_optuna_tuning import configuration_fingerprint
from models.lightgbm.run_seed_confirmation import LOCKED_INPUT_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute native exact LightGBM TreeSHAP contributions for the "
            "saved five-member final ensemble after final testing."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_model_dataset.parquet",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=PROJECT_ROOT / "data" / "test_model_dataset.parquet",
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
        "--final-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "final_test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "final_test" / "shap_top_score",
    )
    parser.add_argument(
        "--selection",
        choices=[
            "top_score",
            "predicted_positive",
            "actual_positive",
            "stratified",
            "random",
        ],
        default="top_score",
    )
    parser.add_argument("--explain-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_shap_args(args: argparse.Namespace) -> None:
    if args.explain_size < 1:
        raise ValueError("--explain-size must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")


def load_final_evidence(final_dir: Path) -> dict[str, Any]:
    required = {
        "status": final_dir / "status.json",
        "run_spec": final_dir / "run_spec.json",
        "summary": final_dir / "final_test_summary.json",
        "predictions": final_dir / "final_test_predictions.parquet",
        "member_metrics": final_dir / "member_metrics.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Final-test evidence is incomplete: {missing}")
    status = json.loads(required["status"].read_text(encoding="utf-8"))
    run_spec = json.loads(required["run_spec"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    member_metrics = pd.read_csv(required["member_metrics"])
    predictions = pd.read_parquet(required["predictions"], engine="pyarrow")
    if status.get("status") != "complete" or not bool(status.get("rerun_prohibited")):
        raise ValueError("Final test is not irreversibly complete.")
    if summary.get("status") != "complete_final_test_consumed":
        raise ValueError("Unexpected final-test summary status.")
    if not bool(summary.get("test_data_used")):
        raise ValueError("Final-test summary does not record test consumption.")
    if len(member_metrics) != 5 or set(member_metrics["member_index"]) != set(range(1, 6)):
        raise ValueError("Final ensemble does not contain five completed members.")
    if len(predictions) != 111_020:
        raise ValueError("Final prediction file has an unexpected row count.")
    return {
        "status": status,
        "run_spec": run_spec,
        "summary": summary,
        "member_metrics": member_metrics,
        "predictions": predictions,
        "sha256": {
            name: sha256_file(path) for name, path in required.items()
        },
    }


def select_explain_indices(
    *,
    selection: str,
    explain_size: int,
    predictions: pd.DataFrame,
    target: np.ndarray,
    seed: int,
) -> np.ndarray:
    row_count = len(predictions)
    size = min(explain_size, row_count)
    rng = np.random.default_rng(seed)
    if selection == "top_score":
        chosen = predictions.nlargest(size, "fraud_probability")["row_index"]
    elif selection == "predicted_positive":
        pool = predictions[predictions["primary_prediction"] == 1]
        chosen = pool.nlargest(min(size, len(pool)), "fraud_probability")["row_index"]
    elif selection == "actual_positive":
        pool = np.flatnonzero(target == 1)
        chosen = rng.choice(pool, size=min(size, len(pool)), replace=False)
    elif selection == "stratified":
        positive = np.flatnonzero(target == 1)
        negative = np.flatnonzero(target == 0)
        positive_n = min(len(positive), (size + 1) // 2)
        negative_n = min(len(negative), size - positive_n)
        chosen = np.concatenate(
            [
                rng.choice(positive, size=positive_n, replace=False),
                rng.choice(negative, size=negative_n, replace=False),
            ]
        )
    elif selection == "random":
        chosen = rng.choice(row_count, size=size, replace=False)
    else:
        raise ValueError(f"Unknown SHAP selection: {selection}")
    return np.sort(np.asarray(chosen, dtype=np.int64))


def sigmoid(raw_score: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(raw_score, dtype=np.float64), -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def run(args: argparse.Namespace) -> None:
    validate_shap_args(args)
    lgb = import_lightgbm()
    train_path = args.train.resolve()
    test_path = args.test.resolve()
    approved_path = args.approved_features.resolve()
    final_dir = args.final_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final = load_final_evidence(final_dir)
    input_hashes = {
        "train": sha256_file(train_path),
        "test": sha256_file(test_path),
        "approved_features": sha256_file(approved_path),
    }
    expected_hashes = {
        "train": LOCKED_INPUT_SHA256["train"],
        "test": EXPECTED_TEST_SHA256,
        "approved_features": LOCKED_INPUT_SHA256["approved_features"],
    }
    if input_hashes != expected_hashes:
        raise ValueError("SHAP inputs do not match the final-test inputs.")

    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_final_ensemble_native_treeshap",
        "implementation": "LightGBM Booster.predict(pred_contrib=True)",
        "output_space": "raw_score_log_odds",
        "ensemble_aggregation": "arithmetic mean of member SHAP contributions",
        "selection": args.selection,
        "explain_size": args.explain_size,
        "seed": args.seed,
        "final_evidence_sha256": final["sha256"],
        "input_sha256": input_hashes,
    }
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior != run_spec:
            raise ValueError("Existing SHAP directory uses a different specification.")
    write_json(spec_path, run_spec)
    run_fingerprint = configuration_fingerprint(run_spec)
    write_json(
        output_dir / "status.json",
        {
            "status": "running",
            "started_at_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
        },
    )

    approved_features = load_approved_features(approved_path)
    training = load_training_frame(train_path, approved_features)
    test = load_training_frame(test_path, approved_features)
    display_features = test[approved_features].copy()
    schema = infer_feature_schema(training, approved_features)
    training = compact_for_lightgbm(training, schema)
    test = compact_for_lightgbm(test, schema)
    training, test, categorical_alignment = align_test_categories(
        training,
        test,
        schema.categorical_features,
    )
    test_target = test[TARGET_COLUMN].to_numpy(dtype=np.int8)
    predictions = final["predictions"].sort_values("row_index", kind="stable")
    expected_index = np.arange(len(test), dtype=np.int64)
    if not np.array_equal(predictions["row_index"].to_numpy(), expected_index):
        raise ValueError("Final predictions are not in complete test-row order.")
    if not np.array_equal(
        predictions["target"].to_numpy(dtype=np.int8),
        test_target,
    ):
        raise ValueError("Final prediction targets do not match the sealed test data.")
    selected_indices = select_explain_indices(
        selection=args.selection,
        explain_size=args.explain_size,
        predictions=predictions,
        target=test_target,
        seed=args.seed,
    )
    explain_frame = test.iloc[selected_indices][approved_features]
    explain_display = display_features.iloc[selected_indices][approved_features]

    member_shap: list[np.ndarray] = []
    member_bias: list[np.ndarray] = []
    member_raw: list[np.ndarray] = []
    member_probability: list[np.ndarray] = []
    member_checks: list[dict[str, Any]] = []
    metrics = final["member_metrics"].sort_values("member_index", kind="stable")
    for row in metrics.itertuples(index=False):
        member_index = int(row.member_index)
        sampling_seed = int(row.sampling_seed)
        model_path = (
            final_dir
            / "members"
            / f"member_{member_index}_seed_{sampling_seed}"
            / "model.txt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Missing final member model: {model_path}")
        if sha256_file(model_path) != str(row.model_sha256):
            raise ValueError(f"Final member model hash mismatch: {model_path}")
        print(f"Computing native TreeSHAP for member {member_index}/5.")
        booster = lgb.Booster(model_file=str(model_path))
        contributions = booster.predict(
            explain_frame,
            pred_contrib=True,
            num_iteration=248,
        )
        contributions = np.asarray(contributions, dtype=np.float64)
        if contributions.shape != (len(explain_frame), len(approved_features) + 1):
            raise ValueError(
                f"Unexpected SHAP contribution shape for member {member_index}: "
                f"{contributions.shape}"
            )
        raw_score = np.asarray(
            booster.predict(explain_frame, raw_score=True, num_iteration=248),
            dtype=np.float64,
        )
        probability = np.asarray(
            booster.predict(explain_frame, num_iteration=248),
            dtype=np.float64,
        )
        shap_values = contributions[:, :-1]
        bias = contributions[:, -1]
        reconstructed = shap_values.sum(axis=1) + bias
        max_error = float(np.max(np.abs(reconstructed - raw_score)))
        if max_error > 1e-8:
            raise ValueError(
                f"TreeSHAP additivity failed for member {member_index}: {max_error}"
            )
        member_shap.append(shap_values)
        member_bias.append(bias)
        member_raw.append(raw_score)
        member_probability.append(probability)
        member_checks.append(
            {
                "member_index": member_index,
                "sampling_seed": sampling_seed,
                "model_sha256": str(row.model_sha256),
                "max_raw_score_additivity_error": max_error,
            }
        )
        del booster, contributions
        gc.collect()

    shap_cube = np.stack(member_shap, axis=0)
    ensemble_shap = shap_cube.mean(axis=0)
    ensemble_bias = np.stack(member_bias, axis=0).mean(axis=0)
    ensemble_mean_raw = np.stack(member_raw, axis=0).mean(axis=0)
    ensemble_probability = np.stack(member_probability, axis=0).mean(axis=0)
    reconstructed_mean_raw = ensemble_shap.sum(axis=1) + ensemble_bias
    ensemble_additivity_error = np.abs(reconstructed_mean_raw - ensemble_mean_raw)
    max_ensemble_additivity_error = float(ensemble_additivity_error.max())
    if max_ensemble_additivity_error > 1e-8:
        raise ValueError("Averaged TreeSHAP contributions failed raw-score additivity.")

    recorded_probability = predictions.iloc[selected_indices][
        "fraud_probability"
    ].to_numpy(dtype=np.float64)
    max_prediction_reload_error = float(
        np.max(np.abs(ensemble_probability - recorded_probability))
    )
    if max_prediction_reload_error > 1e-10:
        raise ValueError(
            "Reloaded ensemble probabilities differ from final-test predictions: "
            f"{max_prediction_reload_error}"
        )

    shap_wide = pd.DataFrame(ensemble_shap, columns=approved_features)
    shap_wide.insert(0, "source_row_index", selected_indices)
    shap_wide.insert(0, "shap_row_id", np.arange(1, len(shap_wide) + 1))
    write_parquet_atomic(shap_wide, output_dir / "shap_values_wide.parquet")
    shap_long = shap_wide.melt(
        id_vars=["shap_row_id", "source_row_index"],
        value_vars=approved_features,
        var_name="feature",
        value_name="shap_value_raw_score",
    )
    write_parquet_atomic(shap_long, output_dir / "shap_values_long.parquet")

    feature_values = explain_display.copy()
    for feature in approved_features:
        feature_values[feature] = feature_values[feature].astype(str)
    feature_values.insert(0, "source_row_index", selected_indices)
    feature_values.insert(0, "shap_row_id", np.arange(1, len(feature_values) + 1))
    feature_values_long = feature_values.melt(
        id_vars=["shap_row_id", "source_row_index"],
        value_vars=approved_features,
        var_name="feature",
        value_name="feature_value",
    )
    write_parquet_atomic(
        feature_values_long,
        output_dir / "shap_feature_values_long.parquet",
    )

    mean_abs_member = np.mean(np.abs(shap_cube), axis=(0, 1))
    summary_rows: list[dict[str, Any]] = []
    for column_index, feature in enumerate(approved_features):
        values = ensemble_shap[:, column_index]
        summary_rows.append(
            {
                "feature": feature,
                "mean_abs_ensemble_shap": float(np.mean(np.abs(values))),
                "mean_abs_member_shap": float(mean_abs_member[column_index]),
                "mean_ensemble_shap": float(np.mean(values)),
                "sd_ensemble_shap": float(np.std(values, ddof=1)),
                "min_ensemble_shap": float(np.min(values)),
                "q25_ensemble_shap": float(np.quantile(values, 0.25)),
                "median_ensemble_shap": float(np.median(values)),
                "q75_ensemble_shap": float(np.quantile(values, 0.75)),
                "max_ensemble_shap": float(np.max(values)),
                "positive_share": float(np.mean(values > 0.0)),
            }
        )
    shap_summary = pd.DataFrame(summary_rows).sort_values(
        "mean_abs_ensemble_shap", ascending=False, kind="stable"
    ).reset_index(drop=True)
    shap_summary.insert(0, "rank", np.arange(1, len(shap_summary) + 1))
    write_csv_atomic(shap_summary, output_dir / "shap_summary.csv")

    selected_prediction_rows = predictions.iloc[selected_indices]
    row_metadata = pd.DataFrame(
        {
            "shap_row_id": np.arange(1, len(selected_indices) + 1),
            "source_row_index": selected_indices,
            "target": test_target[selected_indices],
            "fraud_probability": recorded_probability,
            "primary_prediction": selected_prediction_rows[
                "primary_prediction"
            ].to_numpy(dtype=np.int8),
            "ensemble_mean_raw_score": ensemble_mean_raw,
            "sigmoid_of_mean_raw_score": sigmoid(ensemble_mean_raw),
            "ensemble_mean_bias": ensemble_bias,
            "raw_score_additivity_error": ensemble_additivity_error,
        }
    )
    write_csv_atomic(row_metadata, output_dir / "shap_explain_rows.csv")
    write_csv_atomic(
        pd.DataFrame(member_checks),
        output_dir / "shap_member_checks.csv",
    )
    write_json(
        output_dir / "shap_metadata.json",
        {
            "schema_version": 1,
            "workflow": "lightgbm_final_ensemble_native_treeshap",
            "status": "complete",
            "generated_at_utc": utc_now(),
            "implementation": "LightGBM native exact TreeSHAP pred_contrib",
            "output_space": "raw_score_log_odds",
            "interpretation_note": (
                "Averaged contributions exactly explain the mean member raw "
                "score. The deployed ensemble averages probabilities, so the "
                "raw-score SHAP sum is not itself an ensemble probability."
            ),
            "dataset": "sealed_final_test_after_metric_consumption",
            "selection": args.selection,
            "explain_size": int(len(selected_indices)),
            "seed": args.seed,
            "feature_count": len(approved_features),
            "ensemble_member_count": 5,
            "max_member_raw_score_additivity_error": float(
                max(row["max_raw_score_additivity_error"] for row in member_checks)
            ),
            "max_ensemble_raw_score_additivity_error": max_ensemble_additivity_error,
            "max_reloaded_probability_error": max_prediction_reload_error,
            "categorical_alignment": categorical_alignment,
            "input_sha256": input_hashes,
            "final_evidence_sha256": final["sha256"],
            "output_files": {
                "shap_values_wide": "shap_values_wide.parquet",
                "shap_values_long": "shap_values_long.parquet",
                "shap_feature_values_long": "shap_feature_values_long.parquet",
                "shap_summary": "shap_summary.csv",
                "shap_explain_rows": "shap_explain_rows.csv",
                "shap_member_checks": "shap_member_checks.csv",
            },
            "model_or_threshold_changed": False,
            "test_metrics_recomputed": False,
        },
    )
    write_json(
        output_dir / "run_context.json",
        {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "paths": {
                "train": project_path(train_path),
                "test": project_path(test_path),
                "approved_features": project_path(approved_path),
                "final_dir": project_path(final_dir),
                "output_dir": project_path(output_dir),
            },
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "lightgbm": package_version("lightgbm"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "pyarrow": package_version("pyarrow"),
            },
            "test_data_role": "post-evaluation_explanation_only",
        },
    )
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "explained_rows": int(len(selected_indices)),
            "model_or_threshold_changed": False,
        },
    )
    print("\nFinal ensemble SHAP completed successfully.")
    print(f"Results: {output_dir}")
    print(shap_summary.head(20).to_string(index=False))


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "status.json",
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_or_threshold_changed": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
