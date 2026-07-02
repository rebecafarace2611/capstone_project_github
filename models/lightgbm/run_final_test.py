from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
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
from models.lightgbm.metrics import binary_metrics_at_threshold, discrimination_metrics
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
from models.lightgbm.run_imbalance_screen import undersample_training_indices
from models.lightgbm.run_optuna_tuning import configuration_fingerprint
from models.lightgbm.run_seed_confirmation import LOCKED_INPUT_SHA256


EXPECTED_TEST_SHA256 = (
    "2e9d2e888505e03dccdd40c70ad9df007d18e760570291d8d1b0c2854f724907"
)
EXPECTED_TEST_ROWS = 111_020
EXPECTED_TEST_FRAUD = 465
FINAL_CONFIRMATION_PHRASE = "I_UNDERSTAND_TEST_IS_ONE_TIME"
EXPECTED_LOCKED_THRESHOLD = 0.23669952663465765


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the five locked LightGBM members on all training data and "
            "evaluate the sealed test set exactly once."
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
        "--lock-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "fixed_oof_lock",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "final_test",
    )
    parser.add_argument(
        "--confirm-final-test",
        default="",
        help=(
            "Required acknowledgement phrase: "
            f"{FINAL_CONFIRMATION_PHRASE}"
        ),
    )
    return parser.parse_args()


def validate_final_args(args: argparse.Namespace) -> None:
    if args.confirm_final_test != FINAL_CONFIRMATION_PHRASE:
        raise ValueError(
            "Final test acknowledgement missing. Pass "
            f"--confirm-final-test {FINAL_CONFIRMATION_PHRASE}"
        )


def load_locked_spec(lock_dir: Path) -> dict[str, Any]:
    required = {
        "status": lock_dir / "status.json",
        "run_spec": lock_dir / "run_spec.json",
        "locked_model_spec": lock_dir / "locked_model_spec.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Fixed OOF lock evidence is incomplete: {missing}")
    status = json.loads(required["status"].read_text(encoding="utf-8"))
    run_spec = json.loads(required["run_spec"].read_text(encoding="utf-8"))
    locked = json.loads(required["locked_model_spec"].read_text(encoding="utf-8"))
    if status.get("status") != "complete" or int(status.get("completed_fits", -1)) != 25:
        raise ValueError("Fixed OOF lock is not complete at 25/25 fits.")
    if locked.get("status") != "locked_pending_single_final_test":
        raise ValueError("Model specification is not locked for final testing.")
    if locked.get("input_sha256") != LOCKED_INPUT_SHA256:
        raise ValueError("Locked model uses different training inputs.")
    if bool(locked.get("test_data_used")):
        raise ValueError("Locked-model metadata already indicates test use.")
    if locked.get("locked_candidate", {}).get("candidate_id") != "C":
        raise ValueError("Locked candidate is not C.")
    if int(locked.get("locked_tree_count", -1)) != 248:
        raise ValueError("Locked tree count is not 248.")
    ensemble = locked.get("locked_ensemble", {})
    seeds = ensemble.get("final_full_training_sampling_seeds")
    if seeds != [105042, 205042, 305042, 405042, 505042]:
        raise ValueError("Unexpected final ensemble sampling seeds.")
    threshold = float(locked.get("locked_threshold", float("nan")))
    if not np.isfinite(threshold) or threshold != EXPECTED_LOCKED_THRESHOLD:
        raise ValueError("Locked threshold does not match the reviewed OOF result.")
    parameters = locked.get("locked_model_parameters", {})
    expected_parameters = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.026485341497747728,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 50,
        "n_estimators": 248,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "class_weight": None,
        "random_state": 42,
        "n_jobs": 24,
        "device_type": "cpu",
        "min_split_gain": 0.1,
        "cat_smooth": 50.0,
        "cat_l2": 1.0,
    }
    mismatched = {
        key: {"expected": expected, "actual": parameters.get(key)}
        for key, expected in expected_parameters.items()
        if parameters.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Locked model parameters changed: {mismatched}")
    return {
        "status": status,
        "run_spec": run_spec,
        "locked": locked,
        "sha256": {name: sha256_file(path) for name, path in required.items()},
    }


def align_test_categories(
    training: pd.DataFrame,
    test: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    metadata: dict[str, dict[str, int]] = {}
    for feature in categorical_features:
        if not isinstance(training[feature].dtype, pd.CategoricalDtype):
            raise ValueError(f"Training feature is not categorical: {feature}")
        categories = training[feature].cat.categories
        unseen_count = int((~test[feature].isin(categories)).sum())
        test[feature] = pd.Categorical(test[feature], categories=categories)
        metadata[feature] = {
            "training_level_count": int(len(categories)),
            "unseen_test_rows_treated_as_missing": unseen_count,
        }
    return training, test, metadata


def save_booster_atomic(booster: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    booster.save_model(str(temporary))
    temporary.replace(path)


def completed_member(
    *,
    metrics_path: Path,
    predictions_path: Path,
    model_path: Path,
    expected_fingerprint: str,
    expected_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    if not metrics_path.exists() or not predictions_path.exists() or not model_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        return None
    if metrics.get("configuration_fingerprint") != expected_fingerprint:
        raise ValueError(f"Final member configuration mismatch: {metrics_path}")
    predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    expected_index = np.arange(expected_rows, dtype=np.int64)
    if not np.array_equal(predictions["row_index"].to_numpy(), expected_index):
        raise ValueError(f"Final member prediction coverage mismatch: {predictions_path}")
    if sha256_file(model_path) != metrics.get("model_sha256"):
        raise ValueError(f"Final member model hash mismatch: {model_path}")
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    validate_final_args(args)
    lgb = import_lightgbm()
    train_path = args.train.resolve()
    test_path = args.test.resolve()
    approved_path = args.approved_features.resolve()
    lock_dir = args.lock_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final_summary_path = output_dir / "final_test_summary.json"
    status_path = output_dir / "status.json"
    if final_summary_path.exists():
        raise RuntimeError(
            "Final test summary already exists. This one-time evaluation may not be rerun."
        )
    if status_path.exists():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("status") == "complete":
            raise RuntimeError("Final test is already complete and may not be rerun.")

    lock_evidence = load_locked_spec(lock_dir)
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
        raise ValueError(
            f"Final train/test inputs do not match the sealed hashes: {input_hashes}"
        )

    locked = lock_evidence["locked"]
    model_parameters = locked["locked_model_parameters"]
    sampling_seeds = locked["locked_ensemble"][
        "final_full_training_sampling_seeds"
    ]
    locked_threshold = float(locked["locked_threshold"])
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_single_final_test",
        "one_time_evaluation": True,
        "locked_model_spec_sha256": lock_evidence["sha256"]["locked_model_spec"],
        "model_parameters": model_parameters,
        "sampling_seeds": sampling_seeds,
        "ensemble_rule": "arithmetic mean of five fraud probabilities",
        "locked_threshold": locked_threshold,
        "input_sha256": input_hashes,
    }
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior != run_spec:
            raise ValueError("Existing final-test directory uses a different specification.")
    write_json(spec_path, run_spec)
    run_fingerprint = configuration_fingerprint(run_spec)

    global_marker_path = lock_dir / "FINAL_TEST_CONSUMPTION.json"
    if global_marker_path.exists():
        global_marker = json.loads(global_marker_path.read_text(encoding="utf-8"))
        if global_marker.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                "The locked model is already associated with a different final-test run."
            )
        if global_marker.get("state") == "complete":
            raise RuntimeError("The locked model's final test has already been consumed.")
        if Path(global_marker.get("output_dir", "")).resolve() != output_dir:
            raise RuntimeError(
                "An interrupted final test must resume in its original output directory."
            )
    else:
        write_json(
            global_marker_path,
            {
                "state": "in_progress_resumable_same_spec_only",
                "opened_at_utc": utc_now(),
                "run_fingerprint": run_fingerprint,
                "output_dir": str(output_dir),
                "test_sha256": input_hashes["test"],
            },
        )

    opened_path = output_dir / "FINAL_TEST_OPENED.json"
    if opened_path.exists():
        opened = json.loads(opened_path.read_text(encoding="utf-8"))
        if opened.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Final-test opened marker belongs to a different run.")
    else:
        write_json(
            opened_path,
            {
                "opened_at_utc": utc_now(),
                "run_fingerprint": run_fingerprint,
                "input_sha256": input_hashes,
                "notice": "The sealed final test has now been accessed.",
            },
        )
    write_json(
        status_path,
        {
            "status": "running",
            "started_or_resumed_at_utc": utc_now(),
            "completed_members": 0,
            "expected_members": len(sampling_seeds),
            "test_data_used": True,
        },
    )

    approved_features = load_approved_features(approved_path)
    training = load_training_frame(train_path, approved_features)
    test = load_training_frame(test_path, approved_features)
    if len(test) != EXPECTED_TEST_ROWS or int(test[TARGET_COLUMN].sum()) != EXPECTED_TEST_FRAUD:
        raise ValueError("Sealed test row or fraud count is unexpected.")
    schema = infer_feature_schema(training, approved_features)
    training = compact_for_lightgbm(training, schema)
    test = compact_for_lightgbm(test, schema)
    training, test, categorical_alignment = align_test_categories(
        training,
        test,
        schema.categorical_features,
    )
    train_features = training[approved_features]
    test_features = test[approved_features]
    train_target = training[TARGET_COLUMN].to_numpy(dtype=np.int8)
    test_target = test[TARGET_COLUMN].to_numpy(dtype=np.int8)

    all_train_index = np.arange(len(training), dtype=np.int64)
    member_metrics: list[dict[str, Any]] = []
    member_predictions: list[pd.DataFrame] = []
    ratio = int(locked["locked_candidate"]["hyperparameters"]["rus_ratio"])
    for member_index, sampling_seed in enumerate(sampling_seeds, start=1):
        member_dir = output_dir / "members" / f"member_{member_index}_seed_{sampling_seed}"
        metrics_path = member_dir / "metrics.json"
        predictions_path = member_dir / "test_predictions.parquet"
        model_path = member_dir / "model.txt"
        identity = {
            "run_fingerprint": run_fingerprint,
            "member_index": member_index,
            "sampling_seed": sampling_seed,
        }
        member_fingerprint = configuration_fingerprint(identity)
        completed = completed_member(
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            model_path=model_path,
            expected_fingerprint=member_fingerprint,
            expected_rows=len(test),
        )
        if completed is None:
            train_index = undersample_training_indices(
                all_train_index,
                train_target,
                legitimate_per_fraud=ratio,
                random_state=int(sampling_seed),
            )
            print(
                f"Final member {member_index}/{len(sampling_seeds)}: "
                f"seed={sampling_seed}, training_rows={len(train_index):,}."
            )
            model = lgb.LGBMClassifier(**model_parameters)
            started = time.perf_counter()
            model.fit(
                train_features.iloc[train_index],
                train_target[train_index],
                categorical_feature=schema.categorical_features,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            fit_seconds = time.perf_counter() - started
            score = model.predict_proba(test_features)[:, 1]
            save_booster_atomic(model.booster_, model_path)
            predictions = pd.DataFrame(
                {
                    "row_index": np.arange(len(test), dtype=np.int64),
                    "target": test_target,
                    "member_index": np.full(len(test), member_index, dtype=np.int8),
                    "sampling_seed": np.full(
                        len(test), sampling_seed, dtype=np.int32
                    ),
                    "fraud_probability": score.astype(np.float64),
                }
            )
            write_parquet_atomic(predictions, predictions_path)
            metrics = {
                "status": "complete",
                "completed_at_utc": utc_now(),
                "configuration_fingerprint": member_fingerprint,
                **identity,
                "training_rows": int(len(train_index)),
                "training_fraud": int(train_target[train_index].sum()),
                "test_rows_scored": int(len(test)),
                "tree_count": int(model_parameters["n_estimators"]),
                "fit_seconds": float(fit_seconds),
                "model_sha256": sha256_file(model_path),
                "test_data_used": True,
            }
            write_json(metrics_path, metrics)
            completed = metrics, predictions
            del model, score, train_index
            gc.collect()

        metrics, predictions = completed
        member_metrics.append(
            {
                key: value
                for key, value in metrics.items()
                if key
                not in {
                    "status",
                    "completed_at_utc",
                    "configuration_fingerprint",
                    "run_fingerprint",
                    "test_data_used",
                }
            }
        )
        member_predictions.append(predictions)
        write_csv_atomic(
            pd.DataFrame(member_metrics),
            output_dir / "member_metrics.csv",
        )
        write_json(
            status_path,
            {
                "status": "running",
                "updated_at_utc": utc_now(),
                "completed_members": len(member_metrics),
                "expected_members": len(sampling_seeds),
                "test_data_used": True,
            },
        )

    stacked = pd.concat(member_predictions, ignore_index=True)
    counts = stacked.groupby("row_index")["member_index"].nunique()
    if len(counts) != len(test) or not (counts == len(sampling_seeds)).all():
        raise ValueError("Every test row must have five member predictions.")
    final_predictions = (
        stacked.groupby("row_index", sort=True)
        .agg(
            target=("target", "first"),
            fraud_probability=("fraud_probability", "mean"),
            probability_std_across_members=("fraud_probability", "std"),
            probability_min_across_members=("fraud_probability", "min"),
            probability_max_across_members=("fraud_probability", "max"),
        )
        .reset_index()
    )
    final_predictions["primary_prediction"] = (
        final_predictions["fraud_probability"] >= locked_threshold
    ).astype("int8")
    write_parquet_atomic(
        final_predictions,
        output_dir / "final_test_predictions.parquet",
    )

    discrimination = discrimination_metrics(
        final_predictions["target"].to_numpy(dtype=np.int8),
        final_predictions["fraud_probability"].to_numpy(dtype=np.float64),
    )
    operating = binary_metrics_at_threshold(
        final_predictions["target"].to_numpy(dtype=np.int8),
        final_predictions["fraud_probability"].to_numpy(dtype=np.float64),
        locked_threshold,
    )
    summary = {
        "schema_version": 1,
        "workflow": "lightgbm_single_final_test",
        "status": "complete_final_test_consumed",
        "completed_at_utc": utc_now(),
        "locked_candidate": locked["locked_candidate"],
        "locked_tree_count": locked["locked_tree_count"],
        "ensemble_member_count": len(sampling_seeds),
        "sampling_seeds": sampling_seeds,
        "locked_threshold": locked_threshold,
        "test_metrics_at_locked_threshold": {
            **discrimination,
            **operating,
        },
        "input_sha256": input_hashes,
        "lock_evidence_sha256": lock_evidence["sha256"],
        "selection_after_test_prohibited": True,
        "test_data_used": True,
    }
    write_json(final_summary_path, summary)
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
                "lock_dir": project_path(lock_dir),
                "output_dir": project_path(output_dir),
            },
            "categorical_alignment": categorical_alignment,
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "lightgbm": package_version("lightgbm"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "pyarrow": package_version("pyarrow"),
                "scikit_learn": package_version("scikit-learn"),
            },
            "confidentiality": {
                "models_private": True,
                "row_level_test_predictions_private": True,
            },
            "test_data_used": True,
        },
    )
    write_json(
        status_path,
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_members": len(member_metrics),
            "expected_members": len(sampling_seeds),
            "test_data_used": True,
            "rerun_prohibited": True,
        },
    )
    write_json(
        global_marker_path,
        {
            "state": "complete",
            "completed_at_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "output_dir": str(output_dir),
            "test_sha256": input_hashes["test"],
            "final_test_summary_sha256": sha256_file(final_summary_path),
            "rerun_prohibited": True,
        },
    )
    print("\nFINAL TEST COMPLETED. The sealed test set is now consumed.")
    print(f"Results: {output_dir}")
    print(
        f"Recall={operating['recall']:.6f}, FPR={operating['fpr']:.6f}, "
        f"precision={operating['precision']:.6f}, "
        f"PR-AUC={discrimination['pr_auc_average_precision']:.6f}, "
        f"ROC-AUC={discrimination['roc_auc']:.6f}."
    )
    print("Do not adjust the model or threshold based on this result.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    try:
        run(args)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / "status.json"
        current: dict[str, Any] = {}
        if status_path.exists():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
        if current.get("status") != "complete":
            write_json(
                status_path,
                {
                    "status": "failed_resumable_if_spec_unchanged",
                    "failed_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "test_data_used": (output_dir / "FINAL_TEST_OPENED.json").exists(),
                },
            )
        raise


if __name__ == "__main__":
    main()
