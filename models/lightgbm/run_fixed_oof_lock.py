from __future__ import annotations

import argparse
import gc
import json
import os
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
    feature_groups,
    infer_feature_schema,
    load_approved_features,
    load_fold_assignments,
    load_training_frame,
    ordered_fold_vector,
    validate_fold_assignments,
)
from models.lightgbm.metrics import (
    binary_metrics_at_threshold,
    discrimination_metrics,
    operating_points,
    threshold_for_minimum_recall,
)
from models.lightgbm.run_baseline import (
    build_model_parameters,
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
from models.lightgbm.run_optuna_tuning import (
    build_trial_model_parameters,
    configuration_fingerprint,
)
from models.lightgbm.run_seed_confirmation import (
    DEFAULT_SEED_OFFSETS,
    LOCKED_INPUT_SHA256,
    confirmation_candidates,
    rus_sampling_seed,
)


LOCKED_CANDIDATE_ID = "C"
LOCKED_TREE_COUNT = 248
LOCKED_PRIMARY_RECALL = 0.80
EXPECTED_CONFIRMATION_FITS = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock fixed-tree five-seed ensemble OOF scores and one pooled "
            "threshold for the confirmed LightGBM candidate."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=PROJECT_ROOT / "data" / "train_model_dataset.parquet",
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
        "--fold-assignments",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rfqc" / "folds" / "fold_assignments.parquet",
    )
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "seed_confirmation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "fixed_oof_lock",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    return parser.parse_args()


def locked_candidate():
    matches = [
        candidate
        for candidate in confirmation_candidates()
        if candidate.candidate_id == LOCKED_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise AssertionError("The locked candidate definition is not unique.")
    return matches[0]


def validate_lock_args(args: argparse.Namespace) -> None:
    if args.threads < 1:
        raise ValueError("--threads must be positive.")


def load_confirmation_evidence(confirmation_dir: Path) -> dict[str, Any]:
    required = {
        "status": confirmation_dir / "status.json",
        "run_spec": confirmation_dir / "run_spec.json",
        "summary": confirmation_dir / "confirmation_summary.json",
        "ranking": confirmation_dir / "candidate_ranking.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Confirmation evidence is incomplete; missing: {missing}"
        )

    status = json.loads(required["status"].read_text(encoding="utf-8"))
    run_spec = json.loads(required["run_spec"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    ranking = pd.read_csv(required["ranking"])
    if status.get("status") != "complete":
        raise ValueError("Seed confirmation is not complete.")
    if int(status.get("completed_fits", -1)) != EXPECTED_CONFIRMATION_FITS:
        raise ValueError("Seed confirmation does not contain all 75 fits.")
    if run_spec.get("input_sha256") != LOCKED_INPUT_SHA256:
        raise ValueError("Confirmation used different locked inputs.")
    if run_spec.get("seed_offsets") != DEFAULT_SEED_OFFSETS:
        raise ValueError("Confirmation used different RUS seed offsets.")
    if bool(run_spec.get("test_data_used")):
        raise ValueError("Confirmation metadata indicates test data use.")
    if summary.get("status") != "confirmation_complete_pending_review":
        raise ValueError("Unexpected confirmation summary status.")
    assessment = summary.get("selection_assessment", {})
    if assessment.get("provisional_candidate_pending_review") != LOCKED_CANDIDATE_ID:
        raise ValueError("Confirmation does not support candidate C as provisional choice.")

    selected = ranking[ranking["candidate_id"] == LOCKED_CANDIDATE_ID]
    if len(selected) != 1:
        raise ValueError("Candidate C is missing or duplicated in confirmation ranking.")
    row = selected.iloc[0]
    if int(row["evidence_rank"]) != 1:
        raise ValueError("Candidate C is not the mean-FPR confirmation leader.")
    if int(row["median_best_iteration_25_fits"]) != LOCKED_TREE_COUNT:
        raise ValueError("Confirmation no longer supports the locked 248-tree rule.")
    if not bool(row["selection_shortlist"]):
        raise ValueError("Candidate C is not on the confirmation selection shortlist.")

    return {
        "status": status,
        "run_spec": run_spec,
        "summary": summary,
        "selected_candidate_row": row.to_dict(),
        "sha256": {
            name: sha256_file(path) for name, path in required.items()
        },
    }


def build_locked_model_parameters(
    *,
    threads: int,
    device: str,
) -> dict[str, Any]:
    candidate = locked_candidate()
    base_args = argparse.Namespace(
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        n_estimators=LOCKED_TREE_COUNT,
        random_state=42,
        threads=threads,
        device=device,
    )
    base_parameters = build_model_parameters(base_args)
    parameters = build_trial_model_parameters(
        base_parameters,
        candidate.hyperparameters,
    )
    parameters["n_estimators"] = LOCKED_TREE_COUNT
    return parameters


def fit_fixed_model(
    *,
    lgb: Any,
    model_parameters: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    categorical_features: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    model = lgb.LGBMClassifier(**model_parameters)
    started = time.perf_counter()
    model.fit(
        features.iloc[train_index],
        target[train_index],
        categorical_feature=categorical_features,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    fit_seconds = time.perf_counter() - started
    score = model.predict_proba(features.iloc[validation_index])[:, 1]
    discrimination = discrimination_metrics(target[validation_index], score)
    primary = threshold_for_minimum_recall(
        target[validation_index],
        score,
        LOCKED_PRIMARY_RECALL,
    )
    metrics = {
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "training_fraud": int(target[train_index].sum()),
        "validation_fraud": int(target[validation_index].sum()),
        "tree_count": LOCKED_TREE_COUNT,
        "fit_seconds": float(fit_seconds),
        **discrimination,
        "diagnostic_fold_threshold": float(primary["threshold"]),
        "diagnostic_fold_recall": float(primary["recall"]),
        "diagnostic_fold_fpr": float(primary["fpr"]),
        "diagnostic_fold_precision": float(primary["precision"]),
    }
    return metrics, np.asarray(score, dtype=np.float64)


def average_seed_predictions(
    predictions: pd.DataFrame,
    *,
    rows: int,
    expected_seed_count: int,
) -> pd.DataFrame:
    required = {"row_index", "target", "fold", "seed_offset", "score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    if predictions.duplicated(["row_index", "seed_offset"]).any():
        raise ValueError("Duplicate row/seed predictions detected.")

    grouped = predictions.groupby("row_index", sort=True)
    counts = grouped["seed_offset"].nunique()
    if len(counts) != rows or not (counts == expected_seed_count).all():
        raise ValueError(
            "Every OOF row must have exactly one prediction from every seed."
        )
    target_counts = grouped["target"].nunique()
    fold_counts = grouped["fold"].nunique()
    if not (target_counts == 1).all() or not (fold_counts == 1).all():
        raise ValueError("Target or fold changed across seed predictions.")

    averaged = grouped.agg(
        target=("target", "first"),
        fold=("fold", "first"),
        score=("score", "mean"),
        score_std_across_seeds=("score", "std"),
        score_min_across_seeds=("score", "min"),
        score_max_across_seeds=("score", "max"),
    ).reset_index()
    expected_index = np.arange(rows, dtype=np.int64)
    if not np.array_equal(averaged["row_index"].to_numpy(), expected_index):
        raise ValueError("Averaged OOF predictions do not cover every row in order.")
    return averaged


def load_completed_fit(
    *,
    metrics_path: Path,
    predictions_path: Path,
    expected_fingerprint: str,
    expected_validation_index: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame] | None:
    if not metrics_path.exists() or not predictions_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        return None
    if metrics.get("configuration_fingerprint") != expected_fingerprint:
        raise ValueError(f"Completed fit configuration mismatch: {metrics_path}")
    predictions = pd.read_parquet(predictions_path, engine="pyarrow")
    actual_index = predictions["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_index, expected_validation_index):
        raise ValueError(f"Completed fit row coverage mismatch: {predictions_path}")
    return metrics, predictions


def run(args: argparse.Namespace) -> None:
    validate_lock_args(args)
    lgb = import_lightgbm()
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    folds_path = args.fold_assignments.resolve()
    confirmation_dir = args.confirmation_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {"status": "running", "started_at_utc": utc_now()},
    )

    confirmation = load_confirmation_evidence(confirmation_dir)
    input_hashes = {
        "train": sha256_file(train_path),
        "approved_features": sha256_file(approved_path),
        "fold_assignments": sha256_file(folds_path),
    }
    if input_hashes != LOCKED_INPUT_SHA256:
        raise ValueError("Fixed OOF inputs do not match the locked tuning inputs.")

    approved_features = load_approved_features(approved_path)
    frame = load_training_frame(train_path, approved_features)
    schema = infer_feature_schema(frame, approved_features)
    target_series = frame[TARGET_COLUMN]
    target = target_series.to_numpy(dtype=np.int8)
    groups = feature_groups(frame, approved_features)
    assignments = load_fold_assignments(folds_path)
    validate_fold_assignments(
        assignments,
        rows=len(frame),
        target=target_series,
        groups=groups,
        n_splits=5,
    )
    fold_vector = ordered_fold_vector(assignments)
    del groups
    gc.collect()

    frame = compact_for_lightgbm(frame, schema)
    features = frame[approved_features]
    del frame
    gc.collect()

    candidate = locked_candidate()
    model_parameters = build_locked_model_parameters(
        threads=args.threads,
        device=args.device,
    )
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_fixed_oof_threshold_lock",
        "candidate": candidate.as_dict(),
        "model_parameters": model_parameters,
        "tree_count": LOCKED_TREE_COUNT,
        "tree_count_rule": (
            "median best iteration across candidate C's 25 confirmation fits"
        ),
        "ensemble": {
            "member_count": len(DEFAULT_SEED_OFFSETS),
            "combination": "arithmetic mean of fraud probabilities",
            "seed_offsets": DEFAULT_SEED_OFFSETS,
            "sampling_seed_policy": (
                "42 + seed_offset + rus_ratio * 1000 + validation_fold"
            ),
        },
        "n_splits": 5,
        "expected_fits": 5 * len(DEFAULT_SEED_OFFSETS),
        "primary_recall": LOCKED_PRIMARY_RECALL,
        "threshold_rule": (
            "highest pooled ensemble OOF score threshold attaining recall >= 0.80"
        ),
        "input_sha256": input_hashes,
        "confirmation_evidence_sha256": confirmation["sha256"],
        "test_data_used": False,
    }
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior != run_spec:
            raise ValueError(
                "Existing fixed OOF directory has a different configuration. "
                "Use a new --output-dir."
            )
    write_json(spec_path, run_spec)
    run_fingerprint = configuration_fingerprint(run_spec)

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    expected_fits = int(run_spec["expected_fits"])
    ratio = int(candidate.hyperparameters["rus_ratio"])
    completed_fits = 0
    for fold in range(5):
        validation_index = np.flatnonzero(fold_vector == fold)
        full_train_index = np.flatnonzero(fold_vector != fold)
        for seed_offset in DEFAULT_SEED_OFFSETS:
            sampling_seed = rus_sampling_seed(
                random_state=42,
                seed_offset=seed_offset,
                rus_ratio=ratio,
                fold=fold,
            )
            fit_dir = (
                output_dir
                / "folds"
                / f"fold_{fold}"
                / f"seed_offset_{seed_offset}"
            )
            metrics_path = fit_dir / "metrics.json"
            predictions_path = fit_dir / "predictions.parquet"
            identity = {
                "run_fingerprint": run_fingerprint,
                "candidate_id": LOCKED_CANDIDATE_ID,
                "tree_count": LOCKED_TREE_COUNT,
                "fold": fold,
                "seed_offset": seed_offset,
                "sampling_seed": sampling_seed,
            }
            fit_fingerprint = configuration_fingerprint(identity)
            completed = load_completed_fit(
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                expected_fingerprint=fit_fingerprint,
                expected_validation_index=validation_index,
            )
            if completed is None:
                train_index = undersample_training_indices(
                    full_train_index,
                    target,
                    legitimate_per_fraud=ratio,
                    random_state=sampling_seed,
                )
                print(
                    f"Fixed OOF fold={fold}, offset={seed_offset}: "
                    f"fitting {len(train_index):,} rows with {LOCKED_TREE_COUNT} trees."
                )
                metrics, score = fit_fixed_model(
                    lgb=lgb,
                    model_parameters=model_parameters,
                    features=features,
                    target=target,
                    train_index=train_index,
                    validation_index=validation_index,
                    categorical_features=schema.categorical_features,
                )
                predictions = pd.DataFrame(
                    {
                        "row_index": validation_index.astype(np.int64),
                        "target": target[validation_index].astype(np.int8),
                        "fold": np.full(len(validation_index), fold, dtype=np.int16),
                        "seed_offset": np.full(
                            len(validation_index), seed_offset, dtype=np.int32
                        ),
                        "score": score,
                    }
                )
                write_parquet_atomic(predictions, predictions_path)
                metrics_payload = {
                    "status": "complete",
                    "completed_at_utc": utc_now(),
                    "configuration_fingerprint": fit_fingerprint,
                    **identity,
                    **metrics,
                    "test_data_used": False,
                }
                write_json(metrics_path, metrics_payload)
                completed = metrics_payload, predictions
                del train_index, score
                gc.collect()

            metrics_payload, predictions = completed
            metric_rows.append(
                {
                    "candidate_id": LOCKED_CANDIDATE_ID,
                    "fold": fold,
                    "seed_offset": seed_offset,
                    "sampling_seed": sampling_seed,
                    **{
                        key: value
                        for key, value in metrics_payload.items()
                        if key
                        not in {
                            "status",
                            "completed_at_utc",
                            "configuration_fingerprint",
                            "run_fingerprint",
                            "candidate_id",
                            "fold",
                            "seed_offset",
                            "sampling_seed",
                            "test_data_used",
                        }
                    },
                }
            )
            prediction_frames.append(predictions)
            completed_fits += 1
            write_csv_atomic(
                pd.DataFrame(metric_rows),
                output_dir / "model_fold_metrics.csv",
            )
            write_json(
                output_dir / "status.json",
                {
                    "status": "running",
                    "updated_at_utc": utc_now(),
                    "completed_fits": completed_fits,
                    "expected_fits": expected_fits,
                },
            )
        del validation_index, full_train_index
        gc.collect()

    model_metrics = pd.DataFrame(metric_rows).sort_values(
        ["fold", "seed_offset"], kind="stable"
    ).reset_index(drop=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    averaged_oof = average_seed_predictions(
        all_predictions,
        rows=len(target),
        expected_seed_count=len(DEFAULT_SEED_OFFSETS),
    )
    write_csv_atomic(model_metrics, output_dir / "model_fold_metrics.csv")
    write_parquet_atomic(
        averaged_oof,
        output_dir / "oof_ensemble_predictions.parquet",
    )

    ensemble_fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        fold_frame = averaged_oof[averaged_oof["fold"] == fold]
        fold_target = fold_frame["target"].to_numpy(dtype=np.int8)
        fold_score = fold_frame["score"].to_numpy(dtype=np.float64)
        discrimination = discrimination_metrics(fold_target, fold_score)
        fold_primary = threshold_for_minimum_recall(
            fold_target,
            fold_score,
            LOCKED_PRIMARY_RECALL,
        )
        ensemble_fold_rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(fold_frame)),
                "validation_fraud": int(fold_target.sum()),
                **discrimination,
                "diagnostic_fold_threshold": float(fold_primary["threshold"]),
                "diagnostic_fold_recall": float(fold_primary["recall"]),
                "diagnostic_fold_fpr": float(fold_primary["fpr"]),
                "diagnostic_fold_precision": float(fold_primary["precision"]),
                "mean_score_std_across_seeds": float(
                    fold_frame["score_std_across_seeds"].mean()
                ),
            }
        )

    pooled_target = averaged_oof["target"].to_numpy(dtype=np.int8)
    pooled_score = averaged_oof["score"].to_numpy(dtype=np.float64)
    pooled_discrimination = discrimination_metrics(pooled_target, pooled_score)
    locked_point = threshold_for_minimum_recall(
        pooled_target,
        pooled_score,
        LOCKED_PRIMARY_RECALL,
    )
    locked_threshold = float(locked_point["threshold"])
    for row in ensemble_fold_rows:
        fold_frame = averaged_oof[averaged_oof["fold"] == row["fold"]]
        at_locked = binary_metrics_at_threshold(
            fold_frame["target"].to_numpy(dtype=np.int8),
            fold_frame["score"].to_numpy(dtype=np.float64),
            locked_threshold,
        )
        row.update(
            {
                "locked_threshold": locked_threshold,
                "recall_at_locked_threshold": float(at_locked["recall"]),
                "fpr_at_locked_threshold": float(at_locked["fpr"]),
                "precision_at_locked_threshold": float(at_locked["precision"]),
                "fp_at_locked_threshold": int(at_locked["fp"]),
                "fn_at_locked_threshold": int(at_locked["fn"]),
            }
        )
    ensemble_fold_metrics = pd.DataFrame(ensemble_fold_rows)
    write_csv_atomic(
        ensemble_fold_metrics,
        output_dir / "ensemble_fold_metrics.csv",
    )
    write_csv_atomic(
        pd.DataFrame(
            operating_points(
                pooled_target,
                pooled_score,
                [LOCKED_PRIMARY_RECALL],
            )
        ),
        output_dir / "oof_operating_points.csv",
    )

    final_sampling_seeds = [
        rus_sampling_seed(
            random_state=42,
            seed_offset=offset,
            rus_ratio=ratio,
            fold=0,
        )
        for offset in DEFAULT_SEED_OFFSETS
    ]
    write_json(
        output_dir / "locked_model_spec.json",
        {
            "schema_version": 1,
            "workflow": "lightgbm_fixed_oof_threshold_lock",
            "status": "locked_pending_single_final_test",
            "locked_candidate": candidate.as_dict(),
            "locked_model_parameters": model_parameters,
            "locked_tree_count": LOCKED_TREE_COUNT,
            "locked_ensemble": {
                "member_count": len(DEFAULT_SEED_OFFSETS),
                "seed_offsets": DEFAULT_SEED_OFFSETS,
                "prediction_combination": "arithmetic_mean_probability",
                "final_full_training_sampling_seeds": final_sampling_seeds,
            },
            "locked_threshold": locked_threshold,
            "locked_threshold_rule": run_spec["threshold_rule"],
            "pooled_oof": {
                **pooled_discrimination,
                **locked_point,
            },
            "fold_summary": {
                "mean_diagnostic_fold_fpr": float(
                    ensemble_fold_metrics["diagnostic_fold_fpr"].mean()
                ),
                "std_diagnostic_fold_fpr": float(
                    ensemble_fold_metrics["diagnostic_fold_fpr"].std(ddof=1)
                ),
                "worst_diagnostic_fold_fpr": float(
                    ensemble_fold_metrics["diagnostic_fold_fpr"].max()
                ),
            },
            "confirmation_reference": confirmation["selected_candidate_row"],
            "input_sha256": input_hashes,
            "test_data_used": False,
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
                "approved_features": project_path(approved_path),
                "fold_assignments": project_path(folds_path),
                "confirmation_dir": project_path(confirmation_dir),
                "output_dir": project_path(output_dir),
            },
            "input_sha256": input_hashes,
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
                "models_saved": False,
                "row_level_oof_predictions_private": True,
                "test_predictions_created": False,
            },
            "test_data_used": False,
        },
    )
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_fits": completed_fits,
            "expected_fits": expected_fits,
            "locked_threshold": locked_threshold,
        },
    )

    print("\nFixed-tree ensemble OOF threshold lock completed successfully.")
    print(f"Results: {output_dir}")
    print(
        f"Pooled OOF: FPR={locked_point['fpr']:.6f}, "
        f"recall={locked_point['recall']:.6f}, "
        f"precision={locked_point['precision']:.6f}, "
        f"PR-AUC={pooled_discrimination['pr_auc_average_precision']:.6f}."
    )
    print(f"Locked threshold: {locked_threshold:.12g}")
    print("The final test set remains unused.")


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
            },
        )
        raise


if __name__ == "__main__":
    main()
