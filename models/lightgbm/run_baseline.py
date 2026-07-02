from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
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
    create_grouped_fold_assignments,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the untouched-distribution LightGBM baseline with grouped "
            "cross-validation, PR-AUC early stopping, and OOF threshold analysis."
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
        help=(
            "Existing grouped folds to reuse. If this file is absent, equivalent "
            "folds are generated inside the run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "baseline",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-evaluation", type=int, default=50)
    parser.add_argument("--primary-recall", type=float, default=0.80)
    parser.add_argument(
        "--recall-targets",
        type=float,
        nargs="+",
        default=[0.70, 0.75, 0.80, 0.85, 0.90],
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed fold artifacts in the same run directory.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(
        temporary,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is not installed. Run "
            "'python -m pip install -r models/lightgbm/requirements.txt' first."
        ) from exc
    return lgb


def validate_args(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be positive.")
    if args.early_stopping_rounds < 1:
        raise ValueError("--early-stopping-rounds must be positive.")
    all_recalls = [args.primary_recall, *args.recall_targets]
    if any(not 0.0 < value <= 1.0 for value in all_recalls):
        raise ValueError("Recall targets must be in (0, 1].")
    args.recall_targets = sorted(set(float(value) for value in args.recall_targets))


def build_model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "n_estimators": args.n_estimators,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "class_weight": None,
        "random_state": args.random_state,
        "n_jobs": args.threads,
        "device_type": args.device,
        "metric": "average_precision",
        "verbosity": -1,
    }
    if args.device == "cpu":
        parameters.update(
            {
                "deterministic": True,
                "force_col_wise": True,
            }
        )
    return parameters


def fold_summary(fold_vector: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    summary = (
        pd.DataFrame({"fold": fold_vector, "target": target})
        .groupby("fold", sort=True)["target"]
        .agg(rows="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
    )
    return summary.to_dict(orient="records")


def completed_fold_artifacts(
    fold_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    return (
        fold_dir / "validation_predictions.parquet",
        fold_dir / "metrics.json",
        fold_dir / "feature_importance.csv",
        fold_dir / "model.txt",
    )


def load_completed_fold(
    fold_dir: Path,
    expected_index: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    prediction_path, metrics_path, importance_path, _ = completed_fold_artifacts(fold_dir)
    predictions = pd.read_parquet(prediction_path, engine="pyarrow")
    required = ["row_index", "fold", TARGET_COLUMN, "score"]
    if list(predictions.columns) != required:
        raise ValueError(f"Unexpected columns in resumed predictions: {prediction_path}")
    predictions = predictions.sort_values("row_index", kind="stable")
    if not np.array_equal(
        predictions["row_index"].to_numpy(dtype=np.int64),
        np.sort(expected_index),
    ):
        raise ValueError(f"Resumed predictions do not match fold rows: {prediction_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    importance = pd.read_csv(importance_path)
    return predictions["score"].to_numpy(dtype=np.float64), metrics, importance


def train_fold(
    *,
    lgb: Any,
    fold: int,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    features: pd.DataFrame,
    target: np.ndarray,
    categorical_features: list[str],
    model_parameters: dict[str, Any],
    early_stopping_rounds: int,
    log_evaluation: int,
    primary_recall: float,
    fold_dir: Path,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    fold_dir.mkdir(parents=True, exist_ok=True)
    model = lgb.LGBMClassifier(**model_parameters)
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=early_stopping_rounds,
            first_metric_only=True,
            verbose=True,
        )
    ]
    if log_evaluation > 0:
        callbacks.append(lgb.log_evaluation(period=log_evaluation))

    started = time.perf_counter()
    model.fit(
        features.iloc[train_index],
        target[train_index],
        eval_set=[(features.iloc[validation_index], target[validation_index])],
        eval_names=["validation"],
        eval_metric="average_precision",
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )
    fit_seconds = time.perf_counter() - started
    best_iteration = int(model.best_iteration_)
    score = model.predict_proba(
        features.iloc[validation_index],
        num_iteration=best_iteration,
    )[:, 1]

    discrimination = discrimination_metrics(target[validation_index], score)
    primary_point = threshold_for_minimum_recall(
        target[validation_index],
        score,
        primary_recall,
    )
    default_point = binary_metrics_at_threshold(target[validation_index], score, 0.5)
    metrics: dict[str, Any] = {
        "fold": fold,
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "training_fraud": int(target[train_index].sum()),
        "validation_fraud": int(target[validation_index].sum()),
        "best_iteration": best_iteration,
        "fit_seconds": float(fit_seconds),
        **discrimination,
        "primary_operating_point": primary_point,
        "fixed_0.5_operating_point": default_point,
        "lightgbm_best_score": model.best_score_,
    }

    predictions = pd.DataFrame(
        {
            "row_index": validation_index.astype(np.int64),
            "fold": np.full(len(validation_index), fold, dtype=np.int16),
            TARGET_COLUMN: target[validation_index].astype(np.int8),
            "score": score.astype(np.float64),
        }
    ).sort_values("row_index", kind="stable")
    importance = pd.DataFrame(
        {
            "feature": model.booster_.feature_name(),
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    )

    prediction_path, metrics_path, importance_path, model_path = (
        completed_fold_artifacts(fold_dir)
    )
    model.booster_.save_model(str(model_path), num_iteration=best_iteration)
    write_parquet_atomic(predictions, prediction_path)
    write_json(metrics_path, metrics)
    write_csv_atomic(importance, importance_path)
    return score, metrics, importance


def flatten_fold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = metrics["primary_operating_point"]
    fixed = metrics["fixed_0.5_operating_point"]
    return {
        "fold": metrics["fold"],
        "training_rows": metrics["training_rows"],
        "validation_rows": metrics["validation_rows"],
        "training_fraud": metrics["training_fraud"],
        "validation_fraud": metrics["validation_fraud"],
        "best_iteration": metrics["best_iteration"],
        "fit_seconds": metrics["fit_seconds"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc_average_precision": metrics["pr_auc_average_precision"],
        "primary_threshold": primary["threshold"],
        "primary_recall": primary["recall"],
        "primary_fpr": primary["fpr"],
        "primary_precision": primary["precision"],
        "primary_fp": primary["fp"],
        "primary_fn": primary["fn"],
        "fixed_0.5_recall": fixed["recall"],
        "fixed_0.5_fpr": fixed["fpr"],
        "fixed_0.5_precision": fixed["precision"],
    }


def aggregate_feature_importance(importances: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(
        [frame.assign(fold=index) for index, frame in enumerate(importances)],
        ignore_index=True,
    )
    summary = (
        combined.groupby("feature", sort=False)
        .agg(
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
            mean_split=("split", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values("mean_gain", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    total_gain = float(summary["mean_gain"].sum())
    summary["gain_share"] = (
        summary["mean_gain"] / total_gain if total_gain > 0.0 else 0.0
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1, dtype=np.int32))
    return summary


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    requested_folds_path = args.fold_assignments.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"status": "running", "started_at_utc": utc_now()})

    print("Loading approved feature list and confidential training data...")
    approved_features = load_approved_features(approved_path)
    frame = load_training_frame(train_path, approved_features)
    schema = infer_feature_schema(frame, approved_features)
    target_series = frame[TARGET_COLUMN]
    target = target_series.to_numpy(dtype=np.int8)
    groups = feature_groups(frame, approved_features)

    generated_folds_path = output_dir / "fold_assignments.parquet"
    if requested_folds_path.exists():
        assignments = load_fold_assignments(requested_folds_path)
        folds_path = requested_folds_path
        folds_source = "provided"
    elif generated_folds_path.exists():
        assignments = load_fold_assignments(generated_folds_path)
        folds_path = generated_folds_path
        folds_source = "generated_resume"
    else:
        print("Existing grouped folds not found; generating equivalent folds...")
        assignments = create_grouped_fold_assignments(
            frame,
            approved_features,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        write_parquet_atomic(assignments, generated_folds_path)
        folds_path = generated_folds_path
        folds_source = "generated"

    validate_fold_assignments(
        assignments,
        rows=len(frame),
        target=target_series,
        groups=groups,
        n_splits=args.n_splits,
    )
    fold_vector = ordered_fold_vector(assignments)
    del groups
    gc.collect()

    print(
        f"Rows: {len(frame):,}; fraud: {int(target.sum()):,}; "
        f"features: {len(approved_features)}; categorical: "
        f"{len(schema.categorical_features)}"
    )
    frame = compact_for_lightgbm(frame, schema)
    features = frame[approved_features]
    del frame
    gc.collect()

    model_parameters = build_model_parameters(args)
    input_hashes = {
        "train": sha256_file(train_path),
        "approved_features": sha256_file(approved_path),
        "fold_assignments": sha256_file(folds_path),
    }
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_untouched_distribution_baseline",
        "model_parameters": model_parameters,
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_metric": "average_precision",
        "primary_recall": args.primary_recall,
        "recall_targets": args.recall_targets,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "input_sha256": input_hashes,
    }
    spec_path = output_dir / "run_spec.json"
    if args.resume and spec_path.exists():
        prior_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior_spec != json_safe(run_spec):
            raise ValueError(
                "The existing run directory was created with a different data or "
                "configuration. Use a new --output-dir or pass --no-resume."
            )
    write_json(spec_path, run_spec)

    lgb = import_lightgbm()
    oof_score = np.full(len(target), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    fold_importances: list[pd.DataFrame] = []

    for fold in range(args.n_splits):
        validation_index = np.flatnonzero(fold_vector == fold)
        train_index = np.flatnonzero(fold_vector != fold)
        fold_dir = output_dir / f"fold_{fold}"
        artifact_paths = completed_fold_artifacts(fold_dir)

        if args.resume and all(path.exists() for path in artifact_paths):
            print(f"Fold {fold}: loading completed checkpoint.")
            score, metrics, importance = load_completed_fold(
                fold_dir,
                validation_index,
            )
        else:
            print(
                f"Fold {fold}: fitting on {len(train_index):,} rows; "
                f"validating on {len(validation_index):,} rows."
            )
            score, metrics, importance = train_fold(
                lgb=lgb,
                fold=fold,
                train_index=train_index,
                validation_index=validation_index,
                features=features,
                target=target,
                categorical_features=schema.categorical_features,
                model_parameters=model_parameters,
                early_stopping_rounds=args.early_stopping_rounds,
                log_evaluation=args.log_evaluation,
                primary_recall=args.primary_recall,
                fold_dir=fold_dir,
            )

        oof_score[validation_index] = score
        fold_metrics.append(metrics)
        fold_importances.append(importance)
        primary = metrics["primary_operating_point"]
        print(
            f"Fold {fold} complete: best_iteration={metrics['best_iteration']}; "
            f"PR-AUC={metrics['pr_auc_average_precision']:.6f}; "
            f"FPR@recall>={args.primary_recall:.2f}={primary['fpr']:.6f}."
        )
        del train_index, validation_index, score, importance
        gc.collect()

    if np.isnan(oof_score).any():
        raise AssertionError("At least one training row is missing an OOF prediction.")

    oof_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(target), dtype=np.int64),
            "fold": fold_vector.astype(np.int16),
            TARGET_COLUMN: target,
            "score": oof_score,
        }
    )
    write_parquet_atomic(oof_predictions, output_dir / "oof_predictions.parquet")

    fold_metrics_frame = pd.DataFrame(
        [flatten_fold_metrics(metrics) for metrics in fold_metrics]
    ).sort_values("fold", kind="stable")
    write_csv_atomic(fold_metrics_frame, output_dir / "fold_metrics.csv")

    points = operating_points(target, oof_score, args.recall_targets)
    points_frame = pd.DataFrame(points)
    preferred_columns = [
        "rule",
        "requested_minimum_recall",
        "threshold",
        "recall",
        "fpr",
        "specificity",
        "precision",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "predicted_positive",
        "predicted_positive_rate",
        "false_positives_per_10000_legitimate",
    ]
    write_csv_atomic(
        points_frame[preferred_columns],
        output_dir / "oof_operating_points.csv",
    )

    pooled_discrimination = discrimination_metrics(target, oof_score)
    pooled_primary = threshold_for_minimum_recall(
        target,
        oof_score,
        args.primary_recall,
    )
    best_iterations = [int(metrics["best_iteration"]) for metrics in fold_metrics]
    summary = {
        "workflow": "lightgbm_untouched_distribution_baseline",
        "evaluation_role": "grouped_5fold_out_of_fold_training_only",
        "rows": len(target),
        "fraud_cases": int(target.sum()),
        "fraud_rate": float(target.mean()),
        "model_parameters": model_parameters,
        "sampling": "none",
        "class_weight": "none",
        "early_stopping": {
            "metric": "average_precision",
            "rounds": args.early_stopping_rounds,
            "best_iterations": best_iterations,
            "recommended_final_iterations_median": int(
                statistics.median(best_iterations)
            ),
        },
        "pooled_oof_discrimination": pooled_discrimination,
        "primary_operating_point": pooled_primary,
        "test_data_used": False,
    }
    write_json(output_dir / "baseline_summary.json", summary)

    importance_summary = aggregate_feature_importance(fold_importances)
    write_csv_atomic(
        importance_summary,
        output_dir / "feature_importance_summary.csv",
    )

    context = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "paths": {
            "train": project_path(train_path),
            "approved_features": project_path(approved_path),
            "fold_assignments": project_path(folds_path),
            "output_dir": project_path(output_dir),
        },
        "folds_source": folds_source,
        "input_sha256": input_hashes,
        "feature_schema": schema.as_dict(),
        "fold_summary": fold_summary(fold_vector, target),
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
            "contains_row_level_oof_predictions": True,
            "git_policy": "Keep the entire runs directory untracked and private.",
        },
        "test_data_used": False,
    }
    write_json(output_dir / "run_context.json", context)
    write_json(
        status_path,
        {
            "status": "complete",
            "completed_at_utc": context["completed_at_utc"],
            "completed_folds": args.n_splits,
        },
    )

    print("\nLightGBM baseline completed successfully.")
    print(f"Results: {output_dir}")
    print(
        f"Pooled OOF PR-AUC: "
        f"{pooled_discrimination['pr_auc_average_precision']:.6f}"
    )
    print(
        f"At recall >= {args.primary_recall:.2f}: "
        f"FPR={pooled_primary['fpr']:.6f}; FP={pooled_primary['fp']:,}; "
        f"threshold={pooled_primary['threshold']:.8f}"
    )
    print(
        "Recommended final tree count (median fold best iteration): "
        f"{summary['early_stopping']['recommended_final_iterations_median']}"
    )


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
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
