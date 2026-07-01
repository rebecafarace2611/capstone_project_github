from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
from dataclasses import dataclass
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
    discrimination_metrics,
    operating_points,
    threshold_for_minimum_recall,
)
from models.lightgbm.run_baseline import (
    aggregate_feature_importance,
    build_model_parameters,
    completed_fold_artifacts,
    flatten_fold_metrics,
    import_lightgbm,
    json_safe,
    load_completed_fold,
    package_version,
    project_path,
    sha256_file,
    train_fold,
    utc_now,
    write_csv_atomic,
    write_json,
    write_parquet_atomic,
)


@dataclass(frozen=True)
class Strategy:
    name: str
    kind: str
    value: float | int | None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "value": self.value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen class weighting and random undersampling strategies using "
            "the locked grouped folds and FPR at a minimum recall constraint."
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
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "baseline",
        help="Completed baseline to import. It is retrained if incompatible or absent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "imbalance_screen",
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
        "--weights",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--undersampling-ratios",
        type=int,
        nargs="+",
        default=[15, 30, 60],
        help="Number of legitimate rows retained per fraud row.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_screen_args(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.n_estimators < 1 or args.early_stopping_rounds < 1:
        raise ValueError("Estimator and early-stopping counts must be positive.")
    if not 0.0 < args.primary_recall <= 1.0:
        raise ValueError("--primary-recall must be in (0, 1].")
    if any(not 0.0 < value <= 1.0 for value in args.recall_targets):
        raise ValueError("Recall targets must be in (0, 1].")
    if any(value <= 0.0 for value in args.weights):
        raise ValueError("All class weights must be positive.")
    if any(value < 1 for value in args.undersampling_ratios):
        raise ValueError("Undersampling ratios must be positive integers.")
    args.recall_targets = sorted(set(float(value) for value in args.recall_targets))
    args.weights = sorted(set(float(value) for value in args.weights))
    args.undersampling_ratios = sorted(
        set(int(value) for value in args.undersampling_ratios)
    )


def number_label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def build_strategies(args: argparse.Namespace) -> list[Strategy]:
    strategies = [Strategy(name="baseline", kind="baseline", value=None)]
    strategies.extend(
        Strategy(
            name=f"weight_{number_label(weight)}",
            kind="scale_pos_weight",
            value=weight,
        )
        for weight in args.weights
    )
    strategies.extend(
        Strategy(
            name=f"rus_1_to_{ratio}",
            kind="random_undersampling",
            value=ratio,
        )
        for ratio in args.undersampling_ratios
    )
    return strategies


def strategy_model_parameters(
    base_parameters: dict[str, Any],
    strategy: Strategy,
) -> dict[str, Any]:
    parameters = dict(base_parameters)
    parameters.pop("scale_pos_weight", None)
    parameters.pop("is_unbalance", None)
    if strategy.kind == "scale_pos_weight":
        parameters["scale_pos_weight"] = float(strategy.value)
    return parameters


def undersample_training_indices(
    train_index: np.ndarray,
    target: np.ndarray,
    *,
    legitimate_per_fraud: int,
    random_state: int,
) -> np.ndarray:
    index = np.asarray(train_index, dtype=np.int64)
    positive = index[target[index] == 1]
    negative = index[target[index] == 0]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Training fold must contain both target classes.")

    requested_negative = legitimate_per_fraud * len(positive)
    if requested_negative >= len(negative):
        return np.sort(index)
    rng = np.random.default_rng(random_state)
    sampled_negative = rng.choice(
        negative,
        size=requested_negative,
        replace=False,
    )
    return np.sort(np.concatenate([positive, sampled_negative]))


def training_indices_for_strategy(
    strategy: Strategy,
    train_index: np.ndarray,
    target: np.ndarray,
    *,
    fold: int,
    random_state: int,
) -> np.ndarray:
    if strategy.kind != "random_undersampling":
        return train_index
    ratio = int(strategy.value)
    sampling_seed = random_state + ratio * 1000 + fold
    return undersample_training_indices(
        train_index,
        target,
        legitimate_per_fraud=ratio,
        random_state=sampling_seed,
    )


def baseline_is_compatible(
    baseline_dir: Path,
    *,
    input_hashes: dict[str, str],
    base_parameters: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    required = [
        baseline_dir / "status.json",
        baseline_dir / "run_spec.json",
        baseline_dir / "fold_metrics.csv",
        baseline_dir / "baseline_summary.json",
    ]
    if not all(path.exists() for path in required):
        return False
    status = json.loads(required[0].read_text(encoding="utf-8"))
    spec = json.loads(required[1].read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        return False
    if spec.get("input_sha256") != input_hashes:
        return False
    if spec.get("n_splits") != args.n_splits:
        return False
    if spec.get("primary_recall") != args.primary_recall:
        return False
    prior_parameters = spec.get("model_parameters", {})
    comparison_keys = [
        "objective",
        "boosting_type",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "class_weight",
        "device_type",
        "metric",
    ]
    return all(
        prior_parameters.get(key) == base_parameters.get(key)
        for key in comparison_keys
    ) and "scale_pos_weight" not in prior_parameters


def validate_fold_metrics_frame(frame: pd.DataFrame, n_splits: int) -> None:
    required = {
        "fold",
        "training_rows",
        "validation_rows",
        "best_iteration",
        "fit_seconds",
        "roc_auc",
        "pr_auc_average_precision",
        "primary_recall",
        "primary_fpr",
        "primary_precision",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {missing}")
    if len(frame) != n_splits or set(frame["fold"]) != set(range(n_splits)):
        raise ValueError("Fold metrics do not contain exactly one row per fold.")


def summarize_strategy(
    strategy: Strategy,
    fold_metrics: pd.DataFrame,
    *,
    source: str,
    full_training_rows_per_fold: float,
    primary_recall: float,
) -> dict[str, Any]:
    validate_fold_metrics_frame(fold_metrics, len(fold_metrics))
    best_iterations = fold_metrics["best_iteration"].astype(int).tolist()
    return {
        "strategy": strategy.name,
        "kind": strategy.kind,
        "value": strategy.value,
        "source": source,
        "completed_folds": int(len(fold_metrics)),
        "all_folds_meet_recall": bool(
            (fold_metrics["primary_recall"] >= primary_recall).all()
        ),
        "mean_fpr_at_primary_recall": float(fold_metrics["primary_fpr"].mean()),
        "std_fpr_at_primary_recall": float(fold_metrics["primary_fpr"].std(ddof=1)),
        "best_fold_fpr": float(fold_metrics["primary_fpr"].min()),
        "worst_fold_fpr": float(fold_metrics["primary_fpr"].max()),
        "mean_recall": float(fold_metrics["primary_recall"].mean()),
        "mean_precision": float(fold_metrics["primary_precision"].mean()),
        "mean_pr_auc": float(fold_metrics["pr_auc_average_precision"].mean()),
        "std_pr_auc": float(fold_metrics["pr_auc_average_precision"].std(ddof=1)),
        "mean_roc_auc": float(fold_metrics["roc_auc"].mean()),
        "std_roc_auc": float(fold_metrics["roc_auc"].std(ddof=1)),
        "best_iterations": best_iterations,
        "median_best_iteration": int(statistics.median(best_iterations)),
        "mean_fit_seconds": float(fold_metrics["fit_seconds"].mean()),
        "mean_training_rows": float(fold_metrics["training_rows"].mean()),
        "training_row_retention": float(
            fold_metrics["training_rows"].mean() / full_training_rows_per_fold
        ),
    }


def execute_strategy(
    *,
    lgb: Any,
    strategy: Strategy,
    strategy_dir: Path,
    features: pd.DataFrame,
    target: np.ndarray,
    fold_vector: np.ndarray,
    categorical_features: list[str],
    base_parameters: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        strategy_dir / "status.json",
        {"status": "running", "started_at_utc": utc_now()},
    )
    parameters = strategy_model_parameters(base_parameters, strategy)
    oof_score = np.full(len(target), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    fold_importances: list[pd.DataFrame] = []

    for fold in range(args.n_splits):
        validation_index = np.flatnonzero(fold_vector == fold)
        full_train_index = np.flatnonzero(fold_vector != fold)
        model_train_index = training_indices_for_strategy(
            strategy,
            full_train_index,
            target,
            fold=fold,
            random_state=args.random_state,
        )
        fold_dir = strategy_dir / f"fold_{fold}"
        artifacts = completed_fold_artifacts(fold_dir)

        if args.resume and all(path.exists() for path in artifacts):
            print(f"{strategy.name} fold {fold}: loading completed checkpoint.")
            score, metrics, importance = load_completed_fold(
                fold_dir,
                validation_index,
            )
        else:
            print(
                f"{strategy.name} fold {fold}: training rows="
                f"{len(model_train_index):,}; validation rows="
                f"{len(validation_index):,}."
            )
            score, metrics, importance = train_fold(
                lgb=lgb,
                fold=fold,
                train_index=model_train_index,
                validation_index=validation_index,
                features=features,
                target=target,
                categorical_features=categorical_features,
                model_parameters=parameters,
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
            f"{strategy.name} fold {fold} complete: "
            f"best_iteration={metrics['best_iteration']}; "
            f"PR-AUC={metrics['pr_auc_average_precision']:.6f}; "
            f"FPR={primary['fpr']:.6f}."
        )
        del full_train_index, model_train_index, validation_index, score, importance
        gc.collect()

    if np.isnan(oof_score).any():
        raise AssertionError(f"{strategy.name} has missing OOF predictions.")

    oof_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(target), dtype=np.int64),
            "fold": fold_vector.astype(np.int16),
            TARGET_COLUMN: target,
            "score": oof_score,
        }
    )
    write_parquet_atomic(oof_predictions, strategy_dir / "oof_predictions.parquet")

    fold_frame = pd.DataFrame(
        [flatten_fold_metrics(metrics) for metrics in fold_metrics]
    ).sort_values("fold", kind="stable")
    write_csv_atomic(fold_frame, strategy_dir / "fold_metrics.csv")

    points = pd.DataFrame(operating_points(target, oof_score, args.recall_targets))
    write_csv_atomic(points, strategy_dir / "oof_operating_points.csv")

    pooled = {
        "discrimination": discrimination_metrics(target, oof_score),
        "primary_operating_point": threshold_for_minimum_recall(
            target,
            oof_score,
            args.primary_recall,
        ),
    }
    write_json(
        strategy_dir / "strategy_summary.json",
        {
            "strategy": strategy.as_dict(),
            "model_parameters": parameters,
            "sampling_applied_to_training_folds_only": (
                strategy.kind == "random_undersampling"
            ),
            "pooled_oof": pooled,
            "test_data_used": False,
        },
    )
    write_csv_atomic(
        aggregate_feature_importance(fold_importances),
        strategy_dir / "feature_importance_summary.csv",
    )
    write_json(
        strategy_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_folds": args.n_splits,
        },
    )
    return fold_frame, pooled


def rank_strategies(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    ranking = pd.DataFrame(summaries)
    baseline_rows = ranking.loc[ranking["strategy"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError("Exactly one baseline summary is required.")
    baseline_fpr = float(baseline_rows.iloc[0]["mean_fpr_at_primary_recall"])
    ranking["absolute_fpr_reduction_vs_baseline"] = (
        baseline_fpr - ranking["mean_fpr_at_primary_recall"]
    )
    ranking["relative_fpr_reduction_vs_baseline"] = (
        ranking["absolute_fpr_reduction_vs_baseline"] / baseline_fpr
    )
    ranking["meets_10pct_relative_improvement"] = (
        ranking["relative_fpr_reduction_vs_baseline"] >= 0.10
    )
    ranking = ranking.sort_values(
        [
            "all_folds_meet_recall",
            "mean_fpr_at_primary_recall",
            "std_fpr_at_primary_recall",
            "training_row_retention",
        ],
        ascending=[False, True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int16))
    return ranking


def run(args: argparse.Namespace) -> None:
    validate_screen_args(args)
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    requested_folds_path = args.fold_assignments.resolve()
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {"status": "running", "started_at_utc": utc_now()},
    )

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

    frame = compact_for_lightgbm(frame, schema)
    features = frame[approved_features]
    del frame
    gc.collect()

    base_parameters = build_model_parameters(args)
    input_hashes = {
        "train": sha256_file(train_path),
        "approved_features": sha256_file(approved_path),
        "fold_assignments": sha256_file(folds_path),
    }
    strategies = build_strategies(args)
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_imbalance_strategy_screen",
        "strategies": [strategy.as_dict() for strategy in strategies],
        "base_model_parameters": base_parameters,
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
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior != json_safe(run_spec):
            raise ValueError(
                "The existing screen directory uses a different configuration. "
                "Use a new --output-dir or pass --no-resume."
            )
    write_json(spec_path, run_spec)

    print(
        f"Screening {len(strategies)} strategies on {len(target):,} rows with "
        f"{args.n_splits} locked grouped folds."
    )
    lgb = import_lightgbm()
    strategy_summaries: list[dict[str, Any]] = []
    pooled_summaries: dict[str, Any] = {}
    full_training_rows_per_fold = len(target) * (args.n_splits - 1) / args.n_splits

    for strategy in strategies:
        if strategy.kind == "baseline" and baseline_is_compatible(
            baseline_dir,
            input_hashes=input_hashes,
            base_parameters=base_parameters,
            args=args,
        ):
            print(f"baseline: importing completed metrics from {baseline_dir}.")
            fold_frame = pd.read_csv(baseline_dir / "fold_metrics.csv")
            validate_fold_metrics_frame(fold_frame, args.n_splits)
            baseline_summary = json.loads(
                (baseline_dir / "baseline_summary.json").read_text(encoding="utf-8")
            )
            pooled_summaries[strategy.name] = {
                "discrimination": baseline_summary["pooled_oof_discrimination"],
                "primary_operating_point": baseline_summary["primary_operating_point"],
            }
            source = "imported_completed_baseline"
        else:
            strategy_dir = output_dir / "strategies" / strategy.name
            fold_frame, pooled = execute_strategy(
                lgb=lgb,
                strategy=strategy,
                strategy_dir=strategy_dir,
                features=features,
                target=target,
                fold_vector=fold_vector,
                categorical_features=schema.categorical_features,
                base_parameters=base_parameters,
                args=args,
            )
            pooled_summaries[strategy.name] = pooled
            source = "trained_in_screen"

        strategy_summaries.append(
            summarize_strategy(
                strategy,
                fold_frame,
                source=source,
                full_training_rows_per_fold=full_training_rows_per_fold,
                primary_recall=args.primary_recall,
            )
        )

    ranking = rank_strategies(strategy_summaries)
    write_csv_atomic(ranking, output_dir / "strategy_ranking.csv")
    recommended = ranking.iloc[0].to_dict()
    baseline_row = ranking.loc[ranking["strategy"] == "baseline"].iloc[0].to_dict()
    screen_summary = {
        "workflow": "lightgbm_imbalance_strategy_screen",
        "evaluation_role": "grouped_5fold_training_only",
        "primary_selection_metric": (
            f"mean fold FPR subject to recall >= {args.primary_recall:.2f}"
        ),
        "recommended_strategy": recommended,
        "baseline": baseline_row,
        "meaningful_improvement_threshold": "10% relative FPR reduction",
        "recommended_strategy_meets_threshold": bool(
            recommended["meets_10pct_relative_improvement"]
        ),
        "pooled_oof_metrics_are_secondary": True,
        "pooled_oof_summaries": pooled_summaries,
        "test_data_used": False,
    }
    write_json(output_dir / "screen_summary.json", screen_summary)
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
                "baseline_dir": project_path(baseline_dir),
                "output_dir": project_path(output_dir),
            },
            "folds_source": folds_source,
            "input_sha256": input_hashes,
            "feature_schema": schema.as_dict(),
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
                "git_policy": "Keep the entire runs directory private and untracked.",
            },
            "test_data_used": False,
        },
    )
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_strategies": len(strategies),
        },
    )

    print("\nImbalance strategy screen completed successfully.")
    print(f"Results: {output_dir}")
    print(
        ranking[
            [
                "rank",
                "strategy",
                "mean_fpr_at_primary_recall",
                "std_fpr_at_primary_recall",
                "mean_pr_auc",
                "relative_fpr_reduction_vs_baseline",
            ]
        ].to_string(index=False)
    )
    print(
        f"Recommended for structural tuning: {recommended['strategy']} "
        f"(mean FPR={recommended['mean_fpr_at_primary_recall']:.6f})."
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
