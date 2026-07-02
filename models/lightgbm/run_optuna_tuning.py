from __future__ import annotations

import argparse
import gc
import hashlib
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
    threshold_for_minimum_recall,
)
from models.lightgbm.run_baseline import (
    build_model_parameters,
    import_lightgbm,
    json_safe,
    package_version,
    project_path,
    sha256_file,
    utc_now,
    write_csv_atomic,
    write_json,
    write_parquet_atomic,
)
from models.lightgbm.run_imbalance_screen import undersample_training_indices


RUS_RATIOS = [5, 10, 15, 20, 30, 60]
TREE_SHAPES = [
    "d3_l7",
    "d4_l15",
    "d5_l15",
    "d5_l31",
    "d6_l31",
    "d6_l63",
    "d8_l31",
    "d8_l63",
    "unlimited_l31",
    "unlimited_l63",
]
MIN_CHILD_SAMPLES = [20, 50, 100, 200, 400]
FEATURE_FRACTIONS = [0.6, 0.7, 0.8, 0.9, 1.0]
REGULARIZATION_VALUES = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0]
MIN_SPLIT_GAINS = [0.0, 0.01, 0.05, 0.1, 0.5]
CAT_SMOOTH_VALUES = [1.0, 5.0, 10.0, 20.0, 50.0]
CAT_L2_VALUES = [1.0, 5.0, 10.0, 20.0]

LOCAL_RUS_RATIOS = [1, 2, 3, 5, 7, 10]
LOCAL_TREE_SHAPES = ["d3_l7", "d4_l15", "d5_l15"]
LOCAL_MIN_CHILD_SAMPLES = [50, 100]
LOCAL_FEATURE_FRACTIONS = [0.8, 0.9, 1.0]
LOCAL_REGULARIZATION_VALUES = [0.0, 0.01, 0.1, 1.0]
LOCAL_MIN_SPLIT_GAINS = [0.05, 0.1, 0.2]
LOCAL_CAT_SMOOTH_VALUES = [20.0, 50.0, 100.0]
LOCAL_CAT_L2_VALUES = [1.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune random-undersampled LightGBM models with persistent Optuna "
            "TPE optimization and grouped five-fold validation."
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
        "--screen-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "imbalance_screen",
    )
    parser.add_argument(
        "--broad-tuning-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "optuna_rus",
        help="Completed broad search used as a reference for local refinement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "optuna_rus",
    )
    parser.add_argument("--study-name", default="lightgbm_rus_fpr_tuning")
    parser.add_argument(
        "--search-profile",
        choices=["broad", "local"],
        default="broad",
    )
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--primary-recall", type=float, default=0.80)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive.")
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.n_estimators < 1 or args.early_stopping_rounds < 1:
        raise ValueError("Estimator and early-stopping counts must be positive.")
    if not 0.0 < args.primary_recall <= 1.0:
        raise ValueError("--primary-recall must be in (0, 1].")


def import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Run "
            "'python -m pip install -r models/lightgbm/requirements.txt' first."
        ) from exc
    return optuna


def search_space(profile: str = "broad") -> dict[str, Any]:
    if profile == "local":
        return {
            "rus_ratio": LOCAL_RUS_RATIOS,
            "learning_rate": {"low": 0.02, "high": 0.10, "log": True},
            "tree_shape": LOCAL_TREE_SHAPES,
            "min_child_samples": LOCAL_MIN_CHILD_SAMPLES,
            "feature_fraction": LOCAL_FEATURE_FRACTIONS,
            "reg_alpha": LOCAL_REGULARIZATION_VALUES,
            "reg_lambda": LOCAL_REGULARIZATION_VALUES,
            "min_split_gain": LOCAL_MIN_SPLIT_GAINS,
            "cat_smooth": LOCAL_CAT_SMOOTH_VALUES,
            "cat_l2": LOCAL_CAT_L2_VALUES,
        }
    if profile != "broad":
        raise ValueError(f"Unknown search profile: {profile}")
    return {
        "rus_ratio": RUS_RATIOS,
        "learning_rate": {"low": 0.01, "high": 0.10, "log": True},
        "tree_shape": TREE_SHAPES,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "feature_fraction": FEATURE_FRACTIONS,
        "reg_alpha": REGULARIZATION_VALUES,
        "reg_lambda": REGULARIZATION_VALUES,
        "min_split_gain": MIN_SPLIT_GAINS,
        "cat_smooth": CAT_SMOOTH_VALUES,
        "cat_l2": CAT_L2_VALUES,
    }


def decode_tree_shape(tree_shape: str) -> tuple[int, int]:
    if tree_shape.startswith("unlimited_l"):
        return -1, int(tree_shape.rsplit("l", maxsplit=1)[1])
    try:
        depth_part, leaves_part = tree_shape.split("_")
        max_depth = int(depth_part.removeprefix("d"))
        num_leaves = int(leaves_part.removeprefix("l"))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid tree shape: {tree_shape}") from exc
    if max_depth < 1 or num_leaves < 2 or num_leaves > 2**max_depth:
        raise ValueError(f"Invalid depth/leaves combination: {tree_shape}")
    return max_depth, num_leaves


def suggest_hyperparameters(
    trial: Any,
    profile: str = "broad",
) -> dict[str, Any]:
    space = search_space(profile)
    learning_rate = space["learning_rate"]
    return {
        "rus_ratio": trial.suggest_categorical("rus_ratio", space["rus_ratio"]),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            learning_rate["low"],
            learning_rate["high"],
            log=learning_rate["log"],
        ),
        "tree_shape": trial.suggest_categorical(
            "tree_shape",
            space["tree_shape"],
        ),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples",
            space["min_child_samples"],
        ),
        "feature_fraction": trial.suggest_categorical(
            "feature_fraction",
            space["feature_fraction"],
        ),
        "reg_alpha": trial.suggest_categorical(
            "reg_alpha",
            space["reg_alpha"],
        ),
        "reg_lambda": trial.suggest_categorical(
            "reg_lambda",
            space["reg_lambda"],
        ),
        "min_split_gain": trial.suggest_categorical(
            "min_split_gain",
            space["min_split_gain"],
        ),
        "cat_smooth": trial.suggest_categorical(
            "cat_smooth",
            space["cat_smooth"],
        ),
        "cat_l2": trial.suggest_categorical(
            "cat_l2",
            space["cat_l2"],
        ),
    }


def build_trial_model_parameters(
    base_parameters: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    max_depth, num_leaves = decode_tree_shape(hyperparameters["tree_shape"])
    parameters = dict(base_parameters)
    parameters.update(
        {
            "learning_rate": float(hyperparameters["learning_rate"]),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": int(hyperparameters["min_child_samples"]),
            "colsample_bytree": float(hyperparameters["feature_fraction"]),
            "reg_alpha": float(hyperparameters["reg_alpha"]),
            "reg_lambda": float(hyperparameters["reg_lambda"]),
            "min_split_gain": float(hyperparameters["min_split_gain"]),
            "cat_smooth": float(hyperparameters["cat_smooth"]),
            "cat_l2": float(hyperparameters["cat_l2"]),
        }
    )
    parameters.pop("scale_pos_weight", None)
    parameters.pop("is_unbalance", None)
    return parameters


def anchor_hyperparameters(profile: str = "broad") -> list[dict[str, Any]]:
    if profile == "local":
        return [
            {
                "rus_ratio": 5,
                "learning_rate": 0.026485341497747728,
                "tree_shape": "d3_l7",
                "min_child_samples": 50,
                "feature_fraction": 0.9,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
                "min_split_gain": 0.1,
                "cat_smooth": 50.0,
                "cat_l2": 1.0,
            },
            {
                "rus_ratio": 5,
                "learning_rate": 0.09751029981938089,
                "tree_shape": "d5_l15",
                "min_child_samples": 50,
                "feature_fraction": 0.9,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
                "min_split_gain": 0.1,
                "cat_smooth": 50.0,
                "cat_l2": 1.0,
            },
        ]
    if profile != "broad":
        raise ValueError(f"Unknown search profile: {profile}")
    common = {
        "learning_rate": 0.05,
        "tree_shape": "unlimited_l31",
        "min_child_samples": 20,
        "feature_fraction": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_split_gain": 0.0,
        "cat_smooth": 10.0,
        "cat_l2": 10.0,
    }
    return [
        {"rus_ratio": 15, **common},
        {"rus_ratio": 30, **common},
    ]


def configuration_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_fold(
    *,
    lgb: Any,
    model_parameters: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    categorical_features: list[str],
    early_stopping_rounds: int,
    primary_recall: float,
) -> dict[str, Any]:
    model = lgb.LGBMClassifier(**model_parameters)
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=early_stopping_rounds,
            first_metric_only=True,
            verbose=False,
        ),
        lgb.log_evaluation(period=0),
    ]
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
    primary = threshold_for_minimum_recall(
        target[validation_index],
        score,
        primary_recall,
    )
    return {
        "training_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "training_fraud": int(target[train_index].sum()),
        "validation_fraud": int(target[validation_index].sum()),
        "best_iteration": best_iteration,
        "fit_seconds": float(fit_seconds),
        **discrimination,
        "threshold": primary["threshold"],
        "recall": primary["recall"],
        "fpr": primary["fpr"],
        "precision": primary["precision"],
        "fp": primary["fp"],
        "fn": primary["fn"],
    }


class TuningObjective:
    def __init__(
        self,
        *,
        lgb: Any,
        features: pd.DataFrame,
        target: np.ndarray,
        fold_vector: np.ndarray,
        categorical_features: list[str],
        base_parameters: dict[str, Any],
        output_dir: Path,
        n_splits: int,
        random_state: int,
        early_stopping_rounds: int,
        primary_recall: float,
        search_profile: str,
    ) -> None:
        self.lgb = lgb
        self.features = features
        self.target = target
        self.fold_vector = fold_vector
        self.categorical_features = categorical_features
        self.base_parameters = base_parameters
        self.output_dir = output_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.primary_recall = primary_recall
        self.search_profile = search_profile

    def __call__(self, trial: Any) -> float:
        hyperparameters = suggest_hyperparameters(trial, self.search_profile)
        model_parameters = build_trial_model_parameters(
            self.base_parameters,
            hyperparameters,
        )
        ratio = int(hyperparameters["rus_ratio"])
        trial_dir = self.output_dir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            trial_dir / "trial_spec.json",
            {
                "trial_number": trial.number,
                "hyperparameters": hyperparameters,
                "model_parameters": model_parameters,
                "started_at_utc": utc_now(),
            },
        )

        fold_metrics: list[dict[str, Any]] = []
        try:
            for fold in range(self.n_splits):
                validation_index = np.flatnonzero(self.fold_vector == fold)
                full_train_index = np.flatnonzero(self.fold_vector != fold)
                train_index = undersample_training_indices(
                    full_train_index,
                    self.target,
                    legitimate_per_fraud=ratio,
                    random_state=self.random_state + ratio * 1000 + fold,
                )
                metrics = evaluate_fold(
                    lgb=self.lgb,
                    model_parameters=model_parameters,
                    features=self.features,
                    target=self.target,
                    train_index=train_index,
                    validation_index=validation_index,
                    categorical_features=self.categorical_features,
                    early_stopping_rounds=self.early_stopping_rounds,
                    primary_recall=self.primary_recall,
                )
                metrics["fold"] = fold
                fold_metrics.append(metrics)
                write_csv_atomic(
                    pd.DataFrame(fold_metrics),
                    trial_dir / "fold_metrics.csv",
                )
                trial.report(
                    float(np.mean([row["fpr"] for row in fold_metrics])),
                    step=fold,
                )
                del validation_index, full_train_index, train_index
                gc.collect()

            fold_frame = pd.DataFrame(fold_metrics).sort_values("fold", kind="stable")
            mean_fpr = float(fold_frame["fpr"].mean())
            std_fpr = float(fold_frame["fpr"].std(ddof=1))
            mean_pr_auc = float(fold_frame["pr_auc_average_precision"].mean())
            mean_roc_auc = float(fold_frame["roc_auc"].mean())
            mean_precision = float(fold_frame["precision"].mean())
            best_iterations = fold_frame["best_iteration"].astype(int).tolist()
            trial.set_user_attr("mean_fpr", mean_fpr)
            trial.set_user_attr("std_fpr", std_fpr)
            trial.set_user_attr("worst_fold_fpr", float(fold_frame["fpr"].max()))
            trial.set_user_attr("mean_pr_auc", mean_pr_auc)
            trial.set_user_attr("mean_roc_auc", mean_roc_auc)
            trial.set_user_attr("mean_precision", mean_precision)
            trial.set_user_attr("best_iterations", best_iterations)
            trial.set_user_attr(
                "median_best_iteration",
                int(np.median(best_iterations)),
            )
            trial.set_user_attr(
                "all_folds_meet_recall",
                bool((fold_frame["recall"] >= self.primary_recall).all()),
            )
            trial.set_user_attr(
                "mean_training_rows",
                float(fold_frame["training_rows"].mean()),
            )
            write_json(
                trial_dir / "trial_summary.json",
                {
                    "status": "complete",
                    "completed_at_utc": utc_now(),
                    "objective_mean_fpr": mean_fpr,
                    "std_fpr": std_fpr,
                    "mean_pr_auc": mean_pr_auc,
                    "mean_roc_auc": mean_roc_auc,
                    "mean_precision": mean_precision,
                    "best_iterations": best_iterations,
                    "all_folds_meet_recall": bool(
                        (fold_frame["recall"] >= self.primary_recall).all()
                    ),
                    "test_data_used": False,
                },
            )
            return mean_fpr
        except Exception as exc:
            write_json(
                trial_dir / "trial_summary.json",
                {
                    "status": "failed",
                    "failed_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise


def completed_trial_ranking(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE" or trial.value is None:
            continue
        row: dict[str, Any] = {
            "trial": trial.number,
            "mean_fpr": float(trial.value),
            "std_fpr": trial.user_attrs.get("std_fpr"),
            "worst_fold_fpr": trial.user_attrs.get("worst_fold_fpr"),
            "mean_pr_auc": trial.user_attrs.get("mean_pr_auc"),
            "mean_roc_auc": trial.user_attrs.get("mean_roc_auc"),
            "mean_precision": trial.user_attrs.get("mean_precision"),
            "median_best_iteration": trial.user_attrs.get(
                "median_best_iteration"
            ),
            "best_iterations": trial.user_attrs.get("best_iterations"),
            "mean_training_rows": trial.user_attrs.get("mean_training_rows"),
        }
        row.update(trial.params)
        rows.append(row)
    if not rows:
        raise ValueError("No completed Optuna trials are available.")
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_fpr", "std_fpr", "mean_pr_auc"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int16))
    return ranking


def load_screen_reference(
    screen_dir: Path,
    input_hashes: dict[str, str],
) -> dict[str, Any] | None:
    spec_path = screen_dir / "run_spec.json"
    ranking_path = screen_dir / "strategy_ranking.csv"
    if not spec_path.exists() or not ranking_path.exists():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("input_sha256") != input_hashes:
        return None
    ranking = pd.read_csv(ranking_path)
    reference: dict[str, Any] = {}
    for strategy in ["baseline", "rus_1_to_15", "rus_1_to_30"]:
        rows = ranking.loc[ranking["strategy"] == strategy]
        if len(rows) == 1:
            reference[strategy] = {
                "mean_fpr": float(rows.iloc[0]["mean_fpr_at_primary_recall"]),
                "std_fpr": float(rows.iloc[0]["std_fpr_at_primary_recall"]),
                "mean_pr_auc": float(rows.iloc[0]["mean_pr_auc"]),
            }
    return reference or None


def load_broad_tuning_reference(
    broad_tuning_dir: Path,
    input_hashes: dict[str, str],
) -> dict[str, Any] | None:
    spec_path = broad_tuning_dir / "run_spec.json"
    best_path = broad_tuning_dir / "best_configuration.json"
    if not spec_path.exists() or not best_path.exists():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("input_sha256") != input_hashes:
        return None
    if spec.get("search_profile", "broad") != "broad":
        return None
    best = json.loads(best_path.read_text(encoding="utf-8"))
    return {
        "best_trial": int(best["best_trial"]),
        "mean_fpr": float(best["best_mean_fpr"]),
        "std_fpr": float(best["best_std_fpr"]),
        "hyperparameters": best["best_hyperparameters"],
    }


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    optuna = import_optuna()
    lgb = import_lightgbm()
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    requested_folds_path = args.fold_assignments.resolve()
    screen_dir = args.screen_dir.resolve()
    broad_tuning_dir = args.broad_tuning_dir.resolve()
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

    base_args = argparse.Namespace(
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        threads=args.threads,
        device=args.device,
    )
    base_parameters = build_model_parameters(base_args)
    input_hashes = {
        "train": sha256_file(train_path),
        "approved_features": sha256_file(approved_path),
        "fold_assignments": sha256_file(folds_path),
    }
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_optuna_rus_tuning",
        "study_name": args.study_name,
        "search_profile": args.search_profile,
        "direction": "minimize",
        "objective": f"mean fold FPR subject to recall >= {args.primary_recall:.2f}",
        "search_space": search_space(args.search_profile),
        "base_model_parameters": base_parameters,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "early_stopping_rounds": args.early_stopping_rounds,
        "primary_recall": args.primary_recall,
        "input_sha256": input_hashes,
    }
    fingerprint = configuration_fingerprint(run_spec)
    compatible_fingerprints = {fingerprint}
    if args.search_profile == "broad":
        legacy_spec = dict(run_spec)
        legacy_spec.pop("search_profile", None)
        compatible_fingerprints.add(configuration_fingerprint(legacy_spec))
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        prior_normalized = dict(prior)
        prior_normalized.setdefault("search_profile", "broad")
        if prior_normalized != json_safe(run_spec):
            raise ValueError(
                "The existing tuning directory uses a different configuration. "
                "Use a new --output-dir."
            )
    write_json(spec_path, run_spec)

    database_path = output_dir / "optuna_study.sqlite3"
    storage_url = f"sqlite:///{database_path.resolve().as_posix()}"
    sampler = optuna.samplers.TPESampler(
        seed=args.random_state,
        n_startup_trials=10,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        direction="minimize",
        load_if_exists=True,
    )
    existing_fingerprint = study.user_attrs.get("configuration_fingerprint")
    if existing_fingerprint is None:
        study.set_user_attr("configuration_fingerprint", fingerprint)
        study.set_user_attr("run_spec", json_safe(run_spec))
    elif existing_fingerprint not in compatible_fingerprints:
        raise ValueError(
            "The Optuna database belongs to a different configuration. "
            "Use a new --output-dir or study name."
        )

    for anchor in anchor_hyperparameters(args.search_profile):
        study.enqueue_trial(anchor, skip_if_exists=True)

    complete_state = optuna.trial.TrialState.COMPLETE
    completed_before = sum(trial.state == complete_state for trial in study.trials)
    remaining = max(0, args.n_trials - completed_before)
    print(
        f"Optuna study '{args.study_name}': {completed_before} completed; "
        f"target={args.n_trials}; remaining={remaining}."
    )
    reference = load_screen_reference(screen_dir, input_hashes)
    if reference:
        print(f"Screen reference: {reference}")
    broad_reference = None
    if args.search_profile == "local":
        broad_reference = load_broad_tuning_reference(
            broad_tuning_dir,
            input_hashes,
        )
        if broad_reference:
            print(f"Broad tuning reference: {broad_reference}")

    objective = TuningObjective(
        lgb=lgb,
        features=features,
        target=target,
        fold_vector=fold_vector,
        categorical_features=schema.categorical_features,
        base_parameters=base_parameters,
        output_dir=output_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
        primary_recall=args.primary_recall,
        search_profile=args.search_profile,
    )
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    ranking = completed_trial_ranking(study)
    baseline_fpr = None
    screen_best_fpr = None
    if reference:
        if "baseline" in reference:
            baseline_fpr = reference["baseline"]["mean_fpr"]
            ranking["relative_fpr_reduction_vs_baseline"] = (
                baseline_fpr - ranking["mean_fpr"]
            ) / baseline_fpr
        if "rus_1_to_15" in reference:
            screen_best_fpr = reference["rus_1_to_15"]["mean_fpr"]
            ranking["relative_fpr_reduction_vs_screen_rus15"] = (
                screen_best_fpr - ranking["mean_fpr"]
            ) / screen_best_fpr
    if broad_reference:
        broad_best_fpr = broad_reference["mean_fpr"]
        ranking["relative_fpr_reduction_vs_broad_best"] = (
            broad_best_fpr - ranking["mean_fpr"]
        ) / broad_best_fpr
    write_csv_atomic(ranking, output_dir / "trial_ranking.csv")
    write_csv_atomic(
        study.trials_dataframe(multi_index=False),
        output_dir / "optuna_trials.csv",
    )

    best_row = ranking.iloc[0].to_dict()
    best_trial = next(
        trial for trial in study.trials if trial.number == int(best_row["trial"])
    )
    best_model_parameters = build_trial_model_parameters(
        base_parameters,
        best_trial.params,
    )
    result = {
        "workflow": "lightgbm_optuna_rus_tuning",
        "search_profile": args.search_profile,
        "status": "tuning_complete_pending_confirmation",
        "completed_trials": int(len(ranking)),
        "best_trial": int(best_trial.number),
        "best_mean_fpr": float(best_row["mean_fpr"]),
        "best_std_fpr": float(best_row["std_fpr"]),
        "best_hyperparameters": best_trial.params,
        "best_model_parameters": best_model_parameters,
        "best_trial_user_attrs": best_trial.user_attrs,
        "screen_reference": reference,
        "broad_tuning_reference": broad_reference,
        "selection_note": (
            "This configuration must pass a fixed-configuration confirmation "
            "run before any final model or threshold is locked."
        ),
        "test_data_used": False,
    }
    write_json(output_dir / "best_configuration.json", result)
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
                "screen_dir": project_path(screen_dir),
                "broad_tuning_dir": project_path(broad_tuning_dir),
                "output_dir": project_path(output_dir),
                "optuna_database": project_path(database_path),
            },
            "folds_source": folds_source,
            "input_sha256": input_hashes,
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "lightgbm": package_version("lightgbm"),
                "optuna": package_version("optuna"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "pyarrow": package_version("pyarrow"),
                "scikit_learn": package_version("scikit-learn"),
            },
            "confidentiality": {
                "trial_models_saved": False,
                "row_level_predictions_saved": False,
                "optuna_database_private": True,
            },
            "test_data_used": False,
        },
    )
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_trials": int(len(ranking)),
            "target_trials": args.n_trials,
        },
    )

    print("\nOptuna RUS tuning completed successfully.")
    print(f"Results: {output_dir}")
    print(
        ranking[
            [
                "rank",
                "trial",
                "mean_fpr",
                "std_fpr",
                "mean_pr_auc",
                "rus_ratio",
                "tree_shape",
                "learning_rate",
                "min_child_samples",
            ]
        ].head(10).to_string(index=False)
    )
    print(
        f"Best trial {best_trial.number}: mean FPR={best_row['mean_fpr']:.6f}; "
        f"std={best_row['std_fpr']:.6f}. Confirmation is required next."
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
