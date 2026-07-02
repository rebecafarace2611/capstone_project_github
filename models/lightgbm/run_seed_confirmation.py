from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
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
    feature_groups,
    infer_feature_schema,
    load_approved_features,
    load_fold_assignments,
    load_training_frame,
    ordered_fold_vector,
    validate_fold_assignments,
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
)
from models.lightgbm.run_imbalance_screen import undersample_training_indices
from models.lightgbm.run_optuna_tuning import (
    build_trial_model_parameters,
    configuration_fingerprint,
    evaluate_fold,
)


# These offsets are deliberately different from the zero offset used during
# Optuna model selection. They are fixed before confirmation results are seen.
DEFAULT_SEED_OFFSETS = [100_000, 200_000, 300_000, 400_000, 500_000]
LOCKED_INPUT_SHA256 = {
    "train": "823ec10fbdb9f9ab5cb05144ddda5219b8cdeb41354b02b6e7fd5738cc0da2e3",
    "approved_features": "df92a415d9261858c2d6b500bc1bf0d07dfb0468969643cdb19221102c376fd0",
    "fold_assignments": "5a92aa39bd0d753f68a1c941a04f85ae130d2490b65fc4c8b31389bfe2116d5d",
}


@dataclass(frozen=True)
class ConfirmationCandidate:
    candidate_id: str
    label: str
    source: str
    source_trial: int
    tuning_mean_fpr: float
    hyperparameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "source": self.source,
            "source_trial": self.source_trial,
            "tuning_mean_fpr": self.tuning_mean_fpr,
            "hyperparameters": self.hyperparameters,
        }


def confirmation_candidates() -> list[ConfirmationCandidate]:
    return [
        ConfirmationCandidate(
            candidate_id="A",
            label="local_trial_34_rus_1_to_7",
            source="local_refinement",
            source_trial=34,
            tuning_mean_fpr=0.18657478895360402,
            hyperparameters={
                "rus_ratio": 7,
                "learning_rate": 0.09885115038768211,
                "tree_shape": "d3_l7",
                "min_child_samples": 50,
                "feature_fraction": 0.8,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
                "min_split_gain": 0.1,
                "cat_smooth": 100.0,
                "cat_l2": 5.0,
            },
        ),
        ConfirmationCandidate(
            candidate_id="B",
            label="local_trial_8_rus_1_to_10",
            source="local_refinement",
            source_trial=8,
            tuning_mean_fpr=0.18884743732003445,
            hyperparameters={
                "rus_ratio": 10,
                "learning_rate": 0.08519129283094373,
                "tree_shape": "d3_l7",
                "min_child_samples": 100,
                "feature_fraction": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "min_split_gain": 0.2,
                "cat_smooth": 100.0,
                "cat_l2": 5.0,
            },
        ),
        ConfirmationCandidate(
            candidate_id="C",
            label="broad_trial_43_rus_1_to_5",
            source="broad_tuning",
            source_trial=43,
            tuning_mean_fpr=0.190593244792907,
            hyperparameters={
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
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Confirm three fixed LightGBM/RUS candidates across five new "
            "undersampling seeds and the locked grouped folds."
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
        help="Locked grouped folds from the completed baseline and tuning runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lightgbm" / "seed_confirmation",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--seed-offsets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEED_OFFSETS),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--primary-recall", type=float, default=0.80)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.002,
        help="Absolute mean FPR gap treated as practically close.",
    )
    return parser.parse_args()


def validate_confirmation_args(args: argparse.Namespace) -> None:
    if args.n_splits != 5:
        raise ValueError("Confirmation must reuse the locked five-fold protocol.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.n_estimators < 1 or args.early_stopping_rounds < 1:
        raise ValueError("Estimator and early-stopping counts must be positive.")
    if not 0.0 < args.primary_recall <= 1.0:
        raise ValueError("--primary-recall must be in (0, 1].")
    if args.equivalence_margin < 0.0:
        raise ValueError("--equivalence-margin must be non-negative.")
    if len(args.seed_offsets) != 5:
        raise ValueError("Exactly five pre-specified RUS seed offsets are required.")
    if len(set(args.seed_offsets)) != len(args.seed_offsets):
        raise ValueError("RUS seed offsets must be unique.")
    if any(offset <= 0 for offset in args.seed_offsets):
        raise ValueError(
            "RUS seed offsets must be positive so the tuning seed is not reused."
        )


def rus_sampling_seed(
    *,
    random_state: int,
    seed_offset: int,
    rus_ratio: int,
    fold: int,
) -> int:
    seed = int(random_state + seed_offset + rus_ratio * 1000 + fold)
    if not 0 <= seed < 2**32:
        raise ValueError(f"Derived RUS seed is outside uint32 range: {seed}")
    return seed


def seed_level_summary(
    fold_metrics: pd.DataFrame,
    *,
    n_splits: int,
    primary_recall: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["candidate_id", "candidate_label", "source", "seed_offset"]
    for key, group in fold_metrics.groupby(keys, sort=False, dropna=False):
        ordered = group.sort_values("fold", kind="stable")
        if len(ordered) != n_splits or set(ordered["fold"]) != set(range(n_splits)):
            raise ValueError(f"Incomplete fold evidence for candidate/seed {key}.")
        best_iterations = ordered["best_iteration"].astype(int).tolist()
        rows.append(
            {
                "candidate_id": key[0],
                "candidate_label": key[1],
                "source": key[2],
                "seed_offset": int(key[3]),
                "mean_fpr": float(ordered["fpr"].mean()),
                "std_fpr_across_folds": float(ordered["fpr"].std(ddof=1)),
                "worst_fold_fpr": float(ordered["fpr"].max()),
                "mean_recall": float(ordered["recall"].mean()),
                "mean_precision": float(ordered["precision"].mean()),
                "mean_pr_auc": float(
                    ordered["pr_auc_average_precision"].mean()
                ),
                "mean_roc_auc": float(ordered["roc_auc"].mean()),
                "median_best_iteration": int(np.median(best_iterations)),
                "best_iterations": json.dumps(best_iterations),
                "all_folds_meet_recall": bool(
                    (ordered["recall"] >= primary_recall).all()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_id", "seed_offset"], kind="stable"
    ).reset_index(drop=True)


def candidate_level_summary(
    seed_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    equivalence_margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["candidate_id", "candidate_label", "source"]
    for key, seeds in seed_summary.groupby(keys, sort=False, dropna=False):
        folds = fold_metrics[fold_metrics["candidate_id"] == key[0]]
        all_iterations = folds["best_iteration"].astype(int).tolist()
        rows.append(
            {
                "candidate_id": key[0],
                "candidate_label": key[1],
                "source": key[2],
                "seed_count": int(len(seeds)),
                "fit_count": int(len(folds)),
                "mean_seed_fpr": float(seeds["mean_fpr"].mean()),
                "std_seed_fpr": float(seeds["mean_fpr"].std(ddof=1)),
                "best_seed_mean_fpr": float(seeds["mean_fpr"].min()),
                "worst_seed_mean_fpr": float(seeds["mean_fpr"].max()),
                "mean_fold_fpr": float(folds["fpr"].mean()),
                "std_fpr_across_25_fits_descriptive": float(
                    folds["fpr"].std(ddof=1)
                ),
                "worst_fold_fpr": float(folds["fpr"].max()),
                "mean_recall": float(seeds["mean_recall"].mean()),
                "mean_precision": float(seeds["mean_precision"].mean()),
                "mean_pr_auc": float(seeds["mean_pr_auc"].mean()),
                "mean_roc_auc": float(seeds["mean_roc_auc"].mean()),
                "median_best_iteration_25_fits": int(np.median(all_iterations)),
                "best_iteration_q25": float(np.quantile(all_iterations, 0.25)),
                "best_iteration_q75": float(np.quantile(all_iterations, 0.75)),
                "all_folds_all_seeds_meet_recall": bool(
                    seeds["all_folds_meet_recall"].all()
                ),
            }
        )
    result = pd.DataFrame(rows)
    best_mean = float(result["mean_seed_fpr"].min())
    result["absolute_fpr_gap_from_best"] = result["mean_seed_fpr"] - best_mean
    result["within_equivalence_margin"] = (
        result["absolute_fpr_gap_from_best"] <= equivalence_margin + 1e-15
    )
    result = result.sort_values(
        ["mean_seed_fpr", "std_seed_fpr", "worst_seed_mean_fpr", "mean_pr_auc"],
        ascending=[True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "evidence_rank", np.arange(1, len(result) + 1))
    return result


def paired_seed_comparisons(
    seed_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    candidate_ids = seed_summary["candidate_id"].drop_duplicates().tolist()
    for left, right in itertools.combinations(candidate_ids, 2):
        left_frame = seed_summary[seed_summary["candidate_id"] == left][
            ["seed_offset", "mean_fpr"]
        ].rename(columns={"mean_fpr": "left_mean_fpr"})
        right_frame = seed_summary[seed_summary["candidate_id"] == right][
            ["seed_offset", "mean_fpr"]
        ].rename(columns={"mean_fpr": "right_mean_fpr"})
        paired = left_frame.merge(right_frame, on="seed_offset", validate="one_to_one")
        paired["fpr_difference_left_minus_right"] = (
            paired["left_mean_fpr"] - paired["right_mean_fpr"]
        )
        for row in paired.itertuples(index=False):
            difference = float(row.fpr_difference_left_minus_right)
            detail_rows.append(
                {
                    "left_candidate": left,
                    "right_candidate": right,
                    "seed_offset": int(row.seed_offset),
                    "left_mean_fpr": float(row.left_mean_fpr),
                    "right_mean_fpr": float(row.right_mean_fpr),
                    "fpr_difference_left_minus_right": difference,
                    "winner": left if difference < 0 else right if difference > 0 else "tie",
                }
            )
        differences = paired["fpr_difference_left_minus_right"]
        summary_rows.append(
            {
                "left_candidate": left,
                "right_candidate": right,
                "mean_fpr_difference_left_minus_right": float(differences.mean()),
                "std_difference_across_seeds": float(differences.std(ddof=1)),
                "left_seed_wins": int((differences < 0).sum()),
                "right_seed_wins": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def annotate_selection_evidence(
    candidate_summary: pd.DataFrame,
    paired_detail: pd.DataFrame,
    *,
    equivalence_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = candidate_summary.copy()
    leader = str(result.iloc[0]["candidate_id"])
    leader_wins: list[int] = []
    candidate_wins: list[int] = []
    ties: list[int] = []
    clearly_beaten: list[bool] = []
    for row in result.itertuples(index=False):
        candidate = str(row.candidate_id)
        seed_count = int(row.seed_count)
        if candidate == leader:
            leader_win_count = seed_count
            candidate_win_count = seed_count
            tie_count = 0
            beaten = False
        else:
            pair = paired_detail[
                (
                    (paired_detail["left_candidate"] == leader)
                    & (paired_detail["right_candidate"] == candidate)
                )
                | (
                    (paired_detail["left_candidate"] == candidate)
                    & (paired_detail["right_candidate"] == leader)
                )
            ]
            if len(pair) != seed_count:
                raise ValueError(
                    f"Expected {seed_count} paired seeds for {leader} and {candidate}."
                )
            leader_win_count = int((pair["winner"] == leader).sum())
            candidate_win_count = int((pair["winner"] == candidate).sum())
            tie_count = int((pair["winner"] == "tie").sum())
            beaten = bool(
                row.absolute_fpr_gap_from_best >= equivalence_margin
                and leader_win_count >= 4
            )
        leader_wins.append(leader_win_count)
        candidate_wins.append(candidate_win_count)
        ties.append(tie_count)
        clearly_beaten.append(beaten)

    result["mean_fpr_leader"] = leader
    result["leader_seed_wins_in_pair"] = leader_wins
    result["candidate_seed_wins_in_pair"] = candidate_wins
    result["paired_seed_ties"] = ties
    result["clearly_beaten_by_mean_leader"] = clearly_beaten
    result["selection_shortlist"] = ~result["clearly_beaten_by_mean_leader"]

    challengers = result[result["candidate_id"] != leader]
    leader_is_clear = bool(
        len(challengers) > 0 and challengers["clearly_beaten_by_mean_leader"].all()
    )
    if leader_is_clear:
        provisional = leader
        reason = (
            "Mean-FPR leader is ahead by at least the equivalence margin and "
            "wins at least four of five paired seeds against every challenger."
        )
    else:
        shortlist = result[result["selection_shortlist"]].sort_values(
            [
                "std_seed_fpr",
                "worst_seed_mean_fpr",
                "mean_seed_fpr",
                "mean_pr_auc",
            ],
            ascending=[True, True, True, False],
            kind="stable",
        )
        provisional = str(shortlist.iloc[0]["candidate_id"])
        reason = (
            "No candidate has a clear pre-specified mean-FPR win; provisional "
            "choice comes from seed stability, worst seed, mean FPR, then PR-AUC."
        )
    assessment = {
        "mean_fpr_leader": leader,
        "leader_is_clear": leader_is_clear,
        "provisional_candidate_pending_review": provisional,
        "reason": reason,
        "clear_win_rule": (
            "absolute mean-seed FPR advantage >= equivalence margin and at "
            "least 4 of 5 paired seed wins against each challenger"
        ),
    }
    return result, assessment


def completed_fit_result(
    result_path: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("configuration_fingerprint") != expected_fingerprint:
        raise ValueError(f"Completed fit has a different configuration: {result_path}")
    if payload.get("status") != "complete":
        return None
    return payload


def run(args: argparse.Namespace) -> None:
    validate_confirmation_args(args)
    lgb = import_lightgbm()
    train_path = args.train.resolve()
    approved_path = args.approved_features.resolve()
    folds_path = args.fold_assignments.resolve()
    output_dir = args.output_dir.resolve()
    if not folds_path.exists():
        raise FileNotFoundError(
            "The locked fold assignment file is required for confirmation: "
            f"{folds_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {"status": "running", "started_at_utc": utc_now()},
    )

    input_hashes = {
        "train": sha256_file(train_path),
        "approved_features": sha256_file(approved_path),
        "fold_assignments": sha256_file(folds_path),
    }
    if input_hashes != LOCKED_INPUT_SHA256:
        mismatches = {
            name: {"expected": LOCKED_INPUT_SHA256[name], "actual": actual}
            for name, actual in input_hashes.items()
            if actual != LOCKED_INPUT_SHA256[name]
        }
        raise ValueError(
            "Confirmation inputs do not match the archived tuning inputs: "
            f"{mismatches}"
        )

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
    candidates = confirmation_candidates()
    run_spec = {
        "schema_version": 1,
        "workflow": "lightgbm_rus_seed_confirmation",
        "objective": f"mean fold FPR subject to recall >= {args.primary_recall:.2f}",
        "candidates": [candidate.as_dict() for candidate in candidates],
        "seed_offsets": list(args.seed_offsets),
        "seed_policy": (
            "random_state + seed_offset + rus_ratio * 1000 + fold; positive "
            "offsets exclude the zero-offset samples used during tuning"
        ),
        "analysis_unit": "five-fold mean for each candidate and seed offset",
        "n_splits": args.n_splits,
        "expected_fits": len(candidates) * len(args.seed_offsets) * args.n_splits,
        "base_model_parameters": base_parameters,
        "early_stopping_rounds": args.early_stopping_rounds,
        "primary_recall": args.primary_recall,
        "equivalence_margin": args.equivalence_margin,
        "input_sha256": input_hashes,
        "test_data_used": False,
    }
    spec_path = output_dir / "run_spec.json"
    if spec_path.exists():
        prior = json.loads(spec_path.read_text(encoding="utf-8"))
        if prior != run_spec:
            raise ValueError(
                "The existing confirmation directory uses a different "
                "configuration. Use a new --output-dir."
            )
    write_json(spec_path, run_spec)
    run_fingerprint = configuration_fingerprint(run_spec)

    completed_rows: list[dict[str, Any]] = []
    expected_fits = int(run_spec["expected_fits"])
    for candidate in candidates:
        model_parameters = build_trial_model_parameters(
            base_parameters,
            candidate.hyperparameters,
        )
        ratio = int(candidate.hyperparameters["rus_ratio"])
        for seed_offset in args.seed_offsets:
            for fold in range(args.n_splits):
                sampling_seed = rus_sampling_seed(
                    random_state=args.random_state,
                    seed_offset=seed_offset,
                    rus_ratio=ratio,
                    fold=fold,
                )
                fit_dir = (
                    output_dir
                    / "candidates"
                    / candidate.candidate_id
                    / f"seed_offset_{seed_offset}"
                )
                result_path = fit_dir / f"fold_{fold}.json"
                fit_identity = {
                    "run_fingerprint": run_fingerprint,
                    "candidate_id": candidate.candidate_id,
                    "seed_offset": seed_offset,
                    "fold": fold,
                    "sampling_seed": sampling_seed,
                }
                fit_fingerprint = configuration_fingerprint(fit_identity)
                completed = completed_fit_result(
                    result_path,
                    expected_fingerprint=fit_fingerprint,
                )
                if completed is None:
                    validation_index = np.flatnonzero(fold_vector == fold)
                    full_train_index = np.flatnonzero(fold_vector != fold)
                    train_index = undersample_training_indices(
                        full_train_index,
                        target,
                        legitimate_per_fraud=ratio,
                        random_state=sampling_seed,
                    )
                    print(
                        f"Candidate {candidate.candidate_id}, offset={seed_offset}, "
                        f"fold={fold}: fitting {len(train_index):,} rows."
                    )
                    metrics = evaluate_fold(
                        lgb=lgb,
                        model_parameters=model_parameters,
                        features=features,
                        target=target,
                        train_index=train_index,
                        validation_index=validation_index,
                        categorical_features=schema.categorical_features,
                        early_stopping_rounds=args.early_stopping_rounds,
                        primary_recall=args.primary_recall,
                    )
                    completed = {
                        "status": "complete",
                        "completed_at_utc": utc_now(),
                        "configuration_fingerprint": fit_fingerprint,
                        **fit_identity,
                        "candidate_label": candidate.label,
                        "source": candidate.source,
                        "rus_ratio": ratio,
                        **metrics,
                        "test_data_used": False,
                    }
                    write_json(result_path, completed)
                    del validation_index, full_train_index, train_index
                    gc.collect()
                completed_rows.append(
                    {
                        key: value
                        for key, value in completed.items()
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
                fold_frame = pd.DataFrame(completed_rows).sort_values(
                    ["candidate_id", "seed_offset", "fold"], kind="stable"
                )
                write_csv_atomic(fold_frame, output_dir / "all_fold_metrics.csv")
                write_json(
                    output_dir / "status.json",
                    {
                        "status": "running",
                        "updated_at_utc": utc_now(),
                        "completed_fits": len(completed_rows),
                        "expected_fits": expected_fits,
                    },
                )

    fold_frame = pd.DataFrame(completed_rows).sort_values(
        ["candidate_id", "seed_offset", "fold"], kind="stable"
    ).reset_index(drop=True)
    seed_summary = seed_level_summary(
        fold_frame,
        n_splits=args.n_splits,
        primary_recall=args.primary_recall,
    )
    candidate_summary = candidate_level_summary(
        seed_summary,
        fold_frame,
        equivalence_margin=args.equivalence_margin,
    )
    paired_detail, paired_summary = paired_seed_comparisons(seed_summary)
    candidate_summary, selection_assessment = annotate_selection_evidence(
        candidate_summary,
        paired_detail,
        equivalence_margin=args.equivalence_margin,
    )
    write_csv_atomic(fold_frame, output_dir / "all_fold_metrics.csv")
    write_csv_atomic(seed_summary, output_dir / "seed_summary.csv")
    write_csv_atomic(candidate_summary, output_dir / "candidate_ranking.csv")
    write_csv_atomic(paired_detail, output_dir / "paired_seed_differences.csv")
    write_csv_atomic(
        paired_summary,
        output_dir / "paired_comparison_summary.csv",
    )

    leading = candidate_summary.iloc[0]
    write_json(
        output_dir / "confirmation_summary.json",
        {
            "workflow": "lightgbm_rus_seed_confirmation",
            "status": "confirmation_complete_pending_review",
            "completed_fits": int(len(fold_frame)),
            "analysis_unit": "five-fold candidate mean within each seed offset",
            "leading_candidate_by_mean_fpr": str(leading["candidate_id"]),
            "leading_mean_seed_fpr": float(leading["mean_seed_fpr"]),
            "leading_std_seed_fpr": float(leading["std_seed_fpr"]),
            "equivalence_margin": args.equivalence_margin,
            "selection_assessment": selection_assessment,
            "selection_note": (
                "Review seed-level stability and paired wins before locking a "
                "candidate. Do not treat the 75 fits as independent samples."
            ),
            "next_step": (
                "After candidate review, lock one configuration and a unified "
                "tree-count rule, regenerate comparable fixed-tree OOF scores, "
                "and select one pooled OOF threshold before using the test set."
            ),
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
                "row_level_predictions_saved": False,
                "aggregate_metrics_shareable_after_review": True,
            },
            "test_data_used": False,
        },
    )
    write_json(
        output_dir / "status.json",
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "completed_fits": int(len(fold_frame)),
            "expected_fits": expected_fits,
        },
    )

    print("\nRUS seed confirmation completed successfully.")
    print(f"Results: {output_dir}")
    print(
        candidate_summary[
            [
                "evidence_rank",
                "candidate_id",
                "mean_seed_fpr",
                "std_seed_fpr",
                "worst_seed_mean_fpr",
                "mean_pr_auc",
                "within_equivalence_margin",
            ]
        ].to_string(index=False)
    )


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
