import argparse
import unittest

import numpy as np
import pandas as pd

from models.lightgbm.data import validate_fold_assignments
from models.lightgbm.metrics import (
    binary_metrics_at_threshold,
    operating_points,
    threshold_for_minimum_recall,
)
from models.lightgbm.run_baseline import build_model_parameters, validate_args
from models.lightgbm.run_imbalance_screen import (
    Strategy,
    build_strategies,
    rank_strategies,
    strategy_model_parameters,
    undersample_training_indices,
)
from models.lightgbm.run_optuna_tuning import (
    LOCAL_RUS_RATIOS,
    LOCAL_TREE_SHAPES,
    TREE_SHAPES,
    anchor_hyperparameters,
    build_trial_model_parameters,
    completed_trial_ranking,
    configuration_fingerprint,
    decode_tree_shape,
    search_space,
)
from models.lightgbm.run_seed_confirmation import (
    DEFAULT_SEED_OFFSETS,
    annotate_selection_evidence,
    candidate_level_summary,
    confirmation_candidates,
    paired_seed_comparisons,
    rus_sampling_seed,
    seed_level_summary,
    validate_confirmation_args,
)
from models.lightgbm.run_fixed_oof_lock import (
    LOCKED_CANDIDATE_ID,
    LOCKED_PRIMARY_RECALL,
    LOCKED_TREE_COUNT,
    average_seed_predictions,
    build_locked_model_parameters,
    locked_candidate,
)
from models.lightgbm.run_final_test import (
    EXPECTED_LOCKED_THRESHOLD,
    FINAL_CONFIRMATION_PHRASE,
    align_test_categories,
    validate_final_args,
)
from models.lightgbm.run_final_shap import select_explain_indices, sigmoid


class LightGBMMetricsTests(unittest.TestCase):
    def setUp(self):
        self.target = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
        self.score = np.array([0.90, 0.80, 0.70, 0.20, 0.85, 0.60, 0.10, 0.05])

    def test_threshold_is_highest_value_satisfying_recall(self):
        point = threshold_for_minimum_recall(self.target, self.score, 0.75)

        self.assertAlmostEqual(point["threshold"], 0.70)
        self.assertAlmostEqual(point["recall"], 0.75)
        self.assertEqual(point["tp"], 3)
        self.assertEqual(point["fp"], 1)
        self.assertAlmostEqual(point["fpr"], 0.25)

    def test_fixed_threshold_confusion_counts(self):
        metrics = binary_metrics_at_threshold(self.target, self.score, 0.50)

        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["fp"], 2)
        self.assertEqual(metrics["tn"], 2)

    def test_operating_points_include_default_and_recall_rules(self):
        points = operating_points(self.target, self.score, [0.75, 1.0])

        self.assertEqual(points[0]["rule"], "fixed_0.5")
        self.assertEqual(points[1]["rule"], "minimum_recall_0.75")
        self.assertEqual(points[2]["rule"], "minimum_recall_1.00")


class LightGBMFoldValidationTests(unittest.TestCase):
    def test_identical_groups_may_not_cross_folds(self):
        assignments = pd.DataFrame(
            {
                "row_index": np.arange(6, dtype=np.int64),
                "fold": np.array([0, 1, 0, 1, 0, 1], dtype=np.int16),
            }
        )
        target = pd.Series([1, 1, 0, 0, 0, 0], dtype="int8")
        groups = pd.Series([10, 11, 12, 13, 99, 99], dtype="uint64")

        with self.assertRaisesRegex(ValueError, "cross CV folds"):
            validate_fold_assignments(
                assignments,
                rows=6,
                target=target,
                groups=groups,
                n_splits=2,
            )


class LightGBMConfigurationTests(unittest.TestCase):
    def test_baseline_has_no_imbalance_intervention(self):
        args = argparse.Namespace(
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            n_estimators=5000,
            random_state=42,
            threads=16,
            device="cpu",
        )

        parameters = build_model_parameters(args)

        self.assertIsNone(parameters["class_weight"])
        self.assertNotIn("scale_pos_weight", parameters)
        self.assertNotIn("is_unbalance", parameters)
        self.assertTrue(parameters["deterministic"])
        self.assertTrue(parameters["force_col_wise"])
        self.assertEqual(parameters["metric"], "average_precision")

    def test_invalid_recall_target_is_rejected(self):
        args = argparse.Namespace(
            n_splits=5,
            threads=16,
            learning_rate=0.05,
            n_estimators=5000,
            early_stopping_rounds=100,
            primary_recall=0.0,
            recall_targets=[0.80],
        )

        with self.assertRaisesRegex(ValueError, "Recall targets"):
            validate_args(args)


class LightGBMImbalanceScreenTests(unittest.TestCase):
    def test_default_strategy_grid_is_controlled(self):
        args = argparse.Namespace(
            weights=[2.0, 5.0, 10.0, 20.0],
            undersampling_ratios=[15, 30, 60],
        )

        names = [strategy.name for strategy in build_strategies(args)]

        self.assertEqual(
            names,
            [
                "baseline",
                "weight_2",
                "weight_5",
                "weight_10",
                "weight_20",
                "rus_1_to_15",
                "rus_1_to_30",
                "rus_1_to_60",
            ],
        )

    def test_weighting_does_not_enable_other_imbalance_options(self):
        base = {"class_weight": None, "objective": "binary"}
        strategy = Strategy("weight_5", "scale_pos_weight", 5.0)

        parameters = strategy_model_parameters(base, strategy)

        self.assertEqual(parameters["scale_pos_weight"], 5.0)
        self.assertIsNone(parameters["class_weight"])
        self.assertNotIn("is_unbalance", parameters)

    def test_undersampling_keeps_all_fraud_and_is_deterministic(self):
        target = np.array([1, 1, 1, 1] + [0] * 16, dtype=np.int8)
        train_index = np.arange(20, dtype=np.int64)

        first = undersample_training_indices(
            train_index,
            target,
            legitimate_per_fraud=2,
            random_state=123,
        )
        second = undersample_training_indices(
            train_index,
            target,
            legitimate_per_fraud=2,
            random_state=123,
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 12)
        self.assertEqual(int(target[first].sum()), 4)
        self.assertTrue(set(range(4)).issubset(set(first)))
        self.assertTrue(set(first).issubset(set(train_index)))

    def test_ranking_uses_fpr_and_reports_relative_improvement(self):
        common = {
            "kind": "test",
            "value": None,
            "source": "test",
            "completed_folds": 5,
            "all_folds_meet_recall": True,
            "std_fpr_at_primary_recall": 0.01,
            "best_fold_fpr": 0.10,
            "worst_fold_fpr": 0.25,
            "mean_recall": 0.80,
            "mean_precision": 0.02,
            "mean_pr_auc": 0.04,
            "std_pr_auc": 0.01,
            "mean_roc_auc": 0.87,
            "std_roc_auc": 0.01,
            "best_iterations": [10] * 5,
            "median_best_iteration": 10,
            "mean_fit_seconds": 1.0,
            "mean_training_rows": 100.0,
            "training_row_retention": 1.0,
        }
        baseline = {
            **common,
            "strategy": "baseline",
            "mean_fpr_at_primary_recall": 0.23,
        }
        candidate = {
            **common,
            "strategy": "candidate",
            "mean_fpr_at_primary_recall": 0.18,
        }

        ranking = rank_strategies([baseline, candidate])

        self.assertEqual(ranking.iloc[0]["strategy"], "candidate")
        self.assertGreater(
            ranking.iloc[0]["relative_fpr_reduction_vs_baseline"],
            0.20,
        )
        self.assertTrue(ranking.iloc[0]["meets_10pct_relative_improvement"])


class LightGBMOptunaTuningTests(unittest.TestCase):
    def test_tree_shapes_respect_leaf_depth_constraint(self):
        for tree_shape in TREE_SHAPES:
            max_depth, num_leaves = decode_tree_shape(tree_shape)
            self.assertGreaterEqual(num_leaves, 2)
            if max_depth > 0:
                self.assertLessEqual(num_leaves, 2**max_depth)

    def test_trial_parameters_do_not_enable_class_weighting(self):
        base = {
            "objective": "binary",
            "class_weight": None,
            "scale_pos_weight": 20.0,
            "is_unbalance": True,
        }
        hyperparameters = {
            "rus_ratio": 15,
            "learning_rate": 0.03,
            "tree_shape": "d5_l31",
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_split_gain": 0.05,
            "cat_smooth": 20.0,
            "cat_l2": 10.0,
        }

        parameters = build_trial_model_parameters(base, hyperparameters)

        self.assertEqual(parameters["max_depth"], 5)
        self.assertEqual(parameters["num_leaves"], 31)
        self.assertEqual(parameters["min_child_samples"], 100)
        self.assertNotIn("scale_pos_weight", parameters)
        self.assertNotIn("is_unbalance", parameters)
        self.assertIsNone(parameters["class_weight"])

    def test_anchor_trials_reproduce_screen_defaults(self):
        anchors = anchor_hyperparameters()

        self.assertEqual([anchor["rus_ratio"] for anchor in anchors], [15, 30])
        self.assertTrue(all(anchor["learning_rate"] == 0.05 for anchor in anchors))
        self.assertTrue(
            all(anchor["tree_shape"] == "unlimited_l31" for anchor in anchors)
        )

    def test_local_search_expands_ratio_boundary_and_narrows_structure(self):
        space = search_space("local")

        self.assertEqual(space["rus_ratio"], [1, 2, 3, 5, 7, 10])
        self.assertEqual(space["tree_shape"], ["d3_l7", "d4_l15", "d5_l15"])
        self.assertEqual(space["learning_rate"]["low"], 0.02)
        self.assertEqual(space["learning_rate"]["high"], 0.10)
        self.assertTrue(set(LOCAL_RUS_RATIOS).isdisjoint({15, 20, 30, 60}))
        self.assertTrue(set(LOCAL_TREE_SHAPES).issubset(set(TREE_SHAPES)))

    def test_local_anchor_trials_are_inside_local_space(self):
        space = search_space("local")
        anchors = anchor_hyperparameters("local")

        self.assertEqual(len(anchors), 2)
        for anchor in anchors:
            self.assertIn(anchor["rus_ratio"], space["rus_ratio"])
            self.assertIn(anchor["tree_shape"], space["tree_shape"])
            self.assertIn(
                anchor["min_child_samples"],
                space["min_child_samples"],
            )
            self.assertIn(anchor["feature_fraction"], space["feature_fraction"])

    def test_configuration_fingerprint_is_order_independent(self):
        first = configuration_fingerprint({"a": 1, "b": [2, 3]})
        second = configuration_fingerprint({"b": [2, 3], "a": 1})

        self.assertEqual(first, second)

    def test_completed_trials_are_ranked_by_fpr_then_stability(self):
        class FakeState:
            name = "COMPLETE"

        class FakeTrial:
            def __init__(self, number, value, std):
                self.number = number
                self.value = value
                self.state = FakeState()
                self.params = {"rus_ratio": 15}
                self.user_attrs = {
                    "std_fpr": std,
                    "worst_fold_fpr": value + std,
                    "mean_pr_auc": 0.04,
                    "mean_roc_auc": 0.88,
                    "mean_precision": 0.015,
                    "median_best_iteration": 50,
                    "best_iterations": [50] * 5,
                    "mean_training_rows": 23808,
                }

        class FakeStudy:
            trials = [
                FakeTrial(0, 0.20, 0.03),
                FakeTrial(1, 0.19, 0.04),
                FakeTrial(2, 0.19, 0.02),
            ]

        ranking = completed_trial_ranking(FakeStudy())

        self.assertEqual(ranking["trial"].tolist(), [2, 1, 0])


class LightGBMSeedConfirmationTests(unittest.TestCase):
    def test_candidate_protocol_is_fixed_to_archived_trials(self):
        candidates = confirmation_candidates()

        self.assertEqual([candidate.candidate_id for candidate in candidates], ["A", "B", "C"])
        self.assertEqual(
            [candidate.hyperparameters["rus_ratio"] for candidate in candidates],
            [7, 10, 5],
        )
        self.assertEqual([candidate.source_trial for candidate in candidates], [34, 8, 43])

    def test_confirmation_requires_five_new_unique_seed_offsets(self):
        args = argparse.Namespace(
            n_splits=5,
            threads=16,
            n_estimators=5000,
            early_stopping_rounds=100,
            primary_recall=0.80,
            equivalence_margin=0.002,
            seed_offsets=list(DEFAULT_SEED_OFFSETS),
        )

        validate_confirmation_args(args)
        args.seed_offsets = [0, 1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "tuning seed is not reused"):
            validate_confirmation_args(args)

    def test_sampling_seed_changes_only_with_declared_components(self):
        first = rus_sampling_seed(
            random_state=42,
            seed_offset=100_000,
            rus_ratio=7,
            fold=0,
        )
        second = rus_sampling_seed(
            random_state=42,
            seed_offset=100_000,
            rus_ratio=7,
            fold=1,
        )

        self.assertEqual(first, 107_042)
        self.assertEqual(second, first + 1)

    def test_summaries_keep_seed_means_as_primary_analysis_unit(self):
        rows = []
        fprs = {
            "A": {100: [0.10, 0.20], 200: [0.20, 0.30]},
            "B": {100: [0.15, 0.25], 200: [0.25, 0.35]},
        }
        for candidate, by_seed in fprs.items():
            for seed_offset, values in by_seed.items():
                for fold, fpr in enumerate(values):
                    rows.append(
                        {
                            "candidate_id": candidate,
                            "candidate_label": f"candidate_{candidate}",
                            "source": "test",
                            "seed_offset": seed_offset,
                            "fold": fold,
                            "fpr": fpr,
                            "recall": 0.80,
                            "precision": 0.02,
                            "pr_auc_average_precision": 0.05,
                            "roc_auc": 0.90,
                            "best_iteration": 100 + fold,
                        }
                    )
        fold_frame = pd.DataFrame(rows)

        seeds = seed_level_summary(
            fold_frame,
            n_splits=2,
            primary_recall=0.80,
        )
        candidates = candidate_level_summary(
            seeds,
            fold_frame,
            equivalence_margin=0.002,
        )
        _, paired = paired_seed_comparisons(seeds)

        a = candidates[candidates["candidate_id"] == "A"].iloc[0]
        self.assertAlmostEqual(a["mean_seed_fpr"], 0.20)
        self.assertAlmostEqual(a["std_seed_fpr"], np.sqrt(0.005))
        self.assertEqual(candidates.iloc[0]["candidate_id"], "A")
        self.assertEqual(paired.iloc[0]["left_seed_wins"], 2)

    def test_clear_selection_requires_margin_and_four_of_five_seed_wins(self):
        candidate_summary = pd.DataFrame(
            [
                {
                    "evidence_rank": 1,
                    "candidate_id": "A",
                    "seed_count": 5,
                    "mean_seed_fpr": 0.18,
                    "std_seed_fpr": 0.01,
                    "worst_seed_mean_fpr": 0.20,
                    "mean_pr_auc": 0.05,
                    "absolute_fpr_gap_from_best": 0.0,
                },
                {
                    "evidence_rank": 2,
                    "candidate_id": "B",
                    "seed_count": 5,
                    "mean_seed_fpr": 0.185,
                    "std_seed_fpr": 0.02,
                    "worst_seed_mean_fpr": 0.22,
                    "mean_pr_auc": 0.06,
                    "absolute_fpr_gap_from_best": 0.005,
                },
            ]
        )
        paired_detail = pd.DataFrame(
            {
                "left_candidate": ["A"] * 5,
                "right_candidate": ["B"] * 5,
                "winner": ["A", "A", "A", "A", "B"],
            }
        )

        annotated, assessment = annotate_selection_evidence(
            candidate_summary,
            paired_detail,
            equivalence_margin=0.002,
        )

        b = annotated[annotated["candidate_id"] == "B"].iloc[0]
        self.assertTrue(b["clearly_beaten_by_mean_leader"])
        self.assertTrue(assessment["leader_is_clear"])
        self.assertEqual(assessment["provisional_candidate_pending_review"], "A")


class LightGBMFixedOOFLockTests(unittest.TestCase):
    def test_locked_protocol_uses_confirmed_candidate_and_tree_median(self):
        candidate = locked_candidate()
        parameters = build_locked_model_parameters(threads=24, device="cpu")

        self.assertEqual(LOCKED_CANDIDATE_ID, "C")
        self.assertEqual(LOCKED_TREE_COUNT, 248)
        self.assertEqual(LOCKED_PRIMARY_RECALL, 0.80)
        self.assertEqual(candidate.source_trial, 43)
        self.assertEqual(candidate.hyperparameters["rus_ratio"], 5)
        self.assertEqual(parameters["n_estimators"], 248)
        self.assertEqual(parameters["learning_rate"], 0.026485341497747728)
        self.assertIsNone(parameters["class_weight"])
        self.assertNotIn("scale_pos_weight", parameters)

    def test_seed_predictions_are_averaged_once_per_oof_row(self):
        predictions = pd.DataFrame(
            {
                "row_index": [0, 1, 0, 1],
                "target": [0, 1, 0, 1],
                "fold": [0, 1, 0, 1],
                "seed_offset": [100, 100, 200, 200],
                "score": [0.1, 0.8, 0.3, 0.6],
            }
        )

        averaged = average_seed_predictions(
            predictions,
            rows=2,
            expected_seed_count=2,
        )

        self.assertEqual(averaged["row_index"].tolist(), [0, 1])
        self.assertAlmostEqual(averaged.loc[0, "score"], 0.2)
        self.assertAlmostEqual(averaged.loc[1, "score"], 0.7)

    def test_missing_seed_prediction_is_rejected(self):
        predictions = pd.DataFrame(
            {
                "row_index": [0, 1, 0],
                "target": [0, 1, 0],
                "fold": [0, 1, 0],
                "seed_offset": [100, 100, 200],
                "score": [0.1, 0.8, 0.3],
            }
        )

        with self.assertRaisesRegex(ValueError, "every seed"):
            average_seed_predictions(
                predictions,
                rows=2,
                expected_seed_count=2,
            )


class LightGBMFinalStageTests(unittest.TestCase):
    def test_final_test_requires_exact_one_time_acknowledgement(self):
        args = argparse.Namespace(confirm_final_test="")
        with self.assertRaisesRegex(ValueError, "acknowledgement missing"):
            validate_final_args(args)

        args.confirm_final_test = FINAL_CONFIRMATION_PHRASE
        validate_final_args(args)
        self.assertEqual(EXPECTED_LOCKED_THRESHOLD, 0.23669952663465765)

    def test_unseen_test_categories_are_explicitly_treated_as_missing(self):
        training = pd.DataFrame(
            {"category": pd.Series(["a", "b"], dtype="category")}
        )
        test = pd.DataFrame(
            {"category": pd.Series(["a", "c"], dtype="category")}
        )

        _, aligned, metadata = align_test_categories(
            training,
            test,
            ["category"],
        )

        self.assertEqual(metadata["category"]["unseen_test_rows_treated_as_missing"], 1)
        self.assertEqual(aligned["category"].cat.categories.tolist(), ["a", "b"])
        self.assertTrue(pd.isna(aligned.loc[1, "category"]))

    def test_top_score_shap_selection_is_deterministic(self):
        predictions = pd.DataFrame(
            {
                "row_index": [0, 1, 2, 3],
                "fraud_probability": [0.1, 0.9, 0.8, 0.2],
                "primary_prediction": [0, 1, 1, 0],
            }
        )
        target = np.array([0, 1, 0, 1], dtype=np.int8)

        selected = select_explain_indices(
            selection="top_score",
            explain_size=2,
            predictions=predictions,
            target=target,
            seed=42,
        )

        self.assertEqual(selected.tolist(), [1, 2])

    def test_sigmoid_maps_raw_scores_to_probabilities(self):
        probability = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

        self.assertLess(probability[0], 1e-300)
        self.assertAlmostEqual(probability[1], 0.5)
        self.assertGreater(probability[2], 1.0 - 1e-12)


if __name__ == "__main__":
    unittest.main()
