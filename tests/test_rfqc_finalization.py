import csv
import json
import tempfile
import unittest
from pathlib import Path

from models.rfqc.prepare_final_run import build_configuration
from models.rfqc.summarize_results import write_summary


class RFQCFinalizationTests(unittest.TestCase):
    def test_locked_configuration_uses_candidate_three_qstar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ranking = root / "cv_ranking.csv"
            context = root / "run_context.json"
            baseline = root / "baseline_summary.csv"
            ranking.write_text(
                "rank,candidate,ntree,mtry,nodesize,nsplit,splitrule,"
                "threshold_rule,mean_validation_gmean,mean_specificity\n"
                "2,3,500,24,20,10,gini,q_star_prevalence,"
                "0.786249717333699,0.754067030030368\n",
                encoding="utf-8",
            )
            context.write_text(
                json.dumps(
                    {
                        "random_state": 42,
                        "input_sha256": {
                            "train": "train-hash",
                            "test": "test-hash",
                            "approved_features": "features-hash",
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline.write_text("ntree\n3000\n", encoding="utf-8")

            configuration = build_configuration(
                ranking,
                context,
                baseline,
                final_trees=3000,
                training_rows=444074,
                training_fraud=1860,
            )

        self.assertEqual(configuration["status"], "locked_for_final_test")
        self.assertEqual(configuration["candidate"], 3)
        self.assertEqual(configuration["mtry"], 24)
        self.assertEqual(configuration["nodesize"], 20)
        self.assertEqual(configuration["threshold_rule"], "q_star_prevalence")
        self.assertEqual(configuration["expected_test_rows"], 111020)
        self.assertEqual(configuration["expected_test_fraud"], 465)
        self.assertAlmostEqual(
            configuration["locked_threshold"],
            1860 / 444074,
        )

    def test_summary_contains_baseline_cv_and_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.csv"
            ranking = root / "ranking.csv"
            metrics = root / "metrics.json"
            output = root / "summary.csv"
            baseline.write_text(
                "ntree,gmean_optimized_gmean,gmean_optimized_sensitivity,"
                "gmean_optimized_specificity,gmean_optimized_fpr,roc_auc,"
                "pr_auc,q_star_gmean,q_star_sensitivity,q_star_specificity,"
                "q_star_fpr\n"
                "3000,0.76,0.77,0.75,0.25,0.84,0.025,"
                "0.75,0.70,0.80,0.20\n",
                encoding="utf-8",
            )
            ranking.write_text(
                "candidate,ntree,mtry,nodesize,nsplit,splitrule,"
                "threshold_rule,mean_threshold,mean_validation_gmean,"
                "mean_sensitivity,mean_specificity,mean_precision,mean_f1,"
                "mean_p4,mean_roc_auc,mean_pr_auc\n"
                "3,500,24,20,10,gini,gmean_optimized,0.0036,0.787,"
                "0.844,0.734,0.0132,0.0260,0.0082,0.869,0.0283\n"
                "3,500,24,20,10,gini,q_star_prevalence,0.0042,0.786,"
                "0.820,0.754,0.0138,0.0272,0.0085,0.869,0.0283\n",
                encoding="utf-8",
            )
            metrics.write_text(
                json.dumps(
                    {
                        "locked_candidate": 3,
                        "threshold_rule": "q_star_prevalence",
                        "selected_threshold": 0.0042,
                        "parameters": {
                            "ntree": 3000,
                            "mtry": 24,
                            "nodesize": 20,
                            "nsplit": 10,
                            "splitrule": "gini",
                        },
                        "primary_test_metrics": {
                            "gmean": 0.78,
                            "sensitivity": 0.81,
                            "specificity": 0.75,
                            "fpr": 0.25,
                            "precision": 0.014,
                            "f1": 0.027,
                            "p4": 0.008,
                            "roc_auc": 0.87,
                            "pr_auc": 0.03,
                        },
                    }
                ),
                encoding="utf-8",
            )

            write_summary(baseline, ranking, metrics, output)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[-1]["stage"], "final_test")
        self.assertEqual(rows[-1]["threshold_rule"], "q_star_prevalence")


if __name__ == "__main__":
    unittest.main()
