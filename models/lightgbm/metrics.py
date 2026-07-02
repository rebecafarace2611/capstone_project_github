from __future__ import annotations

from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics_at_threshold(
    target: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int8)
    probability = np.asarray(score, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability):
        raise ValueError("target and score must be one-dimensional and equally sized.")
    if not np.isfinite(probability).all():
        raise ValueError("score contains non-finite values.")

    predicted = probability >= float(threshold)
    positive = y == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & negative))

    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    specificity = 1.0 - fpr
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "predicted_positive": tp + fp,
        "predicted_positive_rate": float((tp + fp) / len(y)),
        "false_positives_per_10000_legitimate": float(fpr * 10000.0),
    }


def threshold_for_minimum_recall(
    target: np.ndarray,
    score: np.ndarray,
    minimum_recall: float,
) -> dict[str, float | int]:
    """Return the highest score threshold that attains the requested recall.

    FPR is monotone as the threshold decreases, so this is also the threshold
    with the lowest achievable FPR subject to the recall constraint. Tied
    scores can cause achieved recall to be slightly above the requested value.
    """
    if not 0.0 < minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be in (0, 1].")
    y = np.asarray(target, dtype=np.int8)
    probability = np.asarray(score, dtype=np.float64)
    positive_scores = probability[y == 1]
    if len(positive_scores) == 0:
        raise ValueError("At least one positive target is required.")

    required_true_positives = ceil(minimum_recall * len(positive_scores))
    descending = np.sort(positive_scores)[::-1]
    threshold = float(descending[required_true_positives - 1])
    result = binary_metrics_at_threshold(y, probability, threshold)
    result["requested_minimum_recall"] = float(minimum_recall)
    return result


def discrimination_metrics(
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(target, dtype=np.int8)
    probability = np.asarray(score, dtype=np.float64)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc_average_precision": float(
            average_precision_score(y, probability)
        ),
    }


def operating_points(
    target: np.ndarray,
    score: np.ndarray,
    recall_targets: list[float],
) -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    default_point = binary_metrics_at_threshold(target, score, 0.5)
    default_point["rule"] = "fixed_0.5"
    default_point["requested_minimum_recall"] = float("nan")
    points.append(default_point)

    for recall_target in recall_targets:
        point = threshold_for_minimum_recall(target, score, recall_target)
        point["rule"] = f"minimum_recall_{recall_target:.2f}"
        points.append(point)
    return points
