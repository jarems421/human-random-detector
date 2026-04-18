import numpy as np
from sklearn.metrics import brier_score_loss


def build_calibration_report(y_true, y_prob_human, n_bins=5):
    y_true = np.asarray(y_true)
    y_prob_human = np.asarray(y_prob_human)

    if len(y_true) == 0:
        return {
            "brier_score": None,
            "buckets": [],
            "summary": "No rows were available for calibration.",
        }

    buckets = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for index in range(n_bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        include_upper = index == n_bins - 1
        mask = (y_prob_human >= lower) & (
            y_prob_human <= upper if include_upper else y_prob_human < upper
        )

        if not mask.any():
            buckets.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0,
                    "mean_predicted_human": None,
                    "actual_human_rate": None,
                    "calibration_gap": None,
                }
            )
            continue

        mean_predicted = float(y_prob_human[mask].mean())
        actual_rate = float(y_true[mask].mean())
        buckets.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted_human": mean_predicted,
                "actual_human_rate": actual_rate,
                "calibration_gap": float(mean_predicted - actual_rate),
            }
        )

    brier = float(brier_score_loss(y_true, y_prob_human))
    return {
        "brier_score": brier,
        "buckets": buckets,
        "summary": summarize_calibration(buckets),
    }


def summarize_calibration(buckets):
    populated = [bucket for bucket in buckets if bucket["count"] > 0]

    if not populated:
        return "No populated confidence buckets were available."

    weighted_gap = sum(
        abs(bucket["calibration_gap"]) * bucket["count"]
        for bucket in populated
    ) / sum(bucket["count"] for bucket in populated)

    if weighted_gap <= 0.05:
        return "Predicted probabilities are well aligned with observed labels."

    overconfident = sum(bucket["calibration_gap"] for bucket in populated) > 0

    if overconfident:
        return "Predicted human probabilities are somewhat higher than observed human rates."

    return "Predicted human probabilities are somewhat lower than observed human rates."
