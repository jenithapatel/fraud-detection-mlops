"""Evaluation metrics shared by training and validation."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision). Primary metric."""
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(y_true, y_score, target_precision: float = 0.9) -> tuple[float, float]:
    """Best recall reachable while precision >= target, and the threshold that achieves it.

    Returns (0.0, 1.0) if the target precision is never reached.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # The last precision/recall point has no threshold, so drop it.
    meets_target = precision[:-1] >= target_precision
    if not meets_target.any():
        return 0.0, 1.0
    idx = int(np.argmax(np.where(meets_target, recall[:-1], -1.0)))
    return float(recall[idx]), float(thresholds[idx])
