"""
Advanced Medical Imaging Metrics:
- Multi-class ROC-AUC (One-vs-Rest, macro and weighted)
- Precision-Recall AUC (PR-AUC / Average Precision)
- Clinical diagnostic metrics: Sensitivity (Recall), Specificity, PPV (Precision), NPV
- Probabilistic quality: Multi-class Brier Score and Expected Calibration Error (ECE)
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_clinical_diagnostic_metrics(
    cm: np.ndarray, class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute clinically relevant diagnostic measures from a confusion matrix:
    - Sensitivity (True Positive Rate / Recall): TP / (TP + FN)
    - Specificity (True Negative Rate): TN / (TN + FP)
    - Positive Predictive Value (PPV / Precision): TP / (TP + FP)
    - Negative Predictive Value (NPV): TN / (TN + FN)
    - Diagnostic Odds Ratio (DOR): (TP * TN) / (FP * FN)

    Args:
        cm: Confusion matrix of shape (num_classes, num_classes).
            Rows = True label, Columns = Predicted label.
        class_names: List of class names.

    Returns:
        Dictionary mapping class name to clinical diagnostic measures.
    """
    total = np.sum(cm)
    results = {}

    for i, name in enumerate(class_names):
        tp = float(cm[i, i])
        fn = float(np.sum(cm[i, :]) - tp)
        fp = float(np.sum(cm[:, i]) - tp)
        tn = float(total - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Diagnostic Odds Ratio
        dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float("inf")

        results[name] = {
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "positive_predictive_value": round(ppv, 4),
            "negative_predictive_value": round(npv, 4),
            "diagnostic_odds_ratio": round(dor, 2) if dor != float("inf") else 9999.0,
        }

    return results


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, Dict[str, list]]:
    """
    Compute Expected Calibration Error (ECE) across predictions:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: 1D array of ground truth class indices (0 to C-1).
        y_prob: 2D array of predicted probabilities of shape (N, C).
        num_bins: Number of confidence bins.

    Returns:
        (ece_score, bin_statistics)
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    bin_stats = {
        "bin_edges": bins.tolist(),
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }

    n_samples = len(y_true)
    for i in range(num_bins):
        b_low, b_high = bins[i], bins[i + 1]
        in_bin = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
        count = int(np.sum(in_bin))
        bin_stats["bin_counts"].append(count)

        if count > 0:
            avg_acc = float(np.mean(accuracies[in_bin]))
            avg_conf = float(np.mean(confidences[in_bin]))
            ece += (count / n_samples) * abs(avg_acc - avg_conf)
            bin_stats["bin_accuracies"].append(round(avg_acc, 4))
            bin_stats["bin_confidences"].append(round(avg_conf, 4))
        else:
            bin_stats["bin_accuracies"].append(0.0)
            bin_stats["bin_confidences"].append(round((b_low + b_high) / 2.0, 4))

    return round(float(ece), 4), bin_stats


def compute_multiclass_roc_and_pr_auc(
    y_true_oh: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Compute multi-class ROC-AUC and PR-AUC using One-vs-Rest (OvR) strategy.

    Args:
        y_true_oh: One-hot encoded true labels (N, C).
        y_prob: Predicted probability distribution (N, C).
        class_names: List of class names.

    Returns:
        Dict of macro/weighted ROC-AUC and PR-AUC plus per-class metrics.
    """
    # Multi-class ROC-AUC
    roc_macro = roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="macro")
    roc_weighted = roc_auc_score(y_true_oh, y_prob, multi_class="ovr", average="weighted")

    # Multi-class PR-AUC (Average Precision)
    pr_macro = average_precision_score(y_true_oh, y_prob, average="macro")
    pr_weighted = average_precision_score(y_true_oh, y_prob, average="weighted")

    # Brier Score across classes
    brier_scores = [
        brier_score_loss(y_true_oh[:, c], y_prob[:, c]) for c in range(len(class_names))
    ]
    mean_brier = float(np.mean(brier_scores))

    res = {
        "roc_auc_ovr_macro": round(float(roc_macro), 4),
        "roc_auc_ovr_weighted": round(float(roc_weighted), 4),
        "pr_auc_macro": round(float(pr_macro), 4),
        "pr_auc_weighted": round(float(pr_weighted), 4),
        "mean_brier_score": round(mean_brier, 4),
        "per_class": {},
    }

    for c, name in enumerate(class_names):
        c_roc = roc_auc_score(y_true_oh[:, c], y_prob[:, c])
        c_pr = average_precision_score(y_true_oh[:, c], y_prob[:, c])
        res["per_class"][name] = {
            "roc_auc": round(float(c_roc), 4),
            "pr_auc": round(float(c_pr), 4),
            "brier_score": round(float(brier_scores[c]), 4),
        }

    return res


def plot_roc_and_pr_curves(
    y_true_oh: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: str,
) -> None:
    """Plot publication-quality One-vs-Rest ROC and PR curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curves
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_prob[:, c])
        auc_val = roc_auc_score(y_true_oh[:, c], y_prob[:, c])
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance level (0.50)")
    axes[0].set_title("One-vs-Rest ROC Curves")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR Curves
    for c, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_prob[:, c])
        ap_val = average_precision_score(y_true_oh[:, c], y_prob[:, c])
        axes[1].plot(recall, precision, lw=2, label=f"{name} (AP = {ap_val:.3f})")

    axes[1].set_title("Precision-Recall (PR) Curves")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved ROC and PR curves to %s", save_path)
