"""
Probability Calibration and Reliability Diagrams for Medical Classification.
Assesses whether confidence scores represent true posterior probabilities
and provides temperature scaling for post-hoc calibration.
"""

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from evaluation.advanced_metrics import compute_expected_calibration_error

logger = logging.getLogger(__name__)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    num_bins: int = 10,
    title: str = "Reliability Diagram (Model Calibration)",
) -> float:
    """
    Plot calibration reliability diagram comparing confidence to empirical accuracy.

    Args:
        y_true: Ground truth labels (N,).
        y_prob: Predicted probability distributions (N, C).
        save_path: Output PNG path.
        num_bins: Number of bins.
        title: Title of plot.

    Returns:
        Expected Calibration Error (ECE) value.
    """
    ece, stats = compute_expected_calibration_error(y_true, y_prob, num_bins=num_bins)

    bins = np.array(stats["bin_edges"])
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    accs = np.array(stats["bin_accuracies"])
    confs = np.array(stats["bin_confidences"])
    counts = np.array(stats["bin_counts"])

    plt.figure(figsize=(7, 6))

    # Perfect calibration reference line
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration (Acc = Conf)")

    # Bar chart for bin accuracy
    bar_width = 1.0 / num_bins
    plt.bar(
        bin_centers,
        accs,
        width=bar_width * 0.85,
        alpha=0.65,
        color="#2b8cbe",
        edgecolor="black",
        label="Observed Accuracy",
    )

    # Gap highlighting
    gaps = np.abs(accs - confs)
    for i in range(num_bins):
        if counts[i] > 0:
            plt.plot(
                [confs[i], confs[i]],
                [accs[i], confs[i]],
                color="red",
                lw=2,
                label="Calibration Gap" if i == 0 else "",
            )

    plt.title(f"{title}\nExpected Calibration Error (ECE) = {ece:.4f}", fontsize=12, fontweight="bold")
    plt.xlabel("Confidence (Predicted Probability)", fontsize=11)
    plt.ylabel("Empirical Accuracy", fontsize=11)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved reliability diagram to %s (ECE=%.4f)", save_path, ece)

    return ece


class TemperatureScaling:
    """
    Post-hoc probability calibration via temperature scaling:
    p_i = softmax(z_i / T)
    Optimizes a single scalar parameter T > 0 on the validation set using NLL.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimize temperature T on validation logits using gradient descent on Cross-Entropy.
        """
        T = 1.0
        n_samples = len(y_true)

        for _ in range(max_iter):
            # Softmax with temperature
            scaled_logits = logits / T
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Cross-entropy loss gradient w.r.t. T
            loss = -np.mean(np.log(probs[np.arange(n_samples), y_true] + 1e-12))
            # d(loss)/dT approximation
            grad = np.mean(
                np.sum(probs * (logits - np.sum(probs * logits, axis=1, keepdims=True)), axis=1)
            ) / (T ** 2)

            T = max(0.05, T - lr * grad)

        self.temperature = float(T)
        logger.info("Optimized temperature T = %.4f", self.temperature)
        return self.temperature

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibrated temperature scaling to logits."""
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
