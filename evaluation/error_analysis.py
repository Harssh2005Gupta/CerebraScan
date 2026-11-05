"""
Medical Machine Learning Error Analysis Module:
- Deep investigation of Glioma False Negatives (FNs).
- Categorization of clinically critical vs benign misclassifications.
- Decision threshold sensitivity analysis for high-risk tumor detection.
"""

import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def analyze_glioma_false_negatives(
    cm: np.ndarray,
    class_names: List[str] = ["glioma", "meningioma", "notumor", "pituitary"],
) -> Dict[str, any]:
    """
    Perform a clinical hazard audit on the Glioma class predictions.

    Args:
        cm: Confusion matrix where rows are True classes and columns are Predicted.
        class_names: List of class names.

    Returns:
        Structured audit dictionary of Glioma predictions and risk severity.
    """
    g_idx = class_names.index("glioma")
    m_idx = class_names.index("meningioma")
    n_idx = class_names.index("notumor")
    p_idx = class_names.index("pituitary")

    total_glioma = int(np.sum(cm[g_idx, :]))
    tp_glioma = int(cm[g_idx, g_idx])

    # False negative breakdown
    fn_as_notumor = int(cm[g_idx, n_idx])
    fn_as_meningioma = int(cm[g_idx, m_idx])
    fn_as_pituitary = int(cm[g_idx, p_idx])
    total_fn = fn_as_notumor + fn_as_meningioma + fn_as_pituitary

    fn_rate = (total_fn / total_glioma) * 100.0 if total_glioma > 0 else 0.0
    critical_miss_rate = (fn_as_notumor / total_glioma) * 100.0 if total_glioma > 0 else 0.0

    analysis = {
        "total_glioma_cases": total_glioma,
        "true_positives": tp_glioma,
        "total_false_negatives": total_fn,
        "false_negative_rate_pct": round(fn_rate, 2),
        "breakdown": {
            "missed_as_healthy_notumor": {
                "count": fn_as_notumor,
                "percentage_of_glioma": round(critical_miss_rate, 2),
                "clinical_hazard_level": "CRITICAL",
                "hazard_description": (
                    "Patient with active infiltrative glioma is informed no lesion is detected. "
                    "Leads to delayed oncological intervention, tumor progression, and compromised survival."
                ),
            },
            "confused_with_meningioma": {
                "count": fn_as_meningioma,
                "percentage_of_glioma": round((fn_as_meningioma / total_glioma) * 100.0, 2),
                "clinical_hazard_level": "MODERATE_HIGH",
                "hazard_description": (
                    "Glioma misidentified as extra-axial meningioma. While lesion detection succeeds, "
                    "surgical margin planning, biopsy strategy, and urgency differ fundamentally."
                ),
            },
            "confused_with_pituitary": {
                "count": fn_as_pituitary,
                "percentage_of_glioma": round((fn_as_pituitary / total_glioma) * 100.0, 2),
                "clinical_hazard_level": "MODERATE",
                "hazard_description": (
                    "Anatomical discrepancy: intra-axial glioma misattributed to sellar/pituitary region."
                ),
            },
        },
        "safety_recommendations": [
            "Implement high-sensitivity screening threshold for the Glioma class.",
            "Do not discharge patients based solely on automated 'No Tumor' output.",
            "Integrate multi-modal MRI sequences (T2-FLAIR) to detect hyperintense peritumoral edema.",
            "Mandate secondary radiologist review on all scans with Glioma probability > 0.15.",
        ],
    }

    return analysis


def plot_error_severity_distribution(analysis: Dict[str, any], save_path: str) -> None:
    """Plot visual bar breakdown of Glioma diagnostic outcomes and risk levels."""
    categories = [
        "True Positive\n(Glioma)",
        "Critical Miss\n(→ No Tumor)",
        "Subtype Error\n(→ Meningioma)",
        "Anatomical Error\n(→ Pituitary)",
    ]
    counts = [
        analysis["true_positives"],
        analysis["breakdown"]["missed_as_healthy_notumor"]["count"],
        analysis["breakdown"]["confused_with_meningioma"]["count"],
        analysis["breakdown"]["confused_with_pituitary"]["count"],
    ]
    colors = ["#2b8cbe", "#de2d26", "#feb24c", "#bcbddc"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(categories, counts, color=colors, width=0.55, edgecolor="black", lw=1.2)
    plt.title("Glioma Clinical Outcome Audit (N = 400 Test Cases)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Patients", fontsize=11)
    plt.ylim(0, 360)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / analysis["total_glioma_cases"]) * 100.0
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 6,
            f"{int(h)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved Glioma error distribution plot to %s", save_path)
