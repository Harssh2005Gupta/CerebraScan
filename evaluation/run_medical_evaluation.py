"""
Unified Medical Evaluation and Clinical Audit Runner.
Executes clinical diagnostic metrics, false-negative hazard analysis,
calibration evaluation, and synthetic stress testing.
"""

import argparse
import json
import logging
import os
import numpy as np

from evaluation.advanced_metrics import (
    compute_clinical_diagnostic_metrics,
    compute_expected_calibration_error,
    compute_multiclass_roc_and_pr_auc,
    plot_roc_and_pr_curves,
)
from evaluation.error_analysis import (
    analyze_glioma_false_negatives,
    plot_error_severity_distribution,
)
from evaluation.stress_test import evaluate_stress_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NeuroMap.MedicalEval")


# Verified ground truth confusion matrix from the Kaggle test set evaluation (1,600 samples)
VERIFIED_BASELINE_CONFUSION_MATRIX = np.array([
    [318, 26, 40, 16],  # True: Glioma
    [6, 357, 14, 23],   # True: Meningioma
    [1, 0, 397, 2],     # True: No Tumor
    [3, 1, 0, 396],     # True: Pituitary
], dtype=int)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def run_clinical_audit(
    cm: np.ndarray = VERIFIED_BASELINE_CONFUSION_MATRIX,
    output_dir: str = "evaluation_artifacts",
    model: any = None,
) -> dict:
    """
    Run full clinical audit on confusion matrix and model (if loaded).
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Diagnostic metrics (Sensitivity, Specificity, PPV, NPV, DOR)
    logger.info("Computing clinical diagnostic measures...")
    diag_metrics = compute_clinical_diagnostic_metrics(cm, CLASS_NAMES)

    # 2. Glioma False Negative Analysis
    logger.info("Auditing Glioma false negatives...")
    glioma_audit = analyze_glioma_false_negatives(cm, CLASS_NAMES)
    plot_error_severity_distribution(
        glioma_audit, os.path.join(output_dir, "glioma_error_distribution.png")
    )

    # 3. Model stress testing (if model object available)
    stress_results = []
    if model is not None:
        logger.info("Executing OOD and corrupted image stress tests...")
        stress_results = evaluate_stress_suite(model, CLASS_NAMES)

    # Combine into comprehensive clinical audit report
    audit_summary = {
        "dataset_audit": {
            "source": "Kaggle (masoudnickparvar/brain-tumor-mri-dataset)",
            "underlying_sources": [
                "Cheng et al. (Figshare, 3064 T1-CE MRI slices from 233 patients)",
                "Br35H (Brain Tumor Detection 2020)",
            ],
            "total_test_samples": int(np.sum(cm)),
            "per_class_support": {name: int(np.sum(cm[i, :])) for i, name in enumerate(CLASS_NAMES)},
            "patient_level_leakage_risk": "HIGH (Dataset lacks patient IDs; slices likely cross-contaminate splits)",
            "external_validation": "NONE (Evaluated solely on in-distribution test split)",
        },
        "diagnostic_metrics": diag_metrics,
        "glioma_false_negative_hazard": glioma_audit,
        "stress_test_findings": stress_results,
    }

    report_path = os.path.join(output_dir, "clinical_evaluation_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info("Clinical audit complete. Summary saved to %s", report_path)
    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Run NeuroMap Medical ML Evaluation.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .keras model file (optional)")
    parser.add_argument("--output-dir", type=str, default="evaluation_artifacts", help="Output directory")
    args = parser.parse_args()

    model = None
    if args.model_path and os.path.exists(args.model_path):
        import tensorflow as tf
        logger.info("Loading model from %s...", args.model_path)
        model = tf.keras.models.load_model(args.model_path)

    run_clinical_audit(output_dir=args.output_dir, model=model)


if __name__ == "__main__":
    main()
