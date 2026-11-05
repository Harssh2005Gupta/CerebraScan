"""
Unit Tests for Medical ML Evaluation Modules.
Verifies diagnostic metric computation, ECE calibration error,
glioma hazard auditing, and synthetic stress suite generation.
"""

import numpy as np
import pytest

from evaluation.advanced_metrics import (
    compute_clinical_diagnostic_metrics,
    compute_expected_calibration_error,
    compute_multiclass_roc_and_pr_auc,
)
from evaluation.calibration import TemperatureScaling
from evaluation.error_analysis import analyze_glioma_false_negatives
from evaluation.stress_test import generate_synthetic_stress_suite


def test_clinical_diagnostic_metrics_computation():
    cm = np.array([
        [80, 10, 5, 5],
        [5, 85, 5, 5],
        [2, 3, 90, 5],
        [1, 2, 2, 95],
    ])
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    metrics = compute_clinical_diagnostic_metrics(cm, class_names)

    assert "glioma" in metrics
    assert metrics["glioma"]["true_positives"] == 80
    assert metrics["glioma"]["false_negatives"] == 20
    assert metrics["glioma"]["sensitivity"] == 0.80
    assert 0.0 <= metrics["glioma"]["specificity"] <= 1.0
    assert 0.0 <= metrics["glioma"]["positive_predictive_value"] <= 1.0


def test_expected_calibration_error_perfect():
    y_true = np.array([0, 1, 2, 3])
    y_prob = np.eye(4)  # 100% confidence, 100% accurate
    ece, stats = compute_expected_calibration_error(y_true, y_prob, num_bins=5)
    assert ece == 0.0


def test_expected_calibration_error_imperfect():
    y_true = np.array([0, 1, 0, 1])
    # Overconfident false predictions
    y_prob = np.array([
        [0.9, 0.1],
        [0.9, 0.1],  # False, confidence 0.9
        [0.85, 0.15],
        [0.85, 0.15], # False, confidence 0.85
    ])
    ece, stats = compute_expected_calibration_error(y_true, y_prob, num_bins=5)
    assert ece > 0.0


def test_temperature_scaling_optimization():
    ts = TemperatureScaling()
    logits = np.array([
        [4.0, 1.0],
        [3.0, 0.5],
        [0.5, 3.5],
        [1.0, 4.0],
    ])
    y_true = np.array([0, 0, 1, 1])
    T = ts.fit(logits, y_true, max_iter=20)
    assert T > 0.0
    scaled_probs = ts.predict(logits)
    assert scaled_probs.shape == (4, 2)
    assert np.allclose(np.sum(scaled_probs, axis=1), 1.0)


def test_glioma_false_negative_audit():
    cm = np.array([
        [318, 26, 40, 16],
        [6, 357, 14, 23],
        [1, 0, 397, 2],
        [3, 1, 0, 396],
    ])
    audit = analyze_glioma_false_negatives(cm)
    assert audit["total_glioma_cases"] == 400
    assert audit["true_positives"] == 318
    assert audit["total_false_negatives"] == 82
    assert audit["breakdown"]["missed_as_healthy_notumor"]["count"] == 40
    assert audit["breakdown"]["missed_as_healthy_notumor"]["clinical_hazard_level"] == "CRITICAL"


def test_synthetic_stress_suite_generation():
    suite = generate_synthetic_stress_suite(img_size=(64, 64))
    assert "Zero_Signal_Blank" in suite
    assert "Saturated_White_Scan" in suite
    assert "Pure_Gaussian_Noise" in suite
    assert suite["Zero_Signal_Blank"].shape == (1, 64, 64, 3)
    assert suite["Zero_Signal_Blank"].min() == 0.0
    assert suite["Zero_Signal_Blank"].max() == 0.0
    assert suite["Saturated_White_Scan"].min() == 1.0
