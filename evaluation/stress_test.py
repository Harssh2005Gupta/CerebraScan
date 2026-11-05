"""
Out-of-Distribution (OOD) and Image Corruption Stress-Testing Suite.
Demonstrates the failure modes of uncalibrated Softmax classifiers when exposed
to non-brain inputs, sensor artifacts, and corrupted acquisitions.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_stress_suite(img_size: tuple = (224, 224)) -> Dict[str, np.ndarray]:
    """
    Generate synthetic test patterns representing corrupted or non-brain acquisitions:
    1. Zero-Signal Blank (all black)
    2. Saturated Scanner (all white)
    3. Pure Gaussian Sensor Noise
    4. Salt-and-Pepper Scanner Dropout
    5. Inverted Intensity Gradient
    6. Non-Brain Synthetic Texture (synthetic chest / bone density pattern)

    Returns:
        Dict mapping scenario name to preprocessed numpy array of shape (1, H, W, 3) in [0.0, 1.0].
    """
    h, w = img_size
    suite = {}

    # 1. Zero-Signal Blank (e.g. empty scan or disconnected RF coil)
    suite["Zero_Signal_Blank"] = np.zeros((1, h, w, 3), dtype=np.float32)

    # 2. Fully Saturated Scan (e.g. RF transmitter artifact / amplifier clipping)
    suite["Saturated_White_Scan"] = np.ones((1, h, w, 3), dtype=np.float32)

    # 3. Pure Gaussian Sensor Noise (SNR = 0)
    np.random.seed(42)
    noise = np.random.normal(loc=0.5, scale=0.2, size=(1, h, w, 3)).astype(np.float32)
    suite["Pure_Gaussian_Noise"] = np.clip(noise, 0.0, 1.0)

    # 4. Salt-and-Pepper Spike Noise
    sp = np.zeros((1, h, w, 3), dtype=np.float32)
    mask_white = np.random.rand(1, h, w, 3) < 0.05
    mask_black = np.random.rand(1, h, w, 3) < 0.05
    sp[mask_white] = 1.0
    suite["Salt_Pepper_Dropout"] = sp

    # 5. Synthetic Non-Brain Radiographic Pattern (e.g. concentric bone rings / lung field)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    ring_pattern = (np.sin(r / 8.0) * 0.5 + 0.5).astype(np.float32)
    ring_tensor = np.stack([ring_pattern] * 3, axis=-1)[np.newaxis, ...]
    suite["NonBrain_Geometric_Pattern"] = ring_tensor

    return suite


def evaluate_stress_suite(
    model: Any,
    class_names: List[str] = ["glioma", "meningioma", "notumor", "pituitary"],
) -> List[Dict[str, any]]:
    """
    Evaluate the model on corrupted and OOD inputs to measure vulnerability
    to overconfident false predictions.

    Args:
        model: Loaded Keras model.
        class_names: List of class names.

    Returns:
        List of audit findings for each stress pattern.
    """
    suite = generate_synthetic_stress_suite()
    findings = []

    for name, tensor in suite.items():
        preds = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        predicted_class = class_names[pred_idx]

        # In an ideal medical system, OOD inputs would trigger a rejection mechanism.
        # Standard Softmax outputs an arbitrary class, often with misleading high confidence.
        findings.append({
            "test_scenario": name,
            "predicted_class": predicted_class,
            "confidence_score": round(conf, 4),
            "class_probabilities": {
                class_names[i]: round(float(preds[i]), 4) for i in range(len(class_names))
            },
            "failure_severity": "HIGH" if conf > 0.70 else "MODERATE",
            "vulnerability_note": (
                f"Model attributed corrupted input to '{predicted_class}' with {conf*100:.1f}% confidence. "
                "Demonstrates that Softmax confidence thresholds cannot serve as a reliable out-of-distribution detector."
            ),
        })

    return findings
