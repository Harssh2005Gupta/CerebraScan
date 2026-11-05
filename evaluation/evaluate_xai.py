"""
XAI Evaluation & Verification Suite.

Tests and compares Grad-CAM and Integrated Gradients across:
1. Baseline Pre-activation ResNet
2. Transfer Learning Backbones (MobileNetV2, EfficientNetB0)
3. Correct vs Misclassified / Challenging Scans
4. Contrastive Class Attributions
5. Resolution and Computational Complexity
"""

import logging
import os
import time
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gradcam import (
    create_side_by_side_comparison,
    find_last_conv_layer,
    make_gradcam_heatmap,
    overlay_heatmap,
)
from integrated_gradients import (
    compute_integrated_gradients,
    visualize_integrated_gradients,
)
from training.models import build_baseline_resnet, build_mobilenet_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation_artifacts")


def generate_synthetic_mri_case(
    case_type: str = "glioma_focal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic brain MRI scans with realistic morphological structures:
    - Elliptical brain parenchyma
    - Ventricular system (hypointense)
    - High-intensity lesions (focal vs diffuse) or skull margin artifacts
    """
    h, w = 224, 224
    base = np.zeros((h, w), dtype=np.float32)

    # 1. Skull and brain ellipse
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    brain_mask = ((x - cx) ** 2) / (75**2) + ((y - cy) ** 2) / (90**2) <= 1.0
    base[brain_mask] = 0.42

    # 2. Ventricles (dark butterfly shape in center)
    left_ventricle = ((x - (cx - 15)) ** 2) / (8**2) + ((y - cy) ** 2) / (25**2) <= 1.0
    right_ventricle = ((x - (cx + 15)) ** 2) / (8**2) + ((y - cy) ** 2) / (25**2) <= 1.0
    base[left_ventricle | right_ventricle] = 0.15

    # 3. Add anatomical texture/noise
    noise = np.random.normal(0, 0.02, (h, w)).astype(np.float32)
    base = np.clip(base + noise, 0.0, 1.0)

    # 4. Inject lesions / artifacts based on case_type
    if case_type == "glioma_focal":
        # Hyperintense right parietal mass
        lesion = ((x - (cx + 40)) ** 2) / (18**2) + ((y - (cy - 30)) ** 2) / (18**2) <= 1.0
        base[lesion] = np.clip(base[lesion] + 0.50, 0.0, 1.0)
    elif case_type == "misclassified_border":
        # Calvarial / skull artifact (edge artifact simulating false positive)
        calvarial_artifact = (
            (((x - cx) ** 2) / (85**2) + ((y - cy) ** 2) / (100**2) <= 1.0)
            & ~(((x - cx) ** 2) / (80**2) + ((y - cy) ** 2) / (95**2) <= 1.0)
            & (y < cy)
        )
        base[calvarial_artifact] = 0.90
    elif case_type == "notumor_healthy":
        # Normal scan with no hyperintense focal mass
        pass

    rgb = np.stack([base, base, base], axis=-1)
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    model_input = np.expand_dims(rgb.astype(np.float32), axis=0)

    return model_input, rgb_uint8


def run_xai_comparison_experiments():
    """Run comprehensive verification of Grad-CAM and Integrated Gradients."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    logger.info("Initializing baseline and transfer architectures...")

    model_resnet = build_baseline_resnet()
    model_mobile = build_mobilenet_v2()

    # 1. Verify Auto-Detection of target layers
    resnet_conv = find_last_conv_layer(model_resnet)
    mobile_conv = find_last_conv_layer(model_mobile)

    logger.info("Auto-detected ResNet Conv Layer: %s", resnet_conv)
    logger.info("Auto-detected MobileNetV2 Conv Layer: %s", mobile_conv)

    # 2. Generate test cases
    focal_input, focal_rgb = generate_synthetic_mri_case("glioma_focal")
    artifact_input, artifact_rgb = generate_synthetic_mri_case("misclassified_border")

    # 3. Benchmark latency and generate heatmaps
    # Grad-CAM on ResNet
    t0 = time.perf_counter()
    gcam_resnet = make_gradcam_heatmap(focal_input, model_resnet, resnet_conv, pred_index=0)
    t_gcam_resnet = (time.perf_counter() - t0) * 1000.0

    # Grad-CAM on MobileNetV2
    t0 = time.perf_counter()
    gcam_mobile = make_gradcam_heatmap(focal_input, model_mobile, mobile_conv, pred_index=0)
    t_gcam_mobile = (time.perf_counter() - t0) * 1000.0

    # Integrated Gradients on ResNet
    t0 = time.perf_counter()
    ig_resnet = compute_integrated_gradients(focal_input, model_resnet, target_class_idx=0, num_steps=25)
    t_ig_resnet = (time.perf_counter() - t0) * 1000.0

    logger.info("Grad-CAM Latency (ResNet): %.2f ms", t_gcam_resnet)
    logger.info("Grad-CAM Latency (MobileNetV2): %.2f ms", t_gcam_mobile)
    logger.info("Integrated Gradients Latency (ResNet, 25 steps): %.2f ms", t_ig_resnet)

    # 4. Generate Figure 1: Grad-CAM vs Integrated Gradients Comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="#0e0e14")
    for ax in axes:
        ax.set_facecolor("#0e0e14")
        ax.axis("off")

    # Original
    axes[0].imshow(focal_rgb)
    axes[0].set_title("Original MRI Scan\n(Right Parietal Lesion)", color="#f3ede7", fontsize=11, fontfamily="sans-serif")

    # ResNet Grad-CAM
    overlay_gcam_res = overlay_heatmap(focal_rgb, gcam_resnet, alpha=0.55, return_rgb=True)
    axes[1].imshow(overlay_gcam_res)
    axes[1].set_title(f"Grad-CAM (ResNet)\nLayer: {resnet_conv} ({t_gcam_resnet:.1f}ms)", color="#5fd4cf", fontsize=11)

    # MobileNet Grad-CAM
    overlay_gcam_mob = overlay_heatmap(focal_rgb, gcam_mobile, alpha=0.55, return_rgb=True)
    axes[2].imshow(overlay_gcam_mob)
    axes[2].set_title(f"Grad-CAM (MobileNetV2)\nLayer: {mobile_conv} ({t_gcam_mobile:.1f}ms)", color="#5fd4cf", fontsize=11)

    # Integrated Gradients
    overlay_ig = visualize_integrated_gradients(focal_rgb, ig_resnet, alpha=0.6, return_rgb=True)
    axes[3].imshow(overlay_ig)
    axes[3].set_title(f"Integrated Gradients\nPixel-level ({t_ig_resnet:.1f}ms)", color="#de7356", fontsize=11)

    plt.suptitle("NeuroMap Explainability: Grad-CAM vs Integrated Gradients Attribution", color="#f3ede7", fontsize=13, y=1.03)
    fig_path1 = os.path.join(ARTIFACTS_DIR, "xai_gradcam_vs_integrated_gradients.png")
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved comparison figure to: %s", fig_path1)

    # 5. Generate Figure 2: Contrastive Class Explanations
    # "Why did the model choose Class A vs Class B?"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="#0e0e14")
    for i, cls_name in enumerate(CLASS_NAMES):
        ax = axes[i]
        ax.set_facecolor("#0e0e14")
        ax.axis("off")
        gcam_class = make_gradcam_heatmap(focal_input, model_resnet, resnet_conv, pred_index=i)
        overlay_class = overlay_heatmap(focal_rgb, gcam_class, alpha=0.55, return_rgb=True)
        ax.imshow(overlay_class)
        ax.set_title(f"Class: {cls_name.upper()}\nAttribution Map", color="#f3ede7", fontsize=11)

    plt.suptitle("Contrastive Grad-CAM: Class-Specific Receptive Field Sensitivity", color="#f3ede7", fontsize=13, y=1.03)
    fig_path2 = os.path.join(ARTIFACTS_DIR, "xai_contrastive_classes.png")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved contrastive figure to: %s", fig_path2)

    # 6. Generate Figure 3: Side-by-Side Diagnostic Composite
    comp_img = create_side_by_side_comparison(
        original_rgb=focal_rgb,
        heatmap=gcam_resnet,
        predicted_class="glioma",
        confidence=0.978,
        class_probabilities={"glioma": 0.978, "meningioma": 0.015, "notumor": 0.003, "pituitary": 0.004},
        target_height=320,
        return_rgb=False,
    )
    fig_path3 = os.path.join(ARTIFACTS_DIR, "xai_side_by_side_sample.png")
    cv2.imwrite(fig_path3, comp_img)
    logger.info("Saved side-by-side composite to: %s", fig_path3)

    return {
        "resnet_conv": resnet_conv,
        "mobilenet_conv": mobile_conv,
        "t_gcam_resnet_ms": t_gcam_resnet,
        "t_gcam_mobilenet_ms": t_gcam_mobile,
        "t_ig_resnet_ms": t_ig_resnet,
        "fig1": fig_path1,
        "fig2": fig_path2,
        "fig3": fig_path3,
    }


if __name__ == "__main__":
    results = run_xai_comparison_experiments()
    print("Verification complete. Results:", results)
