"""
Integrated Gradients — Axiomatic Pixel-Level Attribution.

Implements path-integral feature attributions for deep neural networks.

Reference:
    Sundararajan, Taly, & Yan, "Axiomatic Attribution for Deep Networks",
    International Conference on Machine Learning (ICML), 2017.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def interpolate_images(
    baseline: tf.Tensor,
    target: tf.Tensor,
    num_steps: int = 40,
) -> tf.Tensor:
    """
    Generate linear path interpolation between baseline and target image.

    Args:
        baseline: Baseline image tensor of shape (1, H, W, 3).
        target: Target image tensor of shape (1, H, W, 3).
        num_steps: Number of integration steps along the path.

    Returns:
        Interpolated image batch of shape (num_steps + 1, H, W, 3).
    """
    alphas = tf.linspace(0.0, 1.0, num_steps + 1)
    alphas = tf.cast(alphas, dtype=target.dtype)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    delta = target - baseline
    interpolated = baseline + alphas_x * delta
    return interpolated


def compute_integrated_gradients(
    img: np.ndarray,
    model: tf.keras.Model,
    target_class_idx: Optional[int] = None,
    baseline: Optional[np.ndarray] = None,
    num_steps: int = 30,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute Integrated Gradients attributions for an input scan.

    Args:
        img: Preprocessed image array of shape (1, H, W, 3) or (H, W, 3), float32 [0, 1].
        model: Compiled or loaded Keras model.
        target_class_idx: Class index to compute attributions for. If None, top class is used.
        baseline: Baseline image of shape matching `img`. Default is all-zero (black) image.
        num_steps: Integration steps for Riemann approximation (default: 30).
        batch_size: Mini-batch size for processing interpolation steps to prevent OOM.

    Returns:
        2-D normalized attribution heatmap of shape (H, W), values in [0, 1].
    """
    # 1. Format input tensor
    if not isinstance(img, (np.ndarray, tf.Tensor)):
        img = np.array(img)

    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    if img.ndim != 4 or img.shape[-1] != 3:
        raise ValueError(f"Expected input array of shape (1, H, W, 3) or (H, W, 3), got {img.shape}")

    target_tensor = tf.cast(tf.convert_to_tensor(img), tf.float32)

    # 2. Format baseline (default: all black image 0.0)
    if baseline is None:
        baseline_tensor = tf.zeros_like(target_tensor)
    else:
        if baseline.ndim == 3:
            baseline = np.expand_dims(baseline, axis=0)
        baseline_tensor = tf.cast(tf.convert_to_tensor(baseline), tf.float32)

    # 3. Determine target class
    preds = model(target_tensor, training=False)
    if target_class_idx is None:
        target_class = int(tf.argmax(preds[0]))
    else:
        target_class = int(target_class_idx)

    # 4. Generate linear path interpolations
    interpolated_path = interpolate_images(baseline_tensor, target_tensor, num_steps=num_steps)

    # 5. Compute gradients along path in batches
    total_steps = interpolated_path.shape[0]
    all_grads = []

    for i in range(0, total_steps, batch_size):
        batch = interpolated_path[i : i + batch_size]
        with tf.GradientTape() as tape:
            tape.watch(batch)
            predictions = model(batch, training=False)
            # Use log-probability score to maintain gradient stability
            scores = tf.math.log(predictions[:, target_class] + 1e-12)

        grads = tape.gradient(scores, batch)
        if grads is None:
            logger.warning("Gradients w.r.t input returned None. Returning zeros.")
            return np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)
        all_grads.append(grads)

    all_grads = tf.concat(all_grads, axis=0)

    # 6. Riemann approximation: average gradients across path (trapezoidal rule)
    avg_grads = (all_grads[:-1] + all_grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(avg_grads, axis=0, keepdims=True)

    # 7. Multiply by path displacement: (x - x') * avg_grads
    delta = target_tensor - baseline_tensor
    attributions = delta * avg_grads  # Shape: (1, H, W, 3)

    # 8. Collapse color channels to 2D spatial map (L2 norm or channel sum of abs)
    attribution_2d = tf.reduce_sum(tf.abs(attributions), axis=-1)[0].numpy()

    # 9. Outlier clipping at 99th percentile to suppress single-pixel spike noise
    p99 = np.percentile(attribution_2d, 99)
    if p99 > 1e-10:
        attribution_2d = np.clip(attribution_2d, 0, p99)

    max_val = np.max(attribution_2d)
    if max_val < 1e-10:
        logger.warning("Integrated Gradients attributions are near-zero. Returning blank map.")
        return np.zeros(attribution_2d.shape, dtype=np.float32)

    attribution_2d = (attribution_2d / max_val).astype(np.float32)
    return attribution_2d


def visualize_integrated_gradients(
    original_rgb: np.ndarray,
    attributions: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    return_rgb: bool = False,
) -> np.ndarray:
    """
    Overlay Integrated Gradients attribution map onto the original MRI scan.

    Args:
        original_rgb: Original scan in RGB format, shape (H, W, 3).
        attributions: 2D attribution map in [0, 1].
        alpha: Blending weight for original scan.
        colormap: OpenCV colormap (default: cv2.COLORMAP_VIRIDIS).
        return_rgb: Whether to return RGB uint8 instead of BGR uint8.

    Returns:
        Annotated visualization array as uint8.
    """
    if original_rgb.dtype != np.uint8:
        if original_rgb.max() <= 1.0:
            img_uint8 = (np.clip(original_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(original_rgb, 0, 255).astype(np.uint8)
    else:
        img_uint8 = original_rgb.copy()

    h, w = img_uint8.shape[:2]
    attr_resized = cv2.resize(attributions, (w, h), interpolation=cv2.INTER_LINEAR)
    attr_colored_bgr = cv2.applyColorMap(np.uint8(255 * attr_resized), colormap)

    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, alpha, attr_colored_bgr, 1.0 - alpha, 0)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if return_rgb:
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay
