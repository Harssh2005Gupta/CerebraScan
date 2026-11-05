"""
Grad-CAM — Gradient-weighted Class Activation Mapping.

Generates heatmaps highlighting regions of an MRI scan that most influenced
the model's classification decision.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def find_last_conv_layer(
    model: tf.keras.Model,
    candidate_layer_name: Optional[str] = None,
) -> str:
    """
    Locate the name of the last 4D convolutional/activation layer in the model.
    Supports custom CNNs, transfer learning backbones (ResNet, EfficientNet, MobileNet),
    and nested functional models.

    Args:
        model: Compiled or loaded Keras model.
        candidate_layer_name: Optional layer name to validate. If provided, valid,
            and produces a 4D output, it is returned. Otherwise auto-detection is used.

    Returns:
        The validated layer name.

    Raises:
        ValueError: If no suitable 4D convolutional layer is found in the model.
    """
    layer_dict = {layer.name: layer for layer in model.layers}

    # If candidate name provided and valid
    if candidate_layer_name and candidate_layer_name.lower() != "auto":
        if candidate_layer_name in layer_dict:
            target_layer = layer_dict[candidate_layer_name]
            try:
                shape = target_layer.output.shape
                if len(shape) == 4 and shape[1] is not None and shape[1] > 1:
                    logger.info("Validated candidate conv layer: %s", candidate_layer_name)
                    return candidate_layer_name
                else:
                    logger.warning(
                        "Layer '%s' has output shape %s (not 4D spatial). Falling back to auto-detection.",
                        candidate_layer_name,
                        shape,
                    )
            except Exception as e:
                logger.warning(
                    "Could not inspect output of candidate layer '%s': %s. Falling back to auto-detection.",
                    candidate_layer_name,
                    e,
                )
        else:
            logger.warning(
                "Candidate layer '%s' not found in model layers. Falling back to auto-detection.",
                candidate_layer_name,
            )

    # Recursive / nested layer search
    def _collect_4d_layers(m: tf.keras.Model) -> List[Tuple[str, tf.keras.layers.Layer]]:
        conv_layers = []
        for l in m.layers:
            if hasattr(l, "layers") and l.layers:
                # Nested model / backbone container
                conv_layers.extend(_collect_4d_layers(l))
            else:
                try:
                    shape = l.output.shape
                    if len(shape) == 4 and shape[1] is not None and shape[1] > 1 and shape[2] is not None and shape[2] > 1:
                        conv_layers.append((l.name, l))
                except Exception:
                    continue
        return conv_layers

    candidates = _collect_4d_layers(model)
    if candidates:
        last_name, _ = candidates[-1]
        logger.info("Auto-detected last conv layer for Grad-CAM: %s", last_name)
        return last_name

    raise ValueError("Could not find a 4D convolutional layer in the provided model.")


def make_gradcam_heatmap(
    img: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = "auto",
    pred_index: Optional[int] = None,
    use_logits: bool = True,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a target class.

    Args:
        img: Preprocessed image array of shape (1, H, W, 3) or (H, W, 3), float32, [0, 1].
        model: Compiled Keras model.
        last_conv_layer_name: Name of target layer, or "auto" to detect automatically.
        pred_index: Index of the class to explain. If None, uses top predicted class.
        use_logits: If True, uses log-probability formulation to prevent softmax
            gradient saturation when confidence is near 1.0.

    Returns:
        Heatmap as a 2-D numpy array of shape (H_conv, W_conv), values normalized in [0, 1].
    """
    # 1. Validate & format input tensor
    if not isinstance(img, (np.ndarray, tf.Tensor)):
        img = np.array(img)

    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    if img.ndim != 4 or img.shape[-1] != 3:
        raise ValueError(f"Expected input array of shape (1, H, W, 3) or (H, W, 3), got {img.shape}")

    img_tensor = tf.cast(tf.convert_to_tensor(img), tf.float32)

    # 2. Resolve target layer
    target_layer_name = find_last_conv_layer(model, last_conv_layer_name)

    # 3. Construct gradient model
    try:
        target_layer = model.get_layer(target_layer_name)
    except ValueError as e:
        logger.error("Failed to access layer '%s': %s", target_layer_name, e)
        raise

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    # 4. Compute activations and gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)

        # Determine target class
        num_classes = predictions.shape[-1]
        if pred_index is None:
            target_class = tf.argmax(predictions[0])
        else:
            if not (0 <= pred_index < num_classes):
                raise ValueError(
                    f"pred_index {pred_index} is out of bounds for model with {num_classes} classes."
                )
            target_class = int(pred_index)

        # Selvaraju et al. formulate Grad-CAM w.r.t pre-softmax scores (logits) y^c.
        # When model ends with Softmax, confident outputs (S_c -> 1) suffer from vanishing gradients:
        # dS_c/dz_c = S_c(1 - S_c) -> 0.
        # d(log S_c)/dA = (1/S_c) * (dS_c/dA), cancelling the vanishing factor.
        if use_logits:
            score = tf.math.log(predictions[:, target_class] + 1e-12)
        else:
            score = predictions[:, target_class]

    grads = tape.gradient(score, conv_outputs)

    # 5. Handle zero / disconnected gradients safely
    if grads is None:
        logger.warning("Gradient computation returned None (layer is disconnected). Returning blank heatmap.")
        return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]), dtype=np.float32)

    # Global Average Pooling of gradients across spatial dimensions (alpha_k^c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Linear combination of feature maps weighted by alpha_k^c
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: retain only features that positively influence the target class
    heatmap = tf.maximum(heatmap, 0)

    # Safe normalization into [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val < 1e-10:
        logger.warning("Grad-CAM activations are near-zero or negative; returning blank heatmap.")
        return np.zeros(heatmap.shape, dtype=np.float32)

    heatmap = heatmap / max_val
    return heatmap.numpy().astype(np.float32)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
    return_rgb: bool = False,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original MRI scan.

    Color Space Handling:
        - `image`: Expected in RGB format, uint8 [0, 255] or float32 [0, 1].
        - `colormap`: OpenCV colormap (outputs BGR).
        - If `return_rgb=False` (default): Returns BGR uint8, ready for `cv2.imwrite()`.
        - If `return_rgb=True`: Returns RGB uint8, ready for `matplotlib`, `PIL`, or browser canvas.

    Args:
        image: Original MRI image, shape (H, W, 3).
        heatmap: 2-D heatmap array from make_gradcam_heatmap(), shape (h, w), values in [0, 1].
        alpha: Blending weight for the original image (1 - alpha for heatmap).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).
        return_rgb: Whether to return RGB instead of BGR.

    Returns:
        Blended image as uint8 numpy array of shape (H, W, 3).
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    # Ensure heatmap is resized to match original image (width, height)
    target_w, target_h = img_uint8.shape[1], img_uint8.shape[0]
    heatmap_resized = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Colorize heatmap (OpenCV produces BGR)
    heatmap_colored_bgr = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

    # Convert original RGB image to BGR for OpenCV blending
    image_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Weighted blending
    blended_bgr = cv2.addWeighted(image_bgr, alpha, heatmap_colored_bgr, 1.0 - alpha, 0)
    blended_bgr = np.clip(blended_bgr, 0, 255).astype(np.uint8)

    if return_rgb:
        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    return blended_bgr


def create_side_by_side_comparison(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    confidence: float,
    class_probabilities: Optional[Dict[str, float]] = None,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
    target_height: int = 360,
    return_rgb: bool = False,
) -> np.ndarray:
    """
    Create a labeled, side-by-side composite comparison:
    Left: Original MRI Scan.
    Right: Grad-CAM Overlay.
    Includes explicit clinical attribution disclaimer and prediction metrics.

    Args:
        original_rgb: Original scan in RGB format, shape (H, W, 3).
        heatmap: 2-D heatmap in [0, 1].
        predicted_class: Name of predicted class.
        confidence: Prediction confidence float in [0, 1].
        class_probabilities: Optional dict of all class probabilities.
        alpha: Blending weight for overlay.
        colormap: OpenCV colormap enum.
        target_height: Height in pixels for each scan frame.
        return_rgb: Whether to return RGB uint8 instead of BGR uint8.

    Returns:
        Composite annotated image as uint8 numpy array.
    """
    # 1. Prepare base images in BGR
    if original_rgb.dtype != np.uint8:
        if original_rgb.max() <= 1.0:
            base_rgb = (np.clip(original_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            base_rgb = np.clip(original_rgb, 0, 255).astype(np.uint8)
    else:
        base_rgb = original_rgb.copy()

    overlay_bgr = overlay_heatmap(base_rgb, heatmap, alpha=alpha, colormap=colormap, return_rgb=False)
    original_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

    # 2. Resize both frames to standard square height
    target_w = target_height
    target_h = target_height
    orig_resized = cv2.resize(original_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    over_resized = cv2.resize(overlay_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 3. Canvas layout dimensions
    pad = 16
    banner_top_h = 70
    banner_bottom_h = 44
    canvas_w = target_w * 2 + pad * 3
    canvas_h = target_h + banner_top_h + banner_bottom_h + pad

    # Dark background #0e0e14
    canvas = np.full((canvas_h, canvas_w, 3), [20, 14, 14], dtype=np.uint8)

    # 4. Paste scans onto canvas
    y_scan = banner_top_h
    x_orig = pad
    x_over = target_w + pad * 2

    canvas[y_scan : y_scan + target_h, x_orig : x_orig + target_w] = orig_resized
    canvas[y_scan : y_scan + target_h, x_over : x_over + target_w] = over_resized

    # Border outlines
    border_color = (60, 50, 50)
    cv2.rectangle(canvas, (x_orig, y_scan), (x_orig + target_w, y_scan + target_h), border_color, 1)
    cv2.rectangle(canvas, (x_over, y_scan), (x_over + target_w, y_scan + target_h), border_color, 1)

    # 5. Top Banner: Prediction Title & Confidence
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = f"PREDICTION: {predicted_class.upper()} ({confidence * 100:.1f}%)"
    sub_title = "NEUROMAP DEEP LEARNING ATTRIBUTION"

    cv2.putText(canvas, sub_title, (pad, 24), font, 0.45, (180, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, title_text, (pad, 54), font, 0.75, (86, 115, 222), 2, cv2.LINE_AA)  # Terracotta accent

    # Frame labels
    cv2.putText(canvas, "ORIGINAL MRI SCAN", (x_orig + 8, y_scan + 22), font, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GRAD-CAM OVERLAY (JET)", (x_over + 8, y_scan + 22), font, 0.45, (95, 212, 207), 1, cv2.LINE_AA)

    # 6. Bottom Banner: Medical Interpretability Disclaimer
    disclaimer_text = "INTERPRETABILITY NOTICE: Heatmap highlights influential receptive fields. NOT a tumor margin or segmentation."
    y_disclaimer = canvas_h - 16
    cv2.putText(canvas, disclaimer_text, (pad, y_disclaimer), font, 0.38, (140, 140, 160), 1, cv2.LINE_AA)

    if return_rgb:
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return canvas