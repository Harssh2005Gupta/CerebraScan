"""
Model Architectures: Baseline Custom Pre-activation ResNet & Transfer Learning Backbones.
All models are designed to ingest the same standardized input:
(None, 224, 224, 3) float32 normalized in [0.0, 1.0].
"""

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Baseline Custom Pre-activation ResNet
# ---------------------------------------------------------------------------


def residual_block(x: tf.Tensor, filters: int, stride: int = 1, l2_reg: float = 1e-4) -> tf.Tensor:
    """
    Pre-activation residual block (BN -> ReLU -> Conv).
    Shortcut uses 1x1 conv when dimensions or strides change.
    """
    shortcut = x

    # First conv branch
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(l2_reg),
    )(x)

    # Second conv branch
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(l2_reg),
    )(x)

    # Shortcut projection if dimension or stride changes
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(l2_reg),
        )(shortcut)

    x = layers.Add()([x, shortcut])
    return x


def build_baseline_resnet(
    img_size: Tuple[int, int] = (224, 224),
    num_classes: int = 4,
    l2_reg: float = 1e-4,
    dropout_1: float = 0.40,
    dropout_2: float = 0.30,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    Build the exact custom pre-activation ResNet trained in tumor-detector.ipynb.

    Target Grad-CAM layer: 'activation_17' (output shape: None, 7, 7, 512).
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_image")

    # --- Stem ---
    x = layers.Conv2D(
        32,
        7,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(l2_reg),
        name="stem_conv",
    )(inputs)  # 112x112x32
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)  # 56x56x32

    # --- Stage 1: 56x56 ---
    x = residual_block(x, 64, l2_reg=l2_reg)
    x = residual_block(x, 64, l2_reg=l2_reg)

    # --- Stage 2: 28x28 ---
    x = residual_block(x, 128, stride=2, l2_reg=l2_reg)
    x = residual_block(x, 128, l2_reg=l2_reg)

    # --- Stage 3: 14x14 ---
    x = residual_block(x, 256, stride=2, l2_reg=l2_reg)
    x = residual_block(x, 256, l2_reg=l2_reg)

    # --- Stage 4: 7x7 ---
    x = residual_block(x, 512, stride=2, l2_reg=l2_reg)
    x = residual_block(x, 512, l2_reg=l2_reg)  # Ends with activation_17

    # --- Classification Head ---
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_1, name="dropout_head_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="dense_feature",
    )(x)
    x = layers.Dropout(dropout_2, name="dropout_head_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="NeuroMap_Baseline_ResNet")
    return model


# ---------------------------------------------------------------------------
# 2. Transfer Learning: EfficientNetB0
# ---------------------------------------------------------------------------


def build_efficientnet_b0(
    img_size: Tuple[int, int] = (224, 224),
    num_classes: int = 4,
    weights: str = "imagenet",
    trainable_backbone: bool = True,
    dropout: float = 0.30,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    Build EfficientNetB0 transfer learning model.
    Internal rescaling converts [0, 1] inputs to [0, 255] for standard ImageNet weights.
    Target Grad-CAM layer: 'top_activation' (output shape: None, 7, 7, 1280).
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_image")

    # Rescale [0.0, 1.0] -> [0.0, 255.0] expected by tf.keras.applications.EfficientNetB0
    x = layers.Rescaling(255.0, name="rescaling_to_255")(inputs)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    base_model.trainable = trainable_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_feature")(x)
    x = layers.Dropout(dropout * 0.75, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="NeuroMap_EfficientNetB0")
    return model


# ---------------------------------------------------------------------------
# 3. Transfer Learning: MobileNetV2
# ---------------------------------------------------------------------------


def build_mobilenet_v2(
    img_size: Tuple[int, int] = (224, 224),
    num_classes: int = 4,
    weights: str = "imagenet",
    trainable_backbone: bool = True,
    dropout: float = 0.30,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    Build MobileNetV2 transfer learning model.
    Internal rescaling converts [0, 1] inputs to [-1, 1] required by MobileNetV2.
    Target Grad-CAM layer: 'out_relu' (output shape: None, 7, 7, 1280).
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_image")

    # Rescale [0.0, 1.0] -> [-1.0, 1.0] expected by MobileNetV2
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescaling_to_minus1_plus1")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_tensor=x,
    )
    base_model.trainable = trainable_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_feature")(x)
    x = layers.Dropout(dropout * 0.75, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="NeuroMap_MobileNetV2")
    return model


# ---------------------------------------------------------------------------
# 4. Model Factory & Introspection
# ---------------------------------------------------------------------------


def build_model_from_config(config: dict) -> Tuple[tf.keras.Model, str]:
    """
    Factory function to instantiate models and their corresponding Grad-CAM layer.

    Args:
        config: Configuration dictionary.

    Returns:
        (compiled_or_uncompiled_model, default_gradcam_layer_name)
    """
    arch = config.get("model", {}).get("architecture", "baseline_resnet").lower()
    img_size = tuple(config["dataset"]["img_size"])
    num_classes = len(config["dataset"]["class_names"])
    model_cfg = config.get("model", {})

    if arch in ("baseline_resnet", "custom_resnet", "resnet"):
        model = build_baseline_resnet(
            img_size=img_size,
            num_classes=num_classes,
            l2_reg=float(model_cfg.get("l2_reg", 1e-4)),
            dropout_1=float(model_cfg.get("dropout_1", 0.40)),
            dropout_2=float(model_cfg.get("dropout_2", 0.30)),
            dense_units=int(model_cfg.get("dense_units", 256)),
        )
        gradcam_layer = model_cfg.get("target_conv_layer", "activation_17")

    elif arch in ("efficientnet_b0", "efficientnet"):
        model = build_efficientnet_b0(
            img_size=img_size,
            num_classes=num_classes,
            weights=model_cfg.get("weights", "imagenet"),
            trainable_backbone=bool(model_cfg.get("trainable_backbone", True)),
            dropout=float(model_cfg.get("dropout", 0.30)),
            dense_units=int(model_cfg.get("dense_units", 256)),
        )
        gradcam_layer = model_cfg.get("target_conv_layer", "top_activation")

    elif arch in ("mobilenet_v2", "mobilenet"):
        model = build_mobilenet_v2(
            img_size=img_size,
            num_classes=num_classes,
            weights=model_cfg.get("weights", "imagenet"),
            trainable_backbone=bool(model_cfg.get("trainable_backbone", True)),
            dropout=float(model_cfg.get("dropout", 0.30)),
            dense_units=int(model_cfg.get("dense_units", 256)),
        )
        gradcam_layer = model_cfg.get("target_conv_layer", "out_relu")

    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: baseline_resnet, efficientnet_b0, mobilenet_v2.")

    logger.info("Constructed %s model with %s parameters.", arch, f"{model.count_params():,}")
    return model, gradcam_layer
