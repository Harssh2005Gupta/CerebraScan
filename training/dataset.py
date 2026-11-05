"""
Dataset Pipeline: Loading, Stratified Splitting, and Training Augmentations.
"""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# Standard class names order (alphabetical, consistent with sorted(os.listdir()))
DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def discover_image_paths(
    base_dir: str, class_names: List[str]
) -> Tuple[List[str], List[int]]:
    """
    Discover all valid image files and assign integer labels matching class_names order.

    Args:
        base_dir: Directory containing class subfolders.
        class_names: List of class names.

    Returns:
        (image_paths, labels)
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = []
    labels = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning("Class directory not found: %s", class_dir)
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_exts):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)

    return image_paths, labels


def get_stratified_splits(
    train_dir: str,
    class_names: List[str],
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int], Dict[int, float]]:
    """
    Split the training directory into stratified train and validation splits.

    Args:
        train_dir: Path to training images folder.
        class_names: List of class names.
        val_split: Fraction for validation (e.g. 0.15).
        seed: Random state seed.

    Returns:
        (train_paths, val_paths, train_labels, val_labels, class_weights_dict)
    """
    all_paths, all_labels = discover_image_paths(train_dir, class_names)
    if not all_paths:
        raise ValueError(f"No images found in {train_dir}. Please verify dataset directory.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=val_split,
        stratify=all_labels,
        random_state=seed,
    )

    # Balanced class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights_dict = {int(idx): float(w) for idx, w in enumerate(weights)}

    logger.info(
        "Split dataset: %d train, %d val (seed=%d, val_split=%.2f)",
        len(train_paths),
        len(val_paths),
        seed,
        val_split,
    )

    return train_paths, val_paths, train_labels, val_labels, class_weights_dict


def create_dataset_pipeline(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    test_paths: List[str],
    test_labels: List[int],
    config: dict,
):
    """
    Build TensorFlow tf.data pipelines for Train, Validation, and Test.
    Augmentation is strictly applied ONLY to the training dataset.

    Args:
        train_paths, train_labels: Training samples.
        val_paths, val_labels: Validation samples.
        test_paths, test_labels: Test samples.
        config: Configuration dictionary.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    import tensorflow as tf

    img_size = tuple(config["dataset"]["img_size"])
    batch_size = config["dataset"]["batch_size"]
    num_classes = len(config["dataset"]["class_names"])
    autotune = tf.data.AUTOTUNE

    # 1. Deterministic image loading and normalization to [0.0, 1.0]
    def load_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normalized to [0.0, 1.0]
        one_hot = tf.one_hot(label, num_classes)
        return img, one_hot

    # 2. Augmentation layers (training set ONLY)
    aug_cfg = config.get("augmentation", {})
    rot_factor = aug_cfg.get("rotation_factor", 0.10)
    zoom_factor = aug_cfg.get("zoom_factor", 0.10)

    aug_rotate = tf.keras.layers.RandomRotation(rot_factor, fill_mode="reflect")
    aug_zoom = tf.keras.layers.RandomZoom(zoom_factor, fill_mode="reflect")

    def augment(img, label):
        if aug_cfg.get("horizontal_flip", True):
            img = tf.image.random_flip_left_right(img)
        if aug_cfg.get("vertical_flip", True):
            img = tf.image.random_flip_up_down(img)

        b_delta = aug_cfg.get("brightness_delta", 0.15)
        if b_delta > 0:
            img = tf.image.random_brightness(img, b_delta)

        c_low = aug_cfg.get("contrast_lower", 0.85)
        c_up = aug_cfg.get("contrast_upper", 1.15)
        img = tf.image.random_contrast(img, c_low, c_up)

        s_low = aug_cfg.get("saturation_lower", 0.85)
        s_up = aug_cfg.get("saturation_upper", 1.15)
        img = tf.image.random_saturation(img, s_low, s_up)

        # Spatial transforms (require batch dimension)
        img = tf.expand_dims(img, 0)
        img = aug_rotate(img, training=True)
        img = aug_zoom(img, training=True)
        img = tf.squeeze(img, 0)

        # Ensure valid range [0, 1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    # --- Training dataset ---
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(
        len(train_paths),
        seed=config.get("seed", 42),
        reshuffle_each_iteration=True,
    )
    train_ds = train_ds.map(load_image, num_parallel_calls=autotune)
    train_ds = train_ds.cache()
    if aug_cfg.get("enabled", True):
        train_ds = train_ds.map(augment, num_parallel_calls=autotune)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(autotune)

    # --- Validation dataset (NO augmentation) ---
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_image, num_parallel_calls=autotune)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(autotune)

    # --- Test dataset (NO augmentation) ---
    test_ds = None
    if test_paths:
        test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        test_ds = test_ds.map(load_image, num_parallel_calls=autotune)
        test_ds = test_ds.cache()
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds
