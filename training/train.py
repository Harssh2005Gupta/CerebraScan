"""
Unified Training Runner for NeuroMap.
Reads configuration, initializes seeds, builds datasets, trains model,
and outputs evaluated checkpoints with full training logs.
"""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from training.dataset import (
    create_dataset_pipeline,
    discover_image_paths,
    get_stratified_splits,
)
from training.evaluate import evaluate_model_on_dataset
from training.models import build_model_from_config
from training.utils import load_config, save_json, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("NeuroMap.Train")


def train(config_path: str):
    """
    Execute full training pipeline according to configuration.

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_config(config_path)
    exp_name = config.get("experiment_name", "experiment")
    out_dir = config.get("output", {}).get("dir", f"artifacts/{exp_name}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Reproducibility
    seed = config.get("seed", 42)
    set_seeds(seed)

    # 2. Dataset Paths
    ds_cfg = config["dataset"]
    train_dir = ds_cfg.get("train_path", "data/Training")
    test_dir = ds_cfg.get("test_path", "data/Testing")
    class_names = ds_cfg.get(
        "class_names", ["glioma", "meningioma", "notumor", "pituitary"]
    )
    val_split = ds_cfg.get("val_split", 0.15)

    if not os.path.exists(train_dir):
        logger.error(
            "Training directory '%s' does not exist! "
            "Please ensure dataset is mounted or downloaded. "
            "See README.md for instructions.",
            train_dir,
        )
        print(f"\n[ERROR] Training dataset path not found: {train_dir}")
        print("To run training, place the dataset at 'data/Training' and 'data/Testing',")
        print("or run with a modified path: python -m training.train --config <config.yaml>\n")
        return None

    # 3. Stratified Train / Validation Split
    logger.info("Preparing stratified dataset splits from '%s'...", train_dir)
    train_paths, val_paths, train_labels, val_labels, class_weights = get_stratified_splits(
        train_dir=train_dir,
        class_names=class_names,
        val_split=val_split,
        seed=seed,
    )

    test_paths, test_labels = [], []
    if os.path.exists(test_dir):
        test_paths, test_labels = discover_image_paths(test_dir, class_names)
        logger.info("Found %d test samples in '%s'.", len(test_paths), test_dir)
    else:
        logger.warning("Test directory '%s' not found. Evaluation will be skipped.", test_dir)

    # 4. TensorFlow Data Pipelines
    train_ds, val_ds, test_ds = create_dataset_pipeline(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        config=config,
    )

    # 5. Build Model
    model, target_conv_layer = build_model_from_config(config)

    # 6. Optimizer & Loss
    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 3e-4))
    wd = float(t_cfg.get("weight_decay", 1e-4))
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)

    lbl_smooth = float(t_cfg.get("label_smoothing", 0.10))
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=lbl_smooth)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # 7. Callbacks
    cb_cfg = config.get("callbacks", {})
    checkpoint_file = cb_cfg.get("checkpoint", {}).get("filename", "best_model.keras")
    checkpoint_path = os.path.join(out_dir, checkpoint_file)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=cb_cfg.get("early_stopping", {}).get("monitor", "val_accuracy"),
            patience=int(cb_cfg.get("early_stopping", {}).get("patience", 15)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=cb_cfg.get("reduce_lr", {}).get("monitor", "val_loss"),
            factor=float(cb_cfg.get("reduce_lr", {}).get("factor", 0.5)),
            patience=int(cb_cfg.get("reduce_lr", {}).get("patience", 5)),
            min_lr=float(cb_cfg.get("reduce_lr", {}).get("min_lr", 1e-6)),
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=cb_cfg.get("checkpoint", {}).get("monitor", "val_accuracy"),
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(out_dir, "history.csv"),
            separator=",",
            append=False,
        ),
    ]

    # 8. Train
    epochs = int(t_cfg.get("epochs", 60))
    use_class_weights = t_cfg.get("class_weights", "balanced") == "balanced"
    fit_weights = class_weights if use_class_weights else None

    logger.info("Starting training for %d epochs...", epochs)
    start_time = time.perf_counter()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=fit_weights,
    )

    total_train_time = time.perf_counter() - start_time
    logger.info("Training finished in %.2f seconds.", total_train_time)

    # 9. Plot Learning Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Classification Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss (Label Smoothing 0.1)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    curves_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()

    # 10. Save Model Metadata for Inference & Grad-CAM
    metadata = {
        "experiment_name": exp_name,
        "architecture": config["model"]["architecture"],
        "input_shape": [None, ds_cfg["img_size"][0], ds_cfg["img_size"][1], 3],
        "normalization": "[0.0, 1.0]",
        "class_names": class_names,
        "target_conv_layer": target_conv_layer,
        "training_time_seconds": round(total_train_time, 2),
    }
    save_json(metadata, os.path.join(out_dir, "model_metadata.json"))

    # 11. Run Final Evaluation if test dataset exists
    if test_ds is not None:
        logger.info("Running evaluation on test dataset...")
        best_model = tf.keras.models.load_model(checkpoint_path)
        metrics = evaluate_model_on_dataset(best_model, test_ds, class_names, out_dir)
        metrics["training_time_seconds"] = round(total_train_time, 2)
        save_json(metrics, os.path.join(out_dir, "metrics.json"))
        logger.info("Test Accuracy: %.2f%% | Macro F1: %.4f", metrics["test_accuracy"] * 100, metrics["macro_f1"])

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train a NeuroMap classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
