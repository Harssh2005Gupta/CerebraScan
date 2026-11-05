"""
Evaluation and Metrics Reporting Module.
Computes comprehensive test metrics, confusion matrix, and inference benchmark.
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from training.utils import benchmark_inference, save_json

logger = logging.getLogger(__name__)


def evaluate_model_on_dataset(
    model: Any,
    test_ds: Any,
    class_names: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run complete evaluation on test dataset.

    Args:
        model: Loaded Keras model.
        test_ds: Batched test tf.data.Dataset.
        class_names: List of class names.
        output_dir: Directory to save evaluation artifacts.

    Returns:
        Dictionary of all metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Standard evaluate (loss, accuracy)
    eval_res = model.evaluate(test_ds, verbose=0)
    test_loss = float(eval_res[0])
    test_acc = float(eval_res[1])

    # 2. Extract true labels and predictions
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Collect ground truth
    y_true_list = []
    for _, y_batch in test_ds:
        if len(y_batch.shape) > 1 and y_batch.shape[-1] > 1:
            # One-hot
            y_true_list.append(np.argmax(y_batch.numpy(), axis=1))
        else:
            y_true_list.append(y_batch.numpy())
    y_true = np.concatenate(y_true_list, axis=0)

    # 3. Classification report
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # 4. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 5. Measure latency & parameter counts
    bench_results = benchmark_inference(model)
    total_params = int(model.count_params())
    trainable_params = int(
        np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    )

    # 6. Aggregate results
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall": round(report_dict[cls]["recall"], 4),
                "f1_score": round(report_dict[cls]["f1-score"], 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        },
        "inference_benchmark": bench_results,
    }

    # Save to JSON
    json_path = os.path.join(output_dir, "metrics.json")
    save_json(metrics, json_path)
    logger.info("Evaluation complete. Metrics saved to %s", json_path)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a NeuroMap trained model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .keras model")
    parser.add_argument("--test-dir", type=str, default="data/Testing", help="Path to test images")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation", help="Output directory")
    args = parser.parse_args()

    import tensorflow as tf

    from training.dataset import discover_image_paths

    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    test_paths, test_labels = discover_image_paths(args.test_dir, class_names)
    if not test_paths:
        print(f"No test images found in {args.test_dir}. Evaluation cannot proceed.")
        return

    def load_test_image(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        one_hot = tf.one_hot(label, len(class_names))
        return img, one_hot

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(load_test_image).batch(32)

    metrics = evaluate_model_on_dataset(model, test_ds, class_names, args.output_dir)
    print("\n========= Test Results =========")
    print(f"Accuracy : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Latency  : {metrics['inference_benchmark']['mean_latency_ms']} ms/sample")


if __name__ == "__main__":
    main()
