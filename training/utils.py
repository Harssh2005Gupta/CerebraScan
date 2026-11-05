"""
Training Utilities: Reproducibility, Configuration, and Benchmarking Helpers.
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """
    Ensure complete deterministic reproducibility across Python,
    NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Enable deterministic ops where supported
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
        logger.info("Random seeds set to %d with TensorFlow op determinism.", seed)
    except ImportError:
        logger.info("Random seeds set to %d (TensorFlow not installed in current environment).", seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Dictionary of configuration options.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def benchmark_inference(
    model: Any,
    input_shape: tuple = (1, 224, 224, 3),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model inference latency in milliseconds per image.

    Args:
        model: Loaded Keras model.
        input_shape: Input tensor shape.
        num_warmup: Warm-up runs to stabilize caches.
        num_runs: Measured runs.

    Returns:
        Dict with mean_ms, std_ms, p50_ms, p95_ms, throughput_fps.
    """
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = model.predict(dummy_input, verbose=0)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p95_ms = float(np.percentile(latencies, 95))
    throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "throughput_fps": round(throughput, 1),
    }
