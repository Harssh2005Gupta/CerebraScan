"""
Unit Tests for Training Pipeline, Dataset Loader, and Models.
"""

import os
import tempfile
import numpy as np
import pytest
from PIL import Image

from training.utils import load_config, set_seeds, save_json
from training.dataset import discover_image_paths, get_stratified_splits


class TestTrainingUtils:
    def test_set_seeds_runs_cleanly(self):
        set_seeds(42)
        assert os.environ["PYTHONHASHSEED"] == "42"

    def test_load_config_valid(self):
        config_path = "configs/baseline_resnet.yaml"
        if os.path.exists(config_path):
            cfg = load_config(config_path)
            assert cfg["experiment_name"] == "baseline_resnet"
            assert cfg["dataset"]["img_size"] == [224, 224]

    def test_load_config_missing(self):
        with pytest.raises(FileNotFoundError):
            load_config("non_existent_config.yaml")


class TestDatasetDiscovery:
    def test_discover_image_paths_with_mock_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            class_names = ["glioma", "meningioma", "notumor", "pituitary"]
            for cls in class_names:
                cls_dir = os.path.join(tmpdir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                for i in range(5):
                    img = Image.new("RGB", (32, 32), color=(i * 10, i * 20, i * 30))
                    img.save(os.path.join(cls_dir, f"sample_{i}.jpg"))

            paths, labels = discover_image_paths(tmpdir, class_names)
            assert len(paths) == 20
            assert len(labels) == 20
            # 5 of each label
            assert np.bincount(labels).tolist() == [5, 5, 5, 5]

            # Test stratified split
            train_p, val_p, train_l, val_l, weights = get_stratified_splits(
                tmpdir, class_names, val_split=0.20, seed=42
            )
            assert len(train_p) == 16
            assert len(val_p) == 4
            assert len(weights) == 4


class TestGradCAMAutoDetection:
    def test_auto_detection_finds_last_conv_layer(self):
        from gradcam import find_last_conv_layer, make_gradcam_heatmap
        import tensorflow as tf

        inp = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="first_conv")(inp)
        x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="deep_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        out = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inp, out)

        detected_name = find_last_conv_layer(model)
        assert detected_name == "deep_conv"

        dummy_img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        heatmap = make_gradcam_heatmap(dummy_img, model, "auto")
        assert heatmap.shape == (32, 32)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
