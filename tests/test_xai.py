"""
Unit and integration tests for Explainable AI (XAI) modules:
- Grad-CAM (gradcam.py)
- Integrated Gradients (integrated_gradients.py)
"""

import numpy as np
import pytest
import tensorflow as tf

from gradcam import (
    create_side_by_side_comparison,
    find_last_conv_layer,
    make_gradcam_heatmap,
    overlay_heatmap,
)
from integrated_gradients import (
    compute_integrated_gradients,
    interpolate_images,
    visualize_integrated_gradients,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Helper Models
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_conv_model():
    """Build a tiny deterministic CNN model for testing."""
    inp = tf.keras.Input(shape=(32, 32, 3), name="input_scan")
    x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", name="conv_early")(inp)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv_target")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    out = tf.keras.layers.Dense(4, activation="softmax", name="prob_output")(x)
    return tf.keras.Model(inp, out, name="SimpleTestModel")


@pytest.fixture
def nested_conv_model():
    """Build a model with a nested backbone sub-model."""
    backbone_in = tf.keras.Input(shape=(32, 32, 3), name="sub_input")
    b = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", name="nested_conv")(backbone_in)
    backbone = tf.keras.Model(backbone_in, b, name="BackboneSubmodel")

    inp = tf.keras.Input(shape=(32, 32, 3), name="main_input")
    x = backbone(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inp, out, name="MainNestedModel")


@pytest.fixture
def synthetic_scan():
    """Generate a synthetic 32x32 RGB image array in [0.0, 1.0]."""
    np.random.seed(42)
    return np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test Layer Selection & Detection
# ---------------------------------------------------------------------------


class TestConvLayerDetection:
    def test_auto_detection_finds_last_conv_layer(self, simple_conv_model):
        layer_name = find_last_conv_layer(simple_conv_model)
        assert layer_name == "conv_target"

    def test_valid_candidate_layer_accepted(self, simple_conv_model):
        layer_name = find_last_conv_layer(simple_conv_model, candidate_layer_name="conv_early")
        assert layer_name == "conv_early"

    def test_invalid_candidate_falls_back_to_auto(self, simple_conv_model):
        # "nonexistent_layer" does not exist in model
        layer_name = find_last_conv_layer(simple_conv_model, candidate_layer_name="nonexistent_layer")
        assert layer_name == "conv_target"

    def test_non_4d_candidate_falls_back_to_auto(self, simple_conv_model):
        # "gap" is 2D (batch, features), not 4D
        layer_name = find_last_conv_layer(simple_conv_model, candidate_layer_name="gap")
        assert layer_name == "conv_target"

    def test_nested_model_finds_submodel_conv(self, nested_conv_model):
        layer_name = find_last_conv_layer(nested_conv_model)
        assert layer_name == "nested_conv"

    def test_model_without_conv_raises_error(self):
        inp = tf.keras.Input(shape=(10,))
        out = tf.keras.layers.Dense(2)(inp)
        mlp = tf.keras.Model(inp, out)
        with pytest.raises(ValueError, match="Could not find a 4D convolutional layer"):
            find_last_conv_layer(mlp)


# ---------------------------------------------------------------------------
# Test Grad-CAM Heatmap Generation
# ---------------------------------------------------------------------------


class TestGradCAMHeatmap:
    def test_heatmap_shape_and_range(self, simple_conv_model, synthetic_scan):
        heatmap = make_gradcam_heatmap(synthetic_scan, simple_conv_model, "conv_target")
        assert heatmap.shape == (32, 32)
        assert heatmap.dtype == np.float32
        assert np.all(heatmap >= 0.0)
        assert np.all(heatmap <= 1.0)
        assert np.max(heatmap) == pytest.approx(1.0, abs=1e-5)

    def test_target_class_selection(self, simple_conv_model, synthetic_scan):
        heatmap_c0 = make_gradcam_heatmap(synthetic_scan, simple_conv_model, pred_index=0)
        heatmap_c3 = make_gradcam_heatmap(synthetic_scan, simple_conv_model, pred_index=3)

        assert heatmap_c0.shape == (32, 32)
        assert heatmap_c3.shape == (32, 32)
        # Class 0 and Class 3 gradients should generally differ
        assert not np.array_equal(heatmap_c0, heatmap_c3)

    def test_out_of_bounds_pred_index(self, simple_conv_model, synthetic_scan):
        with pytest.raises(ValueError, match="out of bounds"):
            make_gradcam_heatmap(synthetic_scan, simple_conv_model, pred_index=99)

    def test_3d_input_image_handling(self, simple_conv_model, synthetic_scan):
        # Input of shape (32, 32, 3) instead of (1, 32, 32, 3)
        img_3d = synthetic_scan[0]
        heatmap = make_gradcam_heatmap(img_3d, simple_conv_model)
        assert heatmap.shape == (32, 32)

    def test_invalid_input_shape_raises_error(self, simple_conv_model):
        bad_img = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected input array"):
            make_gradcam_heatmap(bad_img, simple_conv_model)

    def test_zero_gradient_graceful_handling(self):
        # Model with constant output disconnected from conv layer
        inp = tf.keras.Input(shape=(16, 16, 3))
        conv = tf.keras.layers.Conv2D(4, 3, padding="same", name="conv")(inp)
        # Output is zeros, disconnected from conv
        out = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 4)))(conv)
        detached_model = tf.keras.Model(inp, out)

        dummy_img = np.ones((1, 16, 16, 3), dtype=np.float32)
        heatmap = make_gradcam_heatmap(dummy_img, detached_model, "conv")
        assert heatmap.shape == (16, 16)
        assert np.all(heatmap == 0.0)


# ---------------------------------------------------------------------------
# Test Heatmap Overlay & Side-by-Side Comparison
# ---------------------------------------------------------------------------


class TestVisualizations:
    def test_overlay_heatmap_shapes_and_types(self):
        original = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        heatmap = np.random.rand(8, 8).astype(np.float32)

        # Default BGR
        overlay_bgr = overlay_heatmap(original, heatmap, return_rgb=False)
        assert overlay_bgr.shape == (64, 64, 3)
        assert overlay_bgr.dtype == np.uint8

        # RGB
        overlay_rgb = overlay_heatmap(original, heatmap, return_rgb=True)
        assert overlay_rgb.shape == (64, 64, 3)
        assert overlay_rgb.dtype == np.uint8

    def test_overlay_heatmap_float_input(self):
        original_float = np.random.rand(64, 64, 3).astype(np.float32)
        heatmap = np.random.rand(8, 8).astype(np.float32)

        overlay = overlay_heatmap(original_float, heatmap)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_side_by_side_comparison(self):
        original = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        heatmap = np.random.rand(12, 12).astype(np.float32)

        comp = create_side_by_side_comparison(
            original_rgb=original,
            heatmap=heatmap,
            predicted_class="glioma",
            confidence=0.984,
            class_probabilities={"glioma": 0.984, "meningioma": 0.01, "notumor": 0.003, "pituitary": 0.003},
            target_height=200,
            return_rgb=True,
        )

        assert comp.ndim == 3
        assert comp.shape[2] == 3
        assert comp.dtype == np.uint8
        # Height: 200 + banner_top(70) + banner_bottom(44) + pad(16) = 330
        assert comp.shape[0] == 330
        # Width: 200 * 2 + pad(16) * 3 = 448
        assert comp.shape[1] == 448


# ---------------------------------------------------------------------------
# Test Integrated Gradients
# ---------------------------------------------------------------------------


class TestIntegratedGradients:
    def test_interpolate_images(self):
        baseline = tf.zeros((1, 16, 16, 3), dtype=tf.float32)
        target = tf.ones((1, 16, 16, 3), dtype=tf.float32)

        steps = 10
        interp = interpolate_images(baseline, target, num_steps=steps)
        assert interp.shape == (steps + 1, 16, 16, 3)
        # Alpha=0 should equal baseline (0)
        assert np.allclose(interp[0].numpy(), 0.0)
        # Alpha=1 should equal target (1)
        assert np.allclose(interp[-1].numpy(), 1.0)

    def test_compute_integrated_gradients(self, simple_conv_model, synthetic_scan):
        ig_map = compute_integrated_gradients(
            synthetic_scan,
            simple_conv_model,
            target_class_idx=0,
            num_steps=10,
        )

        assert ig_map.shape == (32, 32)
        assert ig_map.dtype == np.float32
        assert np.all(ig_map >= 0.0)
        assert np.all(ig_map <= 1.0)
        assert np.max(ig_map) <= 1.0

    def test_visualize_integrated_gradients(self):
        original = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        ig_map = np.random.rand(32, 32).astype(np.float32)

        vis = visualize_integrated_gradients(original, ig_map, return_rgb=True)
        assert vis.shape == (32, 32, 3)
        assert vis.dtype == np.uint8
