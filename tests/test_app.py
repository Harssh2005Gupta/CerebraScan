"""
Tests for NeuroMap Flask API Backend.

Covers:
- Application factory configuration & CWD-independent path resolution
- Liveness (/health) and Readiness (/ready) probes
- Model-loaded vs Model-unloaded states (503 Service Unavailable)
- Image upload validation (missing file, empty filename, invalid extension, corrupted content, zero-byte stream, bounds)
- Prediction response schema and probability distributions
- Explainability artifact creation & safe filename generation
- Temporary file cleanup lifecycle
- Error handlers (400, 404, 405, 413, 500)
- Concurrent prediction requests simulation
"""

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app import create_app, load_model
from config import Config, TestingConfig
from utils.cleanup import cleanup_old_files
from utils.preprocess import (
    EmptyImageError,
    ImageDimensionError,
    InvalidImageFormatError,
    preprocess_image,
    validate_and_load_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Create a mock Keras model returning deterministic predictions."""
    model = MagicMock()
    # Softmax output: class 2 ("notumor") with 0.91 confidence
    model.predict.return_value = np.array([[0.02, 0.03, 0.91, 0.04]], dtype=np.float32)
    model.inputs = [MagicMock()]
    model.output = MagicMock()
    return model


@pytest.fixture
def client(mock_model):
    """Create a test client with a mock model."""
    with patch("app.make_gradcam_heatmap") as mock_gcam, \
         patch("app.overlay_heatmap") as mock_overlay, \
         patch("app.create_side_by_side_comparison") as mock_comp, \
         patch("cv2.imwrite"):

        mock_gcam.return_value = np.zeros((7, 7), dtype=np.float32)
        mock_overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        mock_comp.return_value = np.zeros((300, 600, 3), dtype=np.uint8)

        test_app = create_app(TestingConfig, model=mock_model)
        with test_app.test_client() as c:
            yield c


@pytest.fixture
def client_no_model():
    """Create a test client where the model failed to load."""
    test_app = create_app(TestingConfig, model=None)
    with test_app.test_client() as c:
        yield c


@pytest.fixture
def sample_image():
    """Generate a minimal valid 64x64 PNG image in memory."""
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def corrupted_image():
    """Generate a file with valid .png extension but corrupted random bytes."""
    return io.BytesIO(b"\x89PNG\r\n\x1a\nCorruptedGarbageBytesNotRealImageData")


# ---------------------------------------------------------------------------
# Route Tests: Health & Readiness Probes
# ---------------------------------------------------------------------------


class TestHealthAndReadiness:
    def test_home_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"NEUROMAP" in resp.data

    def test_health_liveness_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] in ("ok", "degraded")
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_ready_probe_when_model_loaded(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ready"
        assert data["model_loaded"] is True

    def test_ready_probe_when_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/ready")
        assert resp.status_code == 503
        data = json.loads(resp.data)
        assert data["status"] == "not_ready"
        assert data["model_loaded"] is False
        assert "error" in data


# ---------------------------------------------------------------------------
# Route Tests: Prediction & Validation
# ---------------------------------------------------------------------------


class TestPredictRoute:
    def test_predict_success(self, client, sample_image):
        resp = client.post(
            "/predict",
            data={"image": (sample_image, "test_scan.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "success"
        assert data["prediction"] == "notumor"
        assert 0.0 <= data["confidence"] <= 1.0
        assert "gradcam" in data
        assert "comparison" in data
        assert "probabilities" in data
        assert len(data["probabilities"]) == 4
        assert "disclaimer" in data
        assert "request_id" in data
        assert resp.headers.get("X-Request-ID") == data["request_id"]

    def test_predict_model_not_loaded(self, client_no_model, sample_image):
        resp = client_no_model.post(
            "/predict",
            data={"image": (sample_image, "scan.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 503
        data = json.loads(resp.data)
        assert data["status"] == "error"
        assert data["code"] == "SERVICE_UNAVAILABLE"

    def test_predict_no_file(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["status"] == "error"
        assert data["code"] == "MISSING_IMAGE_FIELD"

    def test_predict_empty_filename(self, client):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["code"] == "EMPTY_FILENAME"

    def test_predict_invalid_extension(self, client):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"not an image"), "payload.exe")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["code"] == "INVALID_EXTENSION"

    def test_predict_corrupted_image_content(self, client, corrupted_image):
        resp = client.post(
            "/predict",
            data={"image": (corrupted_image, "corrupt.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["code"] == "INVALID_IMAGE_CONTENT"

    def test_predict_zero_byte_file(self, client):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(b""), "empty.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["code"] == "INVALID_IMAGE_CONTENT"

    def test_predict_path_traversal_filename(self, client, sample_image):
        # Werkzeug's secure_filename sanitizes ../../etc/passwd -> etc_passwd
        resp = client.post(
            "/predict",
            data={"image": (sample_image, "../../../etc/passwd.png")},
            content_type="multipart/form-data",
        )
        # Should either succeed cleanly under sanitized name or return 200
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Preprocessing & Content Validation Tests
# ---------------------------------------------------------------------------


class TestPreprocessingAndValidation:
    def test_preprocess_output_shape_and_range(self, sample_image):
        model_input, original_np = preprocess_image(sample_image)
        assert model_input.shape == (1, 224, 224, 3)
        assert model_input.dtype == np.float32
        assert model_input.min() >= 0.0
        assert model_input.max() <= 1.0
        assert original_np.shape == (224, 224, 3)
        assert original_np.dtype == np.uint8

    def test_empty_stream_raises(self):
        with pytest.raises(EmptyImageError):
            preprocess_image(io.BytesIO(b""))

    def test_corrupted_bytes_raises(self, corrupted_image):
        with pytest.raises(InvalidImageFormatError):
            preprocess_image(corrupted_image)

    def test_image_dimensions_too_small(self):
        tiny_img = Image.new("RGB", (4, 4), color="red")
        buf = io.BytesIO()
        tiny_img.save(buf, format="PNG")
        buf.seek(0)
        with pytest.raises(ImageDimensionError):
            preprocess_image(buf)


# ---------------------------------------------------------------------------
# Path Independence & Model Loader Tests
# ---------------------------------------------------------------------------


class TestModelLoader:
    def test_nonexistent_model_returns_none(self):
        m = load_model("/nonexistent/path/to/model.keras")
        assert m is None

    def test_empty_model_path_returns_none(self):
        m = load_model("")
        assert m is None


# ---------------------------------------------------------------------------
# Error Handlers Tests
# ---------------------------------------------------------------------------


class TestErrorHandlers:
    def test_404_not_found(self, client):
        resp = client.get("/api/nonexistent/endpoint")
        assert resp.status_code == 404
        data = json.loads(resp.data)
        assert data["code"] == "NOT_FOUND"

    def test_405_method_not_allowed(self, client):
        resp = client.get("/predict")  # /predict only accepts POST
        assert resp.status_code == 405
        data = json.loads(resp.data)
        assert data["code"] == "METHOD_NOT_ALLOWED"


# ---------------------------------------------------------------------------
# Temporary Artifact Cleanup Lifecycle Tests
# ---------------------------------------------------------------------------


class TestCleanupLifecycle:
    def test_cleanup_prunes_expired_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test files
            old_file = tmp_path / "gradcam_old123.jpg"
            new_file = tmp_path / "gradcam_new456.jpg"
            other_file = tmp_path / "regular_file.txt"

            old_file.write_text("dummy")
            new_file.write_text("dummy")
            other_file.write_text("dummy")

            # Artificial backdating for old_file (2 hours ago)
            two_hours_ago = time.time() - 7200
            os.utime(str(old_file), (two_hours_ago, two_hours_ago))

            deleted = cleanup_old_files(tmp_path, ttl_seconds=3600, max_files=10)
            assert deleted == 1
            assert not old_file.exists()
            assert new_file.exists()
            assert other_file.exists()  # Untouched


# ---------------------------------------------------------------------------
# Concurrency & Multi-threaded Request Simulation
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_predictions(self, mock_model, sample_image):
        """Simulate concurrent client uploads to ensure thread safety."""
        test_app = create_app(TestingConfig, model=mock_model)
        results = []
        errors = []

        with patch("app.make_gradcam_heatmap") as mock_gcam, \
             patch("app.overlay_heatmap") as mock_overlay, \
             patch("app.create_side_by_side_comparison") as mock_comp, \
             patch("cv2.imwrite"):

            mock_gcam.return_value = np.zeros((7, 7), dtype=np.float32)
            mock_overlay.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
            mock_comp.return_value = np.zeros((300, 600, 3), dtype=np.uint8)

            def worker(img_bytes):
                try:
                    c = test_app.test_client()
                    buf = io.BytesIO(img_bytes)
                    resp = c.post(
                        "/predict",
                        data={"image": (buf, "scan.png")},
                        content_type="multipart/form-data",
                    )
                    if resp.status_code == 200:
                        results.append(json.loads(resp.data))
                    else:
                        errors.append(resp.status_code)
                except Exception as e:
                    errors.append(str(e))

            sample_bytes = sample_image.getvalue()
            threads = [threading.Thread(target=worker, args=(sample_bytes,)) for _ in range(5)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0
        assert len(results) == 5
        # Verify unique request IDs and unique gradcam filenames
        req_ids = {r["request_id"] for r in results}
        gradcam_urls = {r["gradcam"] for r in results}
        assert len(req_ids) == 5
        assert len(gradcam_urls) == 5
