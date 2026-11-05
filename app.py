"""
NeuroMap — Brain Tumor Detection & Lesion Classification
Flask Backend Application with Production-Grade Architecture.

Features:
- Application factory pattern (create_app).
- CWD-independent path resolution for model weights and static assets.
- Structured request logging with request correlation IDs.
- Deep image validation and strict training/inference preprocessing parity.
- Grad-CAM explainability and side-by-side diagnostic composites.
- Automated lifecycle management / pruning of generated visualization artifacts.
- Hardened security: Path traversal prevention, upload limits, sanitized error responses.
- Liveness (/health) and readiness (/ready) probes.
"""

import datetime
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, g, jsonify, render_template, request
from werkzeug.utils import secure_filename

from config import Config, get_config
from gradcam import (
    create_side_by_side_comparison,
    make_gradcam_heatmap,
    overlay_heatmap,
)
from utils.cleanup import cleanup_old_files
from utils.preprocess import (
    ImageValidationError,
    preprocess_image,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("neuromap")


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------


def load_model(model_path: Optional[str] = None) -> Optional[tf.keras.Model]:
    """
    Load the Keras model using absolute path resolution independent of CWD.

    Args:
        model_path: Optional path to the .keras model file. If omitted,
            Config.MODEL_PATH is used.

    Returns:
        Compiled Keras model or None if file does not exist or fails to load.
    """
    if model_path is None:
        model_path = Config.MODEL_PATH

    if not model_path:
        logger.warning("No model path specified.")
        return None

    path_obj = Path(model_path)
    if not path_obj.is_absolute():
        path_obj = (Config.BASE_DIR / path_obj).resolve()

    if not path_obj.is_file():
        logger.warning(
            "Model file not found at '%s'. "
            "Please ensure weights are placed in the model/ directory. "
            "See README.md for instructions.",
            path_obj,
        )
        return None

    try:
        loaded_model = tf.keras.models.load_model(str(path_obj))
        logger.info("Model successfully loaded from '%s'.", path_obj)
        return loaded_model
    except Exception as e:
        logger.error("Failed to load model from '%s': %s", path_obj, e)
        return None


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


def create_app(config_class=None, model: Optional[tf.keras.Model] = None) -> Flask:
    """
    Create and configure the Flask application instance.

    Args:
        config_class: Configuration class to load. Defaults to get_config().
        model: Optional pre-loaded or mock Keras model instance.

    Returns:
        Configured Flask application instance.
    """
    if config_class is None:
        config_class = get_config()

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure upload directory exists
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Attach model to app instance
    if model is not None:
        app.model = model
    else:
        app.model = load_model(app.config.get("MODEL_PATH"))

    # -----------------------------------------------------------------------
    # Request Middleware: Correlation ID & Duration Logging
    # -----------------------------------------------------------------------

    @app.before_request
    def before_request():
        # Correlation ID from client or auto-generated
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        g.request_id = req_id
        g.start_time = time.perf_counter()

    @app.after_request
    def after_request(response):
        duration_ms = (time.perf_counter() - getattr(g, "start_time", time.perf_counter())) * 1000
        req_id = getattr(g, "request_id", "none")

        # Security headers
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        logger.info(
            "%s %s -> HTTP %d (%.2fms)",
            request.method,
            request.path,
            response.status_code,
            duration_ms,
            extra={"request_id": req_id},
        )
        return response

    # -----------------------------------------------------------------------
    # Helper: Extension validation
    # -----------------------------------------------------------------------

    def _is_allowed_extension(filename: str) -> bool:
        if "." not in filename:
            return False
        ext = filename.rsplit(".", 1)[1].lower()
        return ext in app.config["ALLOWED_EXTENSIONS"]

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.route("/")
    def home():
        """Serve the main web user interface."""
        return render_template("index.html")

    @app.route("/health")
    def health():
        """
        Liveness check endpoint.
        Returns HTTP 200 indicating the web server process is alive and responsive.
        """
        return jsonify({
            "status": "ok" if app.model is not None else "degraded",
            "service": "neuromap",
            "model_loaded": app.model is not None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }), 200

    @app.route("/ready")
    def ready():
        """
        Readiness check endpoint.
        Returns HTTP 200 if the model is loaded and ready to serve inference,
        or HTTP 503 if the model weights are unavailable.
        """
        is_ready = app.model is not None
        payload = {
            "status": "ready" if is_ready else "not_ready",
            "service": "neuromap",
            "model_loaded": is_ready,
            "model_path": app.config.get("MODEL_PATH", ""),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if is_ready:
            return jsonify(payload), 200
        payload["error"] = "Model weights are not loaded. Service is not ready for inference."
        return jsonify(payload), 503

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accept an MRI image upload, validate its format and dimensions,
        run model inference, generate Grad-CAM attribution, and return JSON.
        """
        req_id = getattr(g, "request_id", uuid.uuid4().hex[:16])

        # Guard: Model must be loaded
        if app.model is None:
            return jsonify({
                "status": "error",
                "error": "Model not loaded. Service is unavailable.",
                "code": "SERVICE_UNAVAILABLE",
                "request_id": req_id,
            }), 503

        # Guard: File field must exist
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "error": "No image file provided. Use form field 'image'.",
                "code": "MISSING_IMAGE_FIELD",
                "request_id": req_id,
            }), 400

        file = request.files["image"]

        # Guard: Non-empty filename
        if not file or file.filename.strip() == "":
            return jsonify({
                "status": "error",
                "error": "Empty filename provided.",
                "code": "EMPTY_FILENAME",
                "request_id": req_id,
            }), 400

        # Guard: Allowed filename extension
        safe_name = secure_filename(file.filename)
        if not _is_allowed_extension(safe_name):
            allowed = ", ".join(sorted(app.config["ALLOWED_EXTENSIONS"]))
            return jsonify({
                "status": "error",
                "error": f"Unsupported file extension. Allowed extensions: {allowed}",
                "code": "INVALID_EXTENSION",
                "request_id": req_id,
            }), 400

        # Guard & Preprocess: Validate file content & normalize
        try:
            img, original_img = preprocess_image(file.stream)
        except ImageValidationError as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "code": "INVALID_IMAGE_CONTENT",
                "request_id": req_id,
            }), 400
        except Exception as e:
            logger.error("Preprocessing failed unexpectedly: %s", e, extra={"request_id": req_id})
            return jsonify({
                "status": "error",
                "error": "Failed to process image content.",
                "code": "IMAGE_PROCESSING_FAILED",
                "request_id": req_id,
            }), 400

        # Inference
        try:
            pred = app.model.predict(img, verbose=0)
            if pred is None or np.isnan(pred).any() or np.isinf(pred).any():
                raise ValueError("Model produced invalid (NaN/Inf) prediction values.")

            idx = int(np.argmax(pred[0]))
            confidence = float(np.max(pred[0]))
            class_names = app.config["CLASS_NAMES"]

            probabilities = {
                class_names[i]: float(pred[0][i]) for i in range(len(class_names))
            }

            # Generate Grad-CAM attribution
            gradcam_layer = app.config.get("GRADCAM_LAYER", "auto")
            heatmap = make_gradcam_heatmap(img, app.model, gradcam_layer, pred_index=idx)
            overlay = overlay_heatmap(original_img, heatmap, alpha=0.55, return_rgb=False)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # Generate Side-by-Side comparison composite
            comparison = create_side_by_side_comparison(
                original_img,
                heatmap,
                class_names[idx],
                confidence,
                class_probabilities=probabilities,
                return_rgb=False,
            )

            # Persist visualizations with unique per-request UUIDs
            token = uuid.uuid4().hex[:12]
            gradcam_filename = f"gradcam_{token}.jpg"
            comparison_filename = f"comparison_{token}.jpg"

            upload_path = Path(app.config["UPLOAD_FOLDER"])
            gradcam_file_path = upload_path / gradcam_filename
            comparison_file_path = upload_path / comparison_filename

            cv2.imwrite(str(gradcam_file_path), overlay)
            cv2.imwrite(str(comparison_file_path), comparison)

            # Run opportunistic cleanup of expired artifacts
            cleanup_old_files(
                upload_path,
                ttl_seconds=app.config.get("TEMP_FILE_TTL_SECONDS", 3600),
                max_files=app.config.get("MAX_TEMP_FILES", 500),
            )

            # Unified response envelope
            return jsonify({
                "status": "success",
                "request_id": req_id,
                "prediction": class_names[idx],
                "confidence": confidence,
                "probabilities": probabilities,
                "gradcam": f"/static/gradcam/{gradcam_filename}",
                "comparison": f"/static/gradcam/{comparison_filename}",
                "disclaimer": (
                    "Model interpretability notice: Heatmaps highlight receptive features that "
                    "most influenced the model's classification decision. They do not represent "
                    "precise tumor boundaries, segmentation masks, or clinical pathology margins."
                ),
            }), 200

        except Exception as e:
            logger.exception("Prediction execution failed: %s", e, extra={"request_id": req_id})
            return jsonify({
                "status": "error",
                "error": "Prediction computation failed on server.",
                "code": "PREDICTION_FAILED",
                "request_id": req_id,
            }), 500

    # -----------------------------------------------------------------------
    # Standardized Error Handlers
    # -----------------------------------------------------------------------

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "status": "error",
            "error": "Bad request.",
            "code": "BAD_REQUEST",
            "request_id": getattr(g, "request_id", "none"),
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "status": "error",
            "error": "The requested resource was not found on this server.",
            "code": "NOT_FOUND",
            "request_id": getattr(g, "request_id", "none"),
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "status": "error",
            "error": "HTTP method not allowed for this endpoint.",
            "code": "METHOD_NOT_ALLOWED",
            "request_id": getattr(g, "request_id", "none"),
        }), 405

    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({
            "status": "error",
            "error": f"Uploaded file exceeds the maximum permitted size of {max_mb} MB.",
            "code": "PAYLOAD_TOO_LARGE",
            "request_id": getattr(g, "request_id", "none"),
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        # Prevent stack trace leakage to client
        return jsonify({
            "status": "error",
            "error": "An internal server error occurred.",
            "code": "INTERNAL_SERVER_ERROR",
            "request_id": getattr(g, "request_id", "none"),
        }), 500

    return app


# ---------------------------------------------------------------------------
# Default Global App Instance
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )