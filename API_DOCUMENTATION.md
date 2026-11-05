# NeuroMap Backend Architecture & API Specification
**Version**: 2.1 (Refactored Backend & Inference Engine)  
**Status**: Educational / Research Prototype (Not for Clinical Diagnostic Use)

---

## 1. Architectural Overview

The NeuroMap backend is built with Python and Flask, implementing enterprise application patterns designed for maintainability, testability, and deterministic behavior:

- **Application Factory Pattern (`create_app`)**: Allows explicit injection of configuration objects (`DevelopmentConfig`, `TestingConfig`, `ProductionConfig`) and mock models for isolated, deterministic unit and integration testing without disk dependencies.
- **Working-Directory Independence**: All asset resolution, configuration paths, and model weight paths derive from `Path(__file__).resolve().parent`, guaranteeing correct execution regardless of the shell invocation directory.
- **Strict Parity with Training Preprocessing**: Preprocessing parameters (RGB conversion, bilinear resize to $224 \times 224$, scaling to $[0.0, 1.0]$ float32) are centralized in `config.py` and strictly synchronized with the training pipeline in `training/dataset.py`.
- **Request Observability & Correlation**: Middleware assigns a unique correlation ID (`X-Request-ID`) to every incoming HTTP request, logging execution duration, client IP, method, route, and status code in a structured format.
- **Automated Lifecycle Management**: Generated explainability assets (Grad-CAM overlays and comparison composites) in `static/gradcam/` are tracked with per-request UUID tokens and pruned periodically by a thread-safe cleanup utility.

---

## 2. API Endpoints

### 2.1 Web User Interface
- **Route**: `GET /`
- **Description**: Serves the primary web interface for scan loading, dual-frame inspection, and differential diagnosis display.
- **Response**: `text/html` (HTTP 200)

---

### 2.2 Liveness Probe
- **Route**: `GET /health`
- **Description**: Lightweight health-check endpoint for load balancers and container orchestrators (Kubernetes liveness probe).
- **Response**: `application/json` (HTTP 200)
```json
{
  "status": "ok",
  "service": "neuromap",
  "model_loaded": true,
  "timestamp": "2026-10-24T17:35:00.123456+00:00"
}
```

---

### 2.3 Readiness Probe
- **Route**: `GET /ready`
- **Description**: Readiness check verifying whether deep learning model weights are loaded and ready to accept inference requests.
- **Responses**:
  - **Ready (HTTP 200)**:
    ```json
    {
      "status": "ready",
      "service": "neuromap",
      "model_loaded": true,
      "model_path": "c:/.../model/best_model.keras",
      "timestamp": "2026-10-24T17:35:00.123456+00:00"
    }
    ```
  - **Not Ready (HTTP 503)**:
    ```json
    {
      "status": "not_ready",
      "service": "neuromap",
      "model_loaded": false,
      "error": "Model weights are not loaded. Service is not ready for inference.",
      "timestamp": "2026-10-24T17:35:00.123456+00:00"
    }
    ```

---

### 2.4 Lesion Classification & Explainability
- **Route**: `POST /predict`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` *(required, binary file)*: Brain MRI scan in JPEG, PNG, or WebP format.

#### Success Response (HTTP 200)
```json
{
  "status": "success",
  "request_id": "8f3b204918e74a12",
  "prediction": "glioma",
  "confidence": 0.9841,
  "probabilities": {
    "glioma": 0.9841,
    "meningioma": 0.0105,
    "notumor": 0.0028,
    "pituitary": 0.0026
  },
  "gradcam": "/static/gradcam/gradcam_8f3b204918e7.jpg",
  "comparison": "/static/gradcam/comparison_8f3b204918e7.jpg",
  "disclaimer": "Model interpretability notice: Heatmaps highlight receptive features that most influenced the model's classification decision. They do not represent precise tumor boundaries, segmentation masks, or clinical pathology margins."
}
```

#### Error Responses

| Scenario | HTTP Status | Response Payload Example |
| :--- | :---: | :--- |
| **Model Not Loaded** | `503` | `{"status": "error", "error": "Model not loaded. Service is unavailable.", "code": "SERVICE_UNAVAILABLE", "request_id": "..."}` |
| **Missing Image Field** | `400` | `{"status": "error", "error": "No image file provided. Use form field 'image'.", "code": "MISSING_IMAGE_FIELD", "request_id": "..."}` |
| **Empty Filename** | `400` | `{"status": "error", "error": "Empty filename provided.", "code": "EMPTY_FILENAME", "request_id": "..."}` |
| **Unsupported Extension** | `400` | `{"status": "error", "error": "Unsupported file extension. Allowed extensions: jpeg, jpg, png, webp", "code": "INVALID_EXTENSION", "request_id": "..."}` |
| **Corrupted Image Content** | `400` | `{"status": "error", "error": "File content is corrupted or not a recognized image...", "code": "INVALID_IMAGE_CONTENT", "request_id": "..."}` |
| **Zero-Byte Upload** | `400` | `{"status": "error", "error": "Uploaded image stream contains 0 bytes.", "code": "INVALID_IMAGE_CONTENT", "request_id": "..."}` |
| **Dimensions Out of Bounds** | `400` | `{"status": "error", "error": "Image dimensions (4x4) are too small...", "code": "INVALID_IMAGE_CONTENT", "request_id": "..."}` |
| **Payload Too Large** | `413` | `{"status": "error", "error": "Uploaded file exceeds the maximum permitted size of 16 MB.", "code": "PAYLOAD_TOO_LARGE", "request_id": "..."}` |
| **Server Computation Fault** | `500` | `{"status": "error", "error": "Prediction computation failed on server.", "code": "PREDICTION_FAILED", "request_id": "..."}` |

---

## 3. Security & Validation Controls

1. **Content-Aware Image Verification**:
   - Validation bypasses superficial filename extensions. Stream headers are verified against PIL magic bytes (`PNG`, `JPEG`, `WEBP`).
   - Binary streams undergo `img.verify()` to catch malformed structures or embedded scripts.
2. **Decompression Bomb Protection**:
   - Minimum dimensions: $16 \times 16$ pixels (prevents zero/micro-dimension compute waste).
   - Maximum dimensions: $8192 \times 8192$ pixels (mitigates memory-exhaustion zip/image bomb attacks).
3. **Hardened HTTP Response Headers**:
   - `X-Content-Type-Options: nosniff`
   - `X-Frame-Options: DENY`
   - `X-XSS-Protection: 1; mode=block`
   - `X-Request-ID: <uuid>`
4. **Information Leakage Prevention**:
   - Client responses in 500 error scenarios suppress Python stack traces, internal paths, and framework versions.
5. **Path Traversal Mitigation**:
   - Uploaded filenames pass through `werkzeug.utils.secure_filename`.
   - Output paths are strictly constructed via `Path(Config.UPLOAD_FOLDER) / filename`.

---

## 4. Configuration Reference

All settings can be overridden via environment variables:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `FLASK_ENV` | `production` | Active environment (`development`, `testing`, `production`) |
| `NEUROMAP_SECRET_KEY` | `dev-key` | Session and CSRF cryptographic signing key |
| `NEUROMAP_MODEL_PATH` | `<BASE_DIR>/model/best_model.keras` | Absolute or relative path to trained Keras model |
| `NEUROMAP_GRADCAM_LAYER` | `auto` | Target conv layer name, or `auto` for dynamic inspection |
| `NEUROMAP_MAX_CONTENT_LENGTH`| `16777216` (16 MB) | Maximum upload payload size in bytes |
| `NEUROMAP_UPLOAD_DIR` | `<BASE_DIR>/static/gradcam` | Filesystem destination for generated heatmaps |
| `NEUROMAP_TEMP_TTL_SECONDS` | `3600` (1 hour) | Retention TTL for generated heatmap artifacts |
| `NEUROMAP_MAX_TEMP_FILES` | `500` | Maximum artifact threshold before forced FIFO purge |
| `NEUROMAP_LOG_LEVEL` | `INFO` | Logging threshold (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 5. Verification & Testing

The backend is verified by an automated test suite located in `tests/test_app.py`:
- **Probe Coverage**: Validates liveness (`/health`) and readiness (`/ready`) across both model-loaded and model-unloaded conditions.
- **Input Validation**: Tests empty streams, malformed bytes, disallowed extensions, directory traversal attempts, and dimension bounds.
- **Concurrency Test**: Simulates concurrent multi-threaded inference requests, verifying thread safety and unique per-request correlation tokens.
- **Pruning Test**: Verifies FIFO and TTL garbage collection of expired visual artifacts.

Full suite status: **51 tests passed, 0 failures**.
