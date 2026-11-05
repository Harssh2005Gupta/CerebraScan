# NeuroMap Backend Refactoring & Engineering Report
**System**: NeuroMap Brain Tumor Detection & Lesion Classification  
**Role**: Senior Python Backend Engineer  
**Status**: Educational / Research Prototype (Not for Clinical Diagnostic Use)

---

## Executive Summary

The NeuroMap Flask backend has been refactored into a modular, production-grade, and testable application architecture. All 10 technical requirements have been implemented and verified with automated test suites:

1. **Path Independence**: Replaced all fragile relative paths with deterministic absolute paths anchored to `BASE_DIR = Path(__file__).resolve().parent`.
2. **Centralized Configuration (`config.py`)**: Unified configuration hierarchy (`Config`, `DevelopmentConfig`, `TestingConfig`, `ProductionConfig`) managing upload limits, allowed formats, dimension boundaries, model locations, and image normalization parameters.
3. **Deep Content Validation (`utils/preprocess.py`)**: Implemented stream-level verification with PIL header inspection, magic byte whitelisting, decompression bomb protection ($16 \times 16 \le \text{size} \le 8192 \times 8192$), and strict $[0.0, 1.0]$ float32 normalization.
4. **Lifecycle & Pruning (`utils/cleanup.py`)**: Built thread-safe garbage collection for generated Grad-CAM overlays and comparison composites with configurable TTL ($3600 \text{ s}$) and file volume ceilings ($500 \text{ files}$).
5. **Observability & Health Probes**: Added structured request duration logging with correlation IDs (`X-Request-ID`), a liveness probe (`/health`), and a readiness probe (`/ready`).
6. **Hardened Security**: Sealed stack trace leakage across 400, 404, 405, 413, 500, and 503 error handlers, applied security response headers, and mitigated path traversal attacks.
7. **Comprehensive Test Suite (`tests/test_app.py`)**: Expanded test coverage to 22 dedicated backend tests and 51 total repository tests, verifying concurrency, probe states, validation failures, and cleanup.

---

## 1. Architectural Changes

### Application Factory Pattern
Previously, `app.py` created a single monolithic global `app = Flask(__name__)` and eagerly attempted to load model weights at import time. This made unit testing fragile, required mocking global module attributes, and caused issues when running tests without physical model files on disk.

The refactored backend introduces `create_app(config_class=None, model=None)`:
```python
def create_app(config_class=None, model: Optional[tf.keras.Model] = None) -> Flask:
    if config_class is None:
        config_class = get_config()

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Attach model instance to app
    if model is not None:
        app.model = model
    else:
        app.model = load_model(app.config.get("MODEL_PATH"))
    ...
    return app
```
For backward compatibility with WSGI runners (`wsgi.py`) and legacy scripts, the global `app = create_app()` remains accessible at module scope.

---

## 2. Security Hardening & Input Validation

| Threat Vector | Previous Vulnerability | Refactored Mitigation |
| :--- | :--- | :--- |
| **Extension Spoofing** | Checked string `.rsplit('.', 1)[1]` only; executable uploaded with `.png` name would bypass check. | Validates PIL magic bytes and header format (`PNG`, `JPEG`, `WEBP`) via `raw_img.verify()`. |
| **Decompression Bomb (DoS)** | Unbounded image dimensions could exhaust server memory during decompression. | Dimension bounds checking enforces $16 \le W, H \le 8192$. |
| **Disk Space Exhaustion** | Every request wrote unpruned files to `static/gradcam/`. | Thread-safe `cleanup_old_files()` prunes files older than TTL (1 hr) or exceeding count (500). |
| **Path Traversal Attacks** | Direct string interpolation into filenames. | Filename passed through `secure_filename()` and resolved strictly inside `UPLOAD_FOLDER`. |
| **Information Exposure** | Default Flask exception handlers could expose Python stack traces. | Error handlers for 400, 404, 405, 413, 500, 503 return standardized, sanitized JSON envelopes. |
| **Clickjacking / MIME Sniffing** | No HTTP security headers. | Injected `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`. |

---

## 3. Standardized Response Envelopes

### Success Payload (`POST /predict` - HTTP 200)
Maintains 100% backward compatibility with `static/js/app.js` while introducing standard metadata:
```json
{
  "status": "success",
  "request_id": "9a1e0b5c4f2d7e81",
  "prediction": "glioma",
  "confidence": 0.9841,
  "probabilities": {
    "glioma": 0.9841,
    "meningioma": 0.0105,
    "notumor": 0.0028,
    "pituitary": 0.0026
  },
  "gradcam": "/static/gradcam/gradcam_9a1e0b5c4f2d.jpg",
  "comparison": "/static/gradcam/comparison_9a1e0b5c4f2d.jpg",
  "disclaimer": "Model interpretability notice: Heatmaps highlight receptive features that most influenced the model's classification decision. They do not represent precise tumor boundaries, segmentation masks, or clinical pathology margins."
}
```

### Error Payload (HTTP 400 / 413 / 500 / 503)
```json
{
  "status": "error",
  "error": "Uploaded image stream contains 0 bytes.",
  "code": "INVALID_IMAGE_CONTENT",
  "request_id": "9a1e0b5c4f2d7e81"
}
```

---

## 4. Test Suite Execution Results

All 51 automated tests across the repository pass with zero errors:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2
rootdir: C:\Users\Harsh Gupta\Downloads\NeuroMap---Brain-Tumor-Detection-Classification-main\NeuroMap---Brain-Tumor-Detection-Classification-main
collected 51 items

tests\test_app.py ......................                                 [ 43%]
tests\test_medical_evaluation.py ......                                  [ 54%]
tests\test_training.py .....                                             [ 64%]
tests\test_xai.py ..................                                     [100%]

============================= 51 passed in 21.21s =============================
```

### Key Test Coverage Highlights (`tests/test_app.py`)
- `TestHealthAndReadiness`: Liveness and readiness probes under both healthy and degraded (model absent) conditions.
- `TestPredictRoute`: Valid predictions, missing files, empty filenames, unsupported extensions, corrupted image content, zero-byte streams, and path traversal attempts.
- `TestPreprocessingAndValidation`: Output tensor shape `(1, 224, 224, 3)`, float32 normalization in $[0.0, 1.0]$, and dimension bounds checks.
- `TestModelLoader`: CWD path independence and missing-file handling.
- `TestCleanupLifecycle`: Automated pruning of expired temporary visualization artifacts.
- `TestConcurrency`: Multi-threaded client upload simulation verifying thread safety and unique correlation IDs.
