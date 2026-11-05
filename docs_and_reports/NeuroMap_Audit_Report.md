# NeuroMap — Full Codebase Audit Report

> **Date**: 2026-10-23 · **Auditor**: Senior AI/ML Engineer (Automated)
> **Scope**: All source files, notebook, model artifacts, API, frontend, dependencies, documentation

---

## 1. Architecture & End-to-End Data Flow

```mermaid
flowchart LR
    A["Browser\n(index.html + app.js)"] -->|POST /predict\nMultipart image| B["Flask API\n(app.py)"]
    B -->|preprocess_image()| C["utils/preprocess.py\nResize 224×224, /255"]
    C -->|model_input (1,224,224,3)| D["TF/Keras Model\n(model/best_model .keras.keras)"]
    D -->|softmax logits| B
    B -->|activation_17| E["gradcam.py\nmake_gradcam_heatmap()"]
    E -->|overlay → gradcam.jpg| B
    B -->|JSON response| A
```

**Files in Project** (10 source files):

| File | Size | Role |
|------|------|------|
| `app.py` | 1.9 KB | Flask server, `/predict` route, model loading |
| `gradcam.py` | 1.9 KB | Grad-CAM heatmap generation + overlay |
| `utils/preprocess.py` | 364 B | Image loading, resize, normalize |
| `tumor-detector.ipynb` | 302 KB | Full training pipeline (Kaggle) |
| `templates/index.html` | 4.7 KB | SPA frontend template |
| `static/js/app.js` | 13.5 KB | Frontend logic (drag-drop, fetch, render) |
| `static/css/style.css` | 11.9 KB | Dark-themed UI styling |
| `requirements.txt` | 44 B | Dependencies (unpinned) |
| `README.md` | 2.0 KB | Basic documentation |
| `.gitignore` | 58 B | Excludes model/, uploads/, .keras |

---

## 2. Model Architecture

### Confirmed from notebook outputs:

- **Architecture**: Custom ResNet-like CNN with pre-activation residual blocks
- **Input**: `(None, 224, 224, 3)` — RGB, float32, normalized [0, 1]
- **Output**: `(None, 4)` with **softmax** activation
- **Total Parameters**: ~8.9M (extracted from model summary)
- **Stages**: Stem (7×7 Conv, stride 2) → MaxPool → 4 residual stages (64→128→256→512) → GAP → Dropout(0.4) → Dense(256, relu) → Dropout(0.3) → Dense(4, softmax)
- **Regularization**: L2(1e-4) on all Conv/Dense, BatchNorm throughout, Dropout (0.4 + 0.3), label smoothing (0.1)
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Loss**: CategoricalCrossentropy with label_smoothing=0.1

### Class Ordering (from `sorted(os.listdir(TRAIN_PATH))`):
```
["glioma", "meningioma", "notumor", "pituitary"]  → indices 0, 1, 2, 3
```

> [!IMPORTANT]
> The class ordering in `app.py` **matches** the training ordering: `["glioma", "meningioma", "notumor", "pituitary"]`. This is correct because training used `sorted(os.listdir())` which produces alphabetical order.

---

## 3. Dataset, Splits & Augmentation

### Dataset
- **Source**: Kaggle — `masoudnickparvar/brain-tumor-mri-dataset`
- **License**: Not documented in README. The Kaggle dataset is commonly CC0/open, but **no LICENSE file** in the repo.
- **Path**: `/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset/`

### Splits (from notebook output)
| Split | Count | Source |
|-------|-------|--------|
| Train | 4,760 | 85% of Training dir |
| Validation | 840 | 15% of Training dir (stratified, seed=42) |
| Test | 1,600 | Separate Testing dir (400 per class) |

### Class Balance
- **Training/Val**: Stratified split; Class weights computed but all equal `{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}` — classes are balanced.
- **Test**: 400 per class — perfectly balanced.

### Augmentation (train only)
- Random horizontal flip, vertical flip
- Random brightness (±0.15), contrast (0.85–1.15), saturation (0.85–1.15)
- Random rotation (±10%), random zoom (±10%), reflect fill
- `tf.clip_by_value(0.0, 1.0)` after augmentation

### Data Leakage Assessment
> [!NOTE]
> **No data leakage detected.** The training and test sets come from separate directories (`Training/` and `Testing/`) in the source dataset. The validation set is split from training only. However, the original Kaggle dataset itself may contain near-duplicate MRI images across its train/test split — this is a known issue with this dataset that the project cannot control.

---

## 4. Training Metrics

### Training Summary
- **Epochs**: 45/60 (early stopping triggered at epoch 45, patience=15)
- **Best checkpoint**: Epoch 30 (val_accuracy = 0.95714)
- **LR schedule**: ReduceLROnPlateau (factor=0.5, patience=5), reduced at epoch 14 (3e-4→1.5e-4) and epoch 44 (1.5e-4→7.5e-5)

### Final Evaluation (Test Set)
| Metric | Value |
|--------|-------|
| **Test Accuracy** | **0.9175** |
| **Test Loss** | **0.7116** |

### Per-Class Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| glioma | 0.97 | 0.80 | 0.87 | 400 |
| meningioma | 0.93 | 0.89 | 0.91 | 400 |
| notumor | 0.88 | 0.99 | 0.93 | 400 |
| pituitary | 0.91 | 0.99 | 0.95 | 400 |
| **macro avg** | **0.92** | **0.92** | **0.92** | 1600 |

### Observations
- **Val accuracy plateau noted**: Best val_accuracy (0.957) significantly higher than test accuracy (0.918) — ~4% gap suggests mild overfitting or distribution shift.
- **Glioma recall is weakest** (0.80) — 20% of gliomas misclassified.
- **Val accuracy volatility**: Large swings observed (0.88→0.96→0.79) indicating instability, possibly from batch-norm sensitivity or small val set.

### Reproducibility
> [!WARNING]
> **Not fully reproducible.** While `random_state=42` is set for train/val split, there is no `tf.random.set_seed()` call, no `PYTHONHASHSEED`, and augmentation uses non-deterministic TF ops. GPU non-determinism adds further variance.

---

## 5. Critical Bugs & Missing Artifacts

### 🔴 CRITICAL — App Cannot Start

| # | Severity | File | Issue |
|---|----------|------|-------|
| **B1** | 🔴 CRITICAL | [app.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L15) | **Model path has typo**: `"model/best_model .keras.keras"` — contains a spurious space and double `.keras` extension. The notebook saves to `/kaggle/working/best_model.keras`. |
| **B2** | 🔴 CRITICAL | `model/` directory | **Model file is missing.** The `model/` directory does not exist locally. `.gitignore` excludes `model/` and `*.keras`. No download instructions, no LFS setup, no fallback. |

### 🟠 HIGH — Runtime Crashes

| # | Severity | File | Issue |
|---|----------|------|-------|
| **B3** | 🟠 HIGH | [app.py#L26](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L26) | **No error handling for missing/invalid upload.** `request.files["image"]` raises `KeyError` (500) if no file is sent. No file-type validation server-side. |
| **B4** | 🟠 HIGH | [app.py#L40](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L40) | **Hardcoded Grad-CAM layer name `"activation_17"`** — correct for this specific model (confirmed: `activation_17` has shape `(None, 7, 7, 512)` = last ReLU before GAP), but fragile if model changes. |
| **B5** | 🟠 HIGH | [notebook Cell 0](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/tumor-detector.ipynb) | **Dead code in notebook**: First `model.compile()` uses `SparseCategoricalCrossentropy()` with a **syntax error** (extra `)` and `,`). It gets overridden by the second compile with `CategoricalCrossentropy`. The code runs only because Python executes sequentially. |

### 🟡 MEDIUM — Functional Issues

| # | Severity | File | Issue |
|---|----------|------|-------|
| **B6** | 🟡 MEDIUM | [app.py#L56-L60](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L56-L60) | **Concurrency race condition**: All requests overwrite the same `static/gradcam/gradcam.jpg`. Under concurrent requests, user A may see user B's heatmap. |
| **B7** | 🟡 MEDIUM | [app.py#L12-L13](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L12-L13) | **UPLOAD_FOLDER created but never used.** Uploaded files are never saved to disk (good for memory, but the folder creation is dead code). |
| **B8** | 🟡 MEDIUM | [app.py#L30-L31](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/app.py#L30-L31), [gradcam.py#L25-L26](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/gradcam.py#L25-L26) | **Debug print statements in production code.** 6 `print()` calls for debug purposes left in `app.py` and `gradcam.py`. |
| **B9** | 🟡 MEDIUM | [app.js#L1-L217](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/static/js/app.js#L1-L217) | **217 lines of commented-out code** — entire duplicate of the active code. Clutters file. |

---

## 6. Grad-CAM Analysis

### Implementation Assessment

| Aspect | Status | Detail |
|--------|--------|--------|
| Layer selection | ✅ Correct | `activation_17` = last ReLU before GAP (7×7×512). Confirmed in model summary. |
| Gradient computation | ✅ Correct | Uses `GradientTape`, computes gradients of predicted class w.r.t. conv outputs |
| Pooling | ✅ Correct | Global average pooling of gradients (`reduce_mean` over spatial dims) |
| Heatmap generation | ✅ Correct | Weighted sum → ReLU → normalize to [0,1] |
| Division-by-zero guard | ✅ Present | Returns zero heatmap if `max_val == 0` |
| Color mapping | ✅ Correct | JET colormap, proper RGB→BGR conversion for overlay |
| Overlay blending | ✅ Correct | `addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)` |

### Potential Issues
| Issue | Severity | Detail |
|-------|----------|--------|
| Float equality check | 🟡 Low | `if max_val == 0` — should use `tf.less(max_val, 1e-10)` for numerical stability |
| Fixed filename | 🟡 Medium | Overwrites same file (see B6 above) |
| Resolution loss | ℹ️ Info | Heatmap is 7×7 upscaled to 224×224 — expected for this architecture depth |

---

## 7. API Reliability & Security

| # | Category | Finding | Severity |
|---|----------|---------|----------|
| S1 | **Input validation** | No file extension/MIME-type check. Any file accepted. | 🟠 HIGH |
| S2 | **File size limit** | No `MAX_CONTENT_LENGTH` set on Flask app. Unlimited upload possible → DoS vector. | 🟠 HIGH |
| S3 | **Error responses** | No try/except around prediction pipeline. Exceptions return raw 500 with tracebacks. | 🟠 HIGH |
| S4 | **CORS** | No CORS headers set. API only usable from same-origin. | ℹ️ Info |
| S5 | **Rate limiting** | No rate limiting. Model inference is GPU/CPU-heavy (~200ms+). Easy to DoS. | 🟡 MEDIUM |
| S6 | **CSRF** | No CSRF protection on POST endpoint. | 🟡 MEDIUM |
| S7 | **Concurrency** | Flask dev server is single-threaded. No production WSGI server (gunicorn/waitress). | 🟠 HIGH |
| S8 | **Model loading** | Model loaded at module level — cold start blocks all requests. No health check endpoint. | 🟡 MEDIUM |

---

## 8. Frontend Assessment

### Functionality
| Aspect | Status | Detail |
|--------|--------|--------|
| File upload (click) | ✅ Works | Click triggers hidden file input |
| Drag & drop | ✅ Works | Proper drag event handling with visual feedback |
| Keyboard accessible | ✅ Works | Dropzone has `tabindex="0"`, `role="button"`, Enter/Space handling |
| Prediction request | ✅ Works | Fetch API, FormData, proper error handling |
| Result rendering | ✅ Works | Animated gauge, confidence %, class name, Grad-CAM image |
| Cache busting | ✅ Present | `?t=${Date.now()}` appended to gradcam URL |
| Loading state | ✅ Works | Button disabled + spinner during prediction |
| Error display | ✅ Works | Error message rendered in panel |

### Issues
| # | Category | Finding | Severity |
|---|----------|---------|----------|
| F1 | **False claim** | HTML says "Accepted formats: JPG, PNG, DICOM-export" but no DICOM support exists anywhere. | 🟡 MEDIUM |
| F2 | **Dead code** | 217 lines of commented-out JavaScript (lines 1–217) | 🟡 LOW |
| F3 | **XSS** | `escapeHTML()` function properly sanitizes output → no XSS risk | ✅ Safe |
| F4 | **a11y** | ARIA labels present. `prefers-reduced-motion` media query respected. Missing `alt` text for Grad-CAM context. | 🟡 LOW |
| F5 | **Responsiveness** | Mobile breakpoint at 720px. Grid collapses to single column. | ✅ Good |
| F6 | **Disclaimer** | Footer includes "Outputs are not a clinical diagnosis" | ✅ Good |

---

## 9. Dependencies & Environment

### requirements.txt Analysis
```
flask           # unpinned
tensorflow      # unpinned — TF 2.x vs 3.x breaking changes
numpy           # unpinned — numpy 2.x incompatible with older TF
opencv-python   # unpinned
pillow          # unpinned
```

| Issue | Severity | Detail |
|-------|----------|--------|
| **No version pins** | 🟠 HIGH | Any `pip install` may pull incompatible versions. TF/NumPy version mismatch is a common cause of silent model corruption. |
| **Missing `scikit-learn`** | 🟡 MEDIUM | Used in notebook for metrics but not in requirements (not needed for inference). |
| **Missing `seaborn`** | 🟡 LOW | Used in notebook only. |
| **Missing `gunicorn`/`waitress`** | 🟡 MEDIUM | No production WSGI server listed. |
| **No `python_requires`** | ℹ️ Info | No Python version specified. |

---

## 10. Testing, Deployment & Documentation Gaps

### Testing
- ❌ **Zero test files.** No unit tests, integration tests, or model validation scripts.
- ❌ No `pytest`, `unittest`, or any test framework referenced.
- ❌ No CI/CD configuration (GitHub Actions, etc.).

### Deployment
- ❌ No `Dockerfile`, `docker-compose.yml`, or cloud deployment config.
- ❌ No `.env` or environment variable handling.
- ❌ No `Procfile` or `gunicorn` configuration.
- ❌ No model download/provision script.
- ⚠️ `host="0.0.0.0"` exposes to all network interfaces (intentional for containers, risky otherwise).

### Documentation
- ❌ No architecture diagrams.
- ❌ No model card (dataset, metrics, limitations, intended use).
- ❌ No API schema documentation (OpenAPI/Swagger).
- ❌ No `CONTRIBUTING.md` or `CHANGELOG.md`.
- ❌ No license file (README says "educational and research purposes" but no formal license).
- ❌ Model download instructions say "use external storage or LFS" but no actual link or setup.

---

## Summary of All Findings

| Severity | Count | Key Items |
|----------|-------|-----------|
| 🔴 **CRITICAL** | 2 | Model path typo (B1), Model file missing (B2) |
| 🟠 **HIGH** | 6 | No upload validation (B3), No error handling (S3), No file size limit (S2), No version pins (deps), No WSGI server (S7), Input validation (S1) |
| 🟡 **MEDIUM** | 7 | Concurrency race (B6), Debug prints (B8), False DICOM claim (F1), Rate limiting (S5), CSRF (S6), Dead code (B7/B9/F2), Model loading (S8) |
| ℹ️ **INFO** | 4 | CORS, Python version, Grad-CAM resolution, Dead code |

---

## Proposed Implementation Plan

> [!CAUTION]
> **Before any changes**: Create a Git checkpoint.
> ```bash
> git init  # if not already
> git add -A
> git commit -m "checkpoint: pre-audit baseline"
> ```

### Phase 1 — App-Breaking Fixes (Day 1)
**Goal**: Make the app startable and functional.

| Task | Files | Detail | Acceptance Criteria |
|------|-------|--------|---------------------|
| 1.1 Fix model path | `app.py` | Correct `"model/best_model .keras.keras"` → `"model/best_model.keras"` | Model loads without error |
| 1.2 Model provision | `README.md`, new `scripts/download_model.py` or docs | Add model download instructions + provision script | User can obtain and place model in `model/` |
| 1.3 Pin dependencies | `requirements.txt` | Add version pins for all packages (e.g., `tensorflow>=2.15,<3.0`) | `pip install -r requirements.txt` reproducible |
| 1.4 Basic error handling | `app.py` | try/except around `/predict`, validate file presence, validate image type | Invalid uploads return 400 JSON, not 500 traceback |

**Risk**: Model file is ~35MB+ and needs external hosting (Google Drive, HuggingFace Hub, GitHub Releases).

### Phase 2 — Security & Reliability (Day 2)
**Goal**: Production-hardened API.

| Task | Files | Detail | Acceptance Criteria |
|------|-------|--------|---------------------|
| 2.1 File size limit | `app.py` | `app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024` (16MB) | >16MB upload returns 413 |
| 2.2 File type validation | `app.py`, `utils/preprocess.py` | Server-side MIME + extension check, reject non-images | Non-image files return 400 |
| 2.3 Unique Grad-CAM paths | `app.py` | UUID-based filenames: `gradcam_{uuid}.jpg` | Concurrent requests don't collide |
| 2.4 Remove debug prints | `app.py`, `gradcam.py` | Replace with `logging` module or remove | No print() calls in production code |
| 2.5 WSGI server | `requirements.txt`, new `wsgi.py` | Add `waitress` (Windows) or `gunicorn` (Linux) | App serves via production server |

**Dependency**: Phase 1 complete.

### Phase 3 — Code Quality & UX (Day 3)
**Goal**: Clean code, accurate UI.

| Task | Files | Detail | Acceptance Criteria |
|------|-------|--------|---------------------|
| 3.1 Remove dead code | `app.js` | Delete lines 1–217 (commented-out code) | No commented-out blocks |
| 3.2 Remove dead code | `app.py` | Remove unused `UPLOAD_FOLDER` | No unused variables |
| 3.3 Fix DICOM claim | `index.html` | Change "JPG, PNG, DICOM-export" → "JPG, PNG" | UI matches capability |
| 3.4 Clean notebook | `tumor-detector.ipynb` | Fix dead first compile, remove syntax error | Notebook runs cleanly |
| 3.5 Dynamic Grad-CAM layer | `gradcam.py`, `app.py` | Auto-detect last conv layer from model | Layer name not hardcoded |

**Risk**: Minimal, cosmetic changes.

### Phase 4 — Testing & Documentation (Day 4–5)
**Goal**: Reproducible, documented, testable.

| Task | Files | Detail | Acceptance Criteria |
|------|-------|--------|---------------------|
| 4.1 Unit tests | New `tests/` directory | Test preprocessing, prediction shape, API routes | `pytest` passes |
| 4.2 Model card | New `MODEL_CARD.md` | Dataset, architecture, metrics, limitations, biases | Document exists and is accurate |
| 4.3 API docs | `README.md` or new docs | Request/response schema, error codes | Complete API reference |
| 4.4 Docker setup | `Dockerfile`, `docker-compose.yml` | Containerized deployment | `docker compose up` works |
| 4.5 Reproducibility | `tumor-detector.ipynb` | Add `tf.random.set_seed(42)`, pin TF version | Training is deterministic |
| 4.6 License file | `LICENSE` | Add appropriate open-source license | License file present |
| 4.7 Add health endpoint | `app.py` | `GET /health` returns model status | Endpoint returns 200 |

**Risk**: Docker setup requires testing across environments.

### Phase 5 — Model Improvements (Future)
**Goal**: Better accuracy and interpretability.

| Task | Detail | Expected Impact |
|------|--------|-----------------|
| 5.1 Address glioma recall gap | Analyze glioma misclassifications, try focal loss or oversampling | Glioma F1 > 0.90 |
| 5.2 Test-time augmentation (TTA) | Average predictions over augmented copies | +1-2% accuracy |
| 5.3 Transfer learning baseline | Compare against ResNet50/EfficientNet pretrained | Potential +3-5% accuracy |
| 5.4 Confidence calibration | Temperature scaling or Platt scaling | Calibrated probabilities |

---

> [!IMPORTANT]
> **Awaiting your approval before proceeding with any changes.** Which phases should I implement, and in what order? I recommend starting with Phase 1 immediately since the app cannot start at all without the model path fix.
