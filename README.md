#  Brain Tumor Detection & Classification

[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Research & Educational Prototype**: NeuroMap is an experimental computer vision and explainable AI (XAI) demonstration system. It is **NOT** a medical diagnostic device, is **NOT** FDA-cleared or CE-marked, and must **NEVER** be used for clinical decision-making.

---

## Architecture Overview

```mermaid
flowchart TD
    Client["Browser / REST Client"] -->|Multipart POST /predict| FlaskApp["Flask Application Factory (create_app)"]
    
    subgraph Security & Validation Pipeline
        FlaskApp --> ValidateSize["Payload Size Guard (MAX_CONTENT_LENGTH 16MB)"]
        ValidateSize --> DeepValidation["Deep Content Validation (PIL verify + format whitelist)"]
        DeepValidation --> DimensionGuard["Dimension Bounds Check (16x16 to 8192x8192)"]
    end

    subgraph Deep Learning & Inference
        DimensionGuard --> Preprocess["Preprocess: RGB 224x224, Normalized [0.0, 1.0]"]
        Preprocess --> TFModel["TensorFlow / Keras CNN Model (ResNet Baseline / Transfer)"]
        TFModel --> Softmax["Softmax Probability Distribution (4 Classes)"]
    end

    subgraph Explainable AI (XAI) Engine
        TFModel --> DynamicDetection["Recursive Layer Search (find_last_conv_layer)"]
        DynamicDetection --> GradCAM["Grad-CAM (Log-Gradient Stabilization)"]
        DynamicDetection --> IntGrad["Integrated Gradients (Path-Integral Attributions)"]
        GradCAM --> OverlayEngine["Color-Mapped Heatmap Overlay (Jet Colormap)"]
    end

    subgraph Lifecycle & Storage
        OverlayEngine --> PruneOld["Automated File Pruning (cleanup_old_files TTL)"]
        PruneOld --> StaticStorage["Ephemeral Storage: static/gradcam/"]
    end

    Softmax --> ResponseAssembler["JSON API Response Builder"]
    StaticStorage --> ResponseAssembler
    ResponseAssembler -->|JSON: class, confidence, gradcam URI, timing| Client
```

---

## Key Features

- **Multi-Class Classification**: Identifies **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** from 2D brain MRI slices.
- **Dynamic Explainability (XAI)**:
  - **Grad-CAM**: Gradient-weighted Class Activation Mapping with log-probability gradient stabilization and dynamic layer discovery across nested architectures.
  - **Integrated Gradients**: Axiomatic path-integral attribution method comparing pixel importance against a black baseline.
- **Production-Ready Application Factory**:
  - Centralized configuration hierarchy (`Config`, `DevelopmentConfig`, `TestingConfig`, `ProductionConfig`).
  - Strict input sanitization preventing path traversal, zip bombs, and payload exhaustion.
  - Ephemeral file lifecycle management with configurable TTL and volume quotas.
  - Kubernetes-compatible observability probes: `/health` (liveness) and `/ready` (readiness).
- **Modern Research Dashboard**:
  - Accessible, responsive UI (desktop, tablet, mobile) with drag-and-drop file upload.
  - Side-by-side comparative inspection of original MRI scans and Grad-CAM heatmaps.
  - Real-time client-side latency timing and composite image download.

---

## Dataset Provenance & Licensing

NeuroMap was evaluated using the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) curated by Masoud Nickparvar, derived from public datasets published by Sartaj et al., Br35H, and Cheng et al. (Figshare, 2017).

- **Total Images**: 7,023 single-slice brain MRI images.
- **Class Breakdown**: Glioma (1,621), Meningioma (1,645), No Tumor (2,000), Pituitary (1,757).
- **Holdout Test Set**: 1,600 images (exactly 400 balanced per class).
- **License**: Public domain / Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Patient Metadata Limitation**: The upstream dataset lacks individual patient IDs. As a result, potential slice leakage across patient scans cannot be fully ruled out. See [MODEL_CARD.md](MODEL_CARD.md) for detailed clinical impact analysis.

---

## Tech Stack

| Domain | Technologies |
|:---|:---|
| **Backend & API** | Python 3.10–3.13, Flask 3.1, Waitress WSGI |
| **Deep Learning** | TensorFlow / Keras 2.15+, NumPy |
| **Computer Vision** | OpenCV (headless compatible), Pillow |
| **Explainability** | Custom Grad-CAM with Log-Gradients, Integrated Gradients |
| **Frontend** | HTML5 Semantic Markup, Modern CSS3 Grid/Flexbox, Vanilla ES6+ JavaScript |
| **Testing & CI** | Pytest 8.4, GitHub Actions CI Workflow |
| **Deployment** | Docker (multi-stage slim image), Waitress |

---

## Local Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/NeuroMap.git
cd NeuroMap

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (Command Prompt):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and adjust settings as needed:
```bash
cp .env.example .env
```

### 4. Acquire Model Artifacts
The trained model weights (`best_model.keras`, ~35 MB) are excluded from git tracking.

- **Option A (Trained Model)**: Place your trained Keras file at `model/best_model.keras`.
- **Option B (Custom Path)**: Set the `NEUROMAP_MODEL_PATH` environment variable:
  ```bash
  export NEUROMAP_MODEL_PATH=/path/to/your/model.keras   # Linux/macOS
  $env:NEUROMAP_MODEL_PATH="C:\path\to\model.keras"      # Windows PowerShell
  ```
- **Option C (Train via Notebook)**: Execute `tumor-detector.ipynb` in Kaggle or Jupyter and export the resulting `best_model.keras` to `model/`.

*Note: If no model weights are present, the application will still launch cleanly; the `/ready` probe will indicate unreadiness, and `/predict` will return a helpful `503 Service Unavailable` error explaining that model weights must be supplied.*

### 5. Run the Application
**Development Mode:**
```bash
python app.py
```

**Production Mode (Waitress WSGI):**
```bash
python wsgi.py
# Or directly via waitress-serve:
waitress-serve --listen=0.0.0.0:5000 wsgi:app
```
Access the dashboard at `http://127.0.0.1:5000`.

---

## API Documentation

For complete schemas, request/response examples, and error specifications, refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Summary of Endpoints

| Method | Endpoint | Description | Status Codes |
|:---|:---|:---|:---|
| `GET` | `/` | Web research dashboard UI | `200` |
| `GET` | `/health` | Liveness health check probe | `200` |
| `GET` | `/ready` | Readiness probe (verifies model loaded) | `200`, `503` |
| `POST` | `/predict` | Predict brain tumor class and generate Grad-CAM | `200`, `400`, `413`, `500`, `503` |

---

## Docker Deployment

NeuroMap includes a container definition running a minimal Python 3.11-slim image with non-root security privileges:

```bash
# Build the Docker image
docker build -t neuromap:latest .

# Run container mounting your local model directory
docker run -d \
  -p 5000:5000 \
  -v "$(pwd)/model:/app/model:ro" \
  --name neuromap-app \
  neuromap:latest

# Check container health status
docker inspect --format='{{json .State.Health.Status}}' neuromap-app
```

---

## Testing & Quality Verification

NeuroMap includes 58 automated unit, integration, and concurrency tests covering all components:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run full test suite with verbose output
pytest -v
```

### Verified Test Suite Breakdown (58 Passed)
- **`tests/test_app.py` (22 tests)**: Liveness/readiness probes, prediction endpoint, payload validation, path traversal prevention, image dimension boundaries, cleanup lifecycle, and multi-thread concurrency.
- **`tests/test_xai.py` (18 tests)**: Dynamic convolutional layer detection, Grad-CAM heatmaps, log-gradient numerical stability, out-of-bounds indices, side-by-side overlays, and Integrated Gradients.
- **`tests/test_config.py` (7 tests)**: Configuration class inheritance, default values, environment variable overrides, and dynamic module reload.
- **`tests/test_medical_evaluation.py` (6 tests)**: Diagnostic metric computations, Expected Calibration Error (ECE), temperature scaling optimization, glioma false negative audits, and synthetic stress suite generation.
- **`tests/test_training.py` (5 tests)**: Seed determinism, config loading, dataset path discovery, and layer detection.

---

## Baseline Evaluation Findings

Evaluated on the 1,600-image holdout test set (see [MODEL_CARD.md](MODEL_CARD.md) for full breakdown):

- **Overall Accuracy**: **91.75%**
- **Macro F1-Score**: **0.92**
- **Glioma Recall (Critical Finding)**: **80.0%** (20% False Negative Rate). Infiltrative low-contrast gliomas represent the primary failure mode; clinicians and researchers must never rely on negative predictions without diagnostic histological confirmation.
- **Pituitary Tumor Recall**: **99.0%**
- **No Tumor Recall**: **99.0%**

---

## Troubleshooting

| Problem | Cause | Solution |
|:---|:---|:---|
| **`/ready` returns 503** | Model weights missing | Ensure `model/best_model.keras` exists or set `NEUROMAP_MODEL_PATH`. |
| **`libGL.so.1: cannot open shared object` (Linux)** | Missing OpenGL system library | Install `libgl1` via `apt-get install libgl1` or use `opencv-python-headless`. |
| **`413 Payload Too Large`** | Image file exceeds 16 MB limit | Compress image or adjust `NEUROMAP_MAX_CONTENT_LENGTH`. |
| **Port 5000 already bound** | Another process is using port 5000 | Set `PORT=5001` or run `waitress-serve --listen=0.0.0.0:5001 wsgi:app`. |

---

## Ethical & Legal Disclaimer

NeuroMap is strictly intended for educational demonstration and machine learning research. It is **not** certified as medical device software under FDA regulations, the EU Medical Device Regulation (MDR), or equivalent international standards. Antigravity and the project contributors assume no liability for medical or diagnostic errors resulting from misuse.
