# NeuroMap — Model Card

Based on *Model Cards for Model Reporting* ([Mitchell et al., 2019](https://arxiv.org/abs/1810.03993)).

---

## 1. Model Details

| Attribute | Specification |
|:---|:---|
| **Model Name** | NeuroMap Brain Tumor Classifier |
| **Model Version** | v1.1 (Multi-Architecture Inference & XAI Pipeline) |
| **Model Type** | Deep Convolutional Neural Network (Classification & Explainability) |
| **Supported Backbones** | Custom Pre-activation ResNet (Baseline), EfficientNetB0, MobileNetV2 |
| **Developers** | NeuroMap Research Contributors |
| **Release Date** | October 2026 |
| **Framework** | TensorFlow / Keras 2.15+ |
| **Model Artifact Format** | Keras v3 SavedModel (`.keras`) or Legacy HDF5 (`.h5`) |
| **License** | MIT License (Code) / CC BY 4.0 (Dataset Attribution) |
| **Model Artifact Size** | ~35 MB (Baseline ResNet, ~8.9M parameters) |

---

## 2. Intended Use & Target Audience

### Primary Intended Uses
- **Educational Demonstration**: Hands-on exploration of deep learning, computer vision, and explainable AI (XAI) pipelines for computer science and biomedical engineering students.
- **Explainability Research**: Comparative benchmarking between gradient-based saliency methods (Grad-CAM) and path-integral attribution methods (Integrated Gradients) on radiological imagery.
- **MLOps & Pipeline Architecture**: Reference implementation for modular Flask backend architecture, robust input validation, automated cleanup lifecycles, and health/readiness observability probes.

### Primary Intended Users
- Machine learning researchers, educators, computer vision engineers, and students studying explainable medical image analysis.

---

## 3. Explicitly Excluded & Prohibited Use Cases

> [!CAUTION]
> **Strict Non-Clinical Status**: NeuroMap is an educational prototype and research artifact. It is **NOT** a medical device, is **NOT** FDA-cleared or CE-marked, and must **NEVER** be used for clinical decision-making.

The following use cases are strictly prohibited and unsupported:
1. **Primary Clinical Diagnosis**: Using model output to diagnose or rule out brain neoplasms in human patients.
2. **Clinical Decision Support**: Integrating into hospital electronic health record (EHR) or Picture Archiving and Communication Systems (PACS) for live diagnostic workflows.
3. **Surgical Planning / Radiation Therapy**: Relying on Grad-CAM or Integrated Gradients attributions to demarcate surgical resection margins or radiation target volumes.
4. **Emergency Triage**: Screening acute or emergency neurological presentations.
5. **Pediatric Imaging**: Deploying on pediatric brain MRI without domain-specific re-training and validation.

---

## 4. Training Data & Provenance

| Parameter | Details |
|:---|:---|
| **Primary Dataset** | [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Curated by Masoud Nickparvar) |
| **Origin & Provenance** | Derived from combined public repositories: Sartaj et al., Br35H, and Cheng et al. (Figshare, 2017) |
| **Total Images** | 7,023 images across 4 classes |
| **Class Distribution** | Glioma (1,621), Meningioma (1,645), No Tumor (2,000), Pituitary (1,757) |
| **Training Split** | 4,760 images (85% of Training partition) |
| **Validation Split** | 840 images (15% stratified split, seed=42) |
| **Holdout Test Set** | 1,600 images (separate Kaggle Testing directory, exactly 400 per class balanced) |
| **Image Formatting** | 2D axial, coronal, and sagittal T1-weighted contrast-enhanced slices |
| **Preprocessing** | Resized to 224×224 pixels, 3 channels (RGB duplicate), normalized to `[0.0, 1.0]` |

---

## 5. Quantitative Evaluation & Baseline Metrics

Evaluated on the independent 1,600-image holdout test partition:

| Metric | Measured Baseline Value |
|:---|:---:|
| **Test Accuracy** | **91.75%** (1,468 / 1,600) |
| **Test Categorical Loss** | **0.7116** |
| **Macro Precision** | **0.923** |
| **Macro Recall** | **0.918** |
| **Macro F1-Score** | **0.920** |
| **Weighted F1-Score** | **0.920** |

### Per-Class Detailed Performance Breakdown

| Class | Precision | Recall | F1-Score | Support | Clinical Risk Profile |
|:---|:---:|:---:|:---:|:---:|:---|
| **Glioma** | **0.97** | **0.80** | **0.87** | 400 | Infiltrative, aggressive malignancy; **20% False Negative Rate** is a critical failure mode |
| **Meningioma** | **0.93** | **0.89** | **0.91** | 400 | Dural/extra-axial lesion; modest confusion with lower-grade gliomas |
| **No Tumor** | **0.88** | **0.99** | **0.93** | 400 | Normal anatomy / benign scans; high recall minimizes false alarms |
| **Pituitary** | **0.91** | **0.99** | **0.95** | 400 | Sellar/suprasellar lesion; high anatomical contrast enables near-perfect recall |

### False Negative Analysis (Glioma)
- **Observed Behavior**: The model misclassified 80 of 400 glioma cases (primarily misattributed to meningioma or no-tumor).
- **Etiology**: Diffuse, non-enhancing, or low-grade infiltrative gliomas exhibit subtle intensity gradients that standard CNN feature extractors may miss without multiscale attention or FLAIR/T2 sequence inputs.

---

## 6. Explainability (XAI) & Interpretability Specifications

### Supported Attribution Methods
1. **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:
   - Computes gradients of the target class logit with respect to the final convolutional feature map.
   - Log-space gradient stabilization prevents numerical underflow in saturated softmax regimes.
   - Automatically detects the deepest 4D convolutional layer recursively across nested models.
2. **Integrated Gradients (Path-Integral Attribution)**:
   - Computes path integrals of gradients along a linear interpolation from a black baseline image to the input scan.
   - Provides pixel-level attribution that satisfies implementation invariance and completeness axioms.

### Interpretability Caution & Guidelines
- **Correlation, Not Tumor Boundaries**: Activation heatmaps indicate which receptive fields influenced the neural network's final activation score. They do **not** represent verified anatomical tumor margins, cellular infiltration boundaries, or histological grading.
- **Receptive Field Smearing**: Due to downsampling across pooling layers, Grad-CAM heatmaps have coarse spatial resolution (~7×7 pixels upsampled to 224×224) and cannot resolve microvascular infiltration.

---

## 7. Known Limitations, Biases & Ethical Considerations

1. **Slice-Level vs. Patient-Level Leakage Risk**:
   - The public dataset lacks patient identification metadata. Different 2D slices from the same 3D patient volume may be distributed across training and testing partitions, potentially inflating benchmark accuracy.
2. **Missing Multi-Sequence MRI Context**:
   - Real-world neuroradiology relies on multi-parametric MRI sequences (T1, T1+Contrast, T2, FLAIR, DWI/ADC). NeuroMap operates on single-slice 2D images, lacking 3D volumetric context and multi-sequence cross-validation.
3. **Automation Bias Risk**:
   - Users may experience automation bias, over-relying on high model confidence scores or visually appealing Grad-CAM overlays. Explicit disclaimers and UI guardrails are implemented to mitigate this risk.
4. **Demographic and Hardware Representation**:
   - Patient age, sex, scanner manufacturer (Siemens, GE, Philips), field strength (1.5T vs 3.0T), and geographic distribution are unrecorded in the upstream dataset. Generalizability to out-of-distribution scanner protocols is not guaranteed.
