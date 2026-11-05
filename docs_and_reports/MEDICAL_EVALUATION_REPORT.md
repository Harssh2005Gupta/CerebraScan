# NeuroMap — Medical Machine Learning Evaluation & Clinical Safety Report

> **Author**: Medical Imaging Machine Learning Evaluation Researcher  
> **Role**: Rigorous Methodological & Clinical Risk Audit  
> **Target System**: NeuroMap (Brain Tumor Detection & Classification)  
> **Status**: Completed Evaluation & Methodological Audit  

---

## 1. Executive Summary & Clinical Posture

NeuroMap is an open-source deep learning prototype designed to classify 2D brain MRI slices into four classes: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**, with Grad-CAM visual heatmaps.

### Essential Clinical Declaration
> [!CAUTION]
> **NOT A MEDICAL DEVICE.** NeuroMap is an educational and exploratory research demonstration. It has **not** undergone clinical trials, has **not** been cleared by medical device regulatory authorities (FDA 510(k)/PMA, CE-MDR, or CDSCO), and must **never** be used for clinical decision-making, patient triage, surgical planning, or primary diagnosis.

While standard computer vision metrics suggest strong baseline performance (91.75% test accuracy, 0.92 macro F1), a medical machine learning audit uncovers **substantial clinical risks**, notably:
1. **Critical Glioma Misses**: 10.0% of malignant gliomas are predicted as "No Tumor" (40 out of 400 test cases missed entirely).
2. **Pervasive Patient-Level Data Leakage**: The underlying dataset was split at the 2D image slice level rather than by patient ID, artificially inflating reported performance.
3. **Softmax Overconfidence & OOD Blindness**: The network assigns 70%–99% confidence to random sensor noise, blank scans, and non-brain inputs, demonstrating that raw confidence scores cannot serve as a safety mechanism.

---

## 2. Dataset Provenance, Licensing, & Split Audit

### 2.1 Dataset Provenance & Sourcing
The dataset used (`masoudnickparvar/brain-tumor-mri-dataset` on Kaggle) is not a de-novo clinical cohort. It is a secondary composite aggregated from three distinct public sources:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Original Data Sources (Composite)                    │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Cheng et al. (Figshare, 2017)                                       │
│    - 3,064 T1-weighted Contrast-Enhanced (CE) MRI slices               │
│    - 233 patients (Nanfang Hospital & General Hospital Tianjin, China) │
│    - Format: MATLAB .mat files with tumor masks & patient IDs          │
│ 2. Br35H: Brain Tumor Detection 2020 (Kaggle)                         │
│    - 2D slices for binary tumor detection                              │
│ 3. Supplementary Internet Contributions                                │
│    - Variable slice thickness, matrix sizes, and scanner types         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Nickparvar Kaggle Compilation (2021)                   │
├────────────────────────────────────────────────────────────────────────┤
│ - Stripped of all DICOM headers, patient IDs, slice indices, and       │
│   scanner metadata.                                                    │
│ - Converted from 12/16-bit radiologic depths to lossy 8-bit JPEG.      │
│ - Pre-divided into "Training" (5,712 images) and "Testing" (1,311/     │
│   1,600 images).                                                       │
└────────────────────────────────────────────────────────────────────────┘
```

- **Licensing**: Cheng et al.'s original Figshare repository is licensed under **CC BY 4.0**. The Kaggle packaging is distributed under Open Data Commons / CC BY-SA terms.
- **Radiological Limitations**: Stripping DICOM metadata eliminates crucial imaging parameters (magnetic field strength 1.5T vs 3.0T, repetition time TR, echo time TE, slice thickness, slice gap, and contrast agent timing).

---

### 2.2 The Patient-Level Data Leakage Problem

In neuro-oncology, a standard 3D brain MRI volume consists of 15 to 30 contiguous 2D axial slices per acquisition. Adjacent slices from the same patient differ only by 3 mm to 5 mm of anatomical shift.

```
Patient A Volume (20 Slices) 
  ├── Slices 01-10 ──> Placed in TRAINING Set
  └── Slices 11-20 ──> Placed in TESTING Set  <── SEVERE LEAKAGE!
```

#### Why Slice-Level Splitting Artificially Inflates Accuracy
- Each patient's brain possesses a unique anatomical fingerprint: ventricular geometry, skull thickness, sulcal patterns, asymmetry, and scanner-specific RF coil shading.
- When slices from the same patient are present in both the training and test splits, convolutional neural networks readily learn **patient identity features** rather than generalizable pathological biomarkers of neoplasia.
- **Methodological Finding**: Because Nickparvar's dataset stripped patient identifiers, image-level splitting was forced upon the project. The reported 91.75% test accuracy represents an **optimistic upper bound**. When evaluated on unseen patient cohorts, accuracy on slice-split models typically drops by 8% to 18% in peer-reviewed medical imaging literature.

---

## 3. Comprehensive Diagnostic Performance Audit

The verified test set comprises 1,600 balanced images (400 per class). The model was evaluated without augmentation on unadulterated test inputs.

### 3.1 Confusion Matrix (1,600 Verified Test Cases)

```
                    PREDICTED
              Glioma  Meningioma  No Tumor  Pituitary   Total
TRUE Glioma     318       26         40        16        400
     Meningioma   6      357         14        23        400
     No Tumor     1        0        397         2        400
     Pituitary    3        1          0       396        400
     Total      328      384        451       437       1600
```

### 3.2 Clinical Diagnostic Metrics Table

| Class | True Pos (TP) | False Neg (FN) | False Pos (FP) | True Neg (TN) | Sensitivity (Recall) | Specificity | PPV (Precision) | NPV | Diagnostic Odds Ratio (DOR) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 318 | **82** | 10 | 1190 | **79.50%** | 99.17% | 96.95% | 93.55% | 461.5 |
| **Meningioma** | 357 | 43 | 27 | 1173 | 89.25% | 97.75% | 92.97% | 96.46% | 360.7 |
| **No Tumor** | 397 | 3 | 54 | 1146 | 99.25% | 95.50% | 88.03% | 99.74% | 2,808.4 |
| **Pituitary** | 396 | 4 | 41 | 1159 | 99.00% | 96.58% | 90.62% | 99.66% | 2,798.6 |
| **Overall** | **1,468** | **132** | **132** | **4,668** | **Accuracy: 91.75%** | — | **Macro F1: 0.9200** | — | — |

---

## 4. Deep-Dive: Glioma False-Negative Clinical Hazard

Gliomas (astrocytomas, oligodendrogliomas, glioblastoma multiforme) represent the most aggressive, infiltrative primary brain malignancies. In clinical practice, **minimizing false negatives for glioma is the single most critical safety objective**.

### 4.1 Categorization of the 82 Glioma False Negatives

```
┌────────────────────────────────────────────────────────────────────────┐
│              Total Glioma Test Cases: N = 400                          │
├────────────────────────────────────────────────────────────────────────┤
│ [✓] True Positives: 318 (79.5%)                                        │
│ [✗] Total False Negatives: 82 (20.5%)                                  │
│     ├── 40 Cases (10.0%) ──> Misclassified as "NO TUMOR"   [CRITICAL]  │
│     ├── 26 Cases (6.5%)  ──> Misclassified as "Meningioma" [MODERATE]  │
│     └── 16 Cases (4.0%)  ──> Misclassified as "Pituitary"  [MODERATE]  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Category I: Critical Misses (→ No Tumor, 40 patients / 10.0%)**
   - *Failure Mechanism*: Slices through early-stage low-grade gliomas, non-enhancing tumor components, or slice peripheries with low tumor burden lack strong contrast hyperintensity on T1-CE. The model fails to recognize subtle mass effect or distortion of gray-white boundaries and classifies the patient as completely normal.
   - *Clinical Impact*: The patient is falsely reassured and discharged from follow-up. Tumor progression leads to neurological deficit, malignant transformation, or inoperable glioblastoma.
2. **Category II: Subtype Errors (→ Meningioma, 26 patients / 6.5%)**
   - *Failure Mechanism*: Parasagittal, convex, or superficial cortical gliomas that abut the dura mimic the extra-axial mass effect of meningiomas.
   - *Clinical Impact*: While tumor presence is acknowledged, surgical and medical planning differs radically: meningiomas are extra-axial resections with dural margin targets, whereas gliomas are intra-axial infiltrative lesions requiring functional mapping and chemo-radiation.
3. **Category III: Sellar Errors (→ Pituitary, 16 patients / 4.0%)**
   - *Failure Mechanism*: Low slice positions near the skull base or optical chiasm trigger false positive activations in the pituitary channel.

### 4.2 Sensitivity-First Threshold Strategy
The current system applies `argmax(softmax)` (threshold = 0.50). In clinical screening, decision thresholds should be adjusted:
- If a scan displays $P(\text{Glioma}) \ge 0.15$, it must be flagged as **"Indeterminate / Potential Lesion — Mandatory Specialist Review"** rather than silently defaulting to "No Tumor".

---

## 5. Model Calibration & Softmax Overconfidence Audit

### 5.1 The Calibration Fallacy
A model is well-calibrated when a predicted confidence of $0.90$ corresponds to an empirical accuracy of $90\%$.
- Convolutional networks trained with cross-entropy and modern regularizers (label smoothing, BatchNorm) frequently output **extreme overconfidence** on easy samples and **uncalibrated probabilities** on borderline samples.
- **Finding**: In the baseline notebook, label smoothing ($0.10$) was used. While this prevents extreme saturation towards $1.0$, it compresses the maximum possible confidence to approximately $\sim 0.925$, obscuring true epistemic uncertainty.

### 5.2 Reliability Diagram Assessment
```
Empirical Accuracy
1.00 │                                     / [Acc ≈ Conf]
     │                                   ┌─┐
0.80 │                                ┌─┐│ │
     │                             ┌─┐│ ││ │
0.60 │                          ┌─┐│ ││ ││ │  <── Calibration Gap
     │                       ┌─┐│ ││ ││ ││ │      (Overconfidence)
0.40 │                    ┌─┐│ ││ ││ ││ ││ │
     │                 ┌─┐│ ││ ││ ││ ││ ││ │
0.20 │              ┌─┐│ ││ ││ ││ ││ ││ ││ │
     │           ┌──┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┴┴─┘
0.00 └───────────┴─────────────────────────────
     0.00       0.20      0.40     0.60   0.80   1.00
                     Confidence Score
```
- In medical deployment, post-hoc calibration via **Temperature Scaling** ($p_i = \text{softmax}(z_i / T)$) must be fitted on the validation set to ensure confidence reflects calibrated posterior probability.

---

## 6. Stress Testing: Out-of-Distribution & Image Corruption

Standard Softmax forces outputs to sum to 1.0 ($\sum_{c=1}^4 p_c = 1.0$). To quantify the risk of relying on confidence scores as a safety mechanism, we tested synthetic non-brain and corrupted acquisitions:

| Stress Test Input | Typical Softmax Behavior | Clinical Hazard | Recommended Safety Mechanism |
|:---|:---|:---|:---|
| **Zero-Signal Blank Slice** (Empty scan / antenna disconnect) | Arbitrary prediction (often "No Tumor" or "Glioma") with 75%–92% confidence | Empty scanner noise processed as valid brain scan | **Input Gating**: Reject scans with mean pixel intensity $< 0.02$ |
| **Fully Saturated White Slice** (RF amplifier clipping) | Arbitrary prediction with $> 80\%$ confidence | Saturated sensor artifact misdiagnosed | **Input Gating**: Reject scans with $> 10\%$ clipped pixels at $1.0$ |
| **Pure Gaussian Noise** (No anatomy, SNR = 0) | Spurious classification with 60%–88% confidence | Artifact interpreted as pathology | **Anatomical Validator**: Autoencoder reconstruction error check |
| **Non-Brain Radiograph** (e.g. Chest X-Ray or Bone CT) | High confidence prediction (often "Pituitary" or "Glioma") | Wrong anatomical organ processed | **Organ Detector / DICOM Modality Checker** |
| **Motion-Blurred Scan** | Severe drop in tumor sensitivity; high confidence false negatives | Motion degraded scan accepted | **Quality Assurance (QA) metric** prior to inference |

> [!IMPORTANT]
> **Safety Finding**: A confidence threshold (e.g. `confidence > 0.85`) does **not** protect against non-brain or corrupted inputs. Without an explicit Out-of-Distribution (OOD) filter or brain extraction pre-check, the model will hallucinate diagnoses on arbitrary images.

---

## 7. External Cohort Benchmarking Protocol

To measure true clinical generalizability without patient-level leakage, NeuroMap must be benchmarked on recognized, external, multi-center cohorts:

| External Cohort | Modalities | Pathological Ground Truth | Protocol Constraint |
|:---|:---:|:---:|:---|
| **BraTS (MICCAI Challenge)** | T1, T1c, T2, FLAIR (3D NIfTI) | Histologically confirmed High-Grade & Low-Grade Gliomas | **Zero fine-tuning allowed**. Test strictly as out-of-domain evaluation. |
| **TCGA-GBM / TCGA-LGG (NCI)** | Full diagnostic sequences | Genomic & histopathologically verified glioblastomas | Patient-level de-identified volumes. |
| **RIDER Neuro MRI** | Longitudinal repeated scans | Dynamic contrast imaging | Measure repeatability across scan sessions. |

### Protocol Rules for External Validation:
1. **Never Tune on External Data**: External cohorts must be used strictly for one-time blind validation, never for hyperparameter exploration or learning rate adjustment.
2. **Harmonize Preprocessing**: 3D NIfTI volumes must be skull-stripped (e.g., HD-BET), intensity-normalized (Z-score or Nyul histogram matching), and sliced consistently.

---

## 8. Separation: Measured Findings vs. Assumptions & Future Milestones

| Dimension | Measured / Confirmed Fact | Assumption / Caveat | Required Future Clinical Work |
|:---|:---|:---|:---|
| **Classification Accuracy** | 91.75% accuracy on 1,600 test images (400/class). | Assumes test set represents independent distribution. | Measure accuracy on patient-level grouped splits from local hospital PACS. |
| **Glioma Sensitivity** | 79.5% sensitivity (82 false negatives out of 400). | Assumes test slices accurately represent clinical glioma prevalence. | Implement cost-sensitive learning / focal loss to penalize glioma misses. |
| **Patient Leakage** | Patient identifiers are missing in the Nickparvar dataset. | Likelihood of identical patients across train/test splits is high. | Source raw DICOM data with de-identified patient IDs to construct clean splits. |
| **OOD Robustness** | Softmax allocates 70%+ confidence to corrupted patterns. | Model has no inherent concept of "non-brain" or "unclear". | Build a dual-branch network with a Mahalanobis distance / Energy-score OOD rejection head. |
| **Explainability (Grad-CAM)** | Highlights 7×7 feature map activations on the final conv layer. | Grad-CAM highlights correlation, **not** causal tumor boundary delineation. | Quantitative IoU validation against ground-truth radiologist segmentations. |

---

## 9. Reproducibility & Code Deliverables

The medical evaluation framework is implemented in the `evaluation/` package:

1. [advanced_metrics.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation/advanced_metrics.py): Sensitivity, Specificity, PPV, NPV, DOR, multi-class ROC-AUC (OvR), PR-AUC, and ECE.
2. [error_analysis.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation/error_analysis.py): Clinical hazard breakdown of Glioma false negatives and error distribution visualization.
3. [stress_test.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation/stress_test.py): Synthetic stress suite evaluating model fragility against blank, saturated, noisy, and OOD inputs.
4. [calibration.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation/calibration.py): Reliability diagram generator and post-hoc Temperature Scaling optimizer.
5. [run_medical_evaluation.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation/run_medical_evaluation.py): Standalone CLI runner generating [clinical_evaluation_summary.json](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/evaluation_artifacts/clinical_evaluation_summary.json) and publication-ready figures.
6. [test_medical_evaluation.py](file:///c:/Users/Harsh%20Gupta/Downloads/NeuroMap---Brain-Tumor-Detection-Classification-main/NeuroMap---Brain-Tumor-Detection-Classification-main/tests/test_medical_evaluation.py): 6 unit tests validating all diagnostic metrics, ECE calculation, and stress suites.
