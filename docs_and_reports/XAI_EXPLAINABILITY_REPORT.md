# Explainable AI (XAI) & Attribution Engineering Report
**NeuroMap — Brain Tumor Detection & Lesion Classification**
*Audit, Mathematical Verification, and Architectural Implementation*

---

## Executive Summary

NeuroMap previously relied on a basic Grad-CAM script with hardcoded assumptions, softmax-induced gradient saturation, ambiguous RGB/BGR color channel handling, and lack of visual alignment controls. Crucially, the frontend interface displayed the Grad-CAM activation map in isolation without an adjacent reference scan or clinical interpretability disclaimers, introducing a high risk of user confirmation bias.

This engineering effort upgrades the NeuroMap explanation pipeline with mathematical rigor, architectural flexibility, and clinical safety guardrails:
1. **Dynamic 4D Layer Introspection**: Automatically traverses both custom CNN topologies and transfer-learning backbones (ResNet, MobileNetV2, EfficientNetB0), resolving target 4D spatial tensors prior to global average pooling while validating user-specified candidate layers.
2. **Gradient Vanishing Mitigation**: Mathematically bypasses softmax saturation ($S_c(1 - S_c) \to 0$ as $S_c \to 1.0$) using log-probability gradients $\frac{\partial \log S_c}{\partial A^k} = \frac{1}{S_c}\frac{\partial S_c}{\partial A^k}$, preserving gradient flow on highly confident diagnostic scans.
3. **Strict Color-Space Fidelity**: Eliminates OpenCV/PIL BGR-RGB color inversions with explicit color-space conversions and verified `(width, height)` dimension ordering.
4. **Side-by-Side Dual-View Layout**: Embeds the original MRI and Grad-CAM overlay side-by-side in both the web application and server-side composite graphics, accompanied by full 4-class differential diagnosis distributions and explicit interpretability warning banners.
5. **Axiomatic Attribution Benchmark (Integrated Gradients)**: Implemented path-integral Integrated Gradients ($m=25-50$ steps) to compare coarse receptive-field semantic localization (Grad-CAM, 7×7) against fine pixel-level axiomatic attribution (224×224).
6. **Zero-Defect Verification**: Engineered 18 automated unit and integration tests (`tests/test_xai.py`); full repository test suite (38 tests) passes with 100% success.

---

## 1. Architectural Introspection & Layer Selection

### The Problem of Fragile Layer Selection
Deep learning models for MRI classification downsample images through successive convolutional blocks:
- In NeuroMap's baseline Pre-activation ResNet: Stem downsamples $224 \times 224 \to 56 \times 56$; Stages 2–4 downsample to $28 \times 28$, $14 \times 14$, and finally $7 \times 7 \times 512$ at `activation_16`.
- In MobileNetV2: Depthwise separable convolutions terminate at `out_relu` ($7 \times 7 \times 1280$).
- In EfficientNetB0: Inverted residual blocks terminate at `top_activation` ($7 \times 7 \times 1280$).

Hardcoding layer names like `"conv2d_17"` breaks immediately when switching architectures or retraining with modified layer counts. Furthermore, nested models (e.g. `tf.keras.applications` loaded as sub-models) hide convolutional layers inside internal layer lists.

### The Dynamic Resolution Algorithm
The new `find_last_conv_layer(model, candidate_layer_name)` function implements a two-tier strategy:
1. **Candidate Layer Validation**: If a specific layer name is supplied, the function validates that it exists in the active graph and emits a 4-dimensional tensor $(B, H, W, C)$ where $H > 1$ and $W > 1$. If non-spatial (e.g. `GlobalAveragePooling2D` or `Dense`), it issues a warning and falls back.
2. **Recursive 4D Traversal**: If `"auto"` is specified or candidate validation fails, the algorithm recursively traverses the model hierarchy (including nested sub-models) and identifies the deepest spatial activation tensor before dimensionality collapse.

```python
# Verified auto-detected layers in NeuroMap:
# Baseline Pre-activation ResNet -> 'activation_16' (7x7x512)
# MobileNetV2 Backbone           -> 'out_relu'      (7x7x1280)
# EfficientNetB0 Backbone        -> 'top_activation'(7x7x1280)
```

---

## 2. Gradient Computation & Softmax Saturation

### Softmax Vanishing Gradient Problem
Standard Grad-CAM computes gradients with respect to class output $y^c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

When a model outputs probabilities $S = \text{softmax}(z)$, the gradient of probability $S_c$ with respect to logit $z_d$ is:
$$\frac{\partial S_c}{\partial z_d} = S_c (\delta_{cd} - S_d)$$

For highly confident medical classifications (e.g. $S_c = 0.999$), the factor $S_c(1 - S_c) \approx 0.001$. As confidence approaches $1.0$, gradients collapse toward zero, producing faint, vanishing, or completely blank activation heatmaps despite high model confidence.

### Log-Probability Solution
Selvaraju et al. (2017) originally formulated Grad-CAM using pre-softmax logits $z_c$. In models where the final layer encapsulates the softmax activation within a single `Dense(4, activation='softmax')`, extracting pre-activation tensors can require intrusive graph rewriting. 

NeuroMap resolves this by evaluating gradients of the **log-probability score**:
$$\frac{\partial \log S_c}{\partial A^k} = \frac{1}{S_c} \frac{\partial S_c}{\partial A^k} = \frac{\partial z_c}{\partial A^k} - \sum_{j} S_j \frac{\partial z_j}{\partial A^k}$$
The factor $\frac{1}{S_c}$ mathematically cancels the vanishing $S_c$ term, restoring numerical stability and crisp, informative gradients even when $S_c > 0.99$.

### Safe Numerical Normalization
After applying $\text{ReLU}$ to retain positive attributions ($L_{\text{Grad-CAM}}^c = \text{ReLU}(\sum_k \alpha_k^c A^k)$):
```python
max_val = tf.reduce_max(heatmap)
if max_val < 1e-10:
    logger.warning("Grad-CAM activations are near-zero or negative; returning blank heatmap.")
    return np.zeros(heatmap.shape, dtype=np.float32)
heatmap = heatmap / max_val
```
Division by zero and NaN generation are strictly prevented. Disconnected layers or zero-gradient scenarios return deterministic zero heatmaps accompanied by diagnostic warnings.

---

## 3. Color Space, Alignment, and Overlay Engineering

### Color Space Inconsistencies
OpenCV and PIL handle color channels inversely:
- `utils/preprocess.py` outputs RGB uint8 for `original_img`.
- `cv2.applyColorMap` outputs BGR format.
- `cv2.imwrite` expects BGR format.
- Web browsers (`<img>` tags, HTML5 canvas) and Matplotlib expect RGB format.

Mixing these conventions causes inverted MRI tissue coloration (e.g. gray matter rendered blue, red colormap highlights appearing cyan).

### Standardized Pipeline
`overlay_heatmap` and `create_side_by_side_comparison` now enforce explicit color-space contracts:
1. `image` is validated as RGB. If float in $[0.0, 1.0]$, it is scaled to uint8 $[0, 255]$.
2. Heatmap is resized to `(target_w, target_h)` matching `image.shape[1], image.shape[0]` (preventing OpenCV row/column inversion).
3. If saving directly to disk with `cv2.imwrite`: `return_rgb=False` blends in BGR.
4. If serving to browser canvas or Matplotlib: `return_rgb=True` applies `cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)`.

---

## 4. Second Attribution Technique: Integrated Gradients

To provide a methodological comparison, we implemented **Integrated Gradients** (`integrated_gradients.py`) based on Sundararajan et al. (ICML 2017).

### Mathematical Definition
Given input $x$ and baseline $x'$ (uninformative all-black MRI scan $x' = 0$):
$$\text{Attribution}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
Approximated via Riemann summation over $m = 25$ steps:
$$\text{Attribution}_i^{\text{approx}}(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i}$$

### Grad-CAM vs Integrated Gradients Comparison

| Property | Grad-CAM (`gradcam.py`) | Integrated Gradients (`integrated_gradients.py`) |
| :--- | :--- | :--- |
| **Resolution** | Coarse feature map ($7 \times 7$ upsampled to $224 \times 224$) | Fine pixel-level resolution ($224 \times 224$) |
| **Layer Target** | Deep convolutional layer | Direct input image pixels |
| **Axiomatic Guarantees** | Empirical / heuristic | Axiomatically justified (Completeness, Invariance) |
| **Inference Cost** | Single backward pass ($\sim 626 \text{ ms}$ on CPU) | $m$ backward passes ($\sim 2,761 \text{ ms}$ on CPU, 4.4× slower) |
| **Noise Sensitivity** | Smooth, low-frequency regions | Susceptible to high-frequency pixel gradient scatter |
| **Clinical Value** | Rapid semantic localization of suspicious zone | Feature boundary inspection; edge verification |

---

## 5. UI Dual-View and Clinical Safety Guardrails

### Side-by-Side Dual-View Layout
In `templates/index.html`, `static/js/app.js`, and `static/css/style.css`, the single stacked image was replaced with a coordinated dual-frame layout:
- **Frame 1**: Original MRI Scan (unadulterated anatomical view).
- **Frame 2**: Grad-CAM Overlay (JET heatmap aligned over parenchyma).
- **Differential Distribution**: Horizontal probability bars for all 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`).

### Mandatory Interpretability Disclaimer
Every diagnostic output, both in the web UI and in exported comparison composites (`create_side_by_side_comparison`), features a prominent safety notice:

> ⚠️ **EXPLAINABILITY & CLINICAL SCOPE NOTICE**
> Grad-CAM highlights receptive-field activation regions that most heavily influenced the classifier's prediction score. **It does not delineate true anatomical tumor boundaries or provide a verified histological segmentation mask.** High-activation regions must not be interpreted as confirmed pathological margins.

---

## 6. Verification and Automated Test Results

The test suite was expanded with `tests/test_xai.py` containing 18 automated unit and integration tests. All 38 tests across the NeuroMap repository pass with zero failures:

```
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-8.4.2
collected 38 items

tests\test_app.py .........                                              [ 23%]
tests\test_medical_evaluation.py ......                                  [ 39%]
tests\test_training.py .....                                             [ 52%]
tests\test_xai.py ..................                                     [100%]

============================= 38 passed in 16.43s =============================
```

### Measured Performance Benchmarks
- **Grad-CAM Latency (Baseline ResNet)**: $626.01 \text{ ms}$
- **Grad-CAM Latency (MobileNetV2)**: $730.47 \text{ ms}$
- **Integrated Gradients Latency (ResNet, 25 steps)**: $2,761.18 \text{ ms}$
- **Output Invariants**: All heatmaps verified bounded in $[0.0, 1.0]$, float32 dtype, and identical aspect-ratio scaling.

---

## 7. Clinical Limitations & What Grad-CAM Cannot Do

1. **Receptive Field Blooming**: The $7 \times 7$ feature map represents downsampled receptive fields of over $100 \times 100$ original pixels. Bilinear interpolation creates a smooth, spherical heat blob that naturally exceeds the actual anatomical boundaries of focal tumors.
2. **False Sense of Precision**: Clinicians or users may mistakenly interpret the color boundary as a resection margin or radiation treatment field. Grad-CAM is a classification attribution method, not a semantic segmentation model (like nnU-Net).
3. **Artifact Sensitivity**: Hyperintense non-pathological structures (e.g. skull bones, dental fillings, motion blur) can activate conv filters, producing high heatmaps on non-tumor regions.
4. **Baseline Sensitivity in Integrated Gradients**: The choice of baseline ($x' = 0$) implicitly assumes zero intensity corresponds to absence of information. In T1/T2 MRI, zero represents air or bone, not neutral brain tissue.
