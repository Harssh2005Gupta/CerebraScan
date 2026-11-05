# NeuroMap Frontend & UI/UX Redesign Report
**System**: NeuroMap Brain Tumor Detection & Lesion Classification  
**Role**: Senior Frontend Engineer & UI/UX Designer  
**Status**: Educational / Research Prototype (Not for Clinical Diagnostic Use)

---

## Executive Summary

The NeuroMap web interface has been completely redesigned into a modern, responsive, and accessible AI research dashboard. The new interface preserves 100% compatibility with the refactored Flask backend while elevating user experience, visual clarity, and clinical interpretability:

1. **Modern Research Visual Identity**: Built on a medical-computing dark aesthetic (`#0a0b10` deep space canvas, terracotta `#e06d53` accent, neural signal cyan `#2dd4bf`, and sky blue `#38bdf8`), featuring Space Grotesk and JetBrains Mono typography.
2. **Accessible Drag-and-Drop Upload**: Dropzone supporting drag-over, drop, file picker, and keyboard accessibility (`Enter`/`Space`), equipped with client-side file-size ($\le 16 \text{ MB}$) and extension (`PNG`, `JPG`, `WEBP`) verification.
3. **Multi-State Diagnostic Workflow**: Dedicated UI states for Awaiting Input, Live Loading (with animated indeterminate progress), Error Alert (with descriptive cause and retry action), and Diagnostic Results.
4. **Coordinated Side-by-Side XAI Display**: Dual-frame visual layout comparing the unadulterated Original MRI scan and the Grad-CAM activation map with a one-click action to export the labeled diagnostic composite.
5. **Real-Time Measured Metrics**: Displays actual model confidence scores, class names, and accurately measured client-server roundtrip inference latency (via `performance.now()`).
6. **Clinical Taxonomy & Educational Scope**: Dedicated section detailing the biological and radiological characteristics of the four classes: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor (Normal)**.
7. **System Limitations & Privacy Guardrails**: Prominent cards explaining receptive field blooming, single 2D slice constraints, dataset demographic biases, ephemeral in-memory processing, and non-clinical research status.
8. **Responsive & Accessible UX**: Fully adaptive from 320px mobile screens to wide monitors, semantic HTML5 (`<header>`, `<main>`, `<article>`, `<section>`, `<footer`), ARIA live regions, and WCAG AA contrast compliance.

---

## 1. Visual Hierarchy & Design System

### Design Tokens
```css
--bg: #0a0b10;              /* Deep obsidian canvas */
--panel: #12141e;           /* Primary container background */
--panel-alt: #171a27;       /* Secondary nested container */
--border: #252a3d;          /* Subtle structural borders */
--border-focus: #2dd4bf;    /* Accessible keyboard focus ring */

--accent: #e06d53;          /* Terracotta / primary interactive accent */
--signal: #2dd4bf;          /* Neural cyan / activation highlights */
--blue: #38bdf8;            /* Sky blue utility accent */
--success: #34d399;         /* Health probe ready */
--warning: #fbbf24;         /* Health probe checking */
--danger: #f87171;          /* Error alert state */

--font-display: 'Space Grotesk', sans-serif;
--font-mono: 'JetBrains Mono', monospace;
```

### Layout Grid
- **Desktop ($\ge 960\text{px}$)**: Asymmetric 2-column workspace (`1fr : 1.15fr`) placing scan input on the left and diagnostic visualization on the right.
- **Tablet & Mobile ($< 960\text{px}$)**: Stacked linear flow prioritizing scan upload followed by diagnostic outputs.

---

## 2. Component Architecture

### 2.1 Live System Readiness Pill
- Located in the site header.
- Polls the backend `/ready` endpoint upon initial load:
  - `● Model Ready` (green) if model weights are active and validated.
  - `○ Standby (No Weights)` (amber) if running in weightless demo mode.
  - `✕ Backend Offline` (red) if connection fails.
- Features a refresh button for manual probe re-execution.

### 2.2 Client-Validated Upload Zone
- Handles file selection via drag-and-drop, native file browser, or keyboard activation.
- Validates file constraints before dispatching HTTP payloads:
  - Rejecting 0-byte files with explanatory alert.
  - Rejecting files exceeding 16 MB.
  - Rejecting unsupported file extensions.
- Displays a real-time metadata bar: Filename, formatted size (KB/MB), and MIME type.
- Features a "Clear Scan" button to cleanly reset state and revoke object URLs.

### 2.3 Diagnostic Output & Explainability Dashboard
- **Circular Confidence Gauge**: Scalable vector gauge with smooth cubic-bezier fill animation.
- **Measured Latency Chip**: Displays client-to-server inference time measured via high-resolution timers (`performance.now()`).
- **Side-by-Side Dual Frame**:
  - Left: Original MRI slice.
  - Right: Grad-CAM JET heatmap aligned over anatomical structures.
  - Export: Clean action link downloading the annotated 2-panel composite image.
- **Differential Diagnosis Distribution**:
  - Horizontal probability bars for all four classes sorted in descending order of likelihood.
- **Clinical Explainability Notice**:
  - Explicit warning that Grad-CAM visualizes receptive field influence rather than proven tumor boundaries or segmentation margins.

### 2.4 Educational Taxonomy Cards
Summarizes the pathophysiology and MRI characteristics of the four target classes:
1. **Glioma**: Primary intra-axial tumors of glial origin with infiltrative margins.
2. **Meningioma**: Typically benign extra-axial tumors of arachnoid cap origin.
3. **Pituitary Tumor**: Sellar and suprasellar neuroendocrine lesions.
4. **No Tumor**: Physiological brain parenchyma without focal mass lesions.

### 2.5 Limitations & Privacy Notice
Highlights critical machine learning and radiological caveats:
- Receptive field blooming from 7×7 feature map upsampling.
- 2D axial slice limitation omitting volumetric 3D context.
- Dataset selection bias.
- Ephemeral in-memory scan processing with automated lifecycle garbage collection.

---

## 3. Accessibility & Usability (WCAG Compliance)

- **Keyboard Navigation**: All interactive elements (`#dropzone`, `#imageInput`, `#predictBtn`, `#resetBtn`, `#refreshStatusBtn`) are navigable via `Tab` and triggerable via `Enter` / `Space`.
- **Visible Focus Indicator**: Prominent `outline: 2px solid var(--border-focus)` on `:focus-visible`.
- **Color Contrast**: All body text and badges exceed WCAG AA 4.5:1 contrast requirements against panel backgrounds.
- **Screen Reader Support**: ARIA live regions (`aria-live="polite"`, `role="status"`, `role="alert"`) notify assistive devices of scan loading, status updates, and error alerts.
- **Reduced Motion**: Honored via `@media (prefers-reduced-motion: reduce)`, disabling pulse loops and transitions.

---

## 4. Verification Results

- Syntax Verification: `node -c static/js/app.js` passed with zero errors.
- Template Verification: Standalone Flask rendering of `templates/index.html` verified with all semantic sections present.
- Automated Test Suite: All 51 tests across `test_app.py`, `test_xai.py`, `test_training.py`, and `test_medical_evaluation.py` passed with 100% success.
