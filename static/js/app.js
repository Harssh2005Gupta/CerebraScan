/**
 * NeuroMap — Frontend Research Dashboard Controller
 *
 * Implements:
 * - Real-time backend readiness and health check.
 * - Accessible drag-and-drop and file upload with client-side validation (format & size).
 * - Multi-state diagnostic output: Awaiting Input, Live Loading, Error Alert, and Results.
 * - Side-by-side visualization of Original MRI Scan and Grad-CAM Activation Overlay.
 * - Accurately measured client-server inference latency (performance.now()).
 * - Full 4-class differential diagnosis probability bars.
 * - Clear reset controls and downloadable high-resolution composite view.
 */

document.addEventListener("DOMContentLoaded", () => {

    // DOM Elements
    const dropzone          = document.getElementById('dropzone');
    const imageInput        = document.getElementById('imageInput');
    const filenameEl        = document.getElementById('filename');
    const validationMessage = document.getElementById('validationMessage');
    const fileMetaBar       = document.getElementById('fileMetaBar');
    const metaName          = document.getElementById('metaName');
    const metaSize          = document.getElementById('metaSize');
    const metaType          = document.getElementById('metaType');
    const predictBtn        = document.getElementById('predictBtn');
    const resetBtn          = document.getElementById('resetBtn');
    const resultBody        = document.getElementById('resultBody');
    const statusDot         = document.getElementById('statusDot');
    const statusText        = document.getElementById('statusText');
    const systemStatusPill  = document.getElementById('systemStatusPill');
    const refreshStatusBtn  = document.getElementById('refreshStatusBtn');

    if (!dropzone || !imageInput) {
        console.error("NeuroMap: Missing required DOM elements.");
        return;
    }

    // Constraints matching backend config.py
    const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB
    const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp'];
    const GAUGE_CIRCUMFERENCE = 238; // 2 * PI * r (r=38 in 88x88 viewBox)

    let selectedFile = null;
    let previewURL = null;

    // -----------------------------------------------------------------------
    // 1. System Readiness & Health Polling
    // -----------------------------------------------------------------------

    async function checkSystemReadiness() {
        if (!statusDot || !statusText) return;

        statusDot.className = "status-indicator status-indicator--checking";
        statusText.textContent = "Checking...";

        try {
            const resp = await fetch('/ready', { method: 'GET', headers: { 'Accept': 'application/json' } });
            const data = await resp.json().catch(() => ({}));

            if (resp.ok && data.model_loaded) {
                statusDot.className = "status-indicator status-indicator--online";
                statusText.textContent = "Model Ready";
                systemStatusPill.title = `Model loaded: ${data.model_path || 'active'}`;
            } else {
                statusDot.className = "status-indicator status-indicator--offline";
                statusText.textContent = "Standby (No Weights)";
                systemStatusPill.title = data.error || "Model file missing or uninitialized.";
            }
        } catch (err) {
            statusDot.className = "status-indicator status-indicator--offline";
            statusText.textContent = "Backend Offline";
            systemStatusPill.title = "Could not reach inference server.";
        }
    }

    checkSystemReadiness();
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', (e) => {
            e.preventDefault();
            checkSystemReadiness();
        });
    }

    // -----------------------------------------------------------------------
    // 2. Dropzone & File Selection Handlers
    // -----------------------------------------------------------------------

    dropzone.addEventListener('click', () => imageInput.click());

    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            imageInput.click();
        }
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add('is-dragover');
        });
    });

    ['dragleave', 'dragend'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove('is-dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('is-dragover');
        const file = e.dataTransfer.files?.[0];
        if (file) handleSelectedFile(file);
    });

    imageInput.addEventListener('change', () => {
        const file = imageInput.files?.[0];
        if (file) handleSelectedFile(file);
    });

    // -----------------------------------------------------------------------
    // 3. Client-Side File Validation
    // -----------------------------------------------------------------------

    function validateFile(file) {
        if (!file) return "No file provided.";

        // Check empty file
        if (file.size === 0) {
            return "The selected file is empty (0 bytes). Please select a valid MRI scan.";
        }

        // Check size limit
        if (file.size > MAX_FILE_SIZE_BYTES) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            return `File size (${sizeMB} MB) exceeds maximum allowed upload size (16 MB).`;
        }

        // Check extension
        const ext = file.name.split('.').pop()?.toLowerCase();
        if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
            return `Unsupported file extension (.${ext || 'unknown'}). Allowed formats: PNG, JPG, JPEG, WEBP.`;
        }

        return null;
    }

    function handleSelectedFile(file) {
        const validationError = validateFile(file);

        if (validationError) {
            showValidationError(validationError);
            clearFileSelection();
            return;
        }

        clearValidationError();
        selectedFile = file;

        if (previewURL) URL.revokeObjectURL(previewURL);
        previewURL = URL.createObjectURL(file);

        // Update UI
        filenameEl.textContent = file.name;
        dropzone.classList.add('has-file');

        if (metaName) metaName.textContent = file.name;
        if (metaSize) metaSize.textContent = formatBytes(file.size);
        if (metaType) metaType.textContent = file.type || 'image/raw';
        if (fileMetaBar) fileMetaBar.style.display = 'grid';
        if (resetBtn) resetBtn.style.display = 'inline-block';

        predictBtn.disabled = false;
        resetResultPanel();
    }

    function showValidationError(message) {
        if (!validationMessage) return;
        validationMessage.textContent = message;
        validationMessage.style.display = 'block';
    }

    function clearValidationError() {
        if (!validationMessage) return;
        validationMessage.textContent = '';
        validationMessage.style.display = 'none';
    }

    function clearFileSelection() {
        selectedFile = null;
        if (previewURL) {
            URL.revokeObjectURL(previewURL);
            previewURL = null;
        }
        imageInput.value = '';
        filenameEl.textContent = 'No file selected';
        dropzone.classList.remove('has-file');

        if (fileMetaBar) fileMetaBar.style.display = 'none';
        if (resetBtn) resetBtn.style.display = 'none';
        predictBtn.disabled = true;
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', (e) => {
            e.preventDefault();
            clearFileSelection();
            clearValidationError();
            resetResultPanel();
        });
    }

    // -----------------------------------------------------------------------
    // 4. Prediction Execution & Latency Measurement
    // -----------------------------------------------------------------------

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true);
        renderLoadingState();

        const formData = new FormData();
        formData.append('image', selectedFile);

        const startTime = performance.now();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const latencyMs = Math.round(performance.now() - startTime);
            const data = await response.json();

            if (!response.ok) {
                const errorMsg = data.error || `Server responded with HTTP ${response.status}`;
                throw new Error(errorMsg);
            }

            renderSuccessResult(data, latencyMs);

        } catch (err) {
            renderErrorState(err);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(state) {
        predictBtn.disabled = state;
        predictBtn.classList.toggle('is-loading', state);
        if (resetBtn) resetBtn.disabled = state;
    }

    // -----------------------------------------------------------------------
    // 5. UI Renderers: Placeholder, Loading, Error, Success
    // -----------------------------------------------------------------------

    function resetResultPanel() {
        resultBody.innerHTML = `
            <div class="placeholder">
                <svg viewBox="0 0 64 64" width="48" height="48" aria-hidden="true">
                    <circle cx="32" cy="32" r="22" fill="none" stroke="currentColor" stroke-width="2"/>
                    <path d="M32 20 L32 34 L40 40" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <p class="placeholder-title">Scan Ready for Analysis</p>
                <p class="placeholder-desc">Click "Run Diagnostic Inference" to compute lesion classification and feature attribution.</p>
            </div>
        `;
    }

    function renderLoadingState() {
        resultBody.innerHTML = `
            <div class="loading-state" role="status" aria-live="polite">
                <div class="loading-bar-track">
                    <div class="loading-bar-fill"></div>
                </div>
                <div class="loading-status-text">
                    Executing Neural Forward Pass &amp; Grad-CAM Gradients...
                </div>
            </div>
        `;
    }

    function renderErrorState(err) {
        resultBody.innerHTML = `
            <div class="error-card" role="alert">
                <div class="error-card__header">
                    <span>⚠️</span>
                    <span>Diagnostic Request Failed</span>
                </div>
                <p class="error-card__body">${escapeHTML(err.message)}</p>
                <button class="reset-btn" id="retryBtn">Retry Inference</button>
            </div>
        `;

        const retryBtn = document.getElementById('retryBtn');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => {
                if (selectedFile) predictBtn.click();
            });
        }
    }

    function renderSuccessResult(data, latencyMs) {
        const confidence = clamp(Number(data.confidence) || 0, 0, 1);
        const prediction = data.prediction || "Unknown";
        const offset = GAUGE_CIRCUMFERENCE * (1 - confidence);

        // Cache-busting URL for live images
        const gradcamUrl = data.gradcam ? `${data.gradcam}?t=${Date.now()}` : null;
        const comparisonUrl = data.comparison ? `${data.comparison}?t=${Date.now()}` : null;

        // Build Differential Diagnosis Bars
        let probHtml = '';
        if (data.probabilities && typeof data.probabilities === 'object') {
            const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
            probHtml = `
                <div class="probabilities-container">
                    <h4 class="probabilities-title">Differential Diagnosis Distribution</h4>
                    <div class="prob-bars">
                        ${sorted.map(([cls, prob]) => {
                            const pct = (clamp(Number(prob) || 0, 0, 1) * 100).toFixed(1);
                            return `
                                <div class="prob-row">
                                    <span class="prob-label">${escapeHTML(cls)}</span>
                                    <div class="prob-bar-track">
                                        <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                                    </div>
                                    <span class="prob-value">${pct}%</span>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        resultBody.innerHTML = `
            <div class="result-readout">

                <!-- Summary Bar: Prediction, Latency, Gauge -->
                <div class="result-summary-bar">
                    <div class="result-main">
                        <span class="result-badge">Top Classification</span>
                        <p class="result-class">${escapeHTML(prediction.toUpperCase())}</p>
                        <div class="result-meta-chips">
                            <span class="result-meta-chip">
                                <strong>${(confidence * 100).toFixed(1)}%</strong> certainty
                            </span>
                            <span>&bull;</span>
                            <span class="result-meta-chip" title="Total client-measured roundtrip latency">
                                ⏱️ ${latencyMs} ms
                            </span>
                        </div>
                    </div>

                    <div class="gauge" aria-label="Confidence gauge: ${(confidence * 100).toFixed(1)}%">
                        <svg viewBox="0 0 88 88" width="88" height="88">
                            <circle class="gauge-track" cx="44" cy="44" r="38"/>
                            <circle class="gauge-value" id="gaugeRing"
                                    cx="44" cy="44" r="38"
                                    stroke-dasharray="${GAUGE_CIRCUMFERENCE}"
                                    stroke-dashoffset="${GAUGE_CIRCUMFERENCE}"/>
                        </svg>
                        <div class="gauge-label">
                            <span class="gauge-label__value" id="gaugePct">0%</span>
                            <span class="gauge-label__sub">SCORE</span>
                        </div>
                    </div>
                </div>

                <!-- Side-by-Side Dual Frame Display -->
                <div class="xai-comparison-container">
                    <div class="xai-frame">
                        <div class="xai-frame__header">
                            <span>Original MRI Scan</span>
                            <span>224&times;224</span>
                        </div>
                        <div class="scan-frame">
                            <img src="${previewURL}" alt="Original uploaded MRI scan">
                        </div>
                    </div>

                    ${gradcamUrl ? `
                    <div class="xai-frame">
                        <div class="xai-frame__header">
                            <span>Grad-CAM Overlay</span>
                            <span>JET Colormap</span>
                        </div>
                        <div class="scan-frame scan-frame--gradcam">
                            <img src="${gradcamUrl}" alt="Grad-CAM activation overlay heatmap">
                        </div>
                    </div>
                    ` : ''}
                </div>

                <!-- Download Composite Action -->
                ${comparisonUrl ? `
                <div class="composite-action-row">
                    <a href="${comparisonUrl}" download="neuromap_diagnostic_composite.jpg" target="_blank" rel="noopener noreferrer" class="composite-link">
                        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>
                        </svg>
                        Export Annotated Diagnostic Composite (JPG)
                    </a>
                </div>
                ` : ''}

                <!-- Differential Probabilities Breakdown -->
                ${probHtml}

                <!-- Clinical Explainability Disclaimer Box -->
                <div class="interpretability-disclaimer" role="note">
                    <div class="disclaimer-title">
                        <span>⚠️</span>
                        <span>ATTRIBUTION &amp; CLINICAL INTERPRETABILITY NOTICE</span>
                    </div>
                    <p class="disclaimer-body">
                        Grad-CAM highlights receptive-field activation regions that most heavily influenced
                        the classifier's prediction score. <strong>It does not delineate true anatomical tumor boundaries
                        or provide a verified histological segmentation mask.</strong> High-activation regions must
                        not be interpreted as confirmed pathological margins.
                    </p>
                </div>

            </div>
        `;

        // Animate circular gauge
        requestAnimationFrame(() => {
            const ring = document.getElementById('gaugeRing');
            const pct = document.getElementById('gaugePct');
            if (ring) ring.style.strokeDashoffset = offset;
            if (pct) animateNumber(pct, confidence);
        });
    }

    // -----------------------------------------------------------------------
    // 6. Utility Functions
    // -----------------------------------------------------------------------

    function animateNumber(el, target) {
        const duration = 800;
        const start = performance.now();

        function tick(now) {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = `${Math.floor(target * eased * 100)}%`;
            if (progress < 1) requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function clamp(val, min, max) {
        return Math.min(Math.max(val, min), max);
    }

    function escapeHTML(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

});