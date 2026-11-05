# ==============================================================================
# NeuroMap — Production Container Definition
# Multi-stage/Slim Python 3.11 image with OpenCV Headless & TensorFlow CPU
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set security and Python runtime flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=5000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system runtime dependencies for headless image processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root system user for security
RUN groupadd -r neuromap && useradd -r -g neuromap -d /app -s /sbin/nologin neuromap

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code into container
COPY . .

# Create runtime directories and set ownership to non-root user
RUN mkdir -p /app/model /app/uploads /app/static/gradcam && \
    chown -R neuromap:neuromap /app

# Switch to unprivileged user
USER neuromap

EXPOSE 5000

# Health probe check against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health').getcode() == 200 else 1)"

# Production deployment with Waitress WSGI server
CMD ["waitress-serve", "--listen=0.0.0.0:5000", "wsgi:app"]
