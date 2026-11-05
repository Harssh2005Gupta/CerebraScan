"""
Centralized Configuration for NeuroMap.

Provides environment-based configuration, standardized path resolution
independent of the current working directory, and consistent hyperparameters
shared across training and inference pipelines.
"""

import os
from pathlib import Path
from typing import Set, Tuple

# Base directory of the application (where app.py resides)
BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Base application configuration."""

    BASE_DIR = BASE_DIR

    # Server settings
    SECRET_KEY = os.environ.get("NEUROMAP_SECRET_KEY", "neuromap-dev-secret-key-change-in-prod")
    DEBUG = False
    TESTING = False

    # Uploads and limits
    # 16 MB maximum upload payload
    MAX_CONTENT_LENGTH = int(os.environ.get("NEUROMAP_MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
    UPLOAD_FOLDER = Path(os.environ.get("NEUROMAP_UPLOAD_DIR", BASE_DIR / "static" / "gradcam"))

    # Image validation parameters
    ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "webp"}
    ALLOWED_MIME_TYPES: Set[str] = {"image/png", "image/jpeg", "image/webp"}
    ALLOWED_PIL_FORMATS: Set[str] = {"PNG", "JPEG", "WEBP"}

    # Dimension bounds (prevents decompression bomb DoS)
    MIN_IMAGE_PIXELS: Tuple[int, int] = (16, 16)
    MAX_IMAGE_PIXELS: Tuple[int, int] = (8192, 8192)

    # Standardized model & preprocessing specifications
    # Must remain strictly synchronized between training and inference
    IMG_SIZE: Tuple[int, int] = (224, 224)
    INPUT_CHANNELS: int = 3
    NORMALIZATION_RANGE: Tuple[float, float] = (0.0, 1.0)
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

    # Model and Explainability settings
    DEFAULT_MODEL_PATH = str(BASE_DIR / "model" / "best_model.keras")
    MODEL_PATH = os.environ.get("NEUROMAP_MODEL_PATH", DEFAULT_MODEL_PATH)
    GRADCAM_LAYER = os.environ.get("NEUROMAP_GRADCAM_LAYER", "auto")

    # Temporary output lifecycle management
    TEMP_FILE_TTL_SECONDS = int(os.environ.get("NEUROMAP_TEMP_TTL_SECONDS", 3600))  # 1 hour
    MAX_TEMP_FILES = int(os.environ.get("NEUROMAP_MAX_TEMP_FILES", 500))

    # Structured logging
    LOG_LEVEL = os.environ.get("NEUROMAP_LOG_LEVEL", "INFO").upper()


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration with isolated temporary paths."""
    TESTING = True
    DEBUG = False
    MODEL_PATH = ""  # Explicitly empty for mock testing
    TEMP_FILE_TTL_SECONDS = 0  # Immediate expiration for test cleanup


class ProductionConfig(Config):
    """Production configuration with strict defaults."""
    DEBUG = False
    TESTING = False


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(env_name: str = None) -> Config:
    """Resolve active configuration by environment name or FLASK_ENV."""
    if not env_name:
        env_name = os.environ.get("FLASK_ENV", "production").lower()
    config_class = CONFIG_MAP.get(env_name, ProductionConfig)
    return config_class
