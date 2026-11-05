"""
Image preprocessing and validation utilities for NeuroMap inference.

Ensures strict parity with training preprocessing, performs deep content
validation on image streams, prevents decompression bombs, and handles color spaces.
"""

import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Exceptions
# ---------------------------------------------------------------------------


class ImageValidationError(ValueError):
    """Base exception for image validation failures."""
    pass


class InvalidImageFormatError(ImageValidationError):
    """Raised when the file format or magic bytes are unsupported or corrupted."""
    pass


class ImageDimensionError(ImageValidationError):
    """Raised when image dimensions are out of acceptable bounds."""
    pass


class EmptyImageError(ImageValidationError):
    """Raised when the uploaded file stream contains zero bytes."""
    pass


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------


def validate_and_load_image(file_stream) -> Image.Image:
    """
    Validate file stream content and load as a PIL Image.

    Performs:
    1. Zero-byte check.
    2. Deep format verification (magic bytes and header structure).
    3. Dimension bounds check (decompression bomb protection).
    4. Format whitelisting (PNG, JPEG, WEBP).

    Args:
        file_stream: File-like binary stream.

    Returns:
        Verified and opened PIL.Image instance.

    Raises:
        EmptyImageError: If the stream is empty.
        InvalidImageFormatError: If the content is corrupted or not a valid image.
        ImageDimensionError: If dimensions exceed max bounds or are sub-minimal.
    """
    # 1. Check for empty stream
    if hasattr(file_stream, "seek") and hasattr(file_stream, "tell"):
        file_stream.seek(0, io.SEEK_END)
        size = file_stream.tell()
        file_stream.seek(0)
        if size == 0:
            raise EmptyImageError("Uploaded image stream contains 0 bytes.")
    elif hasattr(file_stream, "read"):
        peek = file_stream.read(1)
        if not peek:
            raise EmptyImageError("Uploaded image stream contains 0 bytes.")
        file_stream.seek(0)

    # 2. Open and verify image structure
    try:
        raw_img = Image.open(file_stream)
        detected_format = raw_img.format
        if not detected_format or detected_format.upper() not in Config.ALLOWED_PIL_FORMATS:
            raise InvalidImageFormatError(
                f"Unsupported image format '{detected_format}'. Allowed formats: "
                f"{', '.join(sorted(Config.ALLOWED_PIL_FORMATS))}."
            )
        # Deep header integrity check
        raw_img.verify()
    except (ImageValidationError,):
        raise
    except Exception as e:
        logger.warning("PIL image verification failed: %s", e)
        raise InvalidImageFormatError(f"File content is corrupted or not a recognized image: {e}") from e

    # 3. Rewind stream after verify() and reopen for actual processing
    if hasattr(file_stream, "seek"):
        file_stream.seek(0)

    try:
        img = Image.open(file_stream)
        width, height = img.size

        min_w, min_h = Config.MIN_IMAGE_PIXELS
        max_w, max_h = Config.MAX_IMAGE_PIXELS

        if width < min_w or height < min_h:
            raise ImageDimensionError(
                f"Image dimensions ({width}x{height}) are too small. Minimum required: {min_w}x{min_h}."
            )
        if width > max_w or height > max_h:
            raise ImageDimensionError(
                f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_w}x{max_h})."
            )

        return img

    except ImageValidationError:
        raise
    except Exception as e:
        logger.error("Failed to load verified image: %s", e)
        raise InvalidImageFormatError(f"Could not load image: {e}") from e


def preprocess_image(file_stream) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load, validate, and preprocess an uploaded image for model inference.

    Strictly reproduces the preprocessing applied during training:
    - Color: Standard 3-channel RGB.
    - Spatial dimensions: Resized to Config.IMG_SIZE (224, 224) using bilinear interpolation.
    - Intensity: Scaled to float32 range [0.0, 1.0].
    - Batch dimension: Shape (1, 224, 224, 3).

    Args:
        file_stream: Binary file-like object from HTTP request or disk.

    Returns:
        model_input: Normalized float32 numpy array of shape (1, 224, 224, 3) in [0.0, 1.0].
        original_np: uint8 RGB numpy array of shape (224, 224, 3) for visualization overlays.

    Raises:
        ImageValidationError: If image is empty, malformed, or exceeds bounds.
    """
    img = validate_and_load_image(file_stream)

    # Convert to RGB (handles RGBA, Grayscale 'L', Palette 'P', CMYK safely)
    img_rgb = img.convert("RGB")

    # Resize to exact model input dimension
    img_resized = img_rgb.resize(Config.IMG_SIZE, resample=Image.Resampling.BILINEAR)

    original_np = np.array(img_resized, dtype=np.uint8)

    # Normalize to [0.0, 1.0] float32
    model_input = original_np.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    return model_input, original_np