"""Image decoding, grayscale conversion, and pair preparation."""

from __future__ import annotations

from typing import BinaryIO

import cv2
import numpy as np

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DECODED_PIXELS = 20_000_000


def decode_uploaded_image(uploaded_file: BinaryIO | bytes) -> np.ndarray:
    """Decode an uploaded image into an 8-bit grayscale array."""
    if isinstance(uploaded_file, bytes):
        raw_bytes = uploaded_file
    elif hasattr(uploaded_file, "getvalue"):
        raw_bytes = uploaded_file.getvalue()
    else:
        raw_bytes = uploaded_file.read()
    return decode_uploaded_image_details(raw_bytes)["grayscale"]


def decode_uploaded_image_details(
    raw_bytes: bytes,
    preview_max_side: int = 1200,
) -> dict[str, object]:
    """Decode upload bytes and return grayscale data, a display preview, and metadata."""
    if not raw_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Each image must be 20 MB or smaller.")

    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ValueError("The file could not be decoded. Check that it is a valid image.") from error
    if image is None or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("The file could not be decoded. Check that it is a valid image.")

    height, width = image.shape[:2]
    if height * width > MAX_DECODED_PIXELS:
        raise ValueError("The decoded image exceeds the 20-megapixel safety limit.")

    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in (1, 3, 4):
        raise ValueError("Supported images must be grayscale, RGB, or RGBA.")

    if image.ndim == 2 or channels == 1:
        grayscale = image if image.ndim == 2 else image[:, :, 0]
        preview = grayscale
    elif channels == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        preview = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    grayscale = _to_uint8(grayscale)
    preview = _to_uint8(preview)
    preview = _resize_preview(preview, preview_max_side)
    return {
        "grayscale": grayscale,
        "preview": preview,
        "width": width,
        "height": height,
        "channels": channels,
        "file_size": len(raw_bytes),
    }


def prepare_image_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    size_mode: str = "Image 1 dimensions",
    max_side: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert images to normalized grayscale and resize them to one shape."""
    first = _to_grayscale_float(image_a)
    second = _to_grayscale_float(image_b)

    if size_mode == "Smaller dimensions":
        target_height = min(first.shape[0], second.shape[0])
        target_width = min(first.shape[1], second.shape[1])
    elif size_mode == "Image 1 dimensions":
        target_height, target_width = first.shape
    else:
        raise ValueError(f"Unsupported image sizing mode: {size_mode}")

    if max_side is not None:
        if max_side < 2:
            raise ValueError("Maximum processing side must be at least 2 pixels.")
        scale = min(1.0, max_side / max(target_height, target_width))
        target_height = max(2, int(round(target_height * scale)))
        target_width = max(2, int(round(target_width * scale)))

    if min(target_height, target_width) < 2:
        raise ValueError("Images must be at least 2 pixels wide and high.")

    first = _resize(first, (target_width, target_height))
    second = _resize(second, (target_width, target_height))
    return first, second


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Both input images must contain pixel data.")
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError("Images must be grayscale, RGB, or RGBA.")
    elif image.ndim != 2:
        raise ValueError("Images must be two-dimensional or color images.")

    result = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        max_value = float(np.iinfo(image.dtype).max)
        result /= max_value
    elif result.max(initial=0.0) > 1.0 or result.min(initial=0.0) < 0.0:
        result = cv2.normalize(result, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.clip(result, 0.0, 1.0)


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (image.shape[1], image.shape[0]) == size:
        return image.astype(np.float32, copy=False)
    shrinking = size[0] < image.shape[1] or size[1] < image.shape[0]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _resize_preview(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)