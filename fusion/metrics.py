"""Image quality metrics for comparing each source with the fused result."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def calculate_metrics(reference: np.ndarray, fused: np.ndarray) -> dict[str, float | None]:
    """Calculate MSE, PSNR, SSIM, and fused-image entropy."""
    first, second = _as_uint8_scale(reference), _as_uint8_scale(fused)
    if first.shape != second.shape:
        raise ValueError("Metrics require images with identical dimensions.")

    difference = first - second
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse))
    ssim = _safe_ssim(first, second)
    entropy = _entropy(fused)
    return {"MSE": mse, "PSNR (dB)": psnr, "SSIM": ssim, "Entropy (bits)": entropy}


def _safe_ssim(first: np.ndarray, second: np.ndarray) -> float | None:
    min_dimension = min(first.shape)
    if min_dimension < 3:
        return None
    window_size = min(7, min_dimension if min_dimension % 2 else min_dimension - 1)
    try:
        return float(
            structural_similarity(
                first,
                second,
                data_range=255.0,
                win_size=window_size,
                channel_axis=None,
            )
        )
    except (ValueError, ZeroDivisionError):
        return None


def _entropy(image: np.ndarray) -> float:
    values = np.asarray(image, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / values.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _as_uint8_scale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("Metrics cannot be calculated for an empty image.")
    if values.ndim != 2:
        raise ValueError("Metrics expect grayscale images.")
    result = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
    if np.issubdtype(values.dtype, np.unsignedinteger):
        if np.iinfo(values.dtype).max > 255 and result.max() > 255.0:
            result *= 255.0 / np.iinfo(values.dtype).max
    elif np.issubdtype(values.dtype, np.signedinteger):
        if result.min() < 0.0 or result.max() > 255.0:
            limits = np.iinfo(values.dtype)
            result = (result - limits.min) * (255.0 / (limits.max - limits.min))
    else:
        if result.max() <= 1.0 and result.min() >= 0.0:
            result *= 255.0
        elif result.min() < 0.0 or result.max() > 255.0:
            low, high = float(result.min()), float(result.max())
            result = np.zeros_like(result) if high == low else (result - low) * (255.0 / (high - low))
    return np.clip(result, 0.0, 255.0)