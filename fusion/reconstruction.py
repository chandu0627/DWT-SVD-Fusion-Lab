"""Inverse wavelet reconstruction and output conversion."""

from __future__ import annotations

import numpy as np
import pywt


def reconstruct_image(
    coefficients: list,
    wavelet: str = "db2",
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reconstruct a coefficient pyramid and return an 8-bit grayscale image."""
    reconstructed = pywt.waverec2(coefficients, wavelet=wavelet)
    if output_shape is not None:
        if len(output_shape) != 2 or min(output_shape) < 1:
            raise ValueError("Output shape must contain positive height and width values.")
        target_height, target_width = output_shape
        reconstructed = reconstructed[:target_height, :target_width]
        missing_height = max(0, target_height - reconstructed.shape[0])
        missing_width = max(0, target_width - reconstructed.shape[1])
        if missing_height or missing_width:
            reconstructed = np.pad(
                reconstructed,
                ((0, missing_height), (0, missing_width)),
                mode="edge",
            )
    reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=1.0, neginf=0.0)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    return np.rint(reconstructed * 255.0).astype(np.uint8)