"""Multilevel wavelet decomposition and coefficient fusion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pywt

from .svd_fusion import svd_energy_fuse

WaveletCoefficients = list[Any]


def decompose_image(
    image: np.ndarray,
    wavelet: str = "db2",
    level: int = 1,
) -> WaveletCoefficients:
    """Return a 2D wavelet decomposition in PyWavelets coefficient order."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("DWT input must be a non-empty grayscale image.")
    if wavelet not in pywt.wavelist(kind="discrete"):
        raise ValueError(f"Unsupported discrete wavelet: {wavelet}")
    if level < 1:
        raise ValueError("DWT level must be at least 1.")

    maximum_level = pywt.dwtn_max_level(image.shape, wavelet)
    if maximum_level < 1:
        minimum_size = 2 * pywt.Wavelet(wavelet).dec_len
        padded_height = max(image.shape[0], minimum_size)
        padded_width = max(image.shape[1], minimum_size)
        pad_height = padded_height - image.shape[0]
        pad_width = padded_width - image.shape[1]
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width)),
            mode="symmetric",
        )
        maximum_level = pywt.dwtn_max_level(image.shape, wavelet)
        if maximum_level < 1:
            raise ValueError("Image cannot be padded to a supported DWT size.")
    actual_level = min(level, maximum_level)
    return pywt.wavedec2(image.astype(np.float32), wavelet=wavelet, level=actual_level)


def fuse_coefficients(
    coefficients_a: WaveletCoefficients,
    coefficients_b: WaveletCoefficients,
    method: str = "SVD energy + max-absolute details",
    svd_rank: int | None = None,
    svd_analysis: dict | None = None,
) -> WaveletCoefficients:
    """Fuse approximations with SVD and detail bands with a selected rule."""
    if len(coefficients_a) != len(coefficients_b):
        raise ValueError("DWT decompositions must have the same number of levels.")
    if not coefficients_a:
        raise ValueError("DWT coefficient pyramids cannot be empty.")
    if method not in {"SVD energy + max-absolute details", "SVD energy all bands"}:
        raise ValueError(f"Unsupported fusion method: {method}")

    fused: WaveletCoefficients = [
        svd_energy_fuse(
            coefficients_a[0],
            coefficients_b[0],
            rank=svd_rank,
            analysis=svd_analysis,
        )
    ]
    for details_a, details_b in zip(coefficients_a[1:], coefficients_b[1:]):
        if len(details_a) != 3 or len(details_b) != 3:
            raise ValueError("Each DWT level must contain LH, HL, and HH detail bands.")
        fused_details = []
        for band_a, band_b in zip(details_a, details_b):
            if band_a.shape != band_b.shape:
                raise ValueError("DWT coefficient bands have mismatched shapes.")
            if method == "SVD energy all bands":
                fused_band = svd_energy_fuse(band_a, band_b, rank=svd_rank)
            else:
                fused_band = np.where(np.abs(band_a) >= np.abs(band_b), band_a, band_b)
            fused_details.append(fused_band.astype(np.float32, copy=False))
        fused.append(tuple(fused_details))
    return fused


def coefficient_summary(coefficients: WaveletCoefficients) -> dict[str, object]:
    """Describe approximation and detail coefficient array sizes."""
    approximation = coefficients[0]
    details = coefficients[-1]
    return {
        "level": len(coefficients) - 1,
        "approximation_shape": approximation.shape,
        "detail_shapes": {
            "LH": details[0].shape,
            "HL": details[1].shape,
            "HH": details[2].shape,
        },
    }