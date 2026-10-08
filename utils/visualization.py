"""Matplotlib visualizations for DWT bands and image comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _apply_dark_style(figure: plt.Figure, axes: np.ndarray | list) -> None:
    figure.patch.set_facecolor("#10201d")
    for axis in np.asarray(axes, dtype=object).flat:
        axis.set_facecolor("#0b1714")
        axis.title.set_color("#e8f3ef")
        axis.xaxis.label.set_color("#b4c8bf")
        axis.yaxis.label.set_color("#b4c8bf")
        axis.tick_params(colors="#9bb0a8")
        for spine in axis.spines.values():
            spine.set_color("#30453e")


def decomposition_figure(coefficients: list, image_name: str = "Image") -> plt.Figure:
    """Render LL, LH, HL, and HH coefficient bands in a 2×2 layout."""
    approximation = coefficients[0]
    horizontal, vertical, diagonal = coefficients[-1]
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    panels = [
        (approximation, "LL · Approximation", "gray", False),
        (horizontal, "LH · Horizontal details", "coolwarm", True),
        (vertical, "HL · Vertical details", "coolwarm", True),
        (diagonal, "HH · Diagonal details", "coolwarm", True),
    ]
    for axis, (band, title, color_map, diverging) in zip(axes.flat, panels):
        if diverging:
            limit = max(float(np.max(np.abs(band))), np.finfo(np.float32).eps)
            axis.imshow(band, cmap=color_map, vmin=-limit, vmax=limit)
        else:
            axis.imshow(band, cmap=color_map)
        axis.set_title(title, fontsize=10, pad=8)
        axis.axis("off")
    figure.suptitle(f"{image_name} · DWT coefficient bands", fontsize=12, fontweight="bold")
    _apply_dark_style(figure, axes)
    figure._suptitle.set_color("#e8f3ef")
    return figure


def singular_value_figure(
    coefficients_a: list,
    coefficients_b: list,
) -> tuple[plt.Figure, dict[str, float | int]]:
    """Plot LL singular spectra and summarize the retained low-rank energy."""
    approximation_a = coefficients_a[0]
    approximation_b = coefficients_b[0]
    singular_values_a = np.linalg.svd(approximation_a, compute_uv=False)
    singular_values_b = np.linalg.svd(approximation_b, compute_uv=False)
    retained_rank_a = max(1, int(np.ceil(singular_values_a.size * 0.1)))
    retained_rank_b = max(1, int(np.ceil(singular_values_b.size * 0.1)))
    energy_a = float(np.sum(np.square(singular_values_a), dtype=np.float64))
    energy_b = float(np.sum(np.square(singular_values_b), dtype=np.float64))
    retained_a = float(np.sum(np.square(singular_values_a[:retained_rank_a]), dtype=np.float64))
    retained_b = float(np.sum(np.square(singular_values_b[:retained_rank_b]), dtype=np.float64))

    figure, axis = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    axis.plot(np.arange(1, singular_values_a.size + 1), singular_values_a, marker="o", markersize=3, label="Image 1")
    axis.plot(np.arange(1, singular_values_b.size + 1), singular_values_b, marker="o", markersize=3, label="Image 2")
    axis.axvline(retained_rank_a, color="#315f58", linestyle="--", linewidth=1, alpha=0.7)
    axis.set_title("LL singular-value spectrum", loc="left", fontweight="bold")
    axis.set_xlabel("Singular value index")
    axis.set_ylabel("Magnitude")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False)
    return figure, {
        "rank_a": retained_rank_a,
        "rank_b": retained_rank_b,
        "energy_a": retained_a / energy_a if energy_a else 0.0,
        "energy_b": retained_b / energy_b if energy_b else 0.0,
    }


def singular_spectrum_figure(values_a: np.ndarray, values_b: np.ndarray) -> plt.Figure:
    """Plot singular spectra already computed for the fusion operation."""
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    figure, axis = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    axis.plot(np.arange(1, values_a.size + 1), values_a, color="#59ddb0", linewidth=1.8, label="Image A")
    axis.plot(np.arange(1, values_b.size + 1), values_b, color="#83b9ee", linewidth=1.8, label="Image B")
    axis.set_title("LL singular-value spectrum", loc="left", fontweight="bold")
    axis.set_xlabel("Singular value index")
    axis.set_ylabel("Magnitude")
    axis.grid(True, alpha=0.18)
    axis.legend(frameon=False)
    _apply_dark_style(figure, [axis])
    _apply_dark_style(figure, [axis])
    return figure


def svd_factor_figure(factors: dict, image_name: str) -> plt.Figure:
    """Show compact heatmaps of retained U and Vᵀ factors and singular values."""
    left = np.asarray(factors["left"])
    values = np.asarray(factors["singular_values"])
    right = np.asarray(factors["right"])
    rank = int(factors["rank"])
    component_count = min(rank, 32)
    row_indices = np.linspace(0, max(0, left.shape[0] - 1), min(128, left.shape[0])).astype(int)
    column_indices = np.linspace(0, max(0, right.shape[1] - 1), min(128, right.shape[1])).astype(int)
    left_preview = left[row_indices, :component_count]
    right_preview = right[:component_count, :][:, column_indices]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    axes[0].imshow(left_preview, cmap="coolwarm", aspect="auto")
    axes[0].set_title(f"U · retained modes ({component_count})")
    axes[1].plot(np.arange(1, values.size + 1), values, color="#4ad6aa", linewidth=1.8)
    axes[1].axvline(rank, color="#e5b65c", linestyle="--", linewidth=1)
    axes[1].set_title("Singular values")
    axes[1].set_xlabel("Index")
    axes[1].grid(alpha=0.18)
    axes[2].imshow(right_preview, cmap="coolwarm", aspect="auto")
    axes[2].set_title(f"Vᵀ · retained modes ({component_count})")
    for axis in (axes[0], axes[2]):
        axis.set_xlabel("Retained mode")
        axis.set_ylabel("Sampled coefficient index")
    figure.suptitle(f"{image_name} · compact SVD factors", fontsize=12, fontweight="bold")
    _apply_dark_style(figure, axes)
    figure._suptitle.set_color("#e8f3ef")
    return figure


def intensity_histogram_figure(
    image_a: np.ndarray,
    image_b: np.ndarray,
    fused: np.ndarray,
) -> plt.Figure:
    """Compare grayscale intensity distributions of both inputs and the output."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True, constrained_layout=True)
    panels = [
        (image_a, "Image A", "#48c7a1"),
        (image_b, "Image B", "#76a9fa"),
        (fused, "Fused output", "#f0b65a"),
    ]
    for axis, (image, title, color) in zip(axes, panels):
        axis.hist(np.asarray(image, dtype=np.uint8).ravel(), bins=64, range=(0, 256), color=color, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("Intensity")
        axis.grid(axis="y", alpha=0.16)
    axes[0].set_ylabel("Pixel count")
    _apply_dark_style(figure, axes)
    return figure


def comparison_figure(
    image_a: np.ndarray,
    image_b: np.ndarray,
    fused: np.ndarray,
) -> plt.Figure:
    """Render both inputs and their fused result in a consistent layout."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    panels = [(image_a, "Image 1"), (image_b, "Image 2"), (fused, "Fused image")]
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title, fontweight="bold", pad=10)
        axis.axis("off")
    _apply_dark_style(figure, axes)
    figure._suptitle.set_color("#e8f3ef") if figure._suptitle else None
    return figure