"""SVD-based coefficient fusion helpers."""

from __future__ import annotations

import math

import numpy as np


def svd_energy_fuse(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    rank: int | None = None,
    analysis: dict[str, dict[str, np.ndarray | int | float]] | None = None,
) -> np.ndarray:
    """Blend truncated SVD reconstructions, weighted by retained energy."""
    if matrix_a.shape != matrix_b.shape:
        raise ValueError("SVD fusion requires coefficient matrices with equal shapes.")
    if matrix_a.size == 0:
        raise ValueError("SVD fusion cannot process empty coefficient matrices.")
    if rank is not None and rank < 1:
        raise ValueError("SVD rank must be a positive integer.")

    reconstruction_a, energy_a, factors_a = _truncated_reconstruction(matrix_a, rank)
    reconstruction_b, energy_b, factors_b = _truncated_reconstruction(matrix_b, rank)
    total_full_energy = factors_a["total_energy"] + factors_b["total_energy"]
    if analysis is not None:
        analysis["image_a"] = factors_a
        analysis["image_b"] = factors_b
        analysis["weight_a"] = energy_a / (energy_a + energy_b) if energy_a + energy_b else 0.5
        analysis["weight_b"] = energy_b / (energy_a + energy_b) if energy_a + energy_b else 0.5
        analysis["combined_retained_fraction"] = (
            (energy_a + energy_b) / total_full_energy if total_full_energy else 0.0
        )
    total_energy = energy_a + energy_b
    if total_energy <= np.finfo(np.float64).eps:
        return (matrix_a + matrix_b) * 0.5

    weight_a = energy_a / total_energy
    weight_b = energy_b / total_energy
    return (weight_a * reconstruction_a + weight_b * reconstruction_b).astype(np.float32)


def _truncated_reconstruction(
    matrix: np.ndarray,
    rank: int | None,
) -> tuple[np.ndarray, float, dict[str, np.ndarray | int | float]]:
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros_like(matrix, dtype=np.float32), 0.0, {
            "left": left.astype(np.float32),
            "singular_values": singular_values.astype(np.float32),
            "right": right.astype(np.float32),
            "rank": 0,
            "retained_energy_fraction": 0.0,
            "total_energy": 0.0,
        }

    total_energy = float(np.sum(np.square(singular_values), dtype=np.float64))
    full_singular_values = singular_values.copy()
    retained_rank = (
        max(1, math.ceil(singular_values.size * 0.1))
        if rank is None
        else max(1, min(rank, singular_values.size))
    )
    left = left[:, :retained_rank]
    singular_values = singular_values[:retained_rank]
    right = right[:retained_rank, :]
    approximation = (left * singular_values) @ right
    energy = float(np.sum(np.square(singular_values), dtype=np.float64))
    factors: dict[str, np.ndarray | int | float] = {
        "left": left.astype(np.float32),
        "singular_values": full_singular_values.astype(np.float32),
        "right": right.astype(np.float32),
        "rank": retained_rank,
        "retained_energy_fraction": energy / total_energy if total_energy else 0.0,
        "total_energy": total_energy,
    }
    return approximation.astype(np.float32), energy, factors