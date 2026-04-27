"""Sobol index computation from Chebyshev spectral coefficients.

References
----------
- Sobol (2001) — "Global sensitivity indices for nonlinear mathematical models"
- Chebyshev expansion variance: ⟨T_n, T_n⟩ = π/2 (n ≥ 1) or π (n = 0) under
  weight ω(x) = 1/sqrt(1-x²) on [-1, 1]. Multi-D: product of 1-D weights.
"""
from __future__ import annotations

import numpy as np


def _chebyshev_norm_squared(degree: int) -> float:
    """⟨T_n, T_n⟩ for Chebyshev T_n under weight ω(x) = 1/sqrt(1-x²)."""
    if degree == 0:
        return float(np.pi)
    return float(np.pi) / 2.0


def _multi_index_norm_squared(alpha) -> float:
    """Multi-D inner product = product of per-dim norms squared."""
    result = 1.0
    for d in alpha:
        result *= _chebyshev_norm_squared(int(d))
    return result


def _compute_sobol_from_coeffs(
    coeffs: np.ndarray, num_dimensions: int
) -> dict:
    """Compute first-order and total-order Sobol indices from a tensor of
    Chebyshev coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Shape (n_0, ..., n_{d-1}). coeffs[α] = c_α for multi-index α.
    num_dimensions : int

    Returns
    -------
    dict
        {"first_order": {dim: index}, "total_order": {dim: index}, "variance": float}
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if num_dimensions == 1:
        coeffs = coeffs.reshape(-1)
        first_order = {0: 0.0}
        total_order = {0: 0.0}
        variance = 0.0
        for i in range(1, len(coeffs)):
            energy = coeffs[i] * coeffs[i] * _chebyshev_norm_squared(i)
            variance += energy
        if variance > 0:
            first_order[0] = 1.0
            total_order[0] = 1.0
        return {
            "first_order": first_order,
            "total_order": total_order,
            "variance": float(variance),
        }

    first_order_energy = {d: 0.0 for d in range(num_dimensions)}
    total_order_energy = {d: 0.0 for d in range(num_dimensions)}
    variance = 0.0

    for idx in np.ndindex(*coeffs.shape):
        if all(i == 0 for i in idx):
            continue
        c = coeffs[idx]
        if c == 0:
            continue
        energy = float(c * c * _multi_index_norm_squared(idx))
        variance += energy
        nonzero_dims = [d for d, deg in enumerate(idx) if deg > 0]
        if len(nonzero_dims) == 1:
            first_order_energy[nonzero_dims[0]] += energy
        for d in nonzero_dims:
            total_order_energy[d] += energy

    if variance == 0:
        return {
            "first_order": {d: 0.0 for d in range(num_dimensions)},
            "total_order": {d: 0.0 for d in range(num_dimensions)},
            "variance": 0.0,
        }

    return {
        "first_order": {
            d: first_order_energy[d] / variance for d in range(num_dimensions)
        },
        "total_order": {
            d: total_order_energy[d] / variance for d in range(num_dimensions)
        },
        "variance": variance,
    }
