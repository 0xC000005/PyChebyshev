"""Sobol index computation from Chebyshev spectral coefficients.

References
----------
- Sobol (2001) — "Global sensitivity indices for nonlinear mathematical models"
- Chebyshev expansion variance: ⟨T_n, T_n⟩ = π/2 (n ≥ 1) or π (n = 0) under
  weight ω(x) = 1/sqrt(1-x²) on [-1, 1]. Multi-D: product of 1-D weights.
"""
from __future__ import annotations

import numpy as np


def _compute_chebyshev_coefficients(
    tensor_values: np.ndarray, num_dimensions: int
) -> np.ndarray:
    """Convert tensor of function values at Chebyshev Type-I nodes to
    Chebyshev expansion coefficients using the canonical PyChebyshev DCT
    convention.

    Each axis is processed with the same convention used by
    ``ChebyshevApproximation._chebyshev_coefficients_1d``:
    reverse along the axis, apply DCT-II, divide by n, and halve c_0.

    Parameters
    ----------
    tensor_values : np.ndarray
        Shape (n_0, ..., n_{d-1}).  Values at tensor-product Type-I nodes.
    num_dimensions : int
        Number of dimensions (equals ``tensor_values.ndim``).

    Returns
    -------
    np.ndarray
        Chebyshev coefficient tensor of the same shape.
    """
    from scipy.fft import dct

    coeffs = tensor_values.copy().astype(np.float64)
    for axis in range(num_dimensions):
        n = coeffs.shape[axis]
        reversed_along = np.flip(coeffs, axis=axis)
        coeffs = dct(reversed_along, type=2, axis=axis) / n
    # Halve c_0 along every axis (per PyChebyshev convention)
    for d in range(num_dimensions):
        slicer = [slice(None)] * num_dimensions
        slicer[d] = 0
        coeffs[tuple(slicer)] *= 0.5
    return coeffs


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
    if not np.isfinite(coeffs).all():
        raise ValueError(
            "coefficients contain NaN or Inf; sobol_indices() requires finite "
            "spectral coefficients"
        )
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


def _compute_sobol_from_tt_cores(cores: list) -> dict:
    """Compute first-order + total-order Sobol indices from TT coefficient cores.

    Mathematically equivalent to ``_compute_sobol_from_coeffs`` applied to the
    dense coefficient tensor, but contracts through TT cores in coefficient
    space with cost O(d * n * r^2) instead of O(n^d).

    Parameters
    ----------
    cores : list of np.ndarray
        TT cores in Chebyshev coefficient space. Each core has shape
        (r_{k-1}, n_k, r_k) with cores[0] starting from r_0=1 and
        cores[-1] ending at r_d=1.

    Returns
    -------
    dict
        Same shape as ``_compute_sobol_from_coeffs``:
        ``{"first_order": {d: index}, "total_order": {d: index}, "variance": float}``
        where keys are storage-frame dim indices (0..len(cores)-1).
    """
    d = len(cores)
    pi = float(np.pi)
    n_per_dim = [c.shape[1] for c in cores]

    # Per-dim Chebyshev inner-product weights: [pi, pi/2, pi/2, ..., pi/2]
    w_full = []
    for n_k in n_per_dim:
        w = np.full(n_k, pi / 2.0)
        w[0] = pi
        w_full.append(w)

    # ---- total_weighted_squared = sum over all alpha of
    #      coeffs[alpha]^2 * prod_k w_full[k][alpha_k]
    M = np.array([[1.0]])
    for k in range(d):
        A = cores[k]
        Aw = A * w_full[k][None, :, None]
        M = np.einsum("ij,ipa,jpb->ab", M, Aw, A)
    total_weighted_squared = float(M[0, 0])

    # ---- constant term c_0 (alpha = 0) -- contract along all-zero slices
    v = np.array([1.0])
    for k in range(d):
        v = v @ cores[k][:, 0, :]
    c_0 = float(v[0])
    constant_weighted_squared = (c_0 ** 2) * (pi ** d)

    variance = total_weighted_squared - constant_weighted_squared

    if variance <= 0:
        return {
            "first_order": {j: 0.0 for j in range(d)},
            "total_order": {j: 0.0 for j in range(d)},
            "variance": float(max(variance, 0.0)),
        }

    # ---- Precompute left and right partial self-inner-product matrices
    #      to reduce total-order computation from O(d^2 * n * r^2) to
    #      O(d * n * r^2).
    #
    # L[k] has shape (r_k, r_k): partial contraction of dims 0..k-1.
    # R[k] has shape (r_k, r_k): partial contraction of dims k..d-1.
    #
    # L[0] = identity(1); L[k+1] = einsum("ij,ipa,jpb->ab", L[k], Aw_k, A_k)
    # R[d] = identity(1); R[k]   = einsum("ab,ipa,jpb->ij", R[k+1], Aw_k, A_k)
    #
    # For total-order[j]: the "alpha_j = 0" weighted sum decomposes as
    #   L[j] x cores[j][:,0,:] x R[j+1] (contracted with itself and scaled by pi).
    #   sum_alpha_j_zero = pi * einsum("ij,ia,jb,ab->", L[j], c_j0, c_j0, R[j+1])
    #   where c_j0 = cores[j][:, 0, :].

    # Build L[0..d]
    L = [None] * (d + 1)
    L[0] = np.array([[1.0]])
    for k in range(d):
        A_k = cores[k]
        Aw_k = A_k * w_full[k][None, :, None]
        L[k + 1] = np.einsum("ij,ipa,jpb->ab", L[k], Aw_k, A_k)

    # Build R[0..d]
    R = [None] * (d + 1)
    R[d] = np.array([[1.0]])
    for k in range(d - 1, -1, -1):
        A_k = cores[k]
        Aw_k = A_k * w_full[k][None, :, None]
        R[k] = np.einsum("ab,ipa,jpb->ij", R[k + 1], Aw_k, A_k)

    first_order_energy = {}
    total_order_energy = {}

    for j in range(d):
        # ---- first-order energy[j]: alpha_j >= 1 AND alpha_k = 0 for k != j
        # left boundary: row vector formed by chaining cores[k][:, 0, :] for k < j
        left = np.array([1.0])
        for k in range(j):
            left = left @ cores[k][:, 0, :]
        # left has shape (r_j,)

        # right boundary: column vector formed by chaining cores[k][:, 0, :] for k > j
        right = np.array([1.0])
        for k in range(d - 1, j, -1):
            right = cores[k][:, 0, :] @ right
        # right has shape (r_{j+1},)

        G_j = cores[j]
        sum_squared = 0.0
        for m in range(1, n_per_dim[j]):
            coef_m = float(left @ G_j[:, m, :] @ right)
            sum_squared += coef_m * coef_m
        weight_first = (pi / 2.0) * (pi ** (d - 1))
        first_order_energy[j] = sum_squared * weight_first

        # ---- total-order energy[j]: alpha_j >= 1 (other dims unrestricted)
        # = total_weighted_squared - sum_{alpha_j = 0} weighted
        # Using cached L[j] and R[j+1]:
        #   sum_alpha_j_zero = pi * einsum("ij,ia,jb,ab->", L[j], c_j0, c_j0, R[j+1])
        c_j0 = cores[j][:, 0, :]  # shape (r_j, r_{j+1})
        sum_alpha_j_zero_weighted = pi * float(
            np.einsum("ij,ia,jb,ab->", L[j], c_j0, c_j0, R[j + 1])
        )
        total_order_energy[j] = total_weighted_squared - sum_alpha_j_zero_weighted

    return {
        "first_order": {j: first_order_energy[j] / variance for j in range(d)},
        "total_order": {j: total_order_energy[j] / variance for j in range(d)},
        "variance": float(variance),
    }
