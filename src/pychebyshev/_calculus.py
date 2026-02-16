"""Shared helpers for Chebyshev calculus operations (integration, roots, optimization).

References
----------
- Waldvogel (2006), "Fast Construction of the Fejér and Clenshaw–Curtis
  Quadrature Rules", BIT Numerical Mathematics 46(2):195–202.
- Trefethen (2013), "Approximation Theory and Approximation Practice",
  SIAM, Chapters 18–21.
- Good (1961), "The colleague matrix, a Chebyshev analogue of the companion
  matrix", Quarterly J. Mech. 14:195–196.
"""

from __future__ import annotations

import numpy as np


def _compute_fejer1_weights(n: int) -> np.ndarray:
    """Compute Fejér-1 quadrature weights for *n* Type I Chebyshev nodes.

    Returns weights in **ascending** node order (PyChebyshev convention)
    such that ``sum(w * f(nodes)) ≈ ∫_{-1}^{1} f(x) dx``.

    Uses the O(n log n) algorithm of Waldvogel (2006) via DCT-III.

    Parameters
    ----------
    n : int
        Number of Chebyshev Type I nodes.

    Returns
    -------
    ndarray of shape (n,)
        Quadrature weights on [-1, 1], ascending node order.
    """
    from scipy.fft import dct

    # Integration moments: I_k = ∫_{-1}^{1} T_k(x) dx
    #   I_k = 2/(1-k²) for k even, 0 for k odd
    moments = np.zeros(n)
    for k in range(0, n, 2):
        moments[k] = 2.0 / (1.0 - k * k)

    # DCT-III computes: y[j] = x[0] + 2*Σ_{k≥1} x[k]*cos(πk(2j+1)/(2n))
    # Dividing by n gives the Fejér-1 weights in descending node order.
    weights_desc = dct(moments, type=3) / n

    # Reverse to ascending order (PyChebyshev convention)
    return weights_desc[::-1].copy()


def _integrate_tensor_along_dim(tensor: np.ndarray, dim: int,
                                n: int, scale: float) -> np.ndarray:
    """Contract *tensor* along axis *dim* using Fejér-1 quadrature weights.

    Parameters
    ----------
    tensor : ndarray
        Multi-dimensional array of function values.
    dim : int
        Axis to integrate out.
    n : int
        Number of nodes along *dim* (used to compute weights).
    scale : float
        Domain half-width ``(b - a) / 2`` for that dimension.

    Returns
    -------
    ndarray
        Tensor with *dim* removed.
    """
    weights = _compute_fejer1_weights(n)
    return np.tensordot(tensor, weights * scale, axes=([dim], [0]))


def _roots_1d(values: np.ndarray, domain: tuple) -> np.ndarray:
    """Find all real roots of a 1-D Chebyshev interpolant within its domain.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Function values at ascending Chebyshev Type I nodes.
    domain : (float, float)
        Physical domain ``[a, b]``.

    Returns
    -------
    ndarray
        Sorted real roots in ``[a, b]``.
    """
    from numpy.polynomial.chebyshev import chebroots

    from pychebyshev.barycentric import ChebyshevApproximation

    coeffs = ChebyshevApproximation._chebyshev_coefficients_1d(values)
    raw_roots = chebroots(coeffs)

    # Keep only real roots inside [-1, 1]
    tol = 1e-10
    real_roots = []
    for r in raw_roots:
        if abs(r.imag) < tol:
            t = r.real
            if -1.0 - tol <= t <= 1.0 + tol:
                real_roots.append(np.clip(t, -1.0, 1.0))

    if len(real_roots) == 0:
        return np.array([], dtype=float)

    # Map from [-1, 1] to [a, b]
    a, b = domain
    physical = 0.5 * (a + b) + 0.5 * (b - a) * np.array(real_roots)

    # Sort and deduplicate (tolerance for near-identical roots)
    physical = np.sort(physical)
    if len(physical) > 1:
        mask = np.concatenate([[True], np.diff(physical) > 1e-10 * (b - a + 1)])
        physical = physical[mask]

    return physical


def _optimize_1d(values: np.ndarray, nodes: np.ndarray,
                 bary_weights: np.ndarray, diff_matrix: np.ndarray,
                 domain: tuple, mode: str = "min") -> tuple:
    """Find the minimum or maximum of a 1-D Chebyshev interpolant.

    Parameters
    ----------
    values : ndarray of shape (n,)
        Function values at ascending Chebyshev Type I nodes.
    nodes : ndarray of shape (n,)
        Chebyshev nodes in the physical domain.
    bary_weights : ndarray of shape (n,)
        Barycentric weights.
    diff_matrix : ndarray of shape (n, n)
        Spectral differentiation matrix.
    domain : (float, float)
        Physical domain ``[a, b]``.
    mode : {'min', 'max'}
        Whether to find the minimum or maximum.

    Returns
    -------
    (value, location) : (float, float)
    """
    from pychebyshev.barycentric import barycentric_interpolate

    # Derivative values at nodes
    deriv_values = diff_matrix @ values

    # Critical points: roots of the derivative
    critical = _roots_1d(deriv_values, domain)

    # Candidates: critical points + domain endpoints
    a, b = domain
    candidates = np.concatenate([[a], critical, [b]])

    # Evaluate original function at all candidates
    vals = np.array([
        barycentric_interpolate(float(x), nodes, values, bary_weights)
        for x in candidates
    ])

    idx = np.argmin(vals) if mode == "min" else np.argmax(vals)
    return float(vals[idx]), float(candidates[idx])


def _validate_calculus_args(ndim, dim, fixed, domain):
    """Validate arguments for roots/minimize/maximize.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the interpolant.
    dim : int or None
        Target dimension.
    fixed : dict or None
        ``{dim_index: value}`` for all other dimensions.
    domain : list
        Per-dimension ``[lo, hi]`` bounds.

    Returns
    -------
    (dim, slice_params) : (int, list of (int, float))
        Validated target dim and list of slice parameters.

    Raises
    ------
    ValueError
        If *dim* is out of range, *fixed* is incomplete, or values are
        out of domain.
    """
    if ndim == 1:
        dim = 0 if dim is None else dim
        if dim != 0:
            raise ValueError(f"dim must be 0 for 1-D interpolant, got {dim}")
        if fixed and len(fixed) > 0:
            raise ValueError("fixed must be empty for 1-D interpolant")
        return dim, []

    if dim is None:
        raise ValueError("dim is required for multi-D interpolant")
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim {dim} out of range [0, {ndim - 1}]")

    if fixed is None:
        fixed = {}
    expected = set(range(ndim)) - {dim}
    provided = set(fixed.keys())
    if provided != expected:
        missing = expected - provided
        raise ValueError(f"fixed must specify all dims except {dim}; missing {missing}")

    slice_params = []
    for d, v in fixed.items():
        lo, hi = domain[d]
        if v < lo or v > hi:
            raise ValueError(
                f"Fixed value {v} for dim {d} outside domain [{lo}, {hi}]"
            )
        slice_params.append((d, v))

    return dim, slice_params
