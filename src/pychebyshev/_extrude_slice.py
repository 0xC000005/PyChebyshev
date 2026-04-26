"""Shared helpers for Chebyshev extrusion and slicing operations."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.chebyshev import chebpts1


def _normalize_extrusion_params(params, ndim):
    """Validate and normalize extrusion parameters.

    Accepts single tuple (dim_idx, (lo, hi), n_nodes) or list of such tuples.
    Returns sorted list of tuples (ascending by dim_index).
    Validates: dim_index in [0, ndim + len(params) - 1], no duplicates, n >= 2, lo < hi.
    """
    if isinstance(params, tuple) and len(params) == 3 and isinstance(params[0], (int, np.integer)):
        params = [params]
    params = [tuple(p) for p in params]

    new_ndim = ndim + len(params)
    seen = set()
    for dim_idx, bounds, n in params:
        if not isinstance(dim_idx, (int, np.integer)):
            raise TypeError(f"dim_index must be int, got {type(dim_idx).__name__}")
        if dim_idx < 0 or dim_idx >= new_ndim:
            raise ValueError(f"dim_index {dim_idx} out of range [0, {new_ndim - 1}]")
        if dim_idx in seen:
            raise ValueError(f"Duplicate dim_index {dim_idx}")
        seen.add(dim_idx)
        lo, hi = bounds
        if lo >= hi:
            raise ValueError(f"Domain bounds must satisfy lo < hi, got [{lo}, {hi}]")
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError(f"n_nodes must be int >= 2, got {n}")

    return sorted(params, key=lambda p: p[0])


def _normalize_slicing_params(params, ndim):
    """Validate and normalize slicing parameters.

    Accepts single tuple (dim_idx, value) or list of such tuples.
    Returns sorted list of tuples (descending by dim_index, so we can remove from back to front).
    Validates: dim_index in [0, ndim-1], no duplicates, cannot slice all dims.
    """
    if isinstance(params, tuple) and len(params) == 2 and isinstance(params[0], (int, np.integer)):
        params = [params]
    params = [tuple(p) for p in params]

    if len(params) >= ndim:
        raise ValueError(f"Cannot slice all {ndim} dimensions (would produce 0D result)")

    seen = set()
    for dim_idx, value in params:
        if not isinstance(dim_idx, (int, np.integer)):
            raise TypeError(f"dim_index must be int, got {type(dim_idx).__name__}")
        if dim_idx < 0 or dim_idx >= ndim:
            raise ValueError(f"dim_index {dim_idx} out of range [0, {ndim - 1}]")
        if dim_idx in seen:
            raise ValueError(f"Duplicate dim_index {dim_idx}")
        seen.add(dim_idx)

    return sorted(params, key=lambda p: p[0], reverse=True)


def _make_nodes_for_dim(lo, hi, n):
    """Create Chebyshev nodes on [lo, hi] with n points."""
    nodes_std = chebpts1(n)
    nodes = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes_std
    return np.sort(nodes)


def _extrude_tensor(tensor, axis, n_new):
    """Insert a new axis and replicate values."""
    expanded = np.expand_dims(tensor, axis=axis)
    return np.repeat(expanded, n_new, axis=axis)


def _slice_tensor(tensor, axis, nodes, weights, value):
    """Contract tensor along axis at the given value via barycentric interpolation.

    If value coincides with a node (within 1e-14), use np.take (exact).
    Otherwise use barycentric formula: tensordot with normalized weights.
    """
    diff = value - nodes
    exact_idx = np.argmin(np.abs(diff))
    if np.abs(diff[exact_idx]) < 1e-14:
        return np.take(tensor, exact_idx, axis=axis)

    w_over_diff = weights / diff
    w_norm = w_over_diff / np.sum(w_over_diff)
    return np.tensordot(tensor, w_norm, axes=([axis], [0]))


def _extrude_tt_core(coeff_cores, dim_idx, lo, hi, n_new):
    """Insert a constant rank-preserving core at position ``dim_idx`` into a TT.

    The new core encodes the constant function 1 over the new dim. In DCT-II
    coefficient space (PyChebyshev TT convention), the constant 1 has only
    c_0 = 1 (the halved-c_0 convention means the stored coefficient is 1.0).
    The core is rank-preserving: ``core[i, 0, i] = 1`` for all i.

    Parameters
    ----------
    coeff_cores : list[np.ndarray]
        Existing TT coefficient cores.
    dim_idx : int
        Position to insert the new core (0 <= dim_idx <= len(coeff_cores)).
    lo, hi : float
        Domain of the new dim (for validation only).
    n_new : int
        Number of nodes for the new dim.

    Returns
    -------
    list[np.ndarray]
        New cores list with the constant core inserted at position dim_idx.
    """
    if dim_idx < 0 or dim_idx > len(coeff_cores):
        raise ValueError(
            f"dim_idx={dim_idx} out of range [0, {len(coeff_cores)}]"
        )
    if lo >= hi:
        raise ValueError(f"lo ({lo}) must be < hi ({hi})")
    if n_new < 1:
        raise ValueError(f"n_new must be >= 1, got {n_new}")

    # Determine the rank at the insertion point.
    # At boundary positions (0 or end), rank stays 1.
    # At interior positions, use the right rank of core dim_idx - 1.
    if dim_idx == 0:
        r_at = 1
    elif dim_idx == len(coeff_cores):
        r_at = 1
    else:
        r_at = coeff_cores[dim_idx - 1].shape[2]

    # Construct rank-preserving core: shape (r_at, n_new, r_at).
    # core[i, 0, i] = 1.0: c_0 coefficient of constant function 1.
    # All other slots (j > 0 or i != l) remain 0.
    new_core = np.zeros((r_at, n_new, r_at))
    for i in range(r_at):
        new_core[i, 0, i] = 1.0

    return coeff_cores[:dim_idx] + [new_core] + coeff_cores[dim_idx:]
