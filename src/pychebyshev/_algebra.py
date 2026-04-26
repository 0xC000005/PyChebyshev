"""Shared helpers for Chebyshev arithmetic operators."""

from __future__ import annotations

import numpy as np


def _is_scalar(value) -> bool:
    """Return True if *value* is a numeric scalar (int, float, or numpy scalar)."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _check_compatible(a, b) -> None:
    """Validate that two Chebyshev objects can be combined arithmetically.

    Both operands must:
    - be the same type
    - be built (tensor_values is not None, or _built is True)
    - have the same num_dimensions, domain, n_nodes, max_derivative_order
    """
    if type(a) is not type(b):
        raise TypeError(
            f"Cannot combine {type(a).__name__} with {type(b).__name__}; "
            f"operands must be the same type."
        )

    # Check both are built -- ChebyshevApproximation uses tensor_values,
    # ChebyshevSpline and ChebyshevSlider use _built flag
    a_built = (getattr(a, 'tensor_values', None) is not None) or getattr(a, '_built', False)
    b_built = (getattr(b, 'tensor_values', None) is not None) or getattr(b, '_built', False)
    if not a_built:
        raise RuntimeError("Left operand is not built. Call build() first.")
    if not b_built:
        raise RuntimeError("Right operand is not built. Call build() first.")

    if a.num_dimensions != b.num_dimensions:
        raise ValueError(
            f"Dimension mismatch: {a.num_dimensions} vs {b.num_dimensions}"
        )

    if a.n_nodes != b.n_nodes:
        raise ValueError(
            f"Node count mismatch: {a.n_nodes} vs {b.n_nodes}"
        )

    if a.domain != b.domain:
        raise ValueError(
            f"Domain mismatch: {a.domain} vs {b.domain}"
        )

    if a.max_derivative_order != b.max_derivative_order:
        raise ValueError(
            f"max_derivative_order mismatch: "
            f"{a.max_derivative_order} vs {b.max_derivative_order}"
        )


# ----------------------------------------------------------------------
# TT-specific helpers (block-diagonal addition + TT-SVD rounding)
# ----------------------------------------------------------------------


def _tt_add_cores(cores_a, cores_b):
    """Block-diagonal stacking of TT cores → exact TT representation of sum.

    For interior cores ``k in [1, d-2]`` the result is the block-diagonal of
    ``cores_a[k]`` and ``cores_b[k]``. The leftmost core concatenates along
    the right rank (left rank stays 1); the rightmost core concatenates along
    the left rank (right rank stays 1). For ``d == 1`` the lone core is
    a simple elementwise sum (the only way to keep both end-rank-1 invariants).

    Works directly on Chebyshev coefficient cores: the underlying tensor is a
    linear function of the coefficients, so block-diagonal stacking represents
    the sum of the two interpolants exactly.
    """
    d = len(cores_a)
    if d != len(cores_b):
        raise ValueError("cores must have same length")

    # d == 1 special case: cannot block-diagonal because both endpoint
    # invariants (left rank 1 and right rank 1) collide. The only correct
    # representation of f + g for d=1 is the elementwise coefficient sum.
    if d == 1:
        a = cores_a[0]
        b = cores_b[0]
        if a.shape != b.shape:
            raise ValueError(
                f"core 0 shape mismatch: {a.shape} vs {b.shape}"
            )
        return [a + b]

    new_cores = []
    for k in range(d):
        a = cores_a[k]
        b = cores_b[k]
        ra_l, n, ra_r = a.shape
        rb_l, n_b, rb_r = b.shape
        if n != n_b:
            raise ValueError(f"core {k} n_nodes mismatch: {n} vs {n_b}")
        if k == 0:
            # Concatenate along right rank: shape (1, n, ra_r + rb_r)
            new = np.concatenate([a, b], axis=2)
        elif k == d - 1:
            # Concatenate along left rank: shape (ra_l + rb_l, n, 1)
            new = np.concatenate([a, b], axis=0)
        else:
            # Block diagonal across (left_rank, right_rank); same node axis.
            new = np.zeros(
                (ra_l + rb_l, n, ra_r + rb_r),
                dtype=np.result_type(a.dtype, b.dtype),
            )
            new[:ra_l, :, :ra_r] = a
            new[ra_l:, :, ra_r:] = b
        new_cores.append(new)
    return new_cores


def _tt_round_cores(cores, max_rank, tolerance=1e-12):
    """Round TT to lower rank via TT-SVD recompression.

    Right-to-left QR sweep (right-canonicalize) followed by left-to-right
    SVD truncation. Truncation keeps ``min(max_rank, num_above_relative_tol)``
    singular values, where the relative tolerance is ``tolerance * s_max``.
    The represented function is preserved up to combined truncation error.

    Works on Chebyshev coefficient cores because expansion is linear: QR and
    SVD are over rank dimensions, and the node axis is treated identically
    in coefficient and value spaces for the purposes of recompression.
    """
    cores = [c.copy() for c in cores]
    d = len(cores)

    if d == 1:
        # Nothing to recompress: a single core has rank dims (1, 1).
        return cores

    # Right-to-left QR sweep: right-canonicalize cores k = d-1, ..., 1.
    for k in range(d - 1, 0, -1):
        r_l, n, r_r = cores[k].shape
        # Reshape to (r_l, n*r_r); QR of its transpose gives an
        # orthonormal basis for the row space.
        mat = cores[k].reshape(r_l, n * r_r)
        Q, R = np.linalg.qr(mat.T, mode="reduced")
        # mat = R.T @ Q.T  → core[k] becomes Q.T (rows orthonormal)
        Qt = Q.T
        new_r_l = Qt.shape[0]
        cores[k] = Qt.reshape(new_r_l, n, r_r)
        # Push R.T into the previous core's right rank: prev[l, j, s] @ R.T[s, r]
        prev = cores[k - 1]
        cores[k - 1] = np.einsum("ljs,sr->ljr", prev, R.T)

    # Left-to-right SVD truncation: truncate cores k = 0, ..., d-2.
    for k in range(d - 1):
        r_l, n, r_r = cores[k].shape
        mat = cores[k].reshape(r_l * n, r_r)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        # Truncate by max_rank AND by relative tolerance.
        s_max = S[0] if len(S) > 0 else 0.0
        keep = min(max_rank, len(S))
        if s_max > 0 and tolerance > 0:
            cutoff = s_max * tolerance
            keep_by_tol = int(np.sum(S > cutoff))
            # Always keep at least 1 to avoid degenerate empty cores.
            keep = max(1, min(keep, keep_by_tol))
        else:
            keep = max(1, keep)
        U = U[:, :keep]
        S = S[:keep]
        Vt = Vt[:keep, :]
        cores[k] = U.reshape(r_l, n, keep)
        # Push S @ Vt into the next core's left rank.
        SV = S[:, None] * Vt
        cores[k + 1] = np.einsum("lr,rjs->ljs", SV, cores[k + 1])
    return cores
