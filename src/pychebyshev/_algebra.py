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
