"""Tests for error-threshold-driven construction (v0.11)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation


def _sin2d(x, _):
    return math.sin(x[0]) + math.sin(x[1])


def _sin3d(x, _):
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])


class TestConstructorValidation:
    """Signature-level validation — no build required."""

    def test_explicit_n_unchanged(self):
        """Existing API still works: all-int n_nodes, no error_threshold."""
        cheb = ChebyshevApproximation(_sin2d, 2, [[-1, 1], [-1, 1]], [11, 11])
        assert cheb.n_nodes == [11, 11]
        assert cheb.error_threshold is None

    def test_error_threshold_only(self):
        """Auto-N mode: error_threshold without n_nodes."""
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        assert cheb.error_threshold == 1e-6
        # n_nodes populated with sentinel Nones
        assert cheb.n_nodes == [None, None]

    def test_semi_variable(self):
        """Mixed: some None, some int, plus error_threshold."""
        cheb = ChebyshevApproximation(
            _sin3d, 3, [[-1, 1]] * 3,
            n_nodes=[None, 15, 15],
            error_threshold=1e-6,
        )
        assert cheb.n_nodes == [None, 15, 15]
        assert cheb.error_threshold == 1e-6

    def test_none_without_threshold_raises(self):
        """None in n_nodes without error_threshold → ValueError."""
        with pytest.raises(ValueError, match="error_threshold"):
            ChebyshevApproximation(
                _sin2d, 2, [[-1, 1], [-1, 1]], n_nodes=[None, 11],
            )

    def test_all_none_all_dim_auto(self):
        """All-None n_nodes is equivalent to omitting n_nodes."""
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[None, None], error_threshold=1e-6,
        )
        assert cheb.n_nodes == [None, None]

    def test_max_n_default(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]], error_threshold=1e-6,
        )
        assert cheb.max_n == 64

    def test_max_n_custom(self):
        cheb = ChebyshevApproximation(
            _sin2d, 2, [[-1, 1], [-1, 1]],
            error_threshold=1e-6, max_n=128,
        )
        assert cheb.max_n == 128
