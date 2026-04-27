"""Tests for v0.20 — adaptive refinement + cross-language interop."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSpline,
    ChebyshevTT,
)
from pychebyshev._sensitivity import _compute_sobol_from_coeffs


# ============================================================================
# T1: Sobol indices helper
# ============================================================================

class TestSobolHelper:
    def test_constant_function_zero_variance(self):
        coeffs = np.array([7.0, 0.0, 0.0, 0.0])
        result = _compute_sobol_from_coeffs(coeffs, num_dimensions=1)
        assert result["variance"] == pytest.approx(0.0, abs=1e-12)

    def test_univariate_linear_first_order_one(self):
        coeffs = np.array([0.0, 1.0, 0.0, 0.0])
        result = _compute_sobol_from_coeffs(coeffs, num_dimensions=1)
        assert result["first_order"][0] == pytest.approx(1.0, abs=1e-10)
        assert result["total_order"][0] == pytest.approx(1.0, abs=1e-10)

    def test_additive_2d_first_order_split(self):
        coeffs = np.zeros((4, 4))
        coeffs[1, 0] = 1.0
        coeffs[0, 1] = 1.0
        result = _compute_sobol_from_coeffs(coeffs, num_dimensions=2)
        assert result["first_order"][0] == pytest.approx(0.5, abs=1e-6)
        assert result["first_order"][1] == pytest.approx(0.5, abs=1e-6)
