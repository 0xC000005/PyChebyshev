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


# ============================================================================
# T2: sobol_indices() instance method
# ============================================================================

def _t2_f_x(x, _):
    return x[0]

def _t2_f_xy(x, _):
    return x[0] + x[1]

def _t2_f_xy_product(x, _):
    return x[0] * x[1]


class TestSobolIndicesInstance:
    def test_approximation_univariate_linear(self):
        cheb = ChebyshevApproximation(_t2_f_x, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        assert result["first_order"][0] == pytest.approx(1.0, abs=1e-6)
        assert result["total_order"][0] == pytest.approx(1.0, abs=1e-6)
        assert result["variance"] > 0

    def test_approximation_2d_additive(self):
        cheb = ChebyshevApproximation(_t2_f_xy, 2, [[-1, 1], [-1, 1]], [8, 8])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        assert result["first_order"][0] == pytest.approx(0.5, abs=1e-3)
        assert result["first_order"][1] == pytest.approx(0.5, abs=1e-3)

    def test_approximation_2d_product(self):
        cheb = ChebyshevApproximation(_t2_f_xy_product, 2, [[-1, 1], [-1, 1]], [8, 8])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        # f(x,y) = x*y → only c[1,1] is nonzero; pure interaction
        assert result["first_order"][0] == pytest.approx(0.0, abs=1e-3)
        assert result["total_order"][0] == pytest.approx(1.0, abs=1e-3)

    def test_returns_dict_with_keys(self):
        cheb = ChebyshevApproximation(_t2_f_x, 1, [[-1, 1]], [4])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        assert {"first_order", "total_order", "variance"} <= result.keys()

    def test_total_order_geq_first_order(self):
        cheb = ChebyshevApproximation(
            lambda x, _: math.sin(x[0]) + x[0] * x[1] + math.cos(x[1]),
            2, [[-1, 1], [-1, 1]], [10, 10],
        )
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        for d in range(2):
            assert result["total_order"][d] >= result["first_order"][d] - 1e-10

    def test_spline_sobol_returns_dict(self):
        spl = ChebyshevSpline(
            lambda x, _: abs(x[0]), 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8, 8],
        )
        spl.build(verbose=False)
        result = spl.sobol_indices()
        assert "first_order" in result
        assert result["first_order"][0] == pytest.approx(1.0, abs=1e-3)


# ============================================================================
# T3: ChebyshevSpline.auto_knots()
# ============================================================================

def _t3_f_abs(x, _):
    return abs(x[0])

def _t3_f_two_kinks(x, _):
    return max(0, x[0]) + max(0, x[0] - 0.5)

def _t3_f_smooth(x, _):
    return math.sin(x[0])


class TestAutoKnots:
    def test_recovers_single_kink(self):
        spl = ChebyshevSpline.auto_knots(
            _t3_f_abs, 1, [[-1, 1]],
            max_knots_per_dim=3,
            n_scan_points=200,
        )
        assert isinstance(spl, ChebyshevSpline)
        knots = spl.get_special_points()
        assert any(abs(k) < 0.1 for k in knots[0])

    def test_recovers_two_kinks(self):
        spl = ChebyshevSpline.auto_knots(
            _t3_f_two_kinks, 1, [[-1, 1]],
            max_knots_per_dim=4,
            n_scan_points=300,
        )
        knots = spl.get_special_points()[0]
        assert any(abs(k) < 0.1 for k in knots)
        assert any(abs(k - 0.5) < 0.1 for k in knots)

    def test_smooth_function_few_knots(self):
        spl = ChebyshevSpline.auto_knots(
            _t3_f_smooth, 1, [[-1, 1]],
            max_knots_per_dim=3,
            threshold_factor=10.0,
        )
        knots = spl.get_special_points()[0]
        assert len(knots) == 0

    def test_max_knots_per_dim_caps(self):
        spl = ChebyshevSpline.auto_knots(
            _t3_f_two_kinks, 1, [[-1, 1]],
            max_knots_per_dim=1,
        )
        knots = spl.get_special_points()[0]
        assert len(knots) <= 1
