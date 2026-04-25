"""Tests for v0.16 Polish Bundle."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSlider,
    ChebyshevSpline,
    ChebyshevTT,
)


# ============================================================================
# A3: get_max_derivative_order()
# ============================================================================

class TestGetMaxDerivativeOrder:
    def test_approximation_returns_default(self, cheb_sin_3d):
        assert cheb_sin_3d.get_max_derivative_order() == 2

    def test_approximation_returns_custom(self):
        def f(x, _):
            return x[0] ** 4

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [8], max_derivative_order=4)
        cheb.build(verbose=False)
        assert cheb.get_max_derivative_order() == 4

    def test_spline_returns_value(self, spline_abs_1d):
        assert spline_abs_1d.get_max_derivative_order() == 2

    def test_slider_returns_value(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0], max_derivative_order=3,
        )
        slider.build(verbose=False)
        assert slider.get_max_derivative_order() == 3

    def test_tt_returns_value(self, tt_sin_3d):
        assert tt_sin_3d.get_max_derivative_order() == 2


# ============================================================================
# A4: get_error_threshold()
# ============================================================================

class TestGetErrorThreshold:
    def test_approximation_with_threshold(self):
        def f(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], error_threshold=1e-6)
        cheb.build(verbose=False)
        assert cheb.get_error_threshold() == 1e-6

    def test_approximation_without_threshold(self, cheb_sin_3d):
        assert cheb_sin_3d.get_error_threshold() is None

    def test_spline_with_threshold(self):
        def f(x, _):
            return abs(x[0])

        spl = ChebyshevSpline(
            f, 1, [[-1, 1]], knots=[[0.0]], error_threshold=1e-5,
        )
        spl.build(verbose=False)
        assert spl.get_error_threshold() == 1e-5

    def test_spline_without_threshold(self, spline_abs_1d):
        assert spline_abs_1d.get_error_threshold() is None
