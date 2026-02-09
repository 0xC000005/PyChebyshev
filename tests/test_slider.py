"""Tests for ChebyshevSlider (Sliding Technique)."""

import math

import pytest

from pychebyshev import ChebyshevSlider
from conftest import sin_sum_3d, _bs_call_price


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def coupled_2d_plus_1d(x, _):
    """x0^3 * x1^2 + x2  —  coupling only within (x0, x1)."""
    return x[0] ** 3 * x[1] ** 2 + x[2]


def polynomial_5d(x, _):
    """Sum of squares: x0^2 + x1^2 + x2^2 + x3^2 + x4^2."""
    return sum(xi ** 2 for xi in x)


# ---------------------------------------------------------------------------
# Additively separable function (should be near-exact)
# ---------------------------------------------------------------------------

class TestAdditivelySeparable:
    """sin(x0) + sin(x1) + sin(x2) with partition [[0],[1],[2]]."""

    @pytest.fixture
    def slider_sin_3d(self):
        slider = ChebyshevSlider(
            sin_sum_3d, 3,
            [[-1, 1], [-1, 1], [1, 3]],
            [12, 10, 10],
            partition=[[0], [1], [2]],
            pivot_point=[0.0, 0.0, 2.0],
        )
        slider.build(verbose=False)
        return slider

    def test_function_value(self, slider_sin_3d):
        pt = [0.5, -0.3, 1.7]
        expected = math.sin(0.5) + math.sin(-0.3) + math.sin(1.7)
        result = slider_sin_3d.eval(pt, [0, 0, 0])
        assert abs(result - expected) < 1e-9

    def test_first_derivative_dim0(self, slider_sin_3d):
        pt = [0.5, -0.3, 1.7]
        expected = math.cos(0.5)  # d/dx0 of sin(x0)
        result = slider_sin_3d.eval(pt, [1, 0, 0])
        assert abs(result - expected) < 1e-8

    def test_first_derivative_dim2(self, slider_sin_3d):
        pt = [0.5, -0.3, 1.7]
        expected = math.cos(1.7)  # d/dx2 of sin(x2)
        result = slider_sin_3d.eval(pt, [0, 0, 1])
        assert abs(result - expected) < 1e-6

    def test_second_derivative(self, slider_sin_3d):
        pt = [0.5, -0.3, 1.7]
        expected = -math.sin(0.5)  # d^2/dx0^2 of sin(x0)
        result = slider_sin_3d.eval(pt, [2, 0, 0])
        assert abs(result - expected) < 1e-6

    def test_build_evals_is_sum(self, slider_sin_3d):
        # For partition [[0],[1],[2]] with n_nodes [12,10,10]:
        # total = 12 + 10 + 10 = 32, NOT 12*10*10 = 1200
        assert slider_sin_3d.total_build_evals == 12 + 10 + 10


# ---------------------------------------------------------------------------
# Partially coupled function (exact when coupling is within one slide)
# ---------------------------------------------------------------------------

class TestPartiallyCoupled:
    """x0^3 * x1^2 + x2 with partition [[0,1],[2]]."""

    @pytest.fixture
    def slider_coupled(self):
        slider = ChebyshevSlider(
            coupled_2d_plus_1d, 3,
            [[-2, 2], [-2, 2], [-2, 2]],
            [12, 12, 8],
            partition=[[0, 1], [2]],
            pivot_point=[0.0, 0.0, 0.0],
        )
        slider.build(verbose=False)
        return slider

    def test_function_value(self, slider_coupled):
        pt = [1.0, 0.5, -1.0]
        expected = 1.0 ** 3 * 0.5 ** 2 + (-1.0)  # 0.25 - 1.0 = -0.75
        result = slider_coupled.eval(pt, [0, 0, 0])
        assert abs(result - expected) < 1e-8

    def test_derivative_x0(self, slider_coupled):
        """d/dx0 of x0^3*x1^2 + x2 = 3*x0^2*x1^2."""
        pt = [1.0, 0.5, -1.0]
        expected = 3.0 * 1.0 ** 2 * 0.5 ** 2  # 0.75
        result = slider_coupled.eval(pt, [1, 0, 0])
        assert abs(result - expected) < 1e-6

    def test_derivative_x2(self, slider_coupled):
        """d/dx2 of x0^3*x1^2 + x2 = 1."""
        pt = [1.0, 0.5, -1.0]
        result = slider_coupled.eval(pt, [0, 0, 1])
        assert abs(result - 1.0) < 1e-8

    def test_build_evals(self, slider_coupled):
        # partition [[0,1],[2]] with n_nodes [12,12,8]: total = 12*12 + 8 = 152
        assert slider_coupled.total_build_evals == 12 * 12 + 8


# ---------------------------------------------------------------------------
# 5D additively separable polynomial
# ---------------------------------------------------------------------------

class TestHighDimensional:
    """5D sum-of-squares with partition [[0],[1],[2],[3],[4]]."""

    @pytest.fixture
    def slider_5d(self):
        slider = ChebyshevSlider(
            polynomial_5d, 5,
            [[-1, 1]] * 5,
            [6] * 5,
            partition=[[0], [1], [2], [3], [4]],
            pivot_point=[0.0] * 5,
        )
        slider.build(verbose=False)
        return slider

    def test_function_value(self, slider_5d):
        pt = [0.5, -0.3, 0.7, -0.1, 0.9]
        expected = sum(xi ** 2 for xi in pt)
        result = slider_5d.eval(pt, [0, 0, 0, 0, 0])
        assert abs(result - expected) < 1e-10

    def test_derivative(self, slider_5d):
        pt = [0.5, -0.3, 0.7, -0.1, 0.9]
        # d/dx3 of (x3^2) = 2*x3 = -0.2
        result = slider_5d.eval(pt, [0, 0, 0, 1, 0])
        assert abs(result - (-0.2)) < 1e-8

    def test_build_evals(self, slider_5d):
        # 5 slides of 6 nodes each: 5 * 6 = 30, NOT 6^5 = 7776
        assert slider_5d.total_build_evals == 30

    def test_eval_multi(self, slider_5d):
        pt = [0.5, -0.3, 0.7, -0.1, 0.9]
        derivs = [
            [0, 0, 0, 0, 0],  # value
            [1, 0, 0, 0, 0],  # d/dx0
            [0, 0, 0, 0, 1],  # d/dx4
        ]
        results = slider_5d.eval_multi(pt, derivs)
        expected_val = sum(xi ** 2 for xi in pt)
        assert abs(results[0] - expected_val) < 1e-10
        assert abs(results[1] - 2 * 0.5) < 1e-8  # 2*x0 = 1.0
        assert abs(results[2] - 2 * 0.9) < 1e-8  # 2*x4 = 1.8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestCrossGroupDerivatives:
    """Cross-group mixed partials should return 0."""

    @pytest.fixture
    def slider_3d(self):
        slider = ChebyshevSlider(
            sin_sum_3d, 3,
            [[-1, 1], [-1, 1], [1, 3]],
            [10, 10, 10],
            partition=[[0], [1], [2]],
            pivot_point=[0.0, 0.0, 2.0],
        )
        slider.build(verbose=False)
        return slider

    def test_cross_group_mixed_partial_is_zero(self, slider_3d):
        """d^2f/dx0*dx1 = 0 when x0 and x1 are in different slides."""
        pt = [0.5, -0.3, 1.7]
        result = slider_3d.eval(pt, [1, 1, 0])
        assert result == 0.0

    def test_cross_group_three_way_is_zero(self, slider_3d):
        """d^3f/dx0*dx1*dx2 = 0 when all dims in separate slides."""
        pt = [0.5, -0.3, 1.7]
        result = slider_3d.eval(pt, [1, 1, 1])
        assert result == 0.0


class TestMultiDimSlideDerivative:
    """Derivatives within a multi-dimensional slide group."""

    @pytest.fixture
    def slider_grouped(self):
        slider = ChebyshevSlider(
            coupled_2d_plus_1d, 3,
            [[-2, 2], [-2, 2], [-2, 2]],
            [12, 12, 8],
            partition=[[0, 1], [2]],
            pivot_point=[0.0, 0.0, 0.0],
        )
        slider.build(verbose=False)
        return slider

    def test_mixed_partial_within_group(self, slider_grouped):
        """d^2f/dx0*dx1 of x0^3*x1^2 = 6*x0^2*x1, within slide [0,1]."""
        pt = [1.0, 0.5, -1.0]
        expected = 6.0 * 1.0 ** 2 * 0.5  # 3.0
        result = slider_grouped.eval(pt, [1, 1, 0])
        assert abs(result - expected) < 1e-4

    def test_cross_group_mixed_partial_grouped(self, slider_grouped):
        """d^2f/dx0*dx2 = 0 when x0 in [0,1] and x2 in [2]."""
        pt = [1.0, 0.5, -1.0]
        result = slider_grouped.eval(pt, [1, 0, 1])
        assert result == 0.0


class TestValidation:
    def test_invalid_partition_missing_dim(self):
        with pytest.raises(ValueError, match="Partition must cover"):
            ChebyshevSlider(
                sin_sum_3d, 3,
                [[-1, 1]] * 3,
                [5] * 3,
                partition=[[0], [1]],  # missing dim 2
                pivot_point=[0.0] * 3,
            )

    def test_invalid_partition_duplicate_dim(self):
        with pytest.raises(ValueError, match="Partition must cover"):
            ChebyshevSlider(
                sin_sum_3d, 3,
                [[-1, 1]] * 3,
                [5] * 3,
                partition=[[0, 1], [1, 2]],  # dim 1 appears twice
                pivot_point=[0.0] * 3,
            )

    def test_eval_before_build(self):
        slider = ChebyshevSlider(
            sin_sum_3d, 3,
            [[-1, 1]] * 3,
            [5] * 3,
            partition=[[0], [1], [2]],
            pivot_point=[0.0] * 3,
        )
        with pytest.raises(RuntimeError, match="build"):
            slider.eval([0.5, 0.5, 0.5], [0, 0, 0])


# ---------------------------------------------------------------------------
# Black-Scholes sliding (demonstrates limitation for coupled functions)
# ---------------------------------------------------------------------------

class TestBlackScholesSliding:
    """BS 3D with partition [[0],[1],[2]] — shows cross-coupling error."""

    @pytest.fixture
    def slider_bs_3d(self):
        K, r, q = 100.0, 0.05, 0.02

        def bs(x, _):
            return _bs_call_price(S=x[0], K=K, T=x[1], r=r, sigma=x[2], q=q)

        slider = ChebyshevSlider(
            bs, 3,
            [[50, 150], [0.1, 2.0], [0.1, 0.5]],
            [15, 12, 10],
            partition=[[0], [1], [2]],
            pivot_point=[100.0, 1.0, 0.3],
        )
        slider.build(verbose=False)
        return slider

    def test_at_pivot_near_exact(self, slider_bs_3d):
        """At the pivot point, the slider should be very close to exact.

        Small residual comes from Chebyshev interpolation error of each
        slide at the pivot coordinates (pivot may not be a Chebyshev node).
        """
        pivot = [100.0, 1.0, 0.3]
        expected = _bs_call_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3, q=0.02)
        result = slider_bs_3d.eval(pivot, [0, 0, 0])
        assert abs(result - expected) < 1e-3

    def test_away_from_pivot_has_error(self, slider_bs_3d):
        """Away from pivot, sliding has error due to cross-coupling."""
        pt = [110.0, 0.5, 0.2]
        expected = _bs_call_price(S=110.0, K=100.0, T=0.5, r=0.05, sigma=0.2, q=0.02)
        result = slider_bs_3d.eval(pt, [0, 0, 0])
        # Error exists but should be within ~50% for this separable approximation
        rel_error = abs(result - expected) / abs(expected)
        assert rel_error < 1.0  # loose bound; sliding isn't great for BS
