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


# ============================================================================
# A2: get_special_points()
# ============================================================================

class TestGetSpecialPoints:
    def test_approximation_no_special_points(self, cheb_sin_3d):
        assert cheb_sin_3d.get_special_points() is None

    def test_approximation_all_empty_special_points(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        cheb = ChebyshevApproximation(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], special_points=[[], []],
        )
        cheb.build(verbose=False)
        # All-empty: __new__ does NOT dispatch to Spline; we keep an Approximation
        assert isinstance(cheb, ChebyshevApproximation)
        assert cheb.get_special_points() == [[], []]

    def test_spline_returns_knots_per_dim(self, spline_abs_1d):
        # spline_abs_1d is built with knots=[[0.0]]
        sp = spline_abs_1d.get_special_points()
        assert sp == [[0.0]]

    def test_approximation_dispatches_to_spline_when_kink_declared(self):
        def f(x, _):
            return abs(x[0])

        # __new__ dispatch route: special_points with any non-empty list →
        # returns a ChebyshevSpline.
        # n_nodes must be nested when special_points is non-empty (per v0.12 API).
        obj = ChebyshevApproximation(
            f, 1, [[-1, 1]], [[8, 8]], special_points=[[0.0]],
        )
        assert isinstance(obj, ChebyshevSpline)
        assert obj.get_special_points() == [[0.0]]

    def test_round_trip_pickle(self, spline_abs_1d, tmp_path):
        path = tmp_path / "spl.pkl"
        spline_abs_1d.save(str(path))
        loaded = ChebyshevSpline.load(str(path))
        assert loaded.get_special_points() == spline_abs_1d.get_special_points()

    def test_round_trip_pcb(self, spline_abs_1d, tmp_path):
        path = tmp_path / "spl.pcb"
        spline_abs_1d.save(str(path), format="binary")
        loaded = ChebyshevSpline.load(str(path))
        assert loaded.get_special_points() == spline_abs_1d.get_special_points()


# ============================================================================
# A9: is_dimensionality_allowed() static
# ============================================================================

class TestIsDimensionalityAllowed:
    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_positive_dim_allowed(self, cls):
        assert cls.is_dimensionality_allowed(1) is True
        assert cls.is_dimensionality_allowed(2) is True
        assert cls.is_dimensionality_allowed(10) is True

    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_zero_or_negative_disallowed(self, cls):
        assert cls.is_dimensionality_allowed(0) is False
        assert cls.is_dimensionality_allowed(-1) is False

    @pytest.mark.parametrize("cls", [
        ChebyshevApproximation, ChebyshevSpline, ChebyshevSlider, ChebyshevTT,
    ])
    def test_callable_without_instance(self, cls):
        # Static method: callable on the class itself
        assert callable(cls.is_dimensionality_allowed)
        # Type signature
        result = cls.is_dimensionality_allowed(3)
        assert isinstance(result, bool)


# ============================================================================
# A6: get_num_evaluation_points()
# ============================================================================

class TestGetNumEvaluationPoints:
    def test_approximation_product_of_n_nodes(self, cheb_sin_3d):
        # Built with [10, 8, 4]
        assert cheb_sin_3d.get_num_evaluation_points() == 10 * 8 * 4

    def test_spline_sum_across_pieces(self, spline_abs_1d):
        """Spline returns sum of per-piece grid sizes (matching grid, not work)."""
        expected = sum(
            int(np.prod(piece.n_nodes)) for piece in spline_abs_1d._pieces
        )
        assert spline_abs_1d.get_num_evaluation_points() == expected

    def test_slider_matches_total_build_evals(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        assert slider.get_num_evaluation_points() == slider.total_build_evals

    def test_tt_returns_full_grid_size(self, tt_sin_3d):
        """TT returns prod(n_nodes), the full Cartesian grid (not the sparse cross sample count)."""
        expected = int(np.prod(tt_sin_3d.n_nodes))
        assert tt_sin_3d.get_num_evaluation_points() == expected


# ============================================================================
# A5: get_evaluation_points()
# ============================================================================

class TestGetEvaluationPoints:
    def test_approximation_shape(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        assert pts.shape == (10 * 8 * 4, 3)
        assert pts.dtype == np.float64

    def test_approximation_within_domain(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        # Domain is [[-1,1], [-1,1], [1,3]]
        assert pts[:, 0].min() >= -1.0 and pts[:, 0].max() <= 1.0
        assert pts[:, 1].min() >= -1.0 and pts[:, 1].max() <= 1.0
        assert pts[:, 2].min() >= 1.0 and pts[:, 2].max() <= 3.0

    def test_approximation_count_matches_num_eval_points(self, cheb_sin_3d):
        pts = cheb_sin_3d.get_evaluation_points()
        assert len(pts) == cheb_sin_3d.get_num_evaluation_points()

    def test_approximation_unique_nodes_per_dim(self):
        def f(x, _):
            return x[0] + x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [0, 1]], [4, 4])
        cheb.build(verbose=False)
        pts = cheb.get_evaluation_points()
        assert pts.shape == (16, 2)
        assert len(np.unique(pts[:, 0])) == 4
        assert len(np.unique(pts[:, 1])) == 4

    def test_spline_concatenates_pieces(self, spline_abs_1d):
        pts = spline_abs_1d.get_evaluation_points()
        assert pts.shape[1] == 1
        assert len(pts) == spline_abs_1d.get_num_evaluation_points()

    def test_slider_returns_2d_array(self):
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1])

        slider = ChebyshevSlider(
            f, 2, [[-1, 1], [-1, 1]], [8, 8], partition=[[0], [1]],
            pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        pts = slider.get_evaluation_points()
        assert pts.ndim == 2
        assert pts.shape[1] == 2
        assert len(pts) == slider.get_num_evaluation_points()

    def test_tt_returns_2d_array(self, tt_sin_3d):
        pts = tt_sin_3d.get_evaluation_points()
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert len(pts) == tt_sin_3d.get_num_evaluation_points()

    def test_consistency_with_get_num_eval_points(self):
        """The fundamental contract: len(get_evaluation_points) == get_num_evaluation_points."""
        def f(x, _):
            return x[0] * x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 7])
        cheb.build(verbose=False)
        pts = cheb.get_evaluation_points()
        assert len(pts) == cheb.get_num_evaluation_points()
        assert len(pts) == 5 * 7
