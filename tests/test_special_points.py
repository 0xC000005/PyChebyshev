"""Tests for special_points in the core ChebyshevApproximation API (v0.12)."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


def _abs1d(x, _):
    return abs(x[0])


def _abs_sum_2d(x, _):
    return abs(x[0]) + abs(x[1])


class TestDispatch:
    """ChebyshevApproximation.__new__ routes to ChebyshevSpline when
    special_points has any non-empty dim."""

    def test_special_points_none_returns_approximation(self):
        obj = ChebyshevApproximation(
            lambda x, _: x[0] ** 2, 1, [[-1, 1]], [11]
        )
        assert type(obj) is ChebyshevApproximation

    def test_all_empty_special_points_returns_approximation(self):
        obj = ChebyshevApproximation(
            lambda x, _: x[0] ** 2 + x[1] ** 2,
            2, [[-1, 1], [-1, 1]], [11, 11],
            special_points=[[], []],
        )
        assert type(obj) is ChebyshevApproximation

    def test_kink_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            n_nodes=[[11, 11]],
            special_points=[[0.0]],
        )
        assert type(obj) is ChebyshevSpline

    def test_kink_2d_one_dim_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[11, 11], [13]],
            special_points=[[0.0], []],
        )
        assert type(obj) is ChebyshevSpline
        assert obj.knots == [[0.0], []]

    def test_kink_both_dims_returns_spline(self):
        obj = ChebyshevApproximation(
            _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
            n_nodes=[[11, 11], [11, 11]],
            special_points=[[0.0], [0.0]],
        )
        assert type(obj) is ChebyshevSpline
        assert obj._shape == (2, 2)

    def test_dispatch_passes_error_threshold(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            special_points=[[0.0]],
            error_threshold=1e-6,
        )
        assert type(obj) is ChebyshevSpline
        assert obj.error_threshold == 1e-6

    def test_dispatch_passes_max_n(self):
        obj = ChebyshevApproximation(
            _abs1d, 1, [[-1, 1]],
            special_points=[[0.0]],
            error_threshold=1e-6,
            max_n=32,
        )
        assert obj.max_n == 32

    def test_init_signature_accepts_special_points_when_none(self):
        # __init__ must accept special_points as a kwarg (via the
        # single-tensor path, Python calls __init__ with full kwargs).
        obj = ChebyshevApproximation(
            lambda x, _: x[0], 1, [[-1, 1]], [11],
            special_points=None,
        )
        assert type(obj) is ChebyshevApproximation


class TestValidation:
    """Validation errors for special_points + nested n_nodes."""

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have 2 entries"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[0.0]],
            )

    def test_unsorted_points_raises(self):
        with pytest.raises(ValueError, match="must be sorted"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11, 11]],
                special_points=[[0.5, -0.5]],
            )

    def test_point_on_boundary_raises(self):
        with pytest.raises(ValueError, match="not strictly inside"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[1.0]],
            )

    def test_point_outside_domain_raises(self):
        with pytest.raises(ValueError, match="not strictly inside"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11]],
                special_points=[[2.0]],
            )

    def test_coinciding_points_raises(self):
        with pytest.raises(ValueError, match="[Cc]oinciding"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11, 11, 11]],
                special_points=[[0.3, 0.3]],
            )

    def test_flat_n_nodes_with_special_points_raises(self):
        with pytest.raises(ValueError, match="nested"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[11],
                special_points=[[0.0]],
            )

    def test_wrong_nested_inner_length_raises(self):
        with pytest.raises(ValueError, match="must have 2 entries"):
            ChebyshevApproximation(
                _abs1d, 1, [[-1, 1]],
                n_nodes=[[11]],
                special_points=[[0.0]],
            )

    def test_mixed_nested_and_flat_raises(self):
        with pytest.raises(ValueError, match="fully nested"):
            ChebyshevApproximation(
                _abs_sum_2d, 2, [[-1, 1], [-1, 1]],
                n_nodes=[[11, 11], 13],
                special_points=[[0.0], []],
            )
