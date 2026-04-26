"""Tests for v0.19 Build & Diagnostics."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from pychebyshev import (
    ChebyshevApproximation,
    ChebyshevSlider,
    ChebyshevSpline,
    ChebyshevTT,
)
from pychebyshev._progress import _maybe_progress


# ============================================================================
# T2: _maybe_progress() helper
# ============================================================================

class TestMaybeProgress:
    def test_passthrough_when_verbose_false(self):
        items = list(range(5))
        result = list(_maybe_progress(items, desc="test", verbose=False))
        assert result == items

    def test_passthrough_when_verbose_true(self):
        items = list(range(5))
        result = list(_maybe_progress(items, desc="test", verbose=True))
        assert result == items

    def test_wraps_with_tqdm_when_verbose_2(self):
        items = list(range(3))
        result = list(_maybe_progress(items, desc="test", verbose=2))
        assert result == items


# ============================================================================
# T3: Parallel build (Approximation)
# ============================================================================

# Module-level functions (picklable for ProcessPoolExecutor)
def _t3_f_simple(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def _t3_f_with_ad(x, ad):
    return ad["k"] * x[0]


class TestParallelBuildApproximation:
    def test_parallel_matches_sequential(self):
        seq = ChebyshevApproximation(_t3_f_simple, 2, [[-1, 1], [-1, 1]], [10, 10])
        seq.build(verbose=False)
        par = ChebyshevApproximation(
            _t3_f_simple, 2, [[-1, 1], [-1, 1]], [10, 10], n_workers=2
        )
        par.build(verbose=False)
        np.testing.assert_allclose(seq.tensor_values, par.tensor_values, rtol=1e-12)

    def test_n_workers_minus_one_uses_cpu_count(self):
        cheb = ChebyshevApproximation(
            _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=-1
        )
        cheb.build(verbose=False)
        assert cheb.is_construction_finished()

    def test_n_workers_zero_rejected(self):
        with pytest.raises(ValueError, match="n_workers"):
            ChebyshevApproximation(
                _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=0
            )

    def test_n_workers_default_is_none(self):
        cheb = ChebyshevApproximation(_t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4])
        assert cheb.n_workers is None

    def test_n_workers_negative_below_minus_one_rejected(self):
        with pytest.raises(ValueError, match="n_workers"):
            ChebyshevApproximation(
                _t3_f_simple, 2, [[-1, 1], [-1, 1]], [4, 4], n_workers=-5
            )

    def test_parallel_with_additional_data(self):
        sentinel = {"k": 7}
        cheb = ChebyshevApproximation(
            _t3_f_with_ad, 1, [[-1, 1]], [4],
            additional_data=sentinel, n_workers=2,
        )
        cheb.build(verbose=False)
        # f(x, ad) = 7 * x; eval at 0.5 should be 3.5
        assert cheb.eval([0.5], [0]) == pytest.approx(3.5, abs=1e-10)


# ============================================================================
# T4: Parallel build (Spline)
# ============================================================================

def _t4_f_abs(x, _):
    return abs(x[0])


def _t4_f_x_squared(x, _):
    return x[0] ** 2


class TestParallelBuildSpline:
    def test_parallel_matches_sequential(self):
        # n_nodes=[8]: flat form — 8 nodes per dimension for all pieces in this 1D spline
        seq = ChebyshevSpline(_t4_f_abs, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8])
        seq.build(verbose=False)
        par = ChebyshevSpline(
            _t4_f_abs, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[8], n_workers=2,
        )
        par.build(verbose=False)
        for x in [-0.7, -0.3, 0.3, 0.7]:
            assert seq.eval([x], [0]) == pytest.approx(par.eval([x], [0]), abs=1e-10)

    def test_spline_n_workers_propagates_to_pieces(self):
        spl = ChebyshevSpline(
            _t4_f_x_squared, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[6], n_workers=2,
        )
        spl.build(verbose=False)
        for piece in spl._pieces:
            assert piece.n_workers == 2

    def test_spline_n_workers_default_none(self):
        spl = ChebyshevSpline(_t4_f_x_squared, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[4])
        assert spl.n_workers is None
        spl.build(verbose=False)
        # Pieces should also have n_workers=None
        for piece in spl._pieces:
            assert piece.n_workers is None

    def test_spline_n_workers_minus_one(self):
        spl = ChebyshevSpline(
            _t4_f_x_squared, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[4], n_workers=-1,
        )
        spl.build(verbose=False)
        # Each piece should have positive int n_workers (cpu_count)
        for piece in spl._pieces:
            assert isinstance(piece.n_workers, int) and piece.n_workers >= 1


# ============================================================================
# T5: Progress bars (verbose=2) on all 4 classes
# ============================================================================

def _t5_f_simple(x, _):
    return x[0]


def _t5_f_2d(x, _):
    return x[0] + x[1]


class TestProgressBars:
    def test_verbose_2_does_not_break_approximation_build(self):
        cheb = ChebyshevApproximation(_t5_f_simple, 1, [[-1, 1]], [4])
        cheb.build(verbose=2)
        assert cheb.is_construction_finished()

    def test_verbose_2_does_not_break_spline_build(self):
        spl = ChebyshevSpline(
            _t5_f_simple, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[4]
        )
        spl.build(verbose=2)
        assert spl.is_construction_finished()

    def test_verbose_2_does_not_break_slider_build(self):
        slider = ChebyshevSlider(
            _t5_f_2d, 2, [[-1, 1], [-1, 1]], [4, 4],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=2)
        assert slider.is_construction_finished()

    def test_verbose_2_does_not_break_tt_build(self):
        tt = ChebyshevTT(_t5_f_2d, 2, [[-1, 1], [-1, 1]], [4, 4])
        tt.build(verbose=2)
        assert tt.is_construction_finished()

    def test_verbose_false_no_progress_output(self, capsys):
        cheb = ChebyshevApproximation(_t5_f_simple, 1, [[-1, 1]], [4])
        cheb.build(verbose=False)
        captured = capsys.readouterr()
        assert "it/s" not in captured.err and "it/s" not in captured.out

    def test_verbose_true_unchanged(self):
        # Existing verbose=True path should still work
        cheb = ChebyshevApproximation(_t5_f_simple, 1, [[-1, 1]], [4])
        cheb.build(verbose=True)
        assert cheb.is_construction_finished()
