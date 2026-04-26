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


# ============================================================================
# T6: plot_convergence on ChebyshevApproximation
# ============================================================================

def _t6_f_sin(x, _):
    return math.sin(x[0])


def _t6_f_x(x, _):
    return x[0]


class TestPlotConvergence:
    @pytest.fixture
    def matplotlib_or_skip(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            return matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_returns_axes(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t6_f_sin, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        ax = cheb.plot_convergence(target_error=1e-6, max_n=20)
        assert ax is not None

    def test_requires_function(self, matplotlib_or_skip):
        ref = ChebyshevApproximation(_t6_f_x, 1, [[-1, 1]], [4])
        ref.build(verbose=False)
        function_less = ChebyshevApproximation.from_values(
            ref.tensor_values, 1, [[-1, 1]], [4]
        )
        with pytest.raises(RuntimeError, match="function"):
            function_less.plot_convergence()

    def test_target_error_line_drawn(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t6_f_sin, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        ax = cheb.plot_convergence(target_error=1e-6, max_n=12)
        # Verify horizontal line was drawn (axhline adds a Line2D to ax.lines)
        assert any(
            line.get_linestyle() == "--" for line in ax.lines
        )

    def test_max_n_caps_iterations(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t6_f_sin, 1, [[-1, 1]], [4])
        cheb.build(verbose=False)
        ax = cheb.plot_convergence(max_n=8)
        # Plot should have a single line; xdata should not exceed max_n
        line = ax.lines[0]
        assert max(line.get_xdata()) <= 8


# ============================================================================
# T7: plot_1d / plot_2d_surface / plot_2d_contour on all 4 classes
# ============================================================================

def _t7_f_1d(x, _):
    return math.sin(x[0])


def _t7_f_2d(x, _):
    return math.sin(x[0]) + math.cos(x[1])


def _t7_f_3d(x, _):
    return x[0] + x[1] + x[2]


class TestPlot1D:
    @pytest.fixture
    def matplotlib_or_skip(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            return matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_1d_approximation(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_1d, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        ax = cheb.plot_1d()
        assert ax is not None

    def test_plot_1d_spline(self, matplotlib_or_skip):
        spl = ChebyshevSpline(_t7_f_1d, 1, [[-1, 1]], knots=[[0.0]], n_nodes=[6])
        spl.build(verbose=False)
        ax = spl.plot_1d()
        assert ax is not None

    def test_plot_1d_slider_with_fixed(self, matplotlib_or_skip):
        slider = ChebyshevSlider(
            _t7_f_2d, 2, [[-1, 1], [-1, 1]], [6, 6],
            partition=[[0], [1]], pivot_point=[0.0, 0.0],
        )
        slider.build(verbose=False)
        ax = slider.plot_1d(fixed={1: 0.5})
        assert ax is not None

    def test_plot_1d_tt_with_fixed(self, matplotlib_or_skip):
        tt = ChebyshevTT(_t7_f_2d, 2, [[-1, 1], [-1, 1]], [6, 6])
        tt.build(verbose=False)
        ax = tt.plot_1d(fixed={1: 0.5})
        assert ax is not None

    def test_plot_1d_too_many_free_dims_raises(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_2d, 2, [[-1, 1], [-1, 1]], [4, 4])
        cheb.build(verbose=False)
        with pytest.raises(ValueError, match="1 free"):
            cheb.plot_1d()

    def test_plot_1d_n_points_kwarg(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_1d, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        ax = cheb.plot_1d(n_points=50)
        # Verify the line has 50 points
        assert len(ax.lines[0].get_xdata()) == 50


class TestPlot2DSurface:
    @pytest.fixture
    def matplotlib_or_skip(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            return matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_2d_surface_approximation(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_2d, 2, [[-1, 1], [-1, 1]], [8, 8])
        cheb.build(verbose=False)
        ax = cheb.plot_2d_surface(n_points=20)
        assert ax is not None

    def test_plot_2d_surface_requires_2_free_dims(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_1d, 1, [[-1, 1]], [4])
        cheb.build(verbose=False)
        with pytest.raises(ValueError, match="2 free"):
            cheb.plot_2d_surface()

    def test_plot_2d_surface_with_fixed(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_3d, 3, [[-1, 1]] * 3, [4, 4, 4])
        cheb.build(verbose=False)
        ax = cheb.plot_2d_surface(fixed={2: 0.5}, n_points=10)
        assert ax is not None


class TestPlot2DContour:
    @pytest.fixture
    def matplotlib_or_skip(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            return matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_2d_contour_approximation(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_2d, 2, [[-1, 1], [-1, 1]], [8, 8])
        cheb.build(verbose=False)
        ax = cheb.plot_2d_contour(n_points=20, n_levels=10)
        assert ax is not None

    def test_plot_2d_contour_n_levels_kwarg(self, matplotlib_or_skip):
        cheb = ChebyshevApproximation(_t7_f_2d, 2, [[-1, 1], [-1, 1]], [8, 8])
        cheb.build(verbose=False)
        ax = cheb.plot_2d_contour(n_points=15, n_levels=5)
        assert ax is not None


# ============================================================================
# T8: Cross-feature tests
# ============================================================================

def _t8_f(x, _):
    return math.sin(x[0])


def _t8_f_2d(x, _):
    return x[0] + x[1]


class TestCrossFeatures:
    @pytest.fixture
    def matplotlib_or_skip(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            return matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_plot_after_algebra(self, matplotlib_or_skip):
        """v0.19 viz works on algebra-result interpolants (function=None)."""
        a = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [4])
        a.build(verbose=False)
        b = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [4])
        b.build(verbose=False)
        sum_obj = a + b
        ax = sum_obj.plot_1d()
        assert ax is not None

    def test_plot_after_clone(self, matplotlib_or_skip):
        """v0.16 clone() result supports v0.19 plotting."""
        cheb = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [8])
        cheb.build(verbose=False)
        ax = cheb.clone().plot_1d()
        assert ax is not None

    def test_plot_after_tt_extrude(self, matplotlib_or_skip):
        """v0.18 TT extrude() result supports v0.19 plotting."""
        tt = ChebyshevTT(_t8_f, 1, [[-1, 1]], [6])
        tt.build(verbose=False)
        extruded = tt.extrude((1, (0, 1), 4))
        ax = extruded.plot_1d(fixed={1: 0.5})
        assert ax is not None

    def test_plot_convergence_after_algebra_raises(self, matplotlib_or_skip):
        """plot_convergence requires function-bound; algebra result has function=None."""
        a = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [4])
        a.build(verbose=False)
        b = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [4])
        b.build(verbose=False)
        sum_obj = a + b
        with pytest.raises(RuntimeError, match="function"):
            sum_obj.plot_convergence()

    def test_n_workers_with_descriptor_and_additional_data(self):
        """Parallel build threads through additional_data; descriptor preserved."""
        sentinel = {"k": 5}
        cheb = ChebyshevApproximation(
            _t8_f_with_ad, 1, [[-1, 1]], [4],
            additional_data=sentinel, n_workers=2,
        )
        cheb.set_descriptor("source")
        cheb.build(verbose=False)
        assert cheb.get_descriptor() == "source"
        assert cheb.eval([0.5], [0]) == pytest.approx(2.5, abs=1e-10)

    def test_save_load_with_n_workers(self, tmp_path):
        """Save/load preserves the interpolant; n_workers default after load is None."""
        cheb = ChebyshevApproximation(_t8_f, 1, [[-1, 1]], [6], n_workers=2)
        cheb.build(verbose=False)
        path = tmp_path / "cheb.pkl"
        cheb.save(str(path))
        loaded = ChebyshevApproximation.load(str(path))
        assert loaded.eval([0.3], [0]) == pytest.approx(cheb.eval([0.3], [0]), abs=1e-10)


def _t8_f_with_ad(x, ad):
    return ad["k"] * x[0]
