"""Tests for Chebyshev extrusion and slicing (v0.8.0)."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSlider, ChebyshevSpline


# ---------- helper functions ----------

def _sin_1d(x):
    return math.sin(x[0])

def _sincos_2d(x):
    return math.sin(x[0]) + math.cos(x[1])

def _sin_sum_3d(x):
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])


TEST_POINTS_1D = [-0.7, -0.3, 0.0, 0.4, 0.8]
TEST_POINTS_2D = [
    [0.5, 0.3],
    [-0.7, 0.8],
    [0.0, 0.0],
    [0.9, -0.9],
    [-0.2, 0.6],
]


# ======================================================================
# TestApproxExtrude
# ======================================================================

class TestApproxExtrude:
    """Tests for ChebyshevApproximation.extrude()."""

    def test_extrude_1d_to_2d_append(self, extrude_cheb_1d):
        """Extrude 1D sin(x) -> 2D by appending new dim at index 1."""
        ct2 = extrude_cheb_1d.extrude((1, (-2, 2), 7))
        assert ct2.num_dimensions == 2
        assert ct2.n_nodes == [11, 7]

    def test_extrude_1d_to_2d_prepend(self, extrude_cheb_1d):
        """Extrude 1D sin(x) -> 2D by prepending new dim at index 0."""
        ct2 = extrude_cheb_1d.extrude((0, (-2, 2), 7))
        assert ct2.num_dimensions == 2
        assert ct2.n_nodes == [7, 11]

    def test_extrude_2d_to_3d_middle(self, extrude_cheb_2d):
        """Insert dim in the middle position of 2D."""
        ct3 = extrude_cheb_2d.extrude((1, (0, 5), 9))
        assert ct3.num_dimensions == 3
        assert ct3.n_nodes == [11, 9, 11]

    def test_extrude_2d_to_3d_end(self, extrude_cheb_2d):
        """Append dim to 2D."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        assert ct3.num_dimensions == 3
        assert ct3.n_nodes == [11, 11, 9]

    def test_extrude_multi_dim(self, extrude_cheb_1d):
        """Add 2 dims at once: 1D -> 3D."""
        ct3 = extrude_cheb_1d.extrude([(0, (-2, 2), 7), (2, (0, 5), 9)])
        assert ct3.num_dimensions == 3
        assert ct3.n_nodes == [7, 11, 9]

    def test_extrude_value_preserved(self, extrude_cheb_2d):
        """Extruded CT evaluates same as original at all test points."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        for p in TEST_POINTS_2D:
            orig = extrude_cheb_2d.vectorized_eval(p, [0, 0])
            for new_coord in [0.0, 1.5, 3.7, 5.0]:
                extruded = ct3.vectorized_eval(p + [new_coord], [0, 0, 0])
                assert abs(extruded - orig) < 1e-12, (
                    f"Value mismatch at {p}+[{new_coord}]: {extruded} vs {orig}"
                )

    def test_extrude_any_new_coord(self, extrude_cheb_1d):
        """Evaluation same regardless of new coordinate value."""
        ct2 = extrude_cheb_1d.extrude((1, (-10, 10), 11))
        x = 0.5
        ref = extrude_cheb_1d.vectorized_eval([x], [0])
        for y in [-10.0, -3.3, 0.0, 5.5, 10.0]:
            val = ct2.vectorized_eval([x, y], [0, 0])
            assert abs(val - ref) < 1e-12

    def test_extrude_derivative_original_preserved(self, extrude_cheb_2d):
        """Derivatives in original dims unchanged after extrusion."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        for p in TEST_POINTS_2D:
            # d/dx (sin(x)+cos(y)) = cos(x)
            orig = extrude_cheb_2d.vectorized_eval(p, [1, 0])
            ext = ct3.vectorized_eval(p + [2.5], [1, 0, 0])
            assert abs(ext - orig) < 1e-10, (
                f"Derivative mismatch at {p}: {ext} vs {orig}"
            )

    def test_extrude_derivative_new_dim_zero(self, extrude_cheb_2d):
        """Derivative along new dim is zero."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        for p in TEST_POINTS_2D:
            deriv = ct3.vectorized_eval(p + [2.5], [0, 0, 1])
            assert abs(deriv) < 1e-10, (
                f"Derivative along new dim not zero at {p}: {deriv}"
            )

    def test_extrude_metadata(self, extrude_cheb_1d):
        """Extruded result has function=None, build_time=0.0, n_evaluations=0."""
        ct2 = extrude_cheb_1d.extrude((1, (0, 1), 5))
        assert ct2.function is None
        assert ct2.build_time == 0.0
        assert ct2.n_evaluations == 0

    def test_extrude_domain_updated(self, extrude_cheb_1d):
        """Domain has new entry, n_nodes updated."""
        ct2 = extrude_cheb_1d.extrude((1, (0, 5), 9))
        assert ct2.domain == [[-1, 1], [0, 5]]
        assert ct2.n_nodes == [11, 9]

    def test_extrude_error_not_built(self):
        """Raises RuntimeError if not built."""
        def f(x, _): return math.sin(x[0])
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        with pytest.raises(RuntimeError, match="build"):
            cheb.extrude((1, (0, 1), 5))

    def test_extrude_error_bad_dim_index(self, extrude_cheb_1d):
        """ValueError for out of range dim_index."""
        with pytest.raises(ValueError, match="out of range"):
            extrude_cheb_1d.extrude((5, (0, 1), 5))

    def test_extrude_error_duplicate_dim(self, extrude_cheb_1d):
        """ValueError for duplicate dim_index."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            extrude_cheb_1d.extrude([(0, (0, 1), 5), (0, (0, 1), 5)])

    def test_extrude_serialization(self, extrude_cheb_1d):
        """Save/load round-trip works for extruded CT."""
        ct2 = extrude_cheb_1d.extrude((1, (0, 5), 9))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            ct2.save(f.name)
            loaded = ChebyshevApproximation.load(f.name)
        p = [0.5, 2.5]
        assert abs(ct2.vectorized_eval(p, [0, 0]) - loaded.vectorized_eval(p, [0, 0])) < 1e-15


# ======================================================================
# TestApproxSlice
# ======================================================================

class TestApproxSlice:
    """Tests for ChebyshevApproximation.slice()."""

    def test_slice_2d_to_1d(self, extrude_cheb_2d):
        """Slice 2D -> 1D by fixing dim 1, verify against direct eval."""
        y_fixed = 0.3
        ct1 = extrude_cheb_2d.slice((1, y_fixed))
        assert ct1.num_dimensions == 1
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, y_fixed], [0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-10, (
                f"Slice 2D->1D failed at x={x}: {sliced} vs {exact}"
            )

    def test_slice_3d_to_2d(self):
        """Slice 3D -> 2D by fixing dim 1."""
        def f(x, _): return math.sin(x[0]) + math.cos(x[1]) + math.sin(x[2])
        ct3 = ChebyshevApproximation(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [11, 11, 11])
        ct3.build(verbose=False)
        y_fixed = 0.5
        ct2 = ct3.slice((1, y_fixed))
        assert ct2.num_dimensions == 2
        for p in [[0.3, -0.7], [-0.5, 0.8], [0.0, 0.0]]:
            exact = ct3.vectorized_eval([p[0], y_fixed, p[1]], [0, 0, 0])
            sliced = ct2.vectorized_eval(p, [0, 0])
            assert abs(sliced - exact) < 1e-10

    def test_slice_exact_node(self, extrude_cheb_2d):
        """Slice at exact Chebyshev node (fast path via np.take)."""
        # Get a node value from dim 1
        node_val = extrude_cheb_2d.nodes[1][3]
        ct1 = extrude_cheb_2d.slice((1, node_val))
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, node_val], [0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-13

    def test_slice_between_nodes(self, extrude_cheb_2d):
        """Slice at non-node value (general barycentric path)."""
        y_fixed = 0.1234  # unlikely to be a node
        ct1 = extrude_cheb_2d.slice((1, y_fixed))
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, y_fixed], [0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-10

    def test_slice_boundary(self, extrude_cheb_2d):
        """Slice at domain boundary."""
        ct1 = extrude_cheb_2d.slice((1, -1.0))
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, -1.0], [0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-10

    def test_slice_multi_dim(self):
        """Slice two dims at once (3D -> 1D)."""
        def f(x, _): return math.sin(x[0]) + math.cos(x[1]) + math.sin(x[2])
        ct3 = ChebyshevApproximation(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [11, 11, 11])
        ct3.build(verbose=False)
        y_fixed = 0.3
        z_fixed = -0.5
        ct1 = ct3.slice([(1, y_fixed), (2, z_fixed)])
        assert ct1.num_dimensions == 1
        for x in TEST_POINTS_1D:
            exact = ct3.vectorized_eval([x, y_fixed, z_fixed], [0, 0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-10

    def test_slice_derivative_preserved(self, extrude_cheb_2d):
        """Derivatives in remaining dims match after slicing."""
        y_fixed = 0.4
        ct1 = extrude_cheb_2d.slice((1, y_fixed))
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, y_fixed], [1, 0])
            sliced = ct1.vectorized_eval([x], [1])
            assert abs(sliced - exact) < 1e-8

    def test_slice_metadata(self, extrude_cheb_2d):
        """function=None, build_time=0.0 for sliced result."""
        ct1 = extrude_cheb_2d.slice((1, 0.5))
        assert ct1.function is None
        assert ct1.build_time == 0.0

    def test_slice_domain_updated(self, extrude_cheb_2d):
        """Domain shrinks, n_nodes shrinks."""
        ct1 = extrude_cheb_2d.slice((1, 0.5))
        assert ct1.domain == [[-1, 1]]
        assert ct1.n_nodes == [11]

    def test_slice_error_not_built(self):
        """RuntimeError if not built."""
        def f(x, _): return math.sin(x[0]) + math.cos(x[1])
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
        with pytest.raises(RuntimeError, match="build"):
            cheb.slice((1, 0.5))

    def test_slice_error_bad_dim_index(self, extrude_cheb_2d):
        """ValueError for out-of-range dim_index."""
        with pytest.raises(ValueError, match="out of range"):
            extrude_cheb_2d.slice((5, 0.5))

    def test_slice_error_out_of_domain(self, extrude_cheb_2d):
        """ValueError for value outside domain."""
        with pytest.raises(ValueError, match="outside"):
            extrude_cheb_2d.slice((0, 5.0))

    def test_slice_error_all_dims(self, extrude_cheb_2d):
        """ValueError when slicing all dims."""
        with pytest.raises(ValueError, match="[Cc]annot slice all"):
            extrude_cheb_2d.slice([(0, 0.0), (1, 0.0)])

    def test_slice_serialization(self, extrude_cheb_2d):
        """Save/load round-trip works for sliced CT."""
        ct1 = extrude_cheb_2d.slice((1, 0.5))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            ct1.save(f.name)
            loaded = ChebyshevApproximation.load(f.name)
        x = 0.3
        assert abs(ct1.vectorized_eval([x], [0]) - loaded.vectorized_eval([x], [0])) < 1e-15


# ======================================================================
# TestSliceExtrudeInverse
# ======================================================================

class TestSliceExtrudeInverse:
    """Extrude then slice should approximately recover the original."""

    def test_extrude_then_slice_identity(self, extrude_cheb_2d):
        """Extrude dim 2 then slice it back: result ~ original."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        ct2_back = ct3.slice((2, 2.5))
        for p in TEST_POINTS_2D:
            orig = extrude_cheb_2d.vectorized_eval(p, [0, 0])
            back = ct2_back.vectorized_eval(p, [0, 0])
            assert abs(back - orig) < 1e-12, (
                f"Round-trip failed at {p}: {back} vs {orig}"
            )

    def test_round_trip_multiple_points(self, extrude_cheb_1d):
        """Verify round-trip at many random points."""
        rng = np.random.default_rng(42)
        ct2 = extrude_cheb_1d.extrude((0, (-3, 3), 7))
        ct1_back = ct2.slice((0, 1.0))
        xs = rng.uniform(-1, 1, 50)
        for x in xs:
            orig = extrude_cheb_1d.vectorized_eval([x], [0])
            back = ct1_back.vectorized_eval([x], [0])
            assert abs(back - orig) < 1e-12


# ======================================================================
# TestSplineExtrude
# ======================================================================

class TestSplineExtrude:
    """Tests for ChebyshevSpline.extrude()."""

    def test_spline_extrude_basic(self, spline_abs_1d):
        """Extrude 1D spline -> 2D."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        assert sp2.num_dimensions == 2
        assert sp2.n_nodes == [15, 9]

    def test_spline_extrude_knots_preserved(self, spline_abs_1d):
        """Original knots unchanged, new dim has empty knots."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        assert sp2.knots[0] == [0.0]  # original
        assert sp2.knots[1] == []     # new dim

    def test_spline_extrude_pieces_count(self, spline_abs_1d):
        """Number of pieces unchanged (multiplied by 1 for new knotless dim)."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        assert sp2.num_pieces == spline_abs_1d.num_pieces

    def test_spline_extrude_value_preserved(self, spline_abs_1d):
        """Evaluations match original regardless of new coordinate."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        for x in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            orig = spline_abs_1d.eval([x], [0])
            for y in [0.0, 2.5, 5.0]:
                ext = sp2.eval([x, y], [0, 0])
                assert abs(ext - orig) < 1e-10, (
                    f"Value mismatch at ({x}, {y}): {ext} vs {orig}"
                )

    def test_spline_extrude_derivative_new_dim_zero(self, spline_abs_1d):
        """Derivative along new dim is zero."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        for x in [0.3, 0.8]:  # avoid knot at 0
            deriv = sp2.eval([x, 2.5], [0, 1])
            assert abs(deriv) < 1e-10


# ======================================================================
# TestSplineSlice
# ======================================================================

class TestSplineSlice:
    """Tests for ChebyshevSpline.slice()."""

    def test_spline_slice_basic(self, spline_bs_2d):
        """Slice 2D spline -> 1D."""
        S_fixed = 95.0
        sp1 = spline_bs_2d.slice((0, S_fixed))
        assert sp1.num_dimensions == 1

    def test_spline_slice_piece_reduction(self, spline_bs_2d):
        """After slicing, we keep only pieces from the interval containing the value."""
        # spline_bs_2d has knots [[100.0], []], domain [[80,120],[0.25,1.0]]
        # Slicing dim 0 at S=95 -> picks the left piece (80-100)
        sp1 = spline_bs_2d.slice((0, 95.0))
        # Original had 2 pieces along dim 0 (knot at 100), 1 along dim 1
        # After slicing dim 0, should have 1 piece
        assert sp1.num_pieces == 1

    def test_spline_slice_value_matches(self, spline_bs_2d):
        """Evaluation matches direct 2D eval at sliced point."""
        S_fixed = 95.0
        sp1 = spline_bs_2d.slice((0, S_fixed))
        for T in [0.3, 0.5, 0.8]:
            exact = spline_bs_2d.eval([S_fixed, T], [0, 0])
            sliced = sp1.eval([T], [0])
            assert abs(sliced - exact) < 1e-10, (
                f"Spline slice mismatch at T={T}: {sliced} vs {exact}"
            )

    def test_spline_slice_knots_preserved(self, spline_bs_2d):
        """Remaining knots unchanged after slicing."""
        sp1 = spline_bs_2d.slice((0, 95.0))
        assert sp1.knots == [[]]  # dim 1's knots were []

    def test_spline_slice_serialization(self, spline_bs_2d):
        """Save/load round-trip works."""
        sp1 = spline_bs_2d.slice((0, 95.0))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            sp1.save(f.name)
            loaded = ChebyshevSpline.load(f.name)
        val_orig = sp1.eval([0.5], [0])
        val_loaded = loaded.eval([0.5], [0])
        assert abs(val_orig - val_loaded) < 1e-15


# ======================================================================
# TestSliderExtrude
# ======================================================================

class TestSliderExtrude:
    """Tests for ChebyshevSlider.extrude()."""

    def test_slider_extrude_basic(self, algebra_slider_f):
        """Extrude 3D slider -> 4D, verify partition updated."""
        sl4 = algebra_slider_f.extrude((3, (0, 5), 9))
        assert sl4.num_dimensions == 4
        assert len(sl4.partition) == 4  # original 3 groups + 1 new

    def test_slider_extrude_pivot_extended(self, algebra_slider_f):
        """Pivot point has new entry (midpoint of new domain)."""
        sl4 = algebra_slider_f.extrude((3, (0, 5), 9))
        assert len(sl4.pivot_point) == 4
        assert abs(sl4.pivot_point[3] - 2.5) < 1e-14  # midpoint of [0, 5]

    def test_slider_extrude_value_preserved(self, algebra_slider_f):
        """Evaluations match original regardless of new coordinate."""
        sl4 = algebra_slider_f.extrude((3, (0, 5), 9))
        pts = [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2], [0.0, 0.0, 0.0]]
        for p in pts:
            orig = algebra_slider_f.eval(p, [0, 0, 0])
            for new_coord in [0.0, 2.5, 5.0]:
                ext = sl4.eval(p + [new_coord], [0, 0, 0, 0])
                assert abs(ext - orig) < 1e-10, (
                    f"Slider extrude mismatch at {p}+[{new_coord}]"
                )

    def test_slider_extrude_derivative_new_dim_zero(self, algebra_slider_f):
        """Derivative along new dim is zero."""
        sl4 = algebra_slider_f.extrude((3, (0, 5), 9))
        p = [0.5, 0.3, 0.7, 2.5]
        deriv = sl4.eval(p, [0, 0, 0, 1])
        assert abs(deriv) < 1e-10


# ======================================================================
# TestSliderSlice
# ======================================================================

class TestSliderSlice:
    """Tests for ChebyshevSlider.slice()."""

    def test_slider_slice_single_dim_group(self, algebra_slider_f):
        """Slice removes a single-dim group, pivot_value updated."""
        # algebra_slider_f: partition [[0],[1],[2]], 3D sin(x)+sin(y)+sin(z)
        x_fixed = 0.5
        sl2 = algebra_slider_f.slice((0, x_fixed))
        assert sl2.num_dimensions == 2
        assert len(sl2.partition) == 2
        # pivot_value should be s_0(x_fixed) = sin(0.5) approx
        # (the slide evaluates the full function with other dims at pivot)

    def test_slider_slice_multi_dim_group(self):
        """Slice within a multi-dim group."""
        def f(x, _): return math.sin(x[0]) + math.cos(x[1]) + math.sin(x[2])
        sl = ChebyshevSlider(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [8, 8, 8],
                             [[0, 1], [2]], [0.0, 0.0, 0.0])
        sl.build(verbose=False)
        # Slice dim 0 from the 2-dim group [0,1]
        sl2 = sl.slice((0, 0.3))
        assert sl2.num_dimensions == 2
        # Group that was [0,1] should now be just [0] (remapped from original dim 1)
        assert len(sl2.partition) == 2

    def test_slider_slice_value_matches(self, algebra_slider_f):
        """Evaluation matches original at sliced point."""
        x_fixed = 0.5
        sl2 = algebra_slider_f.slice((0, x_fixed))
        for yz in [[0.3, 0.7], [-0.5, 0.2], [0.8, -0.8]]:
            orig = algebra_slider_f.eval([x_fixed] + yz, [0, 0, 0])
            sliced = sl2.eval(yz, [0, 0])
            assert abs(sliced - orig) < 1e-8, (
                f"Slider slice mismatch at yz={yz}: {sliced} vs {orig}"
            )

    def test_slider_slice_partition_remap(self, algebra_slider_f):
        """After slicing dim 0, remaining partition indices remapped correctly."""
        sl2 = algebra_slider_f.slice((0, 0.5))
        # Original partition was [[0],[1],[2]], after removing dim 0:
        # dim 1->0, dim 2->1, so partition should be [[0],[1]]
        assert sl2.partition == [[0], [1]]


# ======================================================================
# TestExtrudeAlgebraUseCase
# ======================================================================

class TestExtrudeAlgebraUseCase:
    """Integration tests: Trade A(x,y) + Trade B(x,z) via extrude."""

    @pytest.fixture(scope="class")
    def trade_portfolio(self):
        """Build Trade A on (spot, rate) and Trade B on (spot, vol).
        Extrude both to 3D (spot, rate, vol) and add via algebra."""
        def trade_a_func(x, _):
            # Price depends on spot (x[0]) and rate (x[1])
            return math.sin(x[0]) + 0.5 * math.cos(x[1])

        def trade_b_func(x, _):
            # Price depends on spot (x[0]) and vol (x[1])
            return math.cos(x[0]) + 0.3 * math.sin(x[1])

        ct_a = ChebyshevApproximation(
            trade_a_func, 2,
            [[-1, 1], [0.01, 0.08]], [11, 11]
        )
        ct_a.build(verbose=False)

        ct_b = ChebyshevApproximation(
            trade_b_func, 2,
            [[-1, 1], [0.15, 0.35]], [11, 11]
        )
        ct_b.build(verbose=False)

        # Extrude to common 3D: (spot, rate, vol)
        ct_a_3d = ct_a.extrude((2, (0.15, 0.35), 11))  # add vol dim
        ct_b_3d = ct_b.extrude((1, (0.01, 0.08), 11))  # add rate dim

        portfolio = ct_a_3d + ct_b_3d

        return ct_a, ct_b, portfolio

    def test_extrude_add_different_variables(self, trade_portfolio):
        """Trade A(spot,rate) + Trade B(spot,vol) via extrude produces 3D CT."""
        _, _, portfolio = trade_portfolio
        assert portfolio.num_dimensions == 3
        assert portfolio.n_nodes == [11, 11, 11]

    def test_portfolio_value_correct(self, trade_portfolio):
        """Portfolio evaluates to f(spot,rate) + g(spot,vol) at test points."""
        ct_a, ct_b, portfolio = trade_portfolio
        test_pts = [
            [0.5, 0.03, 0.25],
            [-0.3, 0.06, 0.2],
            [0.0, 0.04, 0.3],
            [0.8, 0.02, 0.18],
        ]
        for pt in test_pts:
            spot, rate, vol = pt
            val_a = ct_a.vectorized_eval([spot, rate], [0, 0])
            val_b = ct_b.vectorized_eval([spot, vol], [0, 0])
            port_val = portfolio.vectorized_eval(pt, [0, 0, 0])
            assert abs(port_val - (val_a + val_b)) < 1e-10, (
                f"Portfolio mismatch at {pt}: {port_val} vs {val_a + val_b}"
            )

    def test_portfolio_greeks(self, trade_portfolio):
        """Greeks of combined portfolio are correct."""
        ct_a, ct_b, portfolio = trade_portfolio
        pt = [0.5, 0.04, 0.25]
        spot, rate, vol = pt

        # d/dspot of portfolio = d/dspot(trade_a) + d/dspot(trade_b)
        delta_a = ct_a.vectorized_eval([spot, rate], [1, 0])
        delta_b = ct_b.vectorized_eval([spot, vol], [1, 0])
        delta_port = portfolio.vectorized_eval(pt, [1, 0, 0])
        assert abs(delta_port - (delta_a + delta_b)) < 1e-8

        # d/drate of portfolio = d/drate(trade_a) + 0 (trade_b independent of rate)
        rho_a = ct_a.vectorized_eval([spot, rate], [0, 1])
        rho_port = portfolio.vectorized_eval(pt, [0, 1, 0])
        assert abs(rho_port - rho_a) < 1e-8

        # d/dvol of portfolio = 0 + d/dvol(trade_b)
        vega_b = ct_b.vectorized_eval([spot, vol], [0, 1])
        vega_port = portfolio.vectorized_eval(pt, [0, 0, 1])
        assert abs(vega_port - vega_b) < 1e-8


# ======================================================================
# TestEdgeCases — additional coverage
# ======================================================================

class TestEdgeCases:
    """Edge cases for extrude/slice not covered by the main test classes."""

    def test_extrude_min_nodes(self, extrude_cheb_1d):
        """Extrude with minimum n_nodes=2 (linear interpolant)."""
        ct2 = extrude_cheb_1d.extrude((1, (0, 10), 2))
        assert ct2.num_dimensions == 2
        assert ct2.n_nodes == [11, 2]
        # Value still preserved
        ref = extrude_cheb_1d.vectorized_eval([0.5], [0])
        val = ct2.vectorized_eval([0.5, 5.0], [0, 0])
        assert abs(val - ref) < 1e-12

    def test_slice_boundary_right(self, extrude_cheb_2d):
        """Slice at right domain boundary (+1.0)."""
        ct1 = extrude_cheb_2d.slice((1, 1.0))
        for x in TEST_POINTS_1D:
            exact = extrude_cheb_2d.vectorized_eval([x, 1.0], [0, 0])
            sliced = ct1.vectorized_eval([x], [0])
            assert abs(sliced - exact) < 1e-10

    def test_spline_slice_at_knot(self, spline_abs_1d):
        """Slice a 2D spline at exact knot value."""
        sp2 = spline_abs_1d.extrude((1, (0, 5), 9))
        # Slice dim 0 at the knot x=0
        sp1 = sp2.slice((0, 0.0))
        assert sp1.num_dimensions == 1
        # |0| = 0, so value should be ~0 everywhere in new dim
        for y in [0.0, 2.5, 5.0]:
            val = sp1.eval([y], [0])
            assert abs(val) < 1e-10

    def test_error_estimate_extruded(self, extrude_cheb_2d):
        """error_estimate() works on extruded CT (DCT recomputation)."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        err = ct3.error_estimate()
        assert err >= 0.0
        # Extruding a well-resolved function shouldn't increase error much
        orig_err = extrude_cheb_2d.error_estimate()
        assert err < orig_err * 100  # generous bound

    def test_error_estimate_sliced(self, extrude_cheb_2d):
        """error_estimate() works on sliced CT."""
        ct1 = extrude_cheb_2d.slice((1, 0.5))
        err = ct1.error_estimate()
        assert err >= 0.0

    def test_vectorized_eval_batch_extruded(self, extrude_cheb_1d):
        """vectorized_eval_batch works on extruded result."""
        ct2 = extrude_cheb_1d.extrude((1, (-2, 2), 7))
        pts = np.array([[0.5, 0.0], [-0.3, 1.0], [0.8, -1.5]])
        results = ct2.vectorized_eval_batch(pts, [0, 0])
        ref = extrude_cheb_1d.vectorized_eval([0.5], [0])
        assert abs(results[0] - ref) < 1e-12

    def test_vectorized_eval_batch_sliced(self, extrude_cheb_2d):
        """vectorized_eval_batch works on sliced result."""
        ct1 = extrude_cheb_2d.slice((1, 0.3))
        pts = np.array([[0.5], [-0.3], [0.8]])
        results = ct1.vectorized_eval_batch(pts, [0])
        for i, x in enumerate([0.5, -0.3, 0.8]):
            ref = extrude_cheb_2d.vectorized_eval([x, 0.3], [0, 0])
            assert abs(results[i] - ref) < 1e-10

    def test_vectorized_eval_multi_extruded(self, extrude_cheb_2d):
        """vectorized_eval_multi works on extruded result."""
        ct3 = extrude_cheb_2d.extrude((2, (0, 5), 9))
        pt = [0.5, -0.3, 2.5]
        results = ct3.vectorized_eval_multi(pt, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        assert len(results) == 3
        # d/d(new_dim) should be ~0
        assert abs(results[2]) < 1e-10

    def test_vectorized_eval_multi_sliced(self, extrude_cheb_2d):
        """vectorized_eval_multi works on sliced result."""
        ct1 = extrude_cheb_2d.slice((1, 0.4))
        results = ct1.vectorized_eval_multi([0.5], [[0], [1]])
        assert len(results) == 2
        ref_val = extrude_cheb_2d.vectorized_eval([0.5, 0.4], [0, 0])
        assert abs(results[0] - ref_val) < 1e-10

    def test_slider_extrude_multigroup(self):
        """Extrude slider with multi-dim group partition [[0,1],[2]]."""
        def f(x, _): return math.sin(x[0] + x[1]) + math.cos(x[2])
        sl = ChebyshevSlider(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [8, 8, 8],
                             [[0, 1], [2]], [0.0, 0.0, 0.0])
        sl.build(verbose=False)
        sl4 = sl.extrude((3, (0, 5), 7))
        assert sl4.num_dimensions == 4
        assert len(sl4.partition) == 3  # [[0,1],[2],[3]]
        # Value preserved
        ref = sl.eval([0.3, 0.5, -0.2], [0, 0, 0])
        val = sl4.eval([0.3, 0.5, -0.2, 2.5], [0, 0, 0, 0])
        assert abs(val - ref) < 1e-8

    def test_slider_slice_to_single_group(self, algebra_slider_f):
        """Slice slider from 3 groups to 1 group (2 consecutive slices)."""
        # algebra_slider_f: partition [[0],[1],[2]], sin(x)+sin(y)+sin(z)
        sl2 = algebra_slider_f.slice((0, 0.5))
        sl1 = sl2.slice((0, 0.3))  # now dim 0 was original dim 1
        assert sl1.num_dimensions == 1
        assert len(sl1.partition) == 1
        # Verify: sin(0.5) + sin(0.3) + sin(z)
        for z in [-0.5, 0.0, 0.7]:
            exact = algebra_slider_f.eval([0.5, 0.3, z], [0, 0, 0])
            sliced = sl1.eval([z], [0])
            assert abs(sliced - exact) < 1e-8, (
                f"Slider double-slice at z={z}: {sliced} vs {exact}"
            )
