"""Tests for ChebyshevSpline (Piecewise Chebyshev Interpolation)."""

import math
import pathlib

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        """Valid construction with knots should give correct num_pieces and shape."""
        def f(x, _):
            return abs(x[0]) + x[1]

        sp = ChebyshevSpline(f, 2, [[-1, 1], [0, 2]], [10, 10], [[0.0], [1.0]])
        assert sp.num_pieces == 4  # 2 x 2
        assert sp._shape == (2, 2)

    def test_knot_outside_domain_raises(self):
        """Knot outside domain should raise ValueError."""
        def f(x, _):
            return x[0]

        with pytest.raises(ValueError, match="not strictly inside"):
            ChebyshevSpline(f, 1, [[-1, 1]], [10], [[2.0]])

    def test_unsorted_knots_raises(self):
        """Unsorted knots should raise ValueError."""
        def f(x, _):
            return x[0]

        with pytest.raises(ValueError, match="sorted"):
            ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.5, -0.5]])

    def test_empty_knots_gives_one_piece(self):
        """Empty knots in all dimensions should give 1 piece."""
        def f(x, _):
            return x[0] + x[1]

        sp = ChebyshevSpline(f, 2, [[-1, 1], [-1, 1]], [10, 10], [[], []])
        assert sp.num_pieces == 1
        assert sp._shape == (1, 1)


# ---------------------------------------------------------------------------
# Build required
# ---------------------------------------------------------------------------

class TestBuildRequired:
    def test_eval_before_build_raises(self):
        """eval() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="build"):
            sp.eval([0.5], [0])

    def test_error_estimate_before_build_raises(self):
        """error_estimate() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="build"):
            sp.error_estimate()


# ---------------------------------------------------------------------------
# 1D accuracy
# ---------------------------------------------------------------------------

class Test1DAccuracy:
    def test_abs_value_at_half(self, spline_abs_1d):
        """|x| value accuracy at x=0.5 should be near-exact."""
        val = spline_abs_1d.eval([0.5], [0])
        assert abs(val - 0.5) < 1e-10

    def test_abs_value_at_neg(self, spline_abs_1d):
        """|x| value accuracy at x=-0.3 should be near-exact."""
        val = spline_abs_1d.eval([-0.3], [0])
        assert abs(val - 0.3) < 1e-10

    def test_left_piece_derivative(self, spline_abs_1d):
        """Left piece (x<0) derivative of |x| should be -1."""
        deriv = spline_abs_1d.eval([-0.5], [1])
        assert abs(deriv - (-1.0)) < 1e-8

    def test_right_piece_derivative(self, spline_abs_1d):
        """Right piece (x>0) derivative of |x| should be +1."""
        deriv = spline_abs_1d.eval([0.5], [1])
        assert abs(deriv - 1.0) < 1e-8

    def test_spline_vs_global_accuracy(self, spline_abs_1d):
        """Spline should be much better than global for |x|."""
        def f(x, _):
            return abs(x[0])

        # Build global interpolant with same total nodes
        global_cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [30])
        global_cheb.build(verbose=False)

        # Test at several points
        test_pts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.6]
        spline_max_err = 0.0
        global_max_err = 0.0
        for x in test_pts:
            exact = abs(x)
            spline_err = abs(spline_abs_1d.eval([x], [0]) - exact)
            global_err = abs(global_cheb.vectorized_eval([x], [0]) - exact)
            spline_max_err = max(spline_max_err, spline_err)
            global_max_err = max(global_max_err, global_err)

        # Global should have algebraic error > 1e-2 for |x|
        assert global_max_err > 1e-3
        # Spline should have spectral (near-zero) error
        assert spline_max_err < 1e-10


# ---------------------------------------------------------------------------
# 2D accuracy
# ---------------------------------------------------------------------------

class Test2DAccuracy:
    def test_itm_value(self, spline_bs_2d):
        """In-the-money (S=110 > K=100): value should match payoff."""
        S, T = 110.0, 0.5
        exact = max(S - 100.0, 0.0) * math.exp(-0.05 * T)
        val = spline_bs_2d.eval([S, T], [0, 0])
        assert abs(val - exact) < 1e-8

    def test_otm_value(self, spline_bs_2d):
        """Out-of-money (S=90 < K=100): value should be ~0."""
        S, T = 90.0, 0.5
        exact = max(S - 100.0, 0.0) * math.exp(-0.05 * T)
        val = spline_bs_2d.eval([S, T], [0, 0])
        assert abs(val - exact) < 1e-8
        assert abs(val) < 1e-8

    def test_itm_derivative_wrt_S(self, spline_bs_2d):
        """ITM derivative w.r.t. S should be ~exp(-rT)."""
        S, T = 110.0, 0.5
        deriv = spline_bs_2d.eval([S, T], [1, 0])
        expected = math.exp(-0.05 * T)  # d/dS of (S-100)*exp(-rT) = exp(-rT)
        assert abs(deriv - expected) < 1e-8

    def test_otm_derivative_wrt_S(self, spline_bs_2d):
        """OTM derivative w.r.t. S should be ~0."""
        S, T = 90.0, 0.5
        deriv = spline_bs_2d.eval([S, T], [1, 0])
        assert abs(deriv) < 1e-8


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestBatchEval:
    def test_batch_matches_loop(self, spline_abs_1d):
        """eval_batch should match [eval(p) for p in points]."""
        pts = np.array([[-0.7], [-0.3], [0.1], [0.5], [0.9]])
        batch_results = spline_abs_1d.eval_batch(pts, [0])
        loop_results = [spline_abs_1d.eval(p.tolist(), [0]) for p in pts]
        np.testing.assert_allclose(batch_results, loop_results, atol=1e-12)

    def test_batch_spanning_multiple_pieces(self, spline_bs_2d):
        """Points spanning multiple pieces should get correct results."""
        # Points in ITM (S>100) and OTM (S<100) pieces
        pts = np.array([
            [110.0, 0.5],   # ITM
            [90.0, 0.5],    # OTM
            [105.0, 1.0],   # ITM
            [85.0, 0.25],   # OTM
        ])
        batch_results = spline_bs_2d.eval_batch(pts, [0, 0])
        for i, pt in enumerate(pts):
            exact = max(pt[0] - 100.0, 0.0) * math.exp(-0.05 * pt[1])
            assert abs(batch_results[i] - exact) < 1e-8, (
                f"Point {pt}: batch={batch_results[i]}, exact={exact}"
            )

    def test_single_piece_batch(self, spline_abs_1d):
        """Batch within a single piece should match piece's eval_batch."""
        # All points in right piece (x > 0)
        pts = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        batch_results = spline_abs_1d.eval_batch(pts, [0])
        for i, pt in enumerate(pts):
            assert abs(batch_results[i] - pt[0]) < 1e-10


# ---------------------------------------------------------------------------
# Eval multi
# ---------------------------------------------------------------------------

class TestEvalMulti:
    def test_eval_multi_matches_individual(self, spline_abs_1d):
        """eval_multi should return same results as individual eval calls."""
        pt = [0.5]
        derivs = [[0], [1]]
        multi = spline_abs_1d.eval_multi(pt, derivs)
        singles = [spline_abs_1d.eval(pt, d) for d in derivs]
        for m, s in zip(multi, singles):
            assert abs(m - s) < 1e-12

    def test_value_and_derivative(self, spline_bs_2d):
        """Value + derivative at same point should match separate calls."""
        pt = [110.0, 0.5]
        derivs = [[0, 0], [1, 0], [0, 1]]
        multi = spline_bs_2d.eval_multi(pt, derivs)
        singles = [spline_bs_2d.eval(pt, d) for d in derivs]
        for m, s in zip(multi, singles):
            assert abs(m - s) < 1e-12


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

class TestDerivatives:
    def test_analytical_vs_fd_within_piece(self, spline_abs_1d):
        """Analytical derivative should match FD within a piece (away from knots)."""
        pt = [0.5]
        h = 1e-6
        val_up = spline_abs_1d.eval([pt[0] + h], [0])
        val_dn = spline_abs_1d.eval([pt[0] - h], [0])
        fd_deriv = (val_up - val_dn) / (2 * h)
        analytical = spline_abs_1d.eval(pt, [1])
        assert abs(analytical - fd_deriv) < 1e-5

    def test_derivative_at_knot_raises(self, spline_abs_1d):
        """Requesting derivative at a knot should raise ValueError."""
        with pytest.raises(ValueError, match="not defined"):
            spline_abs_1d.eval([0.0], [1])

    def test_pure_value_at_knot_ok(self, spline_abs_1d):
        """Pure function value (derivative_order=0) at knot should be fine."""
        val = spline_abs_1d.eval([0.0], [0])
        assert abs(val - 0.0) < 1e-10

    def test_second_derivative_within_piece(self, spline_abs_1d):
        """Second derivative of |x| on x>0 piece (= x) should be ~0."""
        d2 = spline_abs_1d.eval([0.5], [2])
        assert abs(d2) < 1e-8


# ---------------------------------------------------------------------------
# Multiple knots
# ---------------------------------------------------------------------------

class TestMultipleKnots:
    @pytest.fixture
    def spline_multi_knot(self):
        """2D function with 2 knots in dim 0, 1 knot in dim 1 -> 6 pieces."""
        def f(x, _):
            return abs(x[0] - 1.0) + abs(x[0] + 1.0) + abs(x[1])

        sp = ChebyshevSpline(
            f, 2,
            [[-3, 3], [-2, 2]],
            [10, 10],
            [[-1.0, 1.0], [0.0]],
        )
        sp.build(verbose=False)
        return sp

    def test_correct_piece_count(self, spline_multi_knot):
        """2 knots in dim 0 + 1 knot in dim 1 should give 3x2=6 pieces."""
        assert spline_multi_knot.num_pieces == 6
        assert spline_multi_knot._shape == (3, 2)

    def test_routing_correctness(self, spline_multi_knot):
        """Points in different regions should route to the correct pieces."""
        # Test point in each of the 6 pieces
        test_pts = [
            [-2.0, -1.0],  # dim0: left of -1, dim1: below 0
            [-2.0, 1.0],   # dim0: left of -1, dim1: above 0
            [0.0, -1.0],   # dim0: between -1 and 1, dim1: below 0
            [0.0, 1.0],    # dim0: between -1 and 1, dim1: above 0
            [2.0, -1.0],   # dim0: right of 1, dim1: below 0
            [2.0, 1.0],    # dim0: right of 1, dim1: above 0
        ]
        for pt in test_pts:
            exact = abs(pt[0] - 1.0) + abs(pt[0] + 1.0) + abs(pt[1])
            approx = spline_multi_knot.eval(pt, [0, 0])
            assert abs(approx - exact) < 1e-8, (
                f"Error at {pt}: approx={approx}, exact={exact}"
            )

    def test_accuracy_across_all_pieces(self, spline_multi_knot):
        """Accuracy should be consistent across all pieces."""
        rng = np.random.default_rng(42)
        max_err = 0.0
        for _ in range(50):
            pt = [rng.uniform(-3, 3), rng.uniform(-2, 2)]
            exact = abs(pt[0] - 1.0) + abs(pt[0] + 1.0) + abs(pt[1])
            approx = spline_multi_knot.eval(pt, [0, 0])
            max_err = max(max_err, abs(approx - exact))
        assert max_err < 1e-8


# ---------------------------------------------------------------------------
# Error estimation
# ---------------------------------------------------------------------------

class TestErrorEstimate:
    def test_error_estimate_positive(self, spline_abs_1d):
        """error_estimate() should be > 0 for kinked function."""
        est = spline_abs_1d.error_estimate()
        assert est > 0

    def test_spline_error_less_than_global(self):
        """Spline error_estimate() should be < global for |x|."""
        def f(x, _):
            return abs(x[0])

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
        sp.build(verbose=False)

        global_cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        global_cheb.build(verbose=False)

        assert sp.error_estimate() < global_cheb.error_estimate()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_load_roundtrip(self, spline_abs_1d, tmp_path):
        """save()/load() round-trip: eval should match original."""
        path = tmp_path / "spline.pkl"
        spline_abs_1d.save(path)
        loaded = ChebyshevSpline.load(path)

        test_pts = [[-0.7], [-0.3], [0.1], [0.5], [0.9]]
        for pt in test_pts:
            orig = spline_abs_1d.eval(pt, [0])
            rest = loaded.eval(pt, [0])
            np.testing.assert_allclose(rest, orig, atol=0, rtol=0)

    def test_function_is_none_after_load(self, spline_abs_1d, tmp_path):
        """Loaded object should have function=None."""
        path = tmp_path / "spline.pkl"
        spline_abs_1d.save(path)
        loaded = ChebyshevSpline.load(path)
        assert loaded.function is None

    def test_wrong_type_raises(self, tmp_path):
        """Loading a non-ChebyshevSpline object should raise TypeError."""
        import pickle

        path = tmp_path / "not_spline.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"not": "a spline"}, fh)

        with pytest.raises(TypeError, match="ChebyshevSpline"):
            ChebyshevSpline.load(path)


# ---------------------------------------------------------------------------
# Repr / Str
# ---------------------------------------------------------------------------

class TestReprStr:
    def test_repr_contains_key_info(self, spline_abs_1d):
        """repr() should contain dims, pieces, built."""
        r = repr(spline_abs_1d)
        assert "dims=1" in r
        assert "pieces=2" in r
        assert "built=True" in r

    def test_str_contains_knots_pieces_domain(self, spline_abs_1d):
        """str() should contain knots, pieces, domain, error estimate."""
        s = str(spline_abs_1d)
        assert "Knots:" in s
        assert "Pieces:" in s
        assert "Domain:" in s
        assert "Error est:" in s


# ---------------------------------------------------------------------------
# Spline vs Global comparison
# ---------------------------------------------------------------------------

class TestSplineVsGlobal:
    def test_abs_x_global_poor_spline_excellent(self):
        """|x| with 15 nodes: global error > 1e-2, spline < 1e-10."""
        def f(x, _):
            return abs(x[0])

        global_cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        global_cheb.build(verbose=False)

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
        sp.build(verbose=False)

        # Evaluate at test points away from 0
        test_pts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8]
        global_max_err = max(
            abs(global_cheb.vectorized_eval([x], [0]) - abs(x))
            for x in test_pts
        )
        spline_max_err = max(
            abs(sp.eval([x], [0]) - abs(x))
            for x in test_pts
        )

        assert global_max_err > 1e-2, (
            f"Global error {global_max_err:.2e} should be > 1e-2"
        )
        assert spline_max_err < 1e-10, (
            f"Spline error {spline_max_err:.2e} should be < 1e-10"
        )

    def test_call_payoff_accuracy(self):
        """max(x-100,0): spline with 15 nodes beats global with 15 nodes."""
        def f(x, _):
            return max(x[0] - 100.0, 0.0)

        global_cheb = ChebyshevApproximation(f, 1, [[80, 120]], [15])
        global_cheb.build(verbose=False)

        sp = ChebyshevSpline(f, 1, [[80, 120]], [15], [[100.0]])
        sp.build(verbose=False)

        # Test at several points
        test_pts = [85, 90, 95, 105, 110, 115]
        global_max_err = max(
            abs(global_cheb.vectorized_eval([x], [0]) - max(x - 100.0, 0.0))
            for x in test_pts
        )
        spline_max_err = max(
            abs(sp.eval([x], [0]) - max(x - 100.0, 0.0))
            for x in test_pts
        )

        assert spline_max_err < global_max_err, (
            f"Spline error {spline_max_err:.2e} should be < "
            f"global error {global_max_err:.2e}"
        )

    def test_smooth_function_similar_accuracy(self):
        """Smooth sin: spline and global should have similar accuracy."""
        def f(x, _):
            return math.sin(x[0])

        global_cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        global_cheb.build(verbose=False)

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
        sp.build(verbose=False)

        # Both should be excellent on a smooth function
        test_pts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8]
        global_max_err = max(
            abs(global_cheb.vectorized_eval([x], [0]) - math.sin(x))
            for x in test_pts
        )
        spline_max_err = max(
            abs(sp.eval([x], [0]) - math.sin(x))
            for x in test_pts
        )

        # Both should be < 1e-10 for smooth function with 15 nodes
        assert global_max_err < 1e-10
        assert spline_max_err < 1e-10

    def test_higher_order_kink_max_x_squared(self):
        """max(x,0)^2: C^1 but not C^2 kink. Spline should beat global."""
        def f(x, _):
            return max(x[0], 0.0) ** 2

        global_cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        global_cheb.build(verbose=False)

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
        sp.build(verbose=False)

        test_pts = [0.1, 0.3, 0.5, 0.7, -0.2, -0.4, -0.8]
        global_max_err = max(
            abs(global_cheb.vectorized_eval([x], [0]) - max(x, 0.0) ** 2)
            for x in test_pts
        )
        spline_max_err = max(
            abs(sp.eval([x], [0]) - max(x, 0.0) ** 2)
            for x in test_pts
        )

        assert spline_max_err < global_max_err, (
            f"Spline error {spline_max_err:.2e} should be < "
            f"global error {global_max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# Additional coverage tests (from review)
# ---------------------------------------------------------------------------

class TestBatchEvalDerivatives:
    def test_batch_with_first_derivative(self, spline_abs_1d):
        """eval_batch with derivative_order=[1] should match loop of eval."""
        pts = np.array([[-0.7], [-0.3], [0.3], [0.7]])
        batch_results = spline_abs_1d.eval_batch(pts, [1])
        loop_results = [spline_abs_1d.eval(p.tolist(), [1]) for p in pts]
        np.testing.assert_allclose(batch_results, loop_results, atol=1e-12)

    def test_batch_derivatives_spanning_pieces(self, spline_bs_2d):
        """Batch derivatives across ITM/OTM pieces should be correct."""
        pts = np.array([
            [110.0, 0.5],  # ITM: dV/dS ~ exp(-rT)
            [90.0, 0.5],   # OTM: dV/dS ~ 0
        ])
        batch_results = spline_bs_2d.eval_batch(pts, [1, 0])
        assert abs(batch_results[0] - math.exp(-0.05 * 0.5)) < 1e-8
        assert abs(batch_results[1]) < 1e-8

    def test_batch_at_knot_boundary_values(self, spline_abs_1d):
        """Batch eval at knot boundary (x=0) should route correctly for values."""
        pts = np.array([[0.0], [0.5], [-0.5]])
        results = spline_abs_1d.eval_batch(pts, [0])
        np.testing.assert_allclose(results, [0.0, 0.5, 0.5], atol=1e-10)


class TestDomainBoundary:
    def test_eval_at_left_boundary(self, spline_abs_1d):
        """Evaluation at the left domain boundary x=-1 should work."""
        val = spline_abs_1d.eval([-1.0], [0])
        assert abs(val - 1.0) < 1e-10

    def test_eval_at_right_boundary(self, spline_abs_1d):
        """Evaluation at the right domain boundary x=1 should work."""
        val = spline_abs_1d.eval([1.0], [0])
        assert abs(val - 1.0) < 1e-10

    def test_derivative_at_domain_boundary(self, spline_abs_1d):
        """Derivative at domain boundary should work (not a knot)."""
        deriv = spline_abs_1d.eval([0.99], [1])
        assert abs(deriv - 1.0) < 1e-6


class TestProperties:
    def test_total_build_evals(self, spline_abs_1d):
        """total_build_evals should equal num_pieces * prod(n_nodes)."""
        assert spline_abs_1d.total_build_evals == 2 * 15  # 2 pieces x 15 nodes

    def test_total_build_evals_2d(self, spline_bs_2d):
        """total_build_evals for 2D spline."""
        assert spline_bs_2d.total_build_evals == 2 * 15 * 15  # 2 pieces x 15x15

    def test_build_time_positive(self, spline_abs_1d):
        """build_time should be > 0 after build."""
        assert spline_abs_1d.build_time > 0


class TestBuildRequiredExtended:
    def test_eval_multi_before_build_raises(self):
        """eval_multi() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="build"):
            sp.eval_multi([0.5], [[0]])

    def test_eval_batch_before_build_raises(self):
        """eval_batch() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="build"):
            sp.eval_batch(np.array([[0.5]]), [0])

    def test_save_before_build_raises(self, tmp_path):
        """save() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        with pytest.raises(RuntimeError, match="build"):
            sp.save(tmp_path / "spline.pkl")


class TestReprStrUnbuilt:
    def test_repr_unbuilt(self):
        """repr() of unbuilt spline should show built=False."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        r = repr(sp)
        assert "built=False" in r

    def test_str_unbuilt(self):
        """str() of unbuilt spline should show 'not built' and no error/build info."""
        def f(x, _):
            return x[0]

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [10], [[0.0]])
        s = str(sp)
        assert "not built" in s
        assert "Error est:" not in s
        assert "Build:" not in s


class TestVerboseAndDisplay:
    def test_verbose_build(self, capsys):
        """Build with verbose=True should print progress messages."""
        def f(x, _):
            return abs(x[0])

        sp = ChebyshevSpline(f, 1, [[-1, 1]], [5], [[0.0]])
        sp.build(verbose=True)
        captured = capsys.readouterr()
        assert "Building" in captured.out
        assert "Piece" in captured.out
        assert "Build complete" in captured.out

    def test_version_mismatch_warning(self, spline_abs_1d):
        """Loading with version mismatch should emit a warning."""
        import warnings

        state = spline_abs_1d.__getstate__()
        state["_pychebyshev_version"] = "0.0.0-fake"

        obj = object.__new__(ChebyshevSpline)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.__setstate__(state)
            assert len(w) == 1
            assert "0.0.0-fake" in str(w[0].message)

    def test_high_dim_str_truncation(self):
        """__str__() for 7D+ should truncate nodes, knots, domain display."""
        def f(x, _):
            return sum(abs(v) for v in x)

        sp = ChebyshevSpline(
            f, 7,
            [[-1, 1]] * 7,
            [3] * 7,
            [[0.0]] * 7,
        )
        s = str(sp)
        assert "...]" in s
        assert "..." in s
