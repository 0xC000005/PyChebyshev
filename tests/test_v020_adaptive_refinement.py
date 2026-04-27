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

    def test_sobol_with_constant_and_linear_modes(self):
        """f(x,y) = x + y + xy should give first ≈ (0.4, 0.4), total ≈ (0.6, 0.6).

        Regression test for the v0.20 DCT convention bug — fails with
        norm='ortho' + post-hoc c_0 halving (returns ~0.333 and ~0.667).
        """
        def f(x, _):
            return x[0] + x[1] + x[0] * x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [10, 10])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        assert result["first_order"][0] == pytest.approx(0.4, abs=1e-3)
        assert result["first_order"][1] == pytest.approx(0.4, abs=1e-3)
        assert result["total_order"][0] == pytest.approx(0.6, abs=1e-3)
        assert result["total_order"][1] == pytest.approx(0.6, abs=1e-3)

    def test_sobol_with_three_term_function(self):
        """f(x,y) = sin(x) + cos(y) + x*y — non-trivial mix of all three modes.

        Verifies first_order[0] ≈ 0.584 (the bug returned 0.424).
        """
        def f(x, _):
            return math.sin(x[0]) + math.cos(x[1]) + x[0] * x[1]

        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [12, 12])
        cheb.build(verbose=False)
        result = cheb.sobol_indices()
        # first_order[0] should be ~0.584 under canonical convention;
        # the buggy norm='ortho' path returns ~0.424 (27% error)
        assert result["first_order"][0] == pytest.approx(0.584, abs=0.05)


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

    def test_auto_knots_rejects_nan_function(self):
        def f(x, _):
            if x[0] > 0.5:
                return float("nan")
            return abs(x[0])

        with pytest.raises(ValueError, match="non-finite"):
            ChebyshevSpline.auto_knots(f, 1, [[-1, 1]], max_knots_per_dim=3)


# ============================================================================
# T4: TT auto dim reordering
# ============================================================================

def _t4_f_5d(x, _):
    return math.sin(x[0]) * math.cos(x[2]) + x[1] * x[4]


class TestAutoDimOrder:
    def test_with_auto_order_returns_tt(self):
        tt = ChebyshevTT.with_auto_order(
            _t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5,
            max_rank=8, n_trials=3,
        )
        assert isinstance(tt, ChebyshevTT)
        assert tt.num_dimensions == 5

    def test_dim_order_property_default(self):
        def f(x, _):
            return x[0]
        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        assert tt.dim_order == [0]

    def test_dim_order_after_with_auto_order(self):
        tt = ChebyshevTT.with_auto_order(
            _t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5,
            max_rank=8, n_trials=3,
        )
        assert sorted(tt.dim_order) == [0, 1, 2, 3, 4]

    def test_eval_accepts_original_order(self):
        tt = ChebyshevTT.with_auto_order(
            _t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5,
            max_rank=8, n_trials=3,
        )
        ref = ChebyshevTT(_t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5, max_rank=8)
        ref.build(verbose=False)
        x_test = [0.3, -0.4, 0.5, 0.1, -0.2]
        np.testing.assert_allclose(
            tt.eval(x_test), ref.eval(x_test), atol=1e-5
        )

    def test_save_load_preserves_dim_order(self, tmp_path):
        tt = ChebyshevTT.with_auto_order(
            _t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5,
            max_rank=8, n_trials=3,
        )
        path = tmp_path / "tt.pkl"
        tt.save(str(path))
        loaded = ChebyshevTT.load(str(path))
        assert loaded.dim_order == tt.dim_order

    def test_pre_v020_pickle_backfills_dim_order(self):
        def f(x, _):
            return x[0]
        tt = ChebyshevTT(f, 1, [[-1, 1]], [4])
        tt.build(verbose=False)
        state = tt.__getstate__()
        if "_dim_order" in state:
            del state["_dim_order"]
        restored = ChebyshevTT.__new__(ChebyshevTT)
        restored.__setstate__(state)
        assert restored.dim_order == [0]


# ============================================================================
# T5: Cross-feature tests
# ============================================================================

class TestCrossFeatures:
    def test_auto_knot_spline_supports_sobol(self):
        spl = ChebyshevSpline.auto_knots(
            _t3_f_abs, 1, [[-1, 1]], max_knots_per_dim=3,
        )
        result = spl.sobol_indices()
        assert result["variance"] > 0
        assert result["first_order"][0] == pytest.approx(1.0, abs=1e-3)

    def test_auto_dim_order_tt_supports_integrate(self):
        tt = ChebyshevTT.with_auto_order(
            _t4_f_5d, 5, [[-1, 1]] * 5, [6] * 5,
            max_rank=8, n_trials=2,
        )
        result = tt.integrate()
        assert isinstance(result, float)

    def test_auto_knot_spline_supports_v019_plot_1d(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not installed")

        spl = ChebyshevSpline.auto_knots(
            _t3_f_abs, 1, [[-1, 1]], max_knots_per_dim=3,
        )
        ax = spl.plot_1d()
        assert ax is not None


# ============================================================================
# T6: Test fixtures (read-back via Python self-test)
# ============================================================================

class TestFixtures:
    def test_approx_2d_loads(self):
        from pathlib import Path
        path = Path(__file__).parent / "fixtures" / "approx_2d_simple.pcb"
        cheb = ChebyshevApproximation.load(str(path))
        assert cheb.num_dimensions == 2
        # f(x,y) = x*y; eval(0.5, 0.5) = 0.25
        assert cheb.eval([0.5, 0.5], [0, 0]) == pytest.approx(0.25, abs=1e-6)

    def test_approx_5d_loads(self):
        from pathlib import Path
        path = Path(__file__).parent / "fixtures" / "approx_5d_bs.pcb"
        cheb = ChebyshevApproximation.load(str(path))
        assert cheb.num_dimensions == 5

    def test_spline_kink_loads(self):
        from pathlib import Path
        path = Path(__file__).parent / "fixtures" / "spline_1d_kink.pcb"
        spl = ChebyshevSpline.load(str(path))
        assert spl.num_dimensions == 1
        assert spl.eval([0.5], [0]) == pytest.approx(0.5, abs=1e-3)


# ============================================================================
# T7: dim_order guards (C1-C3 retroactive review fixes)
# ============================================================================

class TestDimOrderGuards:
    """Tests that document the v0.20 limitation: with_auto_order's permuted
    _dim_order is only threaded through eval() and full integrate().
    Other methods raise NotImplementedError until v0.20.1 fixes it."""

    def _build_non_identity_tt(self):
        """Build a 2-D TT with deterministic non-identity _dim_order.

        Constructs f(x0, x1) = sin(x0) + cos(x1) internally as perm_f
        over the permuted coordinate (x1, x0), then stamps _dim_order=[1,0].
        With _dim_order=[1,0], eval([a, b]) permutes to [b, a] before
        contracting the cores, so the TT correctly evaluates f(a, b).
        """
        def perm_f(point, ad):
            # point arrives in PERMUTED order [orig_dim_1, orig_dim_0]
            return math.sin(point[1]) + math.cos(point[0])

        tt = ChebyshevTT(perm_f, 2, [[-1, 1], [-1, 1]], [6, 6], max_rank=4)
        tt.build(verbose=False)
        tt._dim_order = [1, 0]  # orig dim 1 stored at TT position 0
        return tt

    def test_eval_multi_identity_dim_order_works(self):
        """eval_multi should NOT raise when dim_order is identity."""
        def f(x, _):
            return x[0] + x[1]

        # Standard build always has identity dim_order
        tt = ChebyshevTT(f, 2, [[-1, 1], [-1, 1]], [6, 6], max_rank=4)
        tt.build(verbose=False)
        assert tt.dim_order == [0, 1]
        result = tt.eval_multi([0.3, 0.4], [[0, 0]])
        assert result[0] == pytest.approx(0.3 + 0.4, abs=1e-6)

    def test_add_with_non_identity_dim_order_raises(self):
        """tt + tt where BOTH have same non-identity order should raise NotImplementedError."""
        tt1 = self._build_non_identity_tt()
        tt2 = self._build_non_identity_tt()
        with pytest.raises(NotImplementedError, match="dim_order"):
            _ = tt1 + tt2

    def test_iadd_with_non_identity_dim_order_raises(self):
        tt1 = self._build_non_identity_tt()
        tt2 = self._build_non_identity_tt()
        with pytest.raises(NotImplementedError, match="dim_order"):
            tt1 += tt2

    def test_neg_with_non_identity_dim_order_raises(self):
        tt = self._build_non_identity_tt()
        with pytest.raises(NotImplementedError, match="dim_order"):
            _ = -tt

    def test_mul_with_non_identity_dim_order_raises(self):
        tt = self._build_non_identity_tt()
        with pytest.raises(NotImplementedError, match="dim_order"):
            _ = tt * 2.0

    def test_truediv_with_non_identity_dim_order_raises(self):
        tt = self._build_non_identity_tt()
        with pytest.raises(NotImplementedError, match="dim_order"):
            _ = tt / 2.0

    def test_slice_with_non_identity_dim_order_works(self):
        """v0.20.1 lifted the slice dim_order guard.

        Slicing on a permuted TT now succeeds via storage-frame translation.
        Verified in detail by tests/test_v0201_dim_threading.py::TestSliceThreading.
        """
        tt = self._build_non_identity_tt()
        sliced = tt.slice([(0, 0.5)])
        assert sliced.num_dimensions == 1
        assert isinstance(sliced.eval([0.2]), float)

    def test_extrude_with_non_identity_dim_order_works(self):
        """v0.20.1 lifted the extrude dim_order guard.

        Extrude on a permuted TT now succeeds via _dim_order threading.
        Verified in detail by tests/test_v0201_dim_threading.py::TestExtrudeThreading.
        """
        tt = self._build_non_identity_tt()
        ext = tt.extrude([(2, [-1, 1], 4)])
        assert ext.num_dimensions == tt.num_dimensions + 1
        assert isinstance(ext.eval([0.1, 0.2, 0.0]), float)

    def test_to_dense_with_non_identity_dim_order_works(self):
        """v0.20.1 lifted the to_dense dim_order guard.

        Returns a dense tensor in original-dim axis order. Verified in
        detail by tests/test_v0201_dim_threading.py::TestToDenseThreading.
        """
        tt = self._build_non_identity_tt()
        dense = tt.to_dense()
        # Shape is in original-dim axis order: remap n_nodes via _dim_order.
        n_per_orig = [
            tt.n_nodes[tt._dim_order.index(d)] for d in range(tt.num_dimensions)
        ]
        assert dense.shape == tuple(n_per_orig)

    def test_partial_integrate_with_non_identity_dim_order_works(self):
        """v0.20.1 lifted the partial-integrate dim_order guard.

        Translates user-frame ``dims`` to storage positions and renumbers the
        result ``_dim_order``. Verified in detail by
        ``test_v0201_dim_threading.TestPartialIntegrateThreading``; here we
        just confirm the guard no longer raises.
        """
        tt = self._build_non_identity_tt()
        result = tt.integrate(dims=0)
        assert result.num_dimensions == tt.num_dimensions - 1

    def test_full_integrate_works_with_non_identity_dim_order(self):
        """Full integration is dim_order-invariant — should NOT raise."""
        def f(x, _):
            return math.sin(x[0]) * math.cos(x[1])

        tt = ChebyshevTT.with_auto_order(
            f, 2, [[-1, 1], [-1, 1]], [10, 10],
            max_rank=4, n_trials=3,
        )
        # Full integration: integral of sin(x)*cos(y) over [-1,1]^2 = 0
        result = tt.integrate()
        assert result == pytest.approx(0.0, abs=1e-6)


# ============================================================================
# T8: NaN / Inf guards
# ============================================================================

class TestNaNGuards:
    def test_build_rejects_nan_function(self):
        def f(x, _):
            return float("nan") if x[0] > 0 else x[0]

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [6])
        with pytest.raises(ValueError, match="non-finite"):
            cheb.build(verbose=False)

    def test_build_rejects_inf_function(self):
        def f(x, _):
            return float("inf") if x[0] > 0.9 else x[0]

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [10])
        with pytest.raises(ValueError, match="non-finite"):
            cheb.build(verbose=False)

    def test_sobol_rejects_nan_coeffs(self):
        coeffs = np.array([1.0, float("nan"), 0.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            _compute_sobol_from_coeffs(coeffs, num_dimensions=1)

    def test_sobol_rejects_inf_coeffs(self):
        coeffs = np.array([1.0, float("inf"), 0.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            _compute_sobol_from_coeffs(coeffs, num_dimensions=1)
