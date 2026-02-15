"""Tests for ChebyshevTT (Tensor Train Chebyshev interpolation)."""

import math
import os
import tempfile

import numpy as np
import pytest

from pychebyshev import ChebyshevTT
from conftest import (
    _bs_5d_func,
    _bs_call_delta,
    _bs_call_gamma,
    _bs_call_price,
    _TT_5D_BS_DOMAIN,
    sin_sum_3d,
)


# ======================================================================
# Accuracy tests
# ======================================================================


class TestAccuracy:
    """Test approximation accuracy for various functions."""

    def test_1d_sin(self):
        """1D sin(x) — TT is trivially rank 1."""
        def f(x, _):
            return math.sin(x[0])

        tt = ChebyshevTT(f, 1, [[-1, 1]], [11], max_rank=3)
        tt.build(verbose=False, method="svd")
        for x in [-0.9, -0.3, 0.0, 0.5, 0.99]:
            assert abs(tt.eval([x]) - math.sin(x)) < 1e-8

    def test_3d_sin_svd(self, tt_sin_3d_svd):
        """3D sin sum with TT-SVD — should be near-exact."""
        test_pts = [
            [0.5, 0.3, 0.1],
            [-0.7, 0.0, 0.8],
            [0.0, 0.0, 0.0],
            [0.99, -0.99, 0.5],
        ]
        for pt in test_pts:
            exact = sum(math.sin(xi) for xi in pt)
            approx = tt_sin_3d_svd.eval(pt)
            assert abs(approx - exact) < 1e-8, (
                f"SVD error {abs(approx - exact):.2e} at {pt}"
            )

    def test_3d_sin_cross(self, tt_sin_3d):
        """3D sin sum with TT-Cross — should match SVD quality."""
        test_pts = [
            [0.5, 0.3, 0.1],
            [-0.7, 0.0, 0.8],
            [0.99, -0.99, 0.5],
        ]
        for pt in test_pts:
            exact = sum(math.sin(xi) for xi in pt)
            approx = tt_sin_3d.eval(pt)
            assert abs(approx - exact) < 1e-6, (
                f"Cross error {abs(approx - exact):.2e} at {pt}"
            )

    def test_3d_polynomial(self):
        """x0^2*x1 + x2 — polynomial with known structure."""
        def f(x, _):
            return x[0]**2 * x[1] + x[2]

        tt = ChebyshevTT(f, 3, [[-1, 1], [-1, 1], [-1, 1]], [7, 7, 7], max_rank=5)
        tt.build(verbose=False, method="svd")
        test_pts = [[0.5, 0.3, 0.1], [-0.8, 0.7, -0.5], [0.0, 1.0, -1.0]]
        for pt in test_pts:
            exact = pt[0]**2 * pt[1] + pt[2]
            approx = tt.eval(pt)
            assert abs(approx - exact) < 1e-8, (
                f"Poly error {abs(approx - exact):.2e} at {pt}"
            )

    def test_5d_bs_price(self, tt_bs_5d):
        """5D Black-Scholes: price error < 1%."""
        rng = np.random.default_rng(99)
        max_rel = 0.0
        for _ in range(30):
            pt = [rng.uniform(lo, hi) for lo, hi in _TT_5D_BS_DOMAIN]
            exact = _bs_5d_func(pt, None)
            approx = tt_bs_5d.eval(pt)
            if abs(exact) > 0.1:
                rel = abs(approx - exact) / abs(exact)
                max_rel = max(max_rel, rel)
        assert max_rel < 0.01, f"Max relative error {max_rel:.4e} exceeds 1%"


# ======================================================================
# Batch evaluation
# ======================================================================


class TestBatch:
    """Test vectorized batch evaluation."""

    def test_batch_matches_loop(self, tt_sin_3d):
        """eval_batch must match eval called in a loop."""
        rng = np.random.default_rng(77)
        pts = rng.uniform(-1, 1, size=(50, 3))
        batch = tt_sin_3d.eval_batch(pts)
        loop = np.array([tt_sin_3d.eval(list(p)) for p in pts])
        np.testing.assert_allclose(batch, loop, atol=1e-12)

    def test_batch_5d(self, tt_bs_5d):
        """Batch eval on 5D BS matches loop eval."""
        rng = np.random.default_rng(88)
        pts = np.column_stack([
            rng.uniform(lo, hi, size=20) for lo, hi in _TT_5D_BS_DOMAIN
        ])
        batch = tt_bs_5d.eval_batch(pts)
        loop = np.array([tt_bs_5d.eval(list(p)) for p in pts])
        np.testing.assert_allclose(batch, loop, atol=1e-12)


# ======================================================================
# Derivative tests
# ======================================================================


class TestDerivatives:
    """Test finite-difference derivatives via eval_multi."""

    def test_fd_delta(self, tt_bs_5d):
        """FD Delta vs analytical — error < 5%."""
        pt = [100.0, 100.0, 0.5, 0.25, 0.05]
        results = tt_bs_5d.eval_multi(pt, [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        analytical = _bs_call_delta(S=100, K=100, T=0.5, r=0.05, sigma=0.25, q=0.02)
        rel_err = abs(results[1] - analytical) / abs(analytical)
        assert rel_err < 0.05, f"Delta rel error {rel_err:.2e}"

    def test_fd_gamma(self, tt_bs_5d):
        """FD Gamma vs analytical — error < 10%."""
        pt = [100.0, 100.0, 0.5, 0.25, 0.05]
        results = tt_bs_5d.eval_multi(pt, [[2, 0, 0, 0, 0]])
        analytical = _bs_call_gamma(S=100, K=100, T=0.5, r=0.05, sigma=0.25, q=0.02)
        rel_err = abs(results[0] - analytical) / abs(analytical)
        assert rel_err < 0.10, f"Gamma rel error {rel_err:.2e}"

    def test_fd_3d_sin_deriv(self, tt_sin_3d):
        """FD first derivative of sin(x0)+sin(x1)+sin(x2) ~ cos(x0)."""
        pt = [0.5, 0.3, 0.1]
        results = tt_sin_3d.eval_multi(pt, [[0, 0, 0], [1, 0, 0]])
        analytical_dx0 = math.cos(0.5)
        rel_err = abs(results[1] - analytical_dx0) / abs(analytical_dx0)
        assert rel_err < 0.01, f"dsin/dx0 rel error {rel_err:.2e}"


# ======================================================================
# Rank & structure
# ======================================================================


class TestStructure:
    """Test TT rank behavior and compression."""

    def test_separable_rank(self, tt_sin_3d_svd):
        """Separable sin sum should have TT rank <= 2 with SVD."""
        ranks = tt_sin_3d_svd.tt_ranks
        # Boundary ranks are 1, interior should be <= 2 for separable
        assert ranks[0] == 1
        assert ranks[-1] == 1
        for r in ranks[1:-1]:
            assert r <= 2, f"Interior rank {r} > 2 for separable function"

    def test_higher_rank_improves(self):
        """Higher max_rank should give lower error."""
        def f(x, _):
            return math.sin(x[0] * x[1]) + math.cos(x[2])

        errors = {}
        for mr in [3, 8]:
            tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [11, 11, 11], max_rank=mr)
            tt.build(verbose=False, method="svd")
            errs = []
            rng = np.random.default_rng(42)
            for _ in range(20):
                pt = rng.uniform(-1, 1, size=3).tolist()
                exact = f(pt, None)
                approx = tt.eval(pt)
                errs.append(abs(approx - exact))
            errors[mr] = max(errs)
        assert errors[8] <= errors[3] + 1e-14, (
            f"rank 8 error {errors[8]:.2e} > rank 3 error {errors[3]:.2e}"
        )

    def test_compression_ratio_5d(self, tt_bs_5d):
        """5D TT should have compression_ratio > 1."""
        assert tt_bs_5d.compression_ratio > 1.0

    def test_tt_ranks_property(self, tt_sin_3d):
        """tt_ranks returns correct boundary ranks."""
        ranks = tt_sin_3d.tt_ranks
        assert ranks[0] == 1
        assert ranks[-1] == 1
        assert len(ranks) == 4  # d + 1


# ======================================================================
# Infrastructure
# ======================================================================


class TestInfrastructure:
    """Test build guards, serialization, printing."""

    def test_not_built_raises(self):
        """Operations before build() should raise RuntimeError."""
        tt = ChebyshevTT(sin_sum_3d, 3, [[-1, 1]] * 3, [5, 5, 5])
        with pytest.raises(RuntimeError, match="build"):
            tt.eval([0.0, 0.0, 0.0])
        with pytest.raises(RuntimeError, match="build"):
            tt.eval_batch(np.zeros((1, 3)))
        with pytest.raises(RuntimeError, match="build"):
            tt.eval_multi([0.0, 0.0, 0.0], [[0, 0, 0]])
        with pytest.raises(RuntimeError, match="build"):
            _ = tt.tt_ranks
        with pytest.raises(RuntimeError, match="build"):
            tt.error_estimate()
        with pytest.raises(RuntimeError, match="build"):
            tt.save("/tmp/test.pkl")

    def test_invalid_method(self):
        """build() with unknown method should raise ValueError."""
        tt = ChebyshevTT(sin_sum_3d, 3, [[-1, 1]] * 3, [5, 5, 5])
        with pytest.raises(ValueError, match="method"):
            tt.build(method="bad")

    def test_domain_validation(self):
        """Mismatched domain length should raise ValueError."""
        with pytest.raises(ValueError):
            ChebyshevTT(sin_sum_3d, 3, [[-1, 1], [-1, 1]], [5, 5, 5])

    def test_nodes_validation(self):
        """Mismatched n_nodes length should raise ValueError."""
        with pytest.raises(ValueError):
            ChebyshevTT(sin_sum_3d, 3, [[-1, 1]] * 3, [5, 5])

    def test_serialization_roundtrip(self, tt_sin_3d):
        """save/load round-trip preserves evaluation."""
        pt = [0.5, 0.3, 0.1]
        expected = tt_sin_3d.eval(pt)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            tt_sin_3d.save(path)
            loaded = ChebyshevTT.load(path)
            assert loaded.function is None
            assert abs(loaded.eval(pt) - expected) < 1e-14
            assert loaded.tt_ranks == tt_sin_3d.tt_ranks
        finally:
            os.unlink(path)

    def test_error_estimate_positive(self, tt_sin_3d):
        """error_estimate() should return a non-negative value."""
        err = tt_sin_3d.error_estimate()
        assert err >= 0.0

    def test_repr_str(self, tt_sin_3d):
        """__repr__ and __str__ should contain key info."""
        r = repr(tt_sin_3d)
        assert "ChebyshevTT" in r
        assert "dims=3" in r
        assert "built=True" in r

        s = str(tt_sin_3d)
        assert "ChebyshevTT" in s
        assert "TT ranks" in s
        assert "Compression" in s

    def test_total_build_evals(self, tt_sin_3d):
        """total_build_evals should be > 0 after build."""
        assert tt_sin_3d.total_build_evals > 0


# ======================================================================
# Cross vs SVD consistency
# ======================================================================


class TestCrossVsSVD:
    """Verify that TT-Cross and TT-SVD give similar results."""

    def test_3d_sin_cross_vs_svd(self, tt_sin_3d, tt_sin_3d_svd):
        """Cross and SVD should agree on sin sum evaluation."""
        rng = np.random.default_rng(55)
        for _ in range(20):
            pt = rng.uniform(-1, 1, size=3).tolist()
            v_cross = tt_sin_3d.eval(pt)
            v_svd = tt_sin_3d_svd.eval(pt)
            assert abs(v_cross - v_svd) < 1e-6, (
                f"Cross-SVD diff {abs(v_cross - v_svd):.2e} at {pt}"
            )


# ======================================================================
# Additional coverage tests
# ======================================================================


class TestCoverageGaps:
    def test_verbose_cross_build(self, capsys):
        """Build TT-Cross with verbose=True should print progress."""
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

        tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [7, 7, 7], max_rank=3)
        tt.build(verbose=True, method="cross")
        captured = capsys.readouterr()
        assert "Building" in captured.out
        assert "TT-Cross" in captured.out

    def test_verbose_svd_build(self, capsys):
        """Build TT-SVD with verbose=True should print progress."""
        def f(x, _):
            return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

        tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [5, 5, 5], max_rank=3)
        tt.build(verbose=True, method="svd")
        captured = capsys.readouterr()
        assert "Building" in captured.out
        assert "TT-SVD" in captured.out or "full tensor" in captured.out

    def test_cross_derivative_mixed_partial(self, tt_sin_3d):
        """Cross-derivative d^2f/dx0dx1 for separable function should be ~0."""
        pt = [0.5, 0.3, 0.1]
        results = tt_sin_3d.eval_multi(pt, [[1, 1, 0]])
        # For f = sin(x0) + sin(x1) + sin(x2), d^2f/dx0dx1 = 0
        assert abs(results[0]) < 0.01, f"Mixed partial = {results[0]:.4e}"

    def test_fd_derivative_near_boundary(self, tt_sin_3d):
        """FD derivative near domain boundary should use nudged point."""
        # Point very close to left boundary of [-1, 1]
        pt = [-0.999, 0.3, 0.1]
        results = tt_sin_3d.eval_multi(pt, [[1, 0, 0]])
        # cos(-0.999) ~ cos(-1) ~ 0.5403
        analytical = math.cos(-0.999)
        rel_err = abs(results[0] - analytical) / abs(analytical)
        assert rel_err < 0.05, f"Boundary FD rel error {rel_err:.2e}"

    def test_fd_derivative_near_right_boundary(self, tt_sin_3d):
        """FD derivative near right domain boundary should use nudged point."""
        pt = [0.3, 0.1, 0.999]
        results = tt_sin_3d.eval_multi(pt, [[0, 0, 1]])
        analytical = math.cos(0.999)
        rel_err = abs(results[0] - analytical) / abs(analytical)
        assert rel_err < 0.05, f"Boundary FD rel error {rel_err:.2e}"

    def test_derivative_order_3_raises(self, tt_sin_3d):
        """Derivative order > 2 should raise ValueError."""
        pt = [0.5, 0.3, 0.1]
        with pytest.raises(ValueError, match="not supported"):
            tt_sin_3d.eval_multi(pt, [[3, 0, 0]])

    def test_load_wrong_type_raises(self, tmp_path):
        """Loading a non-ChebyshevTT object should raise TypeError."""
        import pickle

        path = tmp_path / "not_tt.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"not": "a tt"}, fh)
        with pytest.raises(TypeError, match="ChebyshevTT"):
            ChebyshevTT.load(path)

    def test_version_mismatch_warning(self, tt_sin_3d):
        """Loading with version mismatch should emit a warning."""
        import warnings

        state = tt_sin_3d.__getstate__()
        state["_pychebyshev_version"] = "0.0.0-fake"

        obj = object.__new__(ChebyshevTT)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.__setstate__(state)
            assert len(w) == 1
            assert "0.0.0-fake" in str(w[0].message)

    def test_str_unbuilt(self):
        """str() of unbuilt TT should show 'not built'."""
        def f(x, _):
            return x[0]

        tt = ChebyshevTT(f, 3, [[-1, 1]] * 3, [5, 5, 5], max_rank=3)
        s = str(tt)
        assert "not built" in s
        assert "Domain:" in s

    def test_high_dim_str_truncation(self):
        """__str__() for 7D+ should truncate nodes and domain display."""
        def f(x, _):
            return sum(x)

        tt = ChebyshevTT(f, 7, [[-1, 1]] * 7, [3] * 7, max_rank=2)
        s = str(tt)
        assert "...]" in s
        assert "..." in s

    def test_higher_order_cross_derivative(self, tt_sin_3d):
        """Higher-order cross derivative [2,1,0] triggers nested FD path."""
        pt = [0.5, 0.3, 0.1]
        # d^3f/dx0^2 dx1 for f = sin(x0)+sin(x1)+sin(x2)
        # = d/dx1 (d^2/dx0^2 sin(x0)) = d/dx1 (-sin(x0)) = 0
        results = tt_sin_3d.eval_multi(pt, [[2, 1, 0]])
        assert abs(results[0]) < 0.1, f"Higher-order cross deriv = {results[0]:.4e}"

    def test_triple_cross_derivative(self, tt_sin_3d):
        """Triple cross [1,1,1] triggers 3-active-dim nested FD path."""
        pt = [0.5, 0.3, 0.1]
        # d^3f/dx0 dx1 dx2 for separable f = 0
        results = tt_sin_3d.eval_multi(pt, [[1, 1, 1]])
        assert abs(results[0]) < 0.1, f"Triple cross deriv = {results[0]:.4e}"
