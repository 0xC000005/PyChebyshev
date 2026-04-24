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


class TestOrthogonalization:
    """Tests for ChebyshevTT.orth_left / orth_right (v0.13)."""

    @pytest.fixture
    def tt_3d(self):
        from pychebyshev.tensor_train import ChebyshevTT
        def f(x, _):
            return np.sin(x[0]) * np.cos(x[1]) + 0.3 * x[2] ** 2
        tt = ChebyshevTT(f, 3, [(-1.0, 1.0)] * 3, [11, 11, 11],
                         tolerance=1e-6, max_rank=6)
        tt.build(verbose=False, method="cross", seed=42)
        return tt

    def test_orth_left_produces_left_orthogonal_cores(self, tt_3d):
        tt_3d.orth_left(position=2)
        # Cores 0 and 1 must satisfy Q^T Q = I after unfolding as (r_k*n_k, r_{k+1})
        for k in (0, 1):
            C = tt_3d._coeff_cores[k]
            r0, n, r1 = C.shape
            Q = C.reshape(r0 * n, r1)
            gram = Q.T @ Q
            assert np.allclose(gram, np.eye(r1), atol=1e-10), \
                f"core {k} not left-orthogonal after orth_left(2)"

    def test_orth_right_produces_right_orthogonal_cores(self, tt_3d):
        tt_3d.orth_right(position=0)
        # Cores 1 and 2 must satisfy Q Q^T = I after unfolding as (r_k, n_k*r_{k+1})
        for k in (1, 2):
            C = tt_3d._coeff_cores[k]
            r0, n, r1 = C.shape
            Q = C.reshape(r0, n * r1)
            gram = Q @ Q.T
            assert np.allclose(gram, np.eye(r0), atol=1e-10), \
                f"core {k} not right-orthogonal after orth_right(0)"

    def test_orth_left_preserves_eval(self, tt_3d):
        pts = np.array([[0.1, -0.2, 0.3], [0.5, 0.5, -0.5], [-0.9, 0.1, 0.7]])
        before = np.array([tt_3d.eval(p.tolist()) for p in pts])
        tt_3d.orth_left(position=2)
        after = np.array([tt_3d.eval(p.tolist()) for p in pts])
        assert np.allclose(before, after, atol=1e-10)

    def test_orth_right_preserves_eval(self, tt_3d):
        pts = np.array([[0.1, -0.2, 0.3], [0.5, 0.5, -0.5], [-0.9, 0.1, 0.7]])
        before = np.array([tt_3d.eval(p.tolist()) for p in pts])
        tt_3d.orth_right(position=0)
        after = np.array([tt_3d.eval(p.tolist()) for p in pts])
        assert np.allclose(before, after, atol=1e-10)

    def test_orth_left_position_zero_raises(self, tt_3d):
        with pytest.raises(ValueError, match="position must be in"):
            tt_3d.orth_left(position=0)

    def test_orth_right_position_last_raises(self, tt_3d):
        with pytest.raises(ValueError, match="position must be in"):
            tt_3d.orth_right(position=2)  # d=3, last valid is d-2=1

    def test_orth_left_out_of_range_raises(self, tt_3d):
        with pytest.raises(ValueError, match="position must be in"):
            tt_3d.orth_left(position=5)

    def test_orth_left_on_unbuilt_raises(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt = ChebyshevTT(lambda x: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5])
        with pytest.raises(RuntimeError, match="Call build"):
            tt.orth_left(position=1)


class TestInnerProduct:
    """Tests for ChebyshevTT.inner_product (v0.13)."""

    def test_inner_product_matches_explicit_contraction_2d(self):
        from pychebyshev.tensor_train import ChebyshevTT
        def f(x, data=None):
            return np.sin(x[0]) + 0.5 * x[1]
        def g(x, data=None):
            return np.cos(x[0]) * x[1]
        domain = [(-1.0, 1.0), (-1.0, 1.0)]
        n_nodes = [8, 8]
        tt_a = ChebyshevTT(f, 2, domain, n_nodes,
                           tolerance=1e-8, max_rank=8)
        tt_b = ChebyshevTT(g, 2, domain, n_nodes,
                           tolerance=1e-8, max_rank=8)
        tt_a.build(verbose=False, method="cross", seed=1)
        tt_b.build(verbose=False, method="cross", seed=2)

        ip = tt_a.inner_product(tt_b)

        # Reference: contract full TT tensors explicitly via einsum on cores
        def full_tensor(tt):
            T = tt._coeff_cores[0]  # (1, n, r1)
            for k in range(1, tt.num_dimensions):
                T = np.einsum("...i,ijk->...jk", T, tt._coeff_cores[k])
            return T.squeeze()

        Ta = full_tensor(tt_a)
        Tb = full_tensor(tt_b)
        ref = np.sum(Ta * Tb)
        assert abs(ip - ref) < 1e-10, f"inner_product {ip} != reference {ref}"

    def test_self_inner_product_is_squared_norm(self):
        from pychebyshev.tensor_train import ChebyshevTT
        def f(x, data=None):
            return np.cos(x[0]) + x[1] ** 2
        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [10, 10],
                         tolerance=1e-8, max_rank=8)
        tt.build(verbose=False, method="cross", seed=0)
        ip = tt.inner_product(tt)
        # T is the full Chebyshev coefficient tensor; sum(T*T) is its squared Frobenius norm
        def full_tensor(tt):
            T = tt._coeff_cores[0]
            for k in range(1, tt.num_dimensions):
                T = np.einsum("...i,ijk->...jk", T, tt._coeff_cores[k])
            return T.squeeze()
        T = full_tensor(tt)
        assert abs(ip - float(np.sum(T * T))) < 1e-10
        assert ip > 0  # squared norm is positive

    def test_inner_product_raises_on_non_tt(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                         tolerance=1e-4, max_rank=3)
        tt.build(verbose=False, method="cross")
        with pytest.raises(ValueError, match="must be a ChebyshevTT"):
            tt.inner_product("not a tt")

    def test_inner_product_raises_on_domain_mismatch(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt_a = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                           tolerance=1e-4, max_rank=3)
        tt_b = ChebyshevTT(lambda x, _=None: x[0], 2, [(-2.0, 2.0)] * 2, [5, 5],
                           tolerance=1e-4, max_rank=3)
        tt_a.build(verbose=False, method="cross")
        tt_b.build(verbose=False, method="cross")
        with pytest.raises(ValueError, match="matching domains"):
            tt_a.inner_product(tt_b)

    def test_inner_product_raises_on_n_nodes_mismatch(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt_a = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                           tolerance=1e-4, max_rank=3)
        tt_b = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [7, 7],
                           tolerance=1e-4, max_rank=3)
        tt_a.build(verbose=False, method="cross")
        tt_b.build(verbose=False, method="cross")
        with pytest.raises(ValueError, match="matching n_nodes"):
            tt_a.inner_product(tt_b)

    def test_inner_product_raises_on_unbuilt_self(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt_a = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5])
        tt_b = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                           tolerance=1e-4, max_rank=3)
        tt_b.build(verbose=False, method="cross")
        with pytest.raises(RuntimeError, match="Call build"):
            tt_a.inner_product(tt_b)

    def test_inner_product_raises_on_unbuilt_other(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt_a = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                           tolerance=1e-4, max_rank=3)
        tt_b = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5])
        tt_a.build(verbose=False, method="cross")
        with pytest.raises(RuntimeError, match="Call build"):
            tt_a.inner_product(tt_b)


class TestALSInternals:
    """White-box tests for the internal ALS sweep primitive."""

    def test_als_sweep_reduces_residual_on_rank1_target(self):
        """On an exactly-rank-1 target tensor, one sweep drives residual to ~0."""
        from pychebyshev.tensor_train import _als_fixed_rank_sweeps
        rng = np.random.default_rng(0)
        # Build an exactly rank-1 target on an 8x8x8 grid
        u0 = rng.standard_normal(8)
        u1 = rng.standard_normal(8)
        u2 = rng.standard_normal(8)
        target = np.einsum("a,b,c->abc", u0, u1, u2)
        # Full-grid evaluator: returns target[i,j,k] for integer grid index
        def evals_at(idx_tuple):
            return target[idx_tuple]
        # Random rank-1 initial cores of matching shape
        cores = [
            rng.standard_normal((1, 8, 1)),
            rng.standard_normal((1, 8, 1)),
            rng.standard_normal((1, 8, 1)),
        ]
        new_cores = _als_fixed_rank_sweeps(
            cores, evals_at, n_nodes=[8, 8, 8],
            tolerance=1e-12, max_iter=5, verbose=False,
        )
        # Reconstruct and compare
        T = new_cores[0]
        for c in new_cores[1:]:
            T = np.einsum("...i,ijk->...jk", T, c)
        T = T.squeeze()
        residual = np.linalg.norm(T - target) / np.linalg.norm(target)
        assert residual < 1e-8, f"residual {residual} exceeds 1e-8"

    def test_als_sweep_refines_rank2_target_at_rank2(self):
        """Rank-2 target at rank-2 TT: ALS should fit to near machine precision."""
        from pychebyshev.tensor_train import _als_fixed_rank_sweeps
        rng = np.random.default_rng(7)
        # Rank-2 target: sum of two rank-1 tensors
        u0a, u1a, u2a = (rng.standard_normal(6) for _ in range(3))
        u0b, u1b, u2b = (rng.standard_normal(6) for _ in range(3))
        target = (np.einsum("a,b,c->abc", u0a, u1a, u2a)
                  + np.einsum("a,b,c->abc", u0b, u1b, u2b))
        def evals_at(idx):
            return target[idx]
        cores = [
            rng.standard_normal((1, 6, 2)),
            rng.standard_normal((2, 6, 2)),
            rng.standard_normal((2, 6, 1)),
        ]
        new_cores = _als_fixed_rank_sweeps(
            cores, evals_at, n_nodes=[6, 6, 6],
            tolerance=1e-12, max_iter=15, verbose=False,
        )
        T = new_cores[0]
        for c in new_cores[1:]:
            T = np.einsum("...i,ijk->...jk", T, c)
        T = T.squeeze()
        residual = np.linalg.norm(T - target) / np.linalg.norm(target)
        assert residual < 1e-6, f"residual {residual} exceeds 1e-6"

    def test_value_coeff_round_trip(self):
        from pychebyshev.tensor_train import (
            _value_core_to_coeff_core, _coeff_core_to_value_core,
        )
        rng = np.random.default_rng(2)
        for shape in [(1, 8, 3), (2, 11, 4), (3, 5, 1)]:
            v = rng.standard_normal(shape)
            c = _value_core_to_coeff_core(v)
            v_back = _coeff_core_to_value_core(c)
            assert np.allclose(v, v_back, atol=1e-12), \
                f"round-trip failed for shape {shape}"


class TestALS:
    """Tests for ChebyshevTT method='als' (v0.13)."""

    def test_als_builds_and_reaches_tolerance_3d(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, _=None):
            return np.sin(x[0]) * np.cos(x[1]) + 0.3 * x[2] ** 2

        tt = ChebyshevTT(
            f, 3, [(-1.0, 1.0)] * 3, [10, 10, 10],
            tolerance=1e-4, max_rank=6,
        )
        tt.build(verbose=False, method="als", seed=42)
        assert tt._built
        # Evaluate at a few test points and compare to f
        pts = [[0.1, -0.2, 0.3], [0.5, 0.5, -0.5], [-0.9, 0.1, 0.7]]
        for p in pts:
            got = tt.eval(p)
            want = f(p)
            assert abs(got - want) < 1e-3, f"ALS eval at {p}: {got} vs {want}"

    def test_als_matches_cross_on_same_fixture(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, _=None):
            return np.exp(-x[0] ** 2) * np.cos(x[1])

        domain = [(-1.0, 1.0), (-1.0, 1.0)]
        n_nodes = [10, 10]
        tt_cross = ChebyshevTT(
            f, 2, domain, n_nodes, tolerance=1e-6, max_rank=8,
        )
        tt_cross.build(verbose=False, method="cross", seed=1)
        tt_als = ChebyshevTT(
            f, 2, domain, n_nodes, tolerance=1e-4, max_rank=8,
        )
        tt_als.build(verbose=False, method="als", seed=1)
        pts = [[0.1, -0.2], [0.5, 0.5], [-0.9, 0.7]]
        for p in pts:
            assert abs(tt_cross.eval(p) - tt_als.eval(p)) < 5e-3

    def test_als_respects_max_rank_cap(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def hard_f(x, _=None):
            # Nearly-discontinuous function, unreachable at low rank
            return np.tanh(50 * (x[0] - x[1]))

        tt = ChebyshevTT(
            hard_f, 2, [(-1.0, 1.0)] * 2, [20, 20],
            tolerance=1e-12, max_rank=3,
        )
        tt.build(verbose=False, method="als", seed=0)
        for r in tt.tt_ranks:
            assert r <= 3

    def test_als_deterministic_given_random_state(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, _=None):
            return x[0] * x[1] + 0.5

        kwargs = dict(tolerance=1e-4, max_rank=4)
        tt_a = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8], **kwargs)
        tt_b = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8], **kwargs)
        tt_a.build(verbose=False, method="als", seed=123)
        tt_b.build(verbose=False, method="als", seed=123)
        assert abs(tt_a.eval([0.3, -0.4]) - tt_b.eval([0.3, -0.4])) < 1e-12

    def test_als_method_attribute_set(self):
        from pychebyshev.tensor_train import ChebyshevTT

        tt = ChebyshevTT(
            lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
            tolerance=1e-2, max_rank=3,
        )
        tt.build(verbose=False, method="als")
        assert tt.method == "als"

    def test_als_total_build_evals_positive(self):
        from pychebyshev.tensor_train import ChebyshevTT

        tt = ChebyshevTT(
            lambda x, _=None: x[0] + x[1], 2, [(-1.0, 1.0)] * 2, [6, 6],
            tolerance=1e-4, max_rank=3,
        )
        tt.build(verbose=False, method="als")
        assert tt.total_build_evals > 0

    def test_invalid_method_raises(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt = ChebyshevTT(lambda x, _=None: x[0], 1, [(-1.0, 1.0)], [5])
        with pytest.raises(ValueError, match="'cross', 'svd', or 'als'"):
            tt.build(verbose=False, method="bogus")


class TestCompletion:
    """Tests for ChebyshevTT.run_completion (v0.13)."""

    def test_completion_refines_cross_build(self):
        from pychebyshev.tensor_train import ChebyshevTT

        # err_before must be well above 1e-14 so the assertion actually has teeth.
        def f(x, _=None):
            return np.exp(x[0] * x[1] * x[2])

        tt = ChebyshevTT(f, 3, [(-1.0, 1.0)] * 3, [10, 10, 10],
                         tolerance=1e-3, max_rank=3)
        tt.build(verbose=False, method="cross", seed=0)
        err_before = tt.error_estimate()
        tt.run_completion(tolerance=1e-12, max_iter=20, verbose=False)
        err_after = tt.error_estimate()
        assert err_after <= err_before * 1.1 + 1e-14, \
            f"completion should not worsen error; {err_before} -> {err_after}"

    def test_completion_refines_svd_build(self):
        from pychebyshev.tensor_train import ChebyshevTT

        # sin(x) + cos(y) is truly rank-2 and SVD at max_rank=5 fits it
        # to machine precision, leaving err_before ~1e-9; the LS solve in
        # completion then has gauge freedom and can inject noise at that
        # level. Use a function whose SVD truncation at rank 5 leaves
        # real headroom so the "does not worsen" check is meaningful.
        def f(x, _=None):
            return np.exp(x[0] * x[1])

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [10, 10],
                         tolerance=1e-3, max_rank=5)
        tt.build(verbose=False, method="svd", seed=0)
        err_before = tt.error_estimate()
        tt.run_completion(tolerance=1e-12, max_iter=10, verbose=False)
        err_after = tt.error_estimate()
        assert err_after <= err_before + 1e-9

    def test_completion_refines_als_build(self):
        from pychebyshev.tensor_train import ChebyshevTT

        # err_before must be well above 1e-14 so the assertion actually has teeth.
        def f(x, _=None):
            return np.exp(x[0] * x[1])

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8],
                         tolerance=1e-3, max_rank=2)
        tt.build(verbose=False, method="als", seed=0)
        err_before = tt.error_estimate()
        tt.run_completion(tolerance=1e-12, max_iter=10, verbose=False)
        err_after = tt.error_estimate()
        assert err_after <= err_before * 1.1 + 1e-14, \
            f"completion should not worsen error; {err_before} -> {err_after}"

    def test_completion_max_iter_respected(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, _=None):
            return np.tanh(10 * x[0]) * x[1]  # hard to fit at low rank

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [10, 10],
                         tolerance=1e-3, max_rank=3)
        tt.build(verbose=False, method="cross")
        # Should not hang - max_iter=1 means at most one outer sweep.
        import time
        t0 = time.time()
        tt.run_completion(tolerance=1e-20, max_iter=1, verbose=False)
        assert time.time() - t0 < 30  # sanity timeout

    def test_completion_raises_on_unbuilt(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5])
        with pytest.raises(RuntimeError, match="Call build"):
            tt.run_completion()

    def test_completion_raises_when_function_missing(self):
        from pychebyshev.tensor_train import ChebyshevTT
        tt = ChebyshevTT(lambda x, _=None: x[0], 2, [(-1.0, 1.0)] * 2, [5, 5],
                         tolerance=1e-2, max_rank=3)
        tt.build(verbose=False, method="cross")
        tt.function = None  # simulate loaded-from-pickle case
        with pytest.raises(RuntimeError, match="requires self.function"):
            tt.run_completion()

    def test_completion_eval_stays_close_to_target(self):
        """Sanity: completion should not diverge."""
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, _=None):
            return np.cos(x[0] + x[1])

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [10, 10],
                         tolerance=1e-4, max_rank=5)
        tt.build(verbose=False, method="cross", seed=0)
        tt.run_completion(tolerance=1e-10, max_iter=10, verbose=False)
        for p in [[0.1, 0.2], [-0.5, 0.7]]:
            assert abs(tt.eval(p) - f(p)) < 1e-3


class TestCrossFeatureALS:
    """Integration: ALS-built TTs through existing features."""

    def test_als_tt_save_load_round_trip(self, tmp_path):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, data=None):
            return np.sin(x[0]) + x[1] ** 2

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8],
                         tolerance=1e-4, max_rank=4)
        tt.build(verbose=False, method="als", seed=0)
        val_before = tt.eval([0.3, -0.4])
        path = tmp_path / "als_tt.pkl"
        tt.save(path)
        tt2 = ChebyshevTT.load(path)
        val_after = tt2.eval([0.3, -0.4])
        assert abs(val_before - val_after) < 1e-12
        assert tt2.method == "als"

    def test_als_tt_eval_batch_matches_eval(self):
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, data=None):
            return x[0] * x[1] + np.sin(x[2])

        tt = ChebyshevTT(f, 3, [(-1.0, 1.0)] * 3, [8, 8, 8],
                         tolerance=1e-3, max_rank=4)
        tt.build(verbose=False, method="als", seed=0)
        pts = np.array([[0.1, -0.2, 0.3], [0.5, 0.0, -0.5]])
        batch = tt.eval_batch(pts)
        for i, p in enumerate(pts):
            assert abs(batch[i] - tt.eval(p.tolist())) < 1e-12

    def test_orth_is_idempotent_on_canonical_cores(self):
        """Calling orth_left twice at the same position should not change eval."""
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, data=None):
            return np.cos(x[0]) * np.sin(x[1])

        tt = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8],
                         tolerance=1e-5, max_rank=5)
        tt.build(verbose=False, method="cross")
        tt.orth_left(position=1)
        val1 = tt.eval([0.2, 0.3])
        tt.orth_left(position=1)  # idempotent
        val2 = tt.eval([0.2, 0.3])
        assert abs(val1 - val2) < 1e-10

    def test_inner_product_then_orth_preserves_value(self):
        """Orthogonalizing both TTs should not change their inner product."""
        from pychebyshev.tensor_train import ChebyshevTT

        def f(x, data=None):
            return np.sin(x[0]) + x[1]

        def g(x, data=None):
            return np.cos(x[0]) * x[1]

        kwargs = dict(tolerance=1e-6, max_rank=5)
        tt_a = ChebyshevTT(f, 2, [(-1.0, 1.0)] * 2, [8, 8], **kwargs)
        tt_b = ChebyshevTT(g, 2, [(-1.0, 1.0)] * 2, [8, 8], **kwargs)
        tt_a.build(verbose=False, method="cross")
        tt_b.build(verbose=False, method="cross")
        ip_before = tt_a.inner_product(tt_b)
        tt_a.orth_left(position=1)
        tt_b.orth_right(position=0)
        ip_after = tt_a.inner_product(tt_b)
        assert abs(ip_before - ip_after) < 1e-9
