"""Tests for ChebyshevApproximation: accuracy, derivatives, and eval methods."""

import math

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation
from conftest import (
    _bs_call_delta,
    _bs_call_gamma,
    _bs_call_price,
    _bs_call_rho,
    _bs_call_vega,
    sin_sum_3d,
)


# ---------------------------------------------------------------------------
# 3D sin tests
# ---------------------------------------------------------------------------

class TestSimple3D:
    def test_price_accuracy(self, cheb_sin_3d):
        p = [0.1, 0.3, 1.7]
        exact = sin_sum_3d(p, None)
        approx = cheb_sin_3d.eval(p, [0, 0, 0])
        assert abs(approx - exact) / abs(exact) * 100 < 1.0

    def test_derivative_dy(self, cheb_sin_3d):
        p = [0.1, 0.3, 1.7]
        exact = math.cos(p[1])
        approx = cheb_sin_3d.eval(p, [0, 1, 0])
        assert abs(approx - exact) < 1e-4

    def test_vectorized_matches_eval(self, cheb_sin_3d):
        p = [0.1, 0.3, 1.7]
        v1 = cheb_sin_3d.eval(p, [0, 0, 0])
        v2 = cheb_sin_3d.vectorized_eval(p, [0, 0, 0])
        assert abs(v1 - v2) < 1e-12


# ---------------------------------------------------------------------------
# 3D Black-Scholes tests
# ---------------------------------------------------------------------------

class TestBlackScholes3D:
    @pytest.mark.parametrize("S,name", [(100, "ATM"), (120, "ITM"), (80, "OTM")])
    def test_price(self, cheb_bs_3d, S, name):
        K, r, q = 100.0, 0.05, 0.02
        p = [S, 1.0, 0.25]
        exact = _bs_call_price(S=S, K=K, T=1.0, r=r, sigma=0.25, q=q)
        approx = cheb_bs_3d.eval(p, [0, 0, 0])
        assert abs(approx - exact) / exact * 100 < 0.5, f"{name} error too large"

    def test_delta(self, cheb_bs_3d):
        K, r, q = 100.0, 0.05, 0.02
        p = [100, 1.0, 0.25]
        exact = _bs_call_delta(S=100, K=K, T=1.0, r=r, sigma=0.25, q=q)
        approx = cheb_bs_3d.eval(p, [1, 0, 0])
        assert abs(approx - exact) / exact * 100 < 1.0


# ---------------------------------------------------------------------------
# 5D Black-Scholes tests
# ---------------------------------------------------------------------------

class TestBlackScholes5D:
    Q = 0.02

    @pytest.mark.parametrize("point,name", [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ])
    def test_price(self, cheb_bs_5d, point, name):
        exact = _bs_call_price(
            S=point[0], K=point[1], T=point[2], r=point[4], sigma=point[3], q=self.Q
        )
        approx = cheb_bs_5d.eval(point, [0, 0, 0, 0, 0])
        err = abs(approx - exact) / exact * 100
        assert err < 0.01, f"{name}: {err:.4f}% error"

    def test_delta(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        exact = _bs_call_delta(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=self.Q)
        approx = cheb_bs_5d.eval(p, [1, 0, 0, 0, 0])
        assert abs(approx - exact) / exact * 100 < 1.0

    def test_gamma(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        exact = _bs_call_gamma(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=self.Q)
        approx = cheb_bs_5d.eval(p, [2, 0, 0, 0, 0])
        assert abs(approx - exact) / exact * 100 < 1.0

    def test_vega(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        exact = _bs_call_vega(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=self.Q)
        approx = cheb_bs_5d.eval(p, [0, 0, 0, 1, 0])
        assert abs(approx - exact) / exact * 100 < 3.0

    def test_rho(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        exact = _bs_call_rho(S=100, K=100, T=1.0, r=0.05, sigma=0.25, q=self.Q)
        approx = cheb_bs_5d.eval(p, [0, 0, 0, 0, 1])
        assert abs(approx - exact) / exact * 100 < 1.0


# ---------------------------------------------------------------------------
# Evaluation method consistency
# ---------------------------------------------------------------------------

class TestEvalMethods:
    def test_vectorized_matches_eval(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        v1 = cheb_bs_5d.eval(p, [0, 0, 0, 0, 0])
        v2 = cheb_bs_5d.vectorized_eval(p, [0, 0, 0, 0, 0])
        assert abs(v1 - v2) < 1e-12

    def test_fast_eval_matches_eval(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        v1 = cheb_bs_5d.eval(p, [0, 0, 0, 0, 0])
        v2 = cheb_bs_5d.fast_eval(p, [0, 0, 0, 0, 0])
        assert abs(v1 - v2) < 1e-12

    def test_multi_matches_single(self, cheb_bs_5d):
        p = [100, 100, 1.0, 0.25, 0.05]
        derivs = [[0,0,0,0,0], [1,0,0,0,0], [2,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1]]
        multi = cheb_bs_5d.vectorized_eval_multi(p, derivs)
        singles = [cheb_bs_5d.vectorized_eval(p, d) for d in derivs]
        for m, s in zip(multi, singles):
            assert abs(m - s) < 1e-12

    def test_vectorized_eval_batch(self, cheb_bs_5d):
        points = np.array([
            [100, 100, 1.0, 0.25, 0.05],
            [110, 100, 1.0, 0.25, 0.05],
        ])
        results = cheb_bs_5d.vectorized_eval_batch(points, [0, 0, 0, 0, 0])
        assert results.shape == (2,)
        for i in range(2):
            single = cheb_bs_5d.vectorized_eval(points[i].tolist(), [0, 0, 0, 0, 0])
            assert abs(results[i] - single) < 1e-12

    def test_node_coincidence(self, cheb_bs_5d):
        """Evaluation at exact Chebyshev nodes should not crash."""
        p = [cheb_bs_5d.nodes[d][5] for d in range(5)]
        v1 = cheb_bs_5d.eval(p, [0, 0, 0, 0, 0])
        v2 = cheb_bs_5d.vectorized_eval(p, [0, 0, 0, 0, 0])
        v3 = cheb_bs_5d.fast_eval(p, [0, 0, 0, 0, 0])
        assert abs(v1 - v2) < 1e-12
        assert abs(v1 - v3) < 1e-12

    def test_build_required(self):
        def f(x, _):
            return x[0]
        cheb = ChebyshevApproximation(f, 1, [[0, 1]], [5])
        with pytest.raises(RuntimeError, match="build"):
            cheb.vectorized_eval([0.5], [0])
