"""Tests for ChebyshevApproximation: accuracy, derivatives, and eval methods."""

import math
import pathlib
import warnings

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


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for save/load round-trip on ChebyshevApproximation."""

    TEST_POINTS = [
        [0.1, 0.3, 1.7],
        [-0.5, 0.0, 2.5],
        [0.9, -0.9, 1.1],
        [0.0, 0.0, 2.0],
        [-0.3, 0.7, 2.9],
    ]

    def test_save_load_roundtrip(self, cheb_sin_3d, tmp_path):
        path = tmp_path / "cheb.pkl"
        cheb_sin_3d.save(path)
        loaded = ChebyshevApproximation.load(path)

        for pt in self.TEST_POINTS:
            orig = cheb_sin_3d.vectorized_eval(pt, [0, 0, 0])
            rest = loaded.vectorized_eval(pt, [0, 0, 0])
            np.testing.assert_allclose(rest, orig, atol=0, rtol=0)

    def test_fast_eval_after_load(self, cheb_sin_3d, tmp_path):
        path = tmp_path / "cheb.pkl"
        cheb_sin_3d.save(path)
        loaded = ChebyshevApproximation.load(path)

        for pt in self.TEST_POINTS:
            orig = cheb_sin_3d.fast_eval(pt, [0, 0, 0])
            rest = loaded.fast_eval(pt, [0, 0, 0])
            np.testing.assert_allclose(rest, orig, atol=1e-12)

    def test_function_is_none_after_load(self, cheb_sin_3d, tmp_path):
        path = tmp_path / "cheb.pkl"
        cheb_sin_3d.save(path)
        loaded = ChebyshevApproximation.load(path)
        assert loaded.function is None

    def test_loaded_state_attributes(self, cheb_sin_3d, tmp_path):
        path = tmp_path / "cheb.pkl"
        cheb_sin_3d.save(path)
        loaded = ChebyshevApproximation.load(path)

        assert loaded.tensor_values is not None
        assert loaded.tensor_values.shape == tuple(cheb_sin_3d.n_nodes)
        assert loaded.weights is not None
        assert len(loaded.weights) == cheb_sin_3d.num_dimensions
        assert loaded.diff_matrices is not None
        assert len(loaded.diff_matrices) == cheb_sin_3d.num_dimensions
        assert len(loaded.nodes) == cheb_sin_3d.num_dimensions
        for d in range(cheb_sin_3d.num_dimensions):
            np.testing.assert_array_equal(
                loaded.nodes[d], cheb_sin_3d.nodes[d]
            )

    def test_save_before_build_raises(self, tmp_path):
        def f(x, _):
            return x[0]
        cheb = ChebyshevApproximation(f, 1, [[0, 1]], [5])
        with pytest.raises(RuntimeError, match="unbuilt"):
            cheb.save(tmp_path / "fail.pkl")

    def test_version_mismatch_warning(self, cheb_sin_3d):
        # Directly test __setstate__ with a tampered version
        state = cheb_sin_3d.__getstate__()
        state["_pychebyshev_version"] = "0.0.0-fake"

        obj = object.__new__(ChebyshevApproximation)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj.__setstate__(state)
            assert len(w) == 1
            assert "0.0.0-fake" in str(w[0].message)

    def test_pathlib_path(self, cheb_sin_3d, tmp_path):
        path = pathlib.Path(tmp_path) / "cheb.pkl"
        cheb_sin_3d.save(path)
        loaded = ChebyshevApproximation.load(path)
        pt = [0.1, 0.3, 1.7]
        orig = cheb_sin_3d.vectorized_eval(pt, [0, 0, 0])
        rest = loaded.vectorized_eval(pt, [0, 0, 0])
        np.testing.assert_allclose(rest, orig, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Repr / Str
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_unbuilt(self):
        def f(x, _):
            return x[0]
        cheb = ChebyshevApproximation(f, 2, [[0, 1], [0, 1]], [11, 11])
        r = repr(cheb)
        assert "built=False" in r
        assert "dims=2" in r
        assert "[11, 11]" in r

    def test_repr_built(self, cheb_sin_3d):
        r = repr(cheb_sin_3d)
        assert "built=True" in r
        assert "dims=3" in r

    def test_str_unbuilt(self):
        def f(x, _):
            return x[0]
        cheb = ChebyshevApproximation(f, 2, [[0, 1], [2, 3]], [11, 11])
        s = str(cheb)
        assert "not built" in s
        assert "2D" in s
        assert "[11, 11]" in s
        assert "[0, 1]" in s
        assert "Build:" not in s

    def test_str_built(self, cheb_sin_3d):
        s = str(cheb_sin_3d)
        assert "built" in s
        assert "3D" in s
        assert "Build:" in s
        assert "evaluations" in s
        assert "Derivatives:" in s


# ---------------------------------------------------------------------------
# Error estimation
# ---------------------------------------------------------------------------

class TestErrorEstimation:
    def test_error_estimate_decreases_with_n(self):
        """Error estimate should decrease monotonically as n increases.

        Uses even node counts to avoid aliasing: sin(x) has only odd
        Chebyshev coefficients, so for odd n the last coefficient can
        land on an even index and be spuriously near zero.
        """
        def sin_1d(x, _):
            return math.sin(x[0])

        n_values = [6, 8, 10, 12, 14]
        estimates = []
        for n in n_values:
            cheb = ChebyshevApproximation(sin_1d, 1, [[-1, 1]], [n])
            cheb.build(verbose=False)
            estimates.append(cheb.error_estimate())

        for i in range(1, len(estimates)):
            assert estimates[i] < estimates[i - 1], (
                f"error_estimate did not decrease: n={n_values[i]} "
                f"gave {estimates[i]:.2e} >= {estimates[i-1]:.2e}"
            )

    def test_error_estimate_tracks_empirical_1d(self):
        """Error estimate should be within 2 orders of magnitude of empirical error."""
        def sin_1d(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(sin_1d, 1, [[-1, 1]], [10])
        cheb.build(verbose=False)

        estimate = cheb.error_estimate()

        # Compute empirical max error on a dense grid
        test_x = np.linspace(-1, 1, 1000)
        max_err = 0.0
        for x in test_x:
            exact = math.sin(x)
            approx = cheb.vectorized_eval([x], [0])
            max_err = max(max_err, abs(exact - approx))

        assert estimate > 0.01 * max_err, (
            f"estimate {estimate:.2e} < 0.01 * empirical {max_err:.2e}"
        )
        assert estimate < 1000 * max_err, (
            f"estimate {estimate:.2e} > 1000 * empirical {max_err:.2e}"
        )

    def test_error_estimate_sin_3d(self, cheb_sin_3d):
        """3D sin interpolant should have small but positive error estimate."""
        est = cheb_sin_3d.error_estimate()
        assert est > 0
        assert est < 0.1

    def test_error_estimate_bs_3d(self, cheb_bs_3d):
        """3D Black-Scholes interpolant error estimate should be bounded."""
        est = cheb_bs_3d.error_estimate()
        assert est > 0
        assert est < 1.0

    def test_error_estimate_bs_5d(self, cheb_bs_5d):
        """5D Black-Scholes interpolant should have small error estimate."""
        est = cheb_bs_5d.error_estimate()
        assert est > 0
        assert est < 1.0

    def test_error_estimate_not_built(self):
        """error_estimate() should raise RuntimeError if not built."""
        def f(x, _):
            return x[0]
        cheb = ChebyshevApproximation(f, 1, [[0, 1]], [5])
        with pytest.raises(RuntimeError, match="build"):
            cheb.error_estimate()

    def test_chebyshev_coefficients_1d_simple(self):
        """Chebyshev coefficients of x^2 should be c_0=0.5, c_2=0.5, rest ~0."""
        def x_squared(x, _):
            return x[0] ** 2

        cheb = ChebyshevApproximation(x_squared, 1, [[-1, 1]], [10])
        cheb.build(verbose=False)

        # Extract 1D values (the only slice)
        values_1d = cheb.tensor_values.ravel()
        coeffs = ChebyshevApproximation._chebyshev_coefficients_1d(values_1d)

        # x^2 = (T_0 + T_2) / 2, so c_0 = 0.5, c_2 = 0.5
        np.testing.assert_allclose(coeffs[0], 0.5, atol=1e-12)
        np.testing.assert_allclose(coeffs[2], 0.5, atol=1e-12)
        # All other coefficients should be near zero
        for i in range(len(coeffs)):
            if i not in (0, 2):
                assert abs(coeffs[i]) < 1e-12, (
                    f"c_{i} = {coeffs[i]:.2e}, expected ~0"
                )

    def test_error_estimate_high_n_near_zero(self):
        """With 30 nodes, sin(x) error estimate should be near machine epsilon."""
        def sin_1d(x, _):
            return math.sin(x[0])

        cheb = ChebyshevApproximation(sin_1d, 1, [[-1, 1]], [30])
        cheb.build(verbose=False)
        assert cheb.error_estimate() < 1e-14


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestCoverageGaps:
    def test_verbose_build(self, capsys):
        """Build with verbose=True should print progress messages."""
        def f(x, _):
            return x[0]

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        cheb.build(verbose=True)
        captured = capsys.readouterr()
        assert "Building" in captured.out
        assert "Built in" in captured.out

    def test_eval_before_build_raises(self):
        """eval() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError, match="build"):
            cheb.eval([0.5], [0])

    def test_vectorized_eval_multi_before_build_raises(self):
        """vectorized_eval_multi() before build() should raise RuntimeError."""
        def f(x, _):
            return x[0]

        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError, match="build"):
            cheb.vectorized_eval_multi([0.5], [[0]])

    def test_get_derivative_id(self, cheb_sin_3d):
        """get_derivative_id() should return the input as-is."""
        assert cheb_sin_3d.get_derivative_id([1, 0, 0]) == [1, 0, 0]

    def test_load_wrong_type_raises(self, tmp_path):
        """Loading a non-ChebyshevApproximation object should raise TypeError."""
        import pickle

        path = tmp_path / "not_cheb.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"not": "a chebyshev"}, fh)
        with pytest.raises(TypeError, match="ChebyshevApproximation"):
            ChebyshevApproximation.load(path)

    def test_derivative_order_3_raises(self):
        """barycentric_derivative_analytical with order>2 should raise."""
        from pychebyshev.barycentric import barycentric_derivative_analytical

        nodes = np.array([0.0, 1.0])
        values = np.array([0.0, 1.0])
        weights = np.array([1.0, -1.0])
        diff_matrix = np.array([[0.0, 1.0], [-1.0, 0.0]])
        with pytest.raises(ValueError, match="not supported"):
            barycentric_derivative_analytical(0.5, nodes, values, weights, diff_matrix, order=3)

    def test_high_dim_str_truncation(self):
        """__str__() for 7D+ should truncate nodes and domain display."""
        def f(x, _):
            return sum(x)

        cheb = ChebyshevApproximation(f, 7, [[-1, 1]] * 7, [3] * 7)
        s = str(cheb)
        assert "...]" in s
        assert "..." in s
