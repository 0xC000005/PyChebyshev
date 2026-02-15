"""Tests for Chebyshev arithmetic operators (v0.7.0)."""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSlider, ChebyshevSpline


# ---------- helper functions (exact) ----------

def _f_2d(x):
    """sin(x) + sin(y)"""
    return math.sin(x[0]) + math.sin(x[1])

def _g_2d(x):
    """cos(x) * cos(y)"""
    return math.cos(x[0]) * math.cos(x[1])

def _df_dx0(x):
    """d/dx0 of sin(x)+sin(y)"""
    return math.cos(x[0])

def _dg_dx0(x):
    """d/dx0 of cos(x)*cos(y)"""
    return -math.sin(x[0]) * math.cos(x[1])

def _d2f_dx0(x):
    """d2/dx0^2 of sin(x)+sin(y)"""
    return -math.sin(x[0])

def _d2g_dx0(x):
    """d2/dx0^2 of cos(x)*cos(y)"""
    return -math.cos(x[0]) * math.cos(x[1])


TEST_POINTS_2D = [
    [0.5, 0.3],
    [-0.7, 0.8],
    [0.0, 0.0],
    [0.9, -0.9],
    [-0.2, 0.6],
]


class TestApproxArithmetic:
    """Tests for ChebyshevApproximation arithmetic operators."""

    def test_add_values(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) + _g_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"Add failed at {p}: {approx} vs {exact}"

    def test_add_derivative(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _df_dx0(p) + _dg_dx0(p)
            approx = c.vectorized_eval(p, [1, 0])
            assert abs(approx - exact) < 1e-8, f"Add deriv failed at {p}"

    def test_add_second_derivative(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _d2f_dx0(p) + _d2g_dx0(p)
            approx = c.vectorized_eval(p, [2, 0])
            assert abs(approx - exact) < 1e-6, f"Add 2nd deriv failed at {p}"

    def test_sub_values(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f - algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) - _g_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"Sub failed at {p}"

    def test_sub_self_is_zero(self, algebra_cheb_f):
        c = algebra_cheb_f - algebra_cheb_f
        for p in TEST_POINTS_2D:
            val = c.vectorized_eval(p, [0, 0])
            assert abs(val) < 1e-14, f"f-f not zero at {p}: {val}"

    def test_mul_scalar(self, algebra_cheb_f):
        c = 3.0 * algebra_cheb_f
        for p in TEST_POINTS_2D:
            exact = 3.0 * _f_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-9, f"Scalar mul failed at {p}"

    def test_rmul_scalar(self, algebra_cheb_f):
        c1 = 3.0 * algebra_cheb_f
        c2 = algebra_cheb_f * 3.0
        for p in TEST_POINTS_2D:
            v1 = c1.vectorized_eval(p, [0, 0])
            v2 = c2.vectorized_eval(p, [0, 0])
            assert abs(v1 - v2) < 1e-15, f"rmul != mul at {p}"

    def test_truediv_scalar(self, algebra_cheb_f):
        c = algebra_cheb_f / 2.0
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) / 2.0
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"Div failed at {p}"

    def test_neg(self, algebra_cheb_f):
        c = -algebra_cheb_f
        for p in TEST_POINTS_2D:
            exact = -_f_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"Neg failed at {p}"

    def test_iadd(self, algebra_cheb_f, algebra_cheb_g):
        # Create a copy via scalar mul by 1.0 to avoid mutating the fixture
        c = algebra_cheb_f * 1.0
        c += algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) + _g_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"iadd failed at {p}"

    def test_isub(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f * 1.0
        c -= algebra_cheb_g
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) - _g_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"isub failed at {p}"

    def test_imul(self, algebra_cheb_f):
        c = algebra_cheb_f * 1.0
        c *= 2.0
        for p in TEST_POINTS_2D:
            exact = 2.0 * _f_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"imul failed at {p}"

    def test_itruediv(self, algebra_cheb_f):
        c = algebra_cheb_f * 1.0
        c /= 2.0
        for p in TEST_POINTS_2D:
            exact = _f_2d(p) / 2.0
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10, f"itruediv failed at {p}"

    def test_result_function_is_none(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        assert c.function is None

    def test_result_build_time_zero(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        assert c.build_time == 0.0

    def test_result_serializable(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            c.save(f.name)
            loaded = ChebyshevApproximation.load(f.name)
        for p in TEST_POINTS_2D[:2]:
            v_orig = c.vectorized_eval(p, [0, 0])
            v_loaded = loaded.vectorized_eval(p, [0, 0])
            assert abs(v_orig - v_loaded) < 1e-15


class TestApproxBatchAndMulti:
    """Tests for batch/multi evaluation of algebraic results."""

    def test_batch_eval(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        pts = np.array(TEST_POINTS_2D)
        batch_vals = c.vectorized_eval_batch(pts, [0, 0])
        for i, p in enumerate(TEST_POINTS_2D):
            single = c.vectorized_eval(p, [0, 0])
            assert abs(batch_vals[i] - single) < 1e-14

    def test_eval_multi(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        p = [0.5, 0.3]
        derivs = [[0, 0], [1, 0], [0, 1]]
        results = c.vectorized_eval_multi(p, derivs)
        for i, d in enumerate(derivs):
            single = c.vectorized_eval(p, d)
            assert abs(results[i] - single) < 1e-14

    def test_mul_scalar_batch(self, algebra_cheb_f):
        c = 3.0 * algebra_cheb_f
        pts = np.array(TEST_POINTS_2D)
        batch_c = c.vectorized_eval_batch(pts, [0, 0])
        batch_f = algebra_cheb_f.vectorized_eval_batch(pts, [0, 0])
        np.testing.assert_allclose(batch_c, 3.0 * batch_f, atol=1e-14)

    def test_chained_ops(self, algebra_cheb_f, algebra_cheb_g):
        """0.5*f + 0.3*g - 0.2*f = 0.3*f + 0.3*g"""
        c = 0.5 * algebra_cheb_f + 0.3 * algebra_cheb_g - 0.2 * algebra_cheb_f
        for p in TEST_POINTS_2D:
            exact = 0.3 * _f_2d(p) + 0.3 * _g_2d(p)
            approx = c.vectorized_eval(p, [0, 0])
            assert abs(approx - exact) < 1e-10


class TestSplineArithmetic:
    """Tests for ChebyshevSpline arithmetic operators."""

    # Test points in both pieces (x < 0 and x > 0) within [-1, 1] domain
    SPLINE_PTS = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]

    def test_add_values(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        for x in self.SPLINE_PTS:
            exact = abs(x) + x ** 2
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10, f"Spline add failed at {x}"

    def test_add_derivative(self, algebra_spline_f, algebra_spline_g):
        """d/dx(|x| + x^2) on right side = 1 + 2x"""
        c = algebra_spline_f + algebra_spline_g
        for x in [0.2, 0.5, 0.8]:
            exact = 1.0 + 2.0 * x
            approx = c.eval([x], [1])
            assert abs(approx - exact) < 1e-8, f"Spline deriv failed at {x}"

    def test_sub_values(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f - algebra_spline_g
        for x in self.SPLINE_PTS:
            exact = abs(x) - x ** 2
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_mul_scalar(self, algebra_spline_f):
        c = 2.5 * algebra_spline_f
        for x in self.SPLINE_PTS:
            exact = 2.5 * abs(x)
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_neg(self, algebra_spline_f):
        c = -algebra_spline_f
        for x in self.SPLINE_PTS:
            exact = -abs(x)
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_truediv_scalar(self, algebra_spline_f):
        c = algebra_spline_f / 2.0
        for x in self.SPLINE_PTS:
            exact = abs(x) / 2.0
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_rmul_scalar(self, algebra_spline_f):
        c1 = 2.5 * algebra_spline_f
        c2 = algebra_spline_f * 2.5
        for x in self.SPLINE_PTS:
            v1 = c1.eval([x], [0])
            v2 = c2.eval([x], [0])
            assert abs(v1 - v2) < 1e-15

    def test_isub(self, algebra_spline_f, algebra_spline_g):
        c = 1.0 * algebra_spline_f
        c -= algebra_spline_g
        for x in [-0.5, 0.5]:
            exact = abs(x) - x ** 2
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_itruediv(self, algebra_spline_f):
        c = 1.0 * algebra_spline_f
        c /= 2.0
        for x in [-0.5, 0.5]:
            exact = abs(x) / 2.0
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_eval_batch(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        pts = np.array(self.SPLINE_PTS)
        batch_vals = c.eval_batch(pts.reshape(-1, 1), [0])
        for i, x in enumerate(self.SPLINE_PTS):
            single = c.eval([x], [0])
            assert abs(batch_vals[i] - single) < 1e-14

    def test_different_knots_raises(self, algebra_spline_f):
        """Splines with different knots cannot be combined."""
        def h(x, _): return x[0] ** 2
        s2 = ChebyshevSpline(h, 1, [[-1, 1]], [15], [[0.5]])
        s2.build(verbose=False)
        with pytest.raises(ValueError, match="[Kk]not"):
            _ = algebra_spline_f + s2

    def test_result_error_estimate(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        assert c.error_estimate() >= 0

    def test_result_num_pieces(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        assert len(c._pieces) == len(algebra_spline_f._pieces)


class TestSliderArithmetic:
    """Tests for ChebyshevSlider arithmetic operators."""

    def test_add_values(self, algebra_slider_f, algebra_slider_g):
        c = algebra_slider_f + algebra_slider_g
        pts = [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2], [0.1, -0.3, 0.9]]
        for p in pts:
            exact = sum(math.sin(v) + math.cos(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6, f"Slider add failed at {p}"

    def test_sub_values(self, algebra_slider_f, algebra_slider_g):
        c = algebra_slider_f - algebra_slider_g
        pts = [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2]]
        for p in pts:
            exact = sum(math.sin(v) - math.cos(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_mul_scalar(self, algebra_slider_f):
        c = 2.0 * algebra_slider_f
        for p in [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2]]:
            exact = 2.0 * sum(math.sin(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_mul_scalar_derivative(self, algebra_slider_f):
        """d/dx0(2*sin(x)+...) = 2*cos(x)"""
        c = 2.0 * algebra_slider_f
        p = [0.5, 0.3, 0.7]
        exact = 2.0 * math.cos(p[0])
        approx = c.eval(p, [1, 0, 0])
        assert abs(approx - exact) < 1e-6

    def test_neg(self, algebra_slider_f):
        c = -algebra_slider_f
        for p in [[0.5, 0.3, 0.7]]:
            exact = -sum(math.sin(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_truediv_scalar(self, algebra_slider_f):
        c = algebra_slider_f / 2.0
        for p in [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2]]:
            exact = sum(math.sin(v) for v in p) / 2.0
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_rmul_scalar(self, algebra_slider_f):
        c1 = 2.0 * algebra_slider_f
        c2 = algebra_slider_f * 2.0
        for p in [[0.5, 0.3, 0.7]]:
            v1 = c1.eval(p, [0, 0, 0])
            v2 = c2.eval(p, [0, 0, 0])
            assert abs(v1 - v2) < 1e-15

    def test_add_derivative(self, algebra_slider_f, algebra_slider_g):
        """d/dx0(sin(x)+cos(x)+...) = cos(x0) - sin(x0)"""
        c = algebra_slider_f + algebra_slider_g
        p = [0.5, 0.3, 0.7]
        exact = math.cos(p[0]) - math.sin(p[0])
        approx = c.eval(p, [1, 0, 0])
        assert abs(approx - exact) < 1e-5

    def test_sub_derivative(self, algebra_slider_f, algebra_slider_g):
        """d/dx0(sin(x)-cos(x)+...) = cos(x0) + sin(x0)"""
        c = algebra_slider_f - algebra_slider_g
        p = [0.5, 0.3, 0.7]
        exact = math.cos(p[0]) + math.sin(p[0])
        approx = c.eval(p, [1, 0, 0])
        assert abs(approx - exact) < 1e-5

    def test_isub(self, algebra_slider_f, algebra_slider_g):
        c = 1.0 * algebra_slider_f
        c -= algebra_slider_g
        for p in [[0.5, 0.3, 0.7]]:
            exact = sum(math.sin(v) - math.cos(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_itruediv(self, algebra_slider_f):
        c = 1.0 * algebra_slider_f
        c /= 2.0
        for p in [[0.5, 0.3, 0.7]]:
            exact = sum(math.sin(v) for v in p) / 2.0
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6

    def test_different_partition_raises(self, algebra_slider_f):
        def h(x, _): return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
        s2 = ChebyshevSlider(h, 3, [[-1, 1]] * 3, [8] * 3,
                             [[0, 1], [2]], [0, 0, 0])
        s2.build(verbose=False)
        with pytest.raises(ValueError, match="[Pp]artition"):
            _ = algebra_slider_f + s2

    def test_different_pivot_raises(self, algebra_slider_f):
        def h(x, _): return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
        s2 = ChebyshevSlider(h, 3, [[-1, 1]] * 3, [8] * 3,
                             [[0], [1], [2]], [0.5, 0, 0])
        s2.build(verbose=False)
        with pytest.raises(ValueError, match="[Pp]ivot"):
            _ = algebra_slider_f + s2

    def test_result_pivot_value(self, algebra_slider_f, algebra_slider_g):
        c = algebra_slider_f + algebra_slider_g
        expected = algebra_slider_f.pivot_value + algebra_slider_g.pivot_value
        assert abs(c.pivot_value - expected) < 1e-14

    def test_iadd(self, algebra_slider_f, algebra_slider_g):
        c = 1.0 * algebra_slider_f
        c += algebra_slider_g
        pts = [[0.5, 0.3, 0.7], [-0.5, 0.8, -0.2]]
        for p in pts:
            exact = sum(math.sin(v) + math.cos(v) for v in p)
            approx = c.eval(p, [0, 0, 0])
            assert abs(approx - exact) < 1e-6, f"Slider iadd failed at {p}"

    def test_serializable(self, algebra_slider_f, algebra_slider_g):
        c = algebra_slider_f + algebra_slider_g
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            c.save(f.name)
            loaded = ChebyshevSlider.load(f.name)
        p = [0.5, 0.3, 0.7]
        v_orig = c.eval(p, [0, 0, 0])
        v_loaded = loaded.eval(p, [0, 0, 0])
        assert abs(v_orig - v_loaded) < 1e-15


class TestSplineExtended:
    """Additional spline tests for in-place ops and serialization."""

    def test_iadd(self, algebra_spline_f, algebra_spline_g):
        c = 1.0 * algebra_spline_f
        c += algebra_spline_g
        for x in [-0.5, 0.5]:
            exact = abs(x) + x ** 2
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_imul(self, algebra_spline_f):
        c = 1.0 * algebra_spline_f
        c *= 3.0
        for x in [-0.5, 0.5]:
            exact = 3.0 * abs(x)
            approx = c.eval([x], [0])
            assert abs(approx - exact) < 1e-10

    def test_serializable(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            c.save(f.name)
            loaded = ChebyshevSpline.load(f.name)
        v_orig = c.eval([0.5], [0])
        v_loaded = loaded.eval([0.5], [0])
        assert abs(v_orig - v_loaded) < 1e-15


class TestEdgeCases:
    """Tests for edge cases: scalar types, repr, identity operations."""

    def test_int_scalar(self, algebra_cheb_f):
        c = 2 * algebra_cheb_f
        p = [0.5, 0.3]
        exact = 2.0 * _f_2d(p)
        approx = c.vectorized_eval(p, [0, 0])
        assert abs(approx - exact) < 1e-10

    def test_numpy_scalar(self, algebra_cheb_f):
        c = np.float64(2.0) * algebra_cheb_f
        p = [0.5, 0.3]
        exact = 2.0 * _f_2d(p)
        approx = c.vectorized_eval(p, [0, 0])
        assert abs(approx - exact) < 1e-10

    def test_mul_zero(self, algebra_cheb_f):
        c = 0.0 * algebra_cheb_f
        for p in TEST_POINTS_2D:
            assert abs(c.vectorized_eval(p, [0, 0])) < 1e-15

    def test_mul_one_identity(self, algebra_cheb_f):
        c = 1.0 * algebra_cheb_f
        for p in TEST_POINTS_2D:
            v1 = algebra_cheb_f.vectorized_eval(p, [0, 0])
            v2 = c.vectorized_eval(p, [0, 0])
            assert abs(v1 - v2) < 1e-15

    def test_double_neg(self, algebra_cheb_f):
        c = -(-algebra_cheb_f)
        for p in TEST_POINTS_2D:
            v1 = algebra_cheb_f.vectorized_eval(p, [0, 0])
            v2 = c.vectorized_eval(p, [0, 0])
            assert abs(v1 - v2) < 1e-15

    def test_div_one_identity(self, algebra_cheb_f):
        c = algebra_cheb_f / 1.0
        for p in TEST_POINTS_2D:
            v1 = algebra_cheb_f.vectorized_eval(p, [0, 0])
            v2 = c.vectorized_eval(p, [0, 0])
            assert abs(v1 - v2) < 1e-15

    def test_repr_algebraic_result(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        r = repr(c)
        assert "built=True" in r
        assert "ChebyshevApproximation" in r

    def test_str_algebraic_result(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        s = str(c)
        assert "built" in s
        assert "0.000s" in s
        assert "0 evaluations" in s

    def test_spline_repr(self, algebra_spline_f, algebra_spline_g):
        c = algebra_spline_f + algebra_spline_g
        r = repr(c)
        assert "built=True" in r
        assert "ChebyshevSpline" in r

    def test_slider_repr(self, algebra_slider_f, algebra_slider_g):
        c = algebra_slider_f + algebra_slider_g
        r = repr(c)
        assert "built=True" in r
        assert "ChebyshevSlider" in r

    def test_error_estimate_on_combined(self, algebra_cheb_f, algebra_cheb_g):
        c = algebra_cheb_f + algebra_cheb_g
        est = c.error_estimate()
        assert est >= 0
        assert est < 1e-5  # reasonable for sin+cos on 11-node grid


class TestCompatibility:
    """Tests for error handling on incompatible operands."""

    def test_different_n_nodes_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        a.build(verbose=False); b.build(verbose=False)
        with pytest.raises(ValueError, match="[Nn]ode"):
            _ = a + b

    def test_different_domain_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f, 1, [[-2, 2]], [11])
        a.build(verbose=False); b.build(verbose=False)
        with pytest.raises(ValueError, match="[Dd]omain"):
            _ = a + b

    def test_different_dimensions_raises(self):
        def f1(x, _): return math.sin(x[0])
        def f2(x, _): return math.sin(x[0]) + math.sin(x[1])
        a = ChebyshevApproximation(f1, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f2, 2, [[-1, 1], [-1, 1]], [11, 11])
        a.build(verbose=False); b.build(verbose=False)
        with pytest.raises(ValueError, match="[Dd]imension"):
            _ = a + b

    def test_different_max_derivative_order_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11], max_derivative_order=2)
        b = ChebyshevApproximation(f, 1, [[-1, 1]], [11], max_derivative_order=3)
        a.build(verbose=False); b.build(verbose=False)
        with pytest.raises(ValueError, match="max_derivative_order"):
            _ = a + b

    def test_unbuilt_left_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b.build(verbose=False)
        with pytest.raises(RuntimeError, match="not built"):
            _ = a + b

    def test_unbuilt_right_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        a.build(verbose=False)
        with pytest.raises(RuntimeError, match="not built"):
            _ = a + b

    def test_unbuilt_spline_raises(self):
        def f(x, _): return abs(x[0])
        a = ChebyshevSpline(f, 1, [[-1, 1]], [15], knots=[[0.0]])
        b = ChebyshevSpline(f, 1, [[-1, 1]], [15], knots=[[0.0]])
        b.build(verbose=False)
        with pytest.raises(RuntimeError, match="not built"):
            _ = a + b

    def test_unbuilt_slider_raises(self):
        def f(x, _): return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
        a = ChebyshevSlider(f, 3, [[-1, 1]] * 3, [8] * 3,
                            [[0], [1], [2]], [0, 0, 0])
        b = ChebyshevSlider(f, 3, [[-1, 1]] * 3, [8] * 3,
                            [[0], [1], [2]], [0, 0, 0])
        b.build(verbose=False)
        with pytest.raises(RuntimeError, match="not built"):
            _ = a + b

    def test_mul_non_scalar_returns_not_implemented(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        a.build(verbose=False)
        with pytest.raises(TypeError):
            _ = a * "hello"

    def test_add_cross_type_raises(self):
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        a.build(verbose=False)
        s = ChebyshevSpline(f, 1, [[-1, 1]], [11], knots=[[0.0]])
        s.build(verbose=False)
        with pytest.raises(TypeError):
            _ = a + s

    def test_mul_two_chebs_raises(self):
        """cheb * cheb is not supported (not element-wise)."""
        def f(x, _): return math.sin(x[0])
        a = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        b = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        a.build(verbose=False); b.build(verbose=False)
        with pytest.raises(TypeError):
            _ = a * b


class TestPortfolioUseCase:
    """Integration tests simulating real portfolio use cases."""

    @pytest.fixture(scope="class")
    def instruments(self):
        """Three 'instrument' CTs on the same 2D grid."""
        def call_like(x, _): return max(x[0] - 0.5, 0.0) * math.exp(-0.05 * x[1])
        def put_like(x, _): return max(0.5 - x[0], 0.0) * math.exp(-0.05 * x[1])
        def straddle(x, _): return abs(x[0] - 0.5) * math.exp(-0.05 * x[1])

        domain = [[0.0, 1.0], [0.0, 1.0]]
        ns = [20, 12]

        c = ChebyshevApproximation(call_like, 2, domain, ns)
        p = ChebyshevApproximation(put_like, 2, domain, ns)
        s = ChebyshevApproximation(straddle, 2, domain, ns)
        c.build(verbose=False); p.build(verbose=False); s.build(verbose=False)
        return c, p, s

    def test_weighted_sum_3_instruments(self, instruments):
        """Portfolio value = weighted sum of individual CT values (exact by linearity)."""
        call, put, straddle = instruments
        portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle

        test_pts = [
            [0.7, 0.5],
            [0.3, 0.5],
            [0.5, 0.5],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.6, 0.3],
        ]
        for p in test_pts:
            # Compare portfolio eval to weighted sum of individual evals
            # This tests the algebra, not interpolation accuracy
            weighted = (
                0.4 * call.vectorized_eval(p, [0, 0])
                + 0.3 * put.vectorized_eval(p, [0, 0])
                + 0.3 * straddle.vectorized_eval(p, [0, 0])
            )
            approx = portfolio.vectorized_eval(p, [0, 0])
            assert abs(approx - weighted) < 1e-14, f"Portfolio algebra failed at {p}"

    def test_portfolio_batch_eval(self, instruments):
        call, put, straddle = instruments
        portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle

        pts = np.array([
            [0.7, 0.5], [0.3, 0.5], [0.5, 0.5],
            [0.8, 0.2], [0.2, 0.8], [0.6, 0.3],
        ])
        batch = portfolio.vectorized_eval_batch(pts, [0, 0])
        for i, p in enumerate(pts):
            single = portfolio.vectorized_eval(list(p), [0, 0])
            assert abs(batch[i] - single) < 1e-14

    def test_portfolio_greeks(self, instruments):
        """Delta of portfolio = weighted sum of individual deltas."""
        call, put, straddle = instruments
        portfolio = 0.4 * call + 0.3 * put + 0.3 * straddle

        # Test at ITM points (avoid kink at x=0.5 where derivative is discontinuous)
        for p in [[0.7, 0.5], [0.3, 0.5], [0.8, 0.2]]:
            delta_port = portfolio.vectorized_eval(p, [1, 0])
            delta_call = call.vectorized_eval(p, [1, 0])
            delta_put = put.vectorized_eval(p, [1, 0])
            delta_strad = straddle.vectorized_eval(p, [1, 0])
            weighted = 0.4 * delta_call + 0.3 * delta_put + 0.3 * delta_strad
            assert abs(delta_port - weighted) < 1e-10, f"Delta mismatch at {p}"
