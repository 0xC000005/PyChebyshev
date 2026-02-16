"""Tests for Chebyshev calculus operations: integrate, roots, minimize, maximize (v0.9.0)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline


# ======================================================================
# TestIntegrateApprox
# ======================================================================

class TestIntegrateApprox:
    """Tests for ChebyshevApproximation.integrate()."""

    def test_integrate_constant(self):
        """Integral of constant 5 on [2, 5] = 15."""
        def f(x, _): return 5.0
        cheb = ChebyshevApproximation(f, 1, [[2, 5]], [4])
        cheb.build(verbose=False)
        result = cheb.integrate()
        assert abs(result - 15.0) < 1e-10, f"Expected 15.0, got {result}"

    def test_integrate_x_squared(self):
        """Integral of x^2 on [-1, 1] = 2/3 (exact for polynomial)."""
        def f(x, _): return x[0] ** 2
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [10])
        cheb.build(verbose=False)
        result = cheb.integrate()
        expected = 2.0 / 3.0
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_integrate_x_cubed(self):
        """Integral of x^3 on [-1, 1] = 0 (odd function, exact)."""
        def f(x, _): return x[0] ** 3
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [10])
        cheb.build(verbose=False)
        result = cheb.integrate()
        assert abs(result) < 1e-10, f"Expected 0.0, got {result}"

    def test_integrate_sin(self, calculus_cheb_sin_1d):
        """Integral of sin(x) on [0, pi] = 2 (spectral accuracy)."""
        result = calculus_cheb_sin_1d.integrate()
        assert abs(result - 2.0) < 1e-10, f"Expected 2.0, got {result}"

    def test_integrate_cos(self):
        """Integral of cos(x) on [-1, 1] = 2*sin(1)."""
        def f(x, _): return math.cos(x[0])
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        cheb.build(verbose=False)
        result = cheb.integrate()
        expected = 2.0 * math.sin(1.0)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_integrate_exp(self):
        """Integral of exp(x) on [-1, 1] = e - 1/e, cross-validate with scipy."""
        from scipy.integrate import quad

        def f(x, _): return math.exp(x[0])
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        cheb.build(verbose=False)
        result = cheb.integrate()
        expected = math.e - 1.0 / math.e
        scipy_val, _ = quad(math.exp, -1, 1)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
        assert abs(result - scipy_val) < 1e-10, f"Scipy mismatch: {result} vs {scipy_val}"

    def test_integrate_2d_full(self, calculus_cheb_2d):
        """Integral of sin(x)+cos(y) on [-1,1]^2 = 4*sin(1).

        int sin(x) dx from -1 to 1 = 0 (odd function).
        int cos(y) dy from -1 to 1 = 2*sin(1).
        Double integral = 0*2 + 2*2*sin(1) = 4*sin(1).
        """
        result = calculus_cheb_2d.integrate()
        expected = 4.0 * math.sin(1.0)
        assert abs(result - expected) < 1e-8, f"Expected {expected}, got {result}"

    def test_integrate_2d_partial_dim0(self, calculus_cheb_2d):
        """Integrate out dim 0 from sin(x)+cos(y): result is 2*cos(y).

        int_{-1}^{1} sin(x) dx = 0, int_{-1}^{1} cos(y) dx = 2*cos(y).
        So int_{-1}^{1} [sin(x)+cos(y)] dx = 2*cos(y).
        """
        result = calculus_cheb_2d.integrate(dims=0)
        assert isinstance(result, ChebyshevApproximation)
        assert result.num_dimensions == 1
        for y in [-0.7, 0.0, 0.3, 0.9]:
            val = result.vectorized_eval([y], [0])
            expected = 2.0 * math.cos(y)
            assert abs(val - expected) < 1e-8, f"At y={y}: {val} vs {expected}"

    def test_integrate_2d_partial_dim1(self, calculus_cheb_2d):
        """Integrate out dim 1 from sin(x)+cos(y): result is 2*sin(x) + 2*sin(1).

        int_{-1}^{1} sin(x) dy = 2*sin(x).
        int_{-1}^{1} cos(y) dy = 2*sin(1).
        So result = 2*sin(x) + 2*sin(1).
        """
        result = calculus_cheb_2d.integrate(dims=1)
        assert isinstance(result, ChebyshevApproximation)
        assert result.num_dimensions == 1
        for x in [-0.7, 0.0, 0.3, 0.9]:
            val = result.vectorized_eval([x], [0])
            expected = 2.0 * math.sin(x) + 2.0 * math.sin(1.0)
            assert abs(val - expected) < 1e-8, f"At x={x}: {val} vs {expected}"

    def test_integrate_scaled_domain(self):
        """Integral of x on [2, 5] = (25-4)/2 = 10.5."""
        def f(x, _): return x[0]
        cheb = ChebyshevApproximation(f, 1, [[2, 5]], [4])
        cheb.build(verbose=False)
        result = cheb.integrate()
        expected = 10.5
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_integrate_before_build_raises(self):
        """integrate() raises RuntimeError if build() has not been called."""
        def f(x, _): return x[0]
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError):
            cheb.integrate()

    def test_integrate_cross_validate_scipy(self):
        """Cross-validate integral of exp(x) on [0, 2] with scipy.integrate.quad."""
        from scipy.integrate import quad

        def f(x, _): return math.exp(x[0])
        cheb = ChebyshevApproximation(f, 1, [[0, 2]], [15])
        cheb.build(verbose=False)
        result = cheb.integrate()
        scipy_val, _ = quad(math.exp, 0, 2)
        assert abs(result - scipy_val) < 1e-10, f"Cheb {result} vs scipy {scipy_val}"


# ======================================================================
# TestIntegrateSpline
# ======================================================================

class TestIntegrateSpline:
    """Tests for ChebyshevSpline.integrate()."""

    def test_spline_integrate_abs(self, calculus_spline_abs):
        """Integral of |x| on [-1, 1] = 1 (exact with knot at 0)."""
        result = calculus_spline_abs.integrate()
        assert abs(result - 1.0) < 1e-10, f"Expected 1.0, got {result}"

    def test_spline_integrate_vs_scipy(self, calculus_spline_abs):
        """Compare spline integral of |x| with scipy.integrate.quad."""
        from scipy.integrate import quad

        result = calculus_spline_abs.integrate()
        scipy_val, _ = quad(abs, -1, 1)
        assert abs(result - scipy_val) < 1e-10, f"Cheb {result} vs scipy {scipy_val}"

    def test_spline_integrate_2d_full(self):
        """2D spline integral of |x| + y^2 on [-1,1]^2 with knot at x=0.

        int_{-1}^{1} int_{-1}^{1} (|x| + y^2) dx dy
          = int_{-1}^{1} [1 + 2*y^2/3... no, let's compute step by step.
          = int_{-1}^{1} y^2 dy * int_{-1}^{1} 1 dx  -- NO, it's a sum not product.
          = int_{-1}^{1} int_{-1}^{1} |x| dx dy + int_{-1}^{1} int_{-1}^{1} y^2 dx dy
          = 2 * 1 + 2 * 2/3
          Actually: int |x| dx on [-1,1] = 1. int 1 dy on [-1,1] = 2.
          int y^2 dy on [-1,1] = 2/3. int 1 dx on [-1,1] = 2.
          Total = 1*2 + 2/3*2 = 2 + 4/3 = 10/3.
        """
        def f(x, _): return abs(x[0]) + x[1] ** 2
        sp = ChebyshevSpline(f, 2, [[-1, 1], [-1, 1]], [11, 11], [[0.0], []])
        sp.build(verbose=False)
        result = sp.integrate()
        expected = 10.0 / 3.0
        assert abs(result - expected) < 1e-8, f"Expected {expected}, got {result}"

    def test_spline_integrate_pieces_sum(self):
        """Sum of piece integrals equals total integral for 1D |x|."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        total = sp.integrate()
        piece_sum = sum(p.integrate() for p in sp._pieces)
        assert abs(total - piece_sum) < 1e-12, f"Total {total} vs pieces {piece_sum}"

    def test_spline_integrate_no_knots(self):
        """Spline with no knots should match ChebyshevApproximation.integrate()."""
        def f(x, _): return math.sin(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[]])
        sp.build(verbose=False)
        ca = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        ca.build(verbose=False)
        sp_val = sp.integrate()
        ca_val = ca.integrate()
        assert abs(sp_val - ca_val) < 1e-12, f"Spline {sp_val} vs CA {ca_val}"

    def test_spline_integrate_before_build_raises(self):
        """integrate() raises RuntimeError before build()."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        with pytest.raises(RuntimeError):
            sp.integrate()

    def test_spline_integrate_partial(self):
        """Partial integration of 2D spline |x| + y^2 over dim 0.

        int_{-1}^{1} (|x| + y^2) dx = 1 + 2*y^2.
        """
        def f(x, _): return abs(x[0]) + x[1] ** 2
        sp = ChebyshevSpline(f, 2, [[-1, 1], [-1, 1]], [11, 11], [[0.0], []])
        sp.build(verbose=False)
        result = sp.integrate(dims=0)
        assert isinstance(result, ChebyshevSpline)
        assert result.num_dimensions == 1
        for y in [-0.5, 0.0, 0.3, 0.8]:
            val = result.eval([y], [0])
            expected = 1.0 + 2.0 * y ** 2
            assert abs(val - expected) < 1e-8, f"At y={y}: {val} vs {expected}"


# ======================================================================
# TestRootsApprox
# ======================================================================

class TestRootsApprox:
    """Tests for ChebyshevApproximation.roots()."""

    def test_roots_sin(self):
        """Roots of sin(x) on [-4, 4] should include {-pi, 0, pi}."""
        def f(x, _): return math.sin(x[0])
        cheb = ChebyshevApproximation(f, 1, [[-4, 4]], [25])
        cheb.build(verbose=False)
        roots = cheb.roots()
        expected = sorted([-math.pi, 0.0, math.pi])
        assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-8, f"Root {r} != expected {e}"

    def test_roots_quadratic(self):
        """Roots of x^2 - 0.25 on [-1, 1] should be {-0.5, 0.5}."""
        def f(x, _): return x[0] ** 2 - 0.25
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [10])
        cheb.build(verbose=False)
        roots = cheb.roots()
        expected = [-0.5, 0.5]
        assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-10, f"Root {r} != expected {e}"

    def test_roots_no_roots(self):
        """exp(x) on [0, 1] has no roots."""
        def f(x, _): return math.exp(x[0])
        cheb = ChebyshevApproximation(f, 1, [[0, 1]], [10])
        cheb.build(verbose=False)
        roots = cheb.roots()
        assert len(roots) == 0, f"Expected no roots, got {roots}"

    def test_roots_constant_nonzero(self):
        """Constant 5.0 has no roots."""
        def f(x, _): return 5.0
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        cheb.build(verbose=False)
        roots = cheb.roots()
        assert len(roots) == 0, f"Expected no roots, got {roots}"

    def test_roots_boundary(self):
        """Function x on [0, 1] has root at 0."""
        def f(x, _): return x[0]
        cheb = ChebyshevApproximation(f, 1, [[0, 1]], [10])
        cheb.build(verbose=False)
        roots = cheb.roots()
        assert len(roots) >= 1, f"Expected root near 0, got {roots}"
        assert any(abs(r) < 1e-8 for r in roots), f"No root near 0 in {roots}"

    def test_roots_2d_fixed(self):
        """2D roots: f(x,y) = x - y, with y fixed at 0.3, root at x=0.3."""
        def f(x, _): return x[0] - x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        cheb.build(verbose=False)
        roots = cheb.roots(dim=0, fixed={1: 0.3})
        assert len(roots) == 1, f"Expected 1 root, got {len(roots)}: {roots}"
        assert abs(roots[0] - 0.3) < 1e-8, f"Root {roots[0]} != 0.3"

    def test_roots_missing_fixed_raises(self):
        """Multi-D without full fixed dict raises ValueError."""
        def f(x, _): return x[0] + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        cheb.build(verbose=False)
        with pytest.raises(ValueError):
            cheb.roots(dim=0)

    def test_roots_fixed_out_of_domain_raises(self):
        """Fixed value outside domain raises ValueError."""
        def f(x, _): return x[0] + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        cheb.build(verbose=False)
        with pytest.raises(ValueError):
            cheb.roots(dim=0, fixed={1: 5.0})

    def test_roots_before_build_raises(self):
        """roots() raises RuntimeError before build()."""
        def f(x, _): return x[0]
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError):
            cheb.roots()

    def test_roots_linear_n2(self):
        """Linear function x - 0.3 on [-1,1] with n=2 nodes, root at 0.3."""
        def f(x, _): return x[0] - 0.3
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [2])
        cheb.build(verbose=False)
        roots = cheb.roots()
        assert len(roots) == 1, f"Expected 1 root, got {len(roots)}: {roots}"
        assert abs(roots[0] - 0.3) < 1e-10, f"Root {roots[0]} != 0.3"


# ======================================================================
# TestRootsSpline
# ======================================================================

class TestRootsSpline:
    """Tests for ChebyshevSpline.roots()."""

    def test_spline_roots_abs_shifted(self):
        """Roots of |x| - 0.5 on [-1,1] with knot at 0: {-0.5, 0.5}."""
        def f(x, _): return abs(x[0]) - 0.5
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        roots = sp.roots()
        expected = [-0.5, 0.5]
        assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-8, f"Root {r} != expected {e}"

    def test_spline_roots_at_knot(self):
        """Function x with knot at 0: root at x=0 (the knot). Verify dedup."""
        def f(x, _): return x[0]
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        roots = sp.roots()
        # Both pieces have x=0 as endpoint root; after dedup, expect exactly 1
        assert len(roots) == 1, f"Expected 1 root after dedup, got {len(roots)}: {roots}"
        assert abs(roots[0]) < 1e-8, f"Root {roots[0]} != 0"

    def test_spline_roots_multi_piece(self):
        """Roots spanning multiple pieces: sin(x) on [-4,4] with knots at -2 and 2."""
        def f(x, _): return math.sin(x[0])
        sp = ChebyshevSpline(f, 1, [[-4, 4]], [15], [[-2.0, 2.0]])
        sp.build(verbose=False)
        roots = sp.roots()
        expected = sorted([-math.pi, 0.0, math.pi])
        assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-8, f"Root {r} != expected {e}"

    def test_spline_roots_2d(self):
        """2D spline roots: f(x,y) = |x| - 0.3 with knot at x=0, fix y=0.5."""
        def f(x, _): return abs(x[0]) - 0.3
        sp = ChebyshevSpline(f, 2, [[-1, 1], [-1, 1]], [11, 5], [[0.0], []])
        sp.build(verbose=False)
        roots = sp.roots(dim=0, fixed={1: 0.5})
        expected = [-0.3, 0.3]
        assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}: {roots}"
        for r, e in zip(roots, expected):
            assert abs(r - e) < 1e-8, f"Root {r} != expected {e}"

    def test_spline_roots_no_knots_matches_ca(self):
        """Spline with no knots matches ChebyshevApproximation.roots()."""
        def f(x, _): return math.sin(x[0])
        sp = ChebyshevSpline(f, 1, [[-4, 4]], [25], [[]])
        sp.build(verbose=False)
        ca = ChebyshevApproximation(f, 1, [[-4, 4]], [25])
        ca.build(verbose=False)
        sp_roots = sp.roots()
        ca_roots = ca.roots()
        assert len(sp_roots) == len(ca_roots), f"Count mismatch: {len(sp_roots)} vs {len(ca_roots)}"
        for r1, r2 in zip(sp_roots, ca_roots):
            assert abs(r1 - r2) < 1e-10, f"Root mismatch: {r1} vs {r2}"

    def test_spline_roots_before_build_raises(self):
        """roots() raises RuntimeError before build()."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        with pytest.raises(RuntimeError):
            sp.roots()


# ======================================================================
# TestMinMaxApprox
# ======================================================================

class TestMinMaxApprox:
    """Tests for ChebyshevApproximation.minimize() and maximize()."""

    def test_minimize_x_squared(self):
        """Min of x^2 on [-1, 1] -> (0, 0)."""
        def f(x, _): return x[0] ** 2
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        cheb.build(verbose=False)
        val, loc = cheb.minimize()
        assert abs(val - 0.0) < 1e-8, f"Min value {val} != 0"
        assert abs(loc - 0.0) < 1e-8, f"Min location {loc} != 0"

    def test_maximize_x_squared(self):
        """Max of x^2 on [-1, 1] -> (1, -1 or 1) (at boundary)."""
        def f(x, _): return x[0] ** 2
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        cheb.build(verbose=False)
        val, loc = cheb.maximize()
        assert abs(val - 1.0) < 1e-8, f"Max value {val} != 1"
        assert abs(abs(loc) - 1.0) < 1e-8, f"Max location {loc} not at +/-1"

    def test_maximize_sin(self):
        """Max of sin(x) on [0, pi] -> (1, pi/2)."""
        def f(x, _): return math.sin(x[0])
        cheb = ChebyshevApproximation(f, 1, [[0, math.pi]], [15])
        cheb.build(verbose=False)
        val, loc = cheb.maximize()
        assert abs(val - 1.0) < 1e-8, f"Max value {val} != 1"
        assert abs(loc - math.pi / 2) < 1e-8, f"Max location {loc} != pi/2"

    def test_minimize_sin_wide(self):
        """Min of sin(x) on [0, 3*pi] -> (-1, 3*pi/2)."""
        def f(x, _): return math.sin(x[0])
        cheb = ChebyshevApproximation(f, 1, [[0, 3 * math.pi]], [25])
        cheb.build(verbose=False)
        val, loc = cheb.minimize()
        assert abs(val - (-1.0)) < 1e-6, f"Min value {val} != -1"
        assert abs(loc - 3 * math.pi / 2) < 1e-6, f"Min location {loc} != 3*pi/2"

    def test_minimize_2d_fixed(self):
        """2D min: f(x,y) = x^2 + y, fix y=0.5, min at x=0, value=0.5."""
        def f(x, _): return x[0] ** 2 + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
        cheb.build(verbose=False)
        val, loc = cheb.minimize(dim=0, fixed={1: 0.5})
        assert abs(val - 0.5) < 1e-8, f"Min value {val} != 0.5"
        assert abs(loc - 0.0) < 1e-8, f"Min location {loc} != 0"

    def test_maximize_2d_fixed(self):
        """2D max: f(x,y) = -x^2 + y, fix y=0.5, max at x=0, value=0.5."""
        def f(x, _): return -x[0] ** 2 + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
        cheb.build(verbose=False)
        val, loc = cheb.maximize(dim=0, fixed={1: 0.5})
        assert abs(val - 0.5) < 1e-8, f"Max value {val} != 0.5"
        assert abs(loc - 0.0) < 1e-8, f"Max location {loc} != 0"

    def test_minimize_constant(self):
        """Constant function: min = max = constant."""
        def f(x, _): return 3.14
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        cheb.build(verbose=False)
        min_val, _ = cheb.minimize()
        max_val, _ = cheb.maximize()
        assert abs(min_val - 3.14) < 1e-8, f"Min {min_val} != 3.14"
        assert abs(max_val - 3.14) < 1e-8, f"Max {max_val} != 3.14"

    def test_minimize_before_build_raises(self):
        """minimize() raises RuntimeError before build()."""
        def f(x, _): return x[0]
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [5])
        with pytest.raises(RuntimeError):
            cheb.minimize()

    def test_maximize_missing_fixed_raises(self):
        """Multi-D maximize without full fixed dict raises ValueError."""
        def f(x, _): return x[0] + x[1]
        cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [5, 5])
        cheb.build(verbose=False)
        with pytest.raises(ValueError):
            cheb.maximize(dim=0)

    def test_minimize_returns_tuple(self):
        """Verify minimize() return type is a tuple of two floats."""
        def f(x, _): return x[0] ** 2
        cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [11])
        cheb.build(verbose=False)
        result = cheb.minimize()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
        assert isinstance(result[0], float), f"Value not float: {type(result[0])}"
        assert isinstance(result[1], float), f"Location not float: {type(result[1])}"


# ======================================================================
# TestMinMaxSpline
# ======================================================================

class TestMinMaxSpline:
    """Tests for ChebyshevSpline.minimize() and maximize()."""

    def test_spline_minimize_abs(self):
        """Min of |x| on [-1, 1] with knot at 0 -> (0, 0)."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        val, loc = sp.minimize()
        assert abs(val) < 1e-8, f"Min value {val} != 0"
        assert abs(loc) < 1e-8, f"Min location {loc} != 0"

    def test_spline_maximize_abs(self):
        """Max of |x| on [-1, 1] -> (1, -1 or 1)."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        val, loc = sp.maximize()
        assert abs(val - 1.0) < 1e-8, f"Max value {val} != 1"
        assert abs(abs(loc) - 1.0) < 1e-8, f"Max location {loc} not at +/-1"

    def test_spline_minimize_multi_piece(self):
        """Global min across pieces: |x| - 0.5 on [-1,1], min at x=0, value=-0.5."""
        def f(x, _): return abs(x[0]) - 0.5
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        val, loc = sp.minimize()
        assert abs(val - (-0.5)) < 1e-8, f"Min value {val} != -0.5"
        assert abs(loc) < 1e-8, f"Min location {loc} != 0"

    def test_spline_maximize_2d(self):
        """2D spline maximize: f(x,y) = -|x| + cos(y) on [-1,1]^2 with knot at x=0.

        Fix y=0, max along x: -|x| + 1, max at x=0, value=1.
        """
        def f(x, _): return -abs(x[0]) + math.cos(x[1])
        sp = ChebyshevSpline(f, 2, [[-1, 1], [-1, 1]], [11, 11], [[0.0], []])
        sp.build(verbose=False)
        val, loc = sp.maximize(dim=0, fixed={1: 0.0})
        assert abs(val - 1.0) < 1e-6, f"Max value {val} != 1.0"
        assert abs(loc) < 1e-6, f"Max location {loc} != 0"

    def test_spline_minimize_no_knots_matches_ca(self):
        """Spline with no knots matches CA.minimize()."""
        def f(x, _): return x[0] ** 2 - 0.5 * x[0]
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[]])
        sp.build(verbose=False)
        ca = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
        ca.build(verbose=False)
        sp_val, sp_loc = sp.minimize()
        ca_val, ca_loc = ca.minimize()
        assert abs(sp_val - ca_val) < 1e-10, f"Value mismatch: {sp_val} vs {ca_val}"
        assert abs(sp_loc - ca_loc) < 1e-10, f"Location mismatch: {sp_loc} vs {ca_loc}"

    def test_spline_minimize_before_build_raises(self):
        """minimize() raises RuntimeError before build()."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        with pytest.raises(RuntimeError):
            sp.minimize()

    def test_spline_maximize_returns_tuple(self):
        """Verify maximize() return type is a tuple of two floats."""
        def f(x, _): return abs(x[0])
        sp = ChebyshevSpline(f, 1, [[-1, 1]], [11], [[0.0]])
        sp.build(verbose=False)
        result = sp.maximize()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
        assert isinstance(result[0], float), f"Value not float: {type(result[0])}"
        assert isinstance(result[1], float), f"Location not float: {type(result[1])}"
