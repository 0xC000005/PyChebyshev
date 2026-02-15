"""
compare_extrude_slice.py -- PyChebyshev vs MoCaX extrusion/slicing comparison (local only).

Requires the MoCaX library (not published; local-only):
    mocax_lib/mocaxpy/Mocax.py
    mocaxextend_lib/shared_libs/libtensorvals.so
    mocaxextend_lib/shared_libs/libhommat.so
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PyChebyshev imports
# ---------------------------------------------------------------------------
from pychebyshev import ChebyshevApproximation

# ---------------------------------------------------------------------------
# MoCaX imports (optional -- skip gracefully if unavailable)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
HAS_MOCAX = False
try:
    # Patch CDLL so the MoCaX shared libs resolve
    _orig_cdll_init = ctypes.CDLL.__init__

    def _patched_cdll_init(self, name, *args, **kwargs):
        if name and ("libtensorvals" in str(name) or "libhommat" in str(name)):
            name = str(ROOT / "mocaxextend_lib" / "shared_libs" / Path(name).name)
        _orig_cdll_init(self, name, *args, **kwargs)

    ctypes.CDLL.__init__ = _patched_cdll_init

    sys.path.insert(0, str(ROOT / "mocax_lib"))
    import mocaxpy  # noqa: E402

    HAS_MOCAX = True
except Exception as exc:
    print(f"MoCaX not available: {exc}")

# ---------------------------------------------------------------------------
# Shared test functions
# ---------------------------------------------------------------------------

def func_sinsin_2d(x, _):
    """sin(x0) + sin(x1)"""
    return math.sin(x[0]) + math.sin(x[1])

def func_3d(x, _):
    """sin(x0) * cos(x1) + x2"""
    return math.sin(x[0]) * math.cos(x[1]) + x[2]

def func_sin_1d(x, _):
    """sin(x0)"""
    return math.sin(x[0])

def func_trade_a(x, _):
    """Trade A: sin(spot/20) + rate  (smooth over [80,120])"""
    return math.sin(x[0] / 20.0) + x[1]

def func_trade_b(x, _):
    """Trade B: cos(spot/20) * vol  (smooth over [80,120])"""
    return math.cos(x[0] / 20.0) * x[1]


DOMAIN_1D = [[-1, 1]]
NS_1D = [15]
DOMAIN_2D = [[-1, 1], [-1, 1]]
NS_2D = [11, 11]
DOMAIN_3D = [[-1, 1], [-1, 1], [-1, 1]]
NS_3D = [11, 11, 11]

SPOT_DOMAIN = [80, 120]
RATE_DOMAIN = [0.01, 0.08]
VOL_DOMAIN = [0.15, 0.35]

np.random.seed(42)
RAND_POINTS_3D = np.random.uniform(-1, 1, size=(100, 3)).tolist()
RAND_POINTS_2D = np.random.uniform(-1, 1, size=(100, 2)).tolist()
RAND_POINTS_1D = np.random.uniform(-1, 1, size=(100, 1)).tolist()


def _build_mocax(func, ndim, domain, ns):
    """Build a MoCaX Mocax object."""
    mx = mocaxpy.Mocax(func, ndim, domain, None, ns, max_derivative_order=2)
    return mx


def _eval_mocax(mx, point, deriv):
    """Evaluate MoCaX object."""
    return mx.eval(point, deriv)


def _report(test_name, py_vals, ref_vals, max_n=10):
    """Print comparison table and return max diff."""
    print(f"\n{'=' * 70}")
    print(f"  {test_name}")
    print(f"{'=' * 70}")
    ref_label = "MoCaX" if HAS_MOCAX else "Exact"
    n_show = min(max_n, len(py_vals))
    print(f"  Showing first {n_show} of {len(py_vals)} points")
    print(f"  {'#':<5} {'PyChebyshev':>15} {ref_label:>15} {'Diff':>12}")
    print(f"  {'-' * 50}")
    max_diff = 0.0
    for i in range(len(py_vals)):
        diff = abs(py_vals[i] - ref_vals[i])
        max_diff = max(max_diff, diff)
        if i < n_show:
            print(f"  {i:<5} {py_vals[i]:>15.10f} {ref_vals[i]:>15.10f} {diff:>12.2e}")
    print(f"  {'-' * 50}")
    tol = 1e-10 if HAS_MOCAX else 1e-8
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"  Max diff: {max_diff:.2e}  [{status}]")
    return max_diff


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_extrude_2d_to_3d():
    """Test 1: Extrude 2D -> 3D.  sin(x0)+sin(x1) extruded along dim 2."""
    print("\n" + "=" * 70)
    print("  TEST 1: Extrude 2D -> 3D")
    print("=" * 70)

    # PyChebyshev
    t0 = time.perf_counter()
    py_2d = ChebyshevApproximation(func_sinsin_2d, 2, DOMAIN_2D, NS_2D)
    py_2d.build(verbose=False)
    py_3d = py_2d.extrude((2, (-1, 1), 11))
    py_time = time.perf_counter() - t0
    py_vals = [py_3d.vectorized_eval(p, [0, 0, 0]) for p in RAND_POINTS_3D]

    if HAS_MOCAX:
        t0 = time.perf_counter()
        mx_2d = _build_mocax(func_sinsin_2d, 2, DOMAIN_2D, NS_2D)
        mx_3d = mx_2d.extrude((2, (-1, 1)))
        mx_time = time.perf_counter() - t0
        mx_vals = [_eval_mocax(mx_3d, p, [0, 0, 0]) for p in RAND_POINTS_3D]
        _report("Extrude 2D->3D", py_vals, mx_vals)
        print(f"  PyChebyshev: {py_time:.4f}s, MoCaX: {mx_time:.4f}s")
    else:
        exact_vals = [func_sinsin_2d(p[:2], None) for p in RAND_POINTS_3D]
        _report("Extrude 2D->3D (vs exact)", py_vals, exact_vals)
        print(f"  PyChebyshev: {py_time:.4f}s")


def test_2_slice_3d_to_2d():
    """Test 2: Slice 3D -> 2D.  Build 3D, slice dim 1 at midpoint."""
    print("\n" + "=" * 70)
    print("  TEST 2: Slice 3D -> 2D")
    print("=" * 70)

    slice_val = 0.0

    # PyChebyshev
    t0 = time.perf_counter()
    py_3d = ChebyshevApproximation(func_3d, 3, DOMAIN_3D, NS_3D)
    py_3d.build(verbose=False)
    py_2d = py_3d.slice((1, slice_val))
    py_time = time.perf_counter() - t0
    py_vals = [py_2d.vectorized_eval(p, [0, 0]) for p in RAND_POINTS_2D]

    if HAS_MOCAX:
        t0 = time.perf_counter()
        mx_3d = _build_mocax(func_3d, 3, DOMAIN_3D, NS_3D)
        mx_2d = mx_3d.slice((1, slice_val))
        mx_time = time.perf_counter() - t0
        mx_vals = [_eval_mocax(mx_2d, p, [0, 0]) for p in RAND_POINTS_2D]
        _report("Slice 3D->2D", py_vals, mx_vals)
        print(f"  PyChebyshev: {py_time:.4f}s, MoCaX: {mx_time:.4f}s")
    else:
        exact_vals = [func_3d([p[0], slice_val, p[1]], None) for p in RAND_POINTS_2D]
        _report("Slice 3D->2D (vs exact)", py_vals, exact_vals)
        print(f"  PyChebyshev: {py_time:.4f}s")


def test_3_multi_extrude():
    """Test 3: Multi-extrude 1D -> 3D (add 2 dims)."""
    print("\n" + "=" * 70)
    print("  TEST 3: Multi-Extrude 1D -> 3D")
    print("=" * 70)

    # PyChebyshev
    t0 = time.perf_counter()
    py_1d = ChebyshevApproximation(func_sin_1d, 1, DOMAIN_1D, NS_1D)
    py_1d.build(verbose=False)
    py_3d = py_1d.extrude([(1, (-1, 1), 11), (2, (-1, 1), 11)])
    py_time = time.perf_counter() - t0
    py_vals = [py_3d.vectorized_eval(p, [0, 0, 0]) for p in RAND_POINTS_3D]

    if HAS_MOCAX:
        t0 = time.perf_counter()
        mx_1d = _build_mocax(func_sin_1d, 1, DOMAIN_1D, NS_1D)
        mx_2d = mx_1d.extrude((1, (-1, 1)))
        mx_3d = mx_2d.extrude((2, (-1, 1)))
        mx_time = time.perf_counter() - t0
        mx_vals = [_eval_mocax(mx_3d, p, [0, 0, 0]) for p in RAND_POINTS_3D]
        _report("Multi-extrude 1D->3D", py_vals, mx_vals)
        print(f"  PyChebyshev: {py_time:.4f}s, MoCaX: {mx_time:.4f}s")
    else:
        exact_vals = [func_sin_1d([p[0]], None) for p in RAND_POINTS_3D]
        _report("Multi-extrude 1D->3D (vs exact)", py_vals, exact_vals)
        print(f"  PyChebyshev: {py_time:.4f}s")


def test_4_multi_slice():
    """Test 4: Multi-slice 3D -> 1D (remove 2 dims)."""
    print("\n" + "=" * 70)
    print("  TEST 4: Multi-Slice 3D -> 1D")
    print("=" * 70)

    slice_vals = [(1, 0.3), (2, -0.5)]

    # PyChebyshev
    t0 = time.perf_counter()
    py_3d = ChebyshevApproximation(func_3d, 3, DOMAIN_3D, NS_3D)
    py_3d.build(verbose=False)
    py_1d = py_3d.slice(slice_vals)
    py_time = time.perf_counter() - t0
    py_vals = [py_1d.vectorized_eval(p, [0]) for p in RAND_POINTS_1D]

    if HAS_MOCAX:
        t0 = time.perf_counter()
        mx_3d = _build_mocax(func_3d, 3, DOMAIN_3D, NS_3D)
        mx_2d = mx_3d.slice((2, -0.5))
        mx_1d = mx_2d.slice((1, 0.3))
        mx_time = time.perf_counter() - t0
        mx_vals = [_eval_mocax(mx_1d, p, [0]) for p in RAND_POINTS_1D]
        _report("Multi-slice 3D->1D", py_vals, mx_vals)
        print(f"  PyChebyshev: {py_time:.4f}s, MoCaX: {mx_time:.4f}s")
    else:
        exact_vals = [func_3d([p[0], 0.3, -0.5], None) for p in RAND_POINTS_1D]
        _report("Multi-slice 3D->1D (vs exact)", py_vals, exact_vals)
        print(f"  PyChebyshev: {py_time:.4f}s")


def test_5_extrude_then_slice():
    """Test 5: Extrude then slice = identity."""
    print("\n" + "=" * 70)
    print("  TEST 5: Extrude-then-Slice Round-Trip")
    print("=" * 70)

    # PyChebyshev
    py_2d = ChebyshevApproximation(func_sinsin_2d, 2, DOMAIN_2D, NS_2D)
    py_2d.build(verbose=False)
    py_3d = py_2d.extrude((2, (-1, 1), 11))
    py_roundtrip = py_3d.slice((2, 0.7))

    py_orig_vals = [py_2d.vectorized_eval(p, [0, 0]) for p in RAND_POINTS_2D[:20]]
    py_rt_vals = [py_roundtrip.vectorized_eval(p, [0, 0]) for p in RAND_POINTS_2D[:20]]

    if HAS_MOCAX:
        mx_2d = _build_mocax(func_sinsin_2d, 2, DOMAIN_2D, NS_2D)
        mx_3d = mx_2d.extrude((2, (-1, 1)))
        mx_roundtrip = mx_3d.slice((2, 0.7))
        mx_rt_vals = [_eval_mocax(mx_roundtrip, p, [0, 0]) for p in RAND_POINTS_2D[:20]]
        _report("Extrude-then-slice (PyChebyshev orig vs round-trip)", py_orig_vals, py_rt_vals, max_n=10)
        _report("Extrude-then-slice (PyChebyshev vs MoCaX round-trip)", py_rt_vals, mx_rt_vals, max_n=10)
    else:
        _report("Extrude-then-slice (orig vs round-trip)", py_orig_vals, py_rt_vals, max_n=10)


def test_6_portfolio_via_extrude():
    """Test 6: Portfolio via extrude + algebra.
    Trade A(spot, rate) + Trade B(spot, vol) -> portfolio(spot, rate, vol).
    """
    print("\n" + "=" * 70)
    print("  TEST 6: Portfolio via Extrude + Algebra")
    print("=" * 70)

    spot_ns = 11
    rate_ns = 11
    vol_ns = 11

    # PyChebyshev
    t0 = time.perf_counter()
    py_a = ChebyshevApproximation(func_trade_a, 2, [SPOT_DOMAIN, RATE_DOMAIN], [spot_ns, rate_ns])
    py_b = ChebyshevApproximation(func_trade_b, 2, [SPOT_DOMAIN, VOL_DOMAIN], [spot_ns, vol_ns])
    py_a.build(verbose=False); py_b.build(verbose=False)
    py_a_3d = py_a.extrude((2, tuple(VOL_DOMAIN), vol_ns))
    py_b_3d = py_b.extrude((1, tuple(RATE_DOMAIN), rate_ns))
    py_portfolio = py_a_3d + py_b_3d
    py_time = time.perf_counter() - t0

    # Random test points in (spot, rate, vol)
    np.random.seed(123)
    test_pts = []
    for _ in range(50):
        s = np.random.uniform(SPOT_DOMAIN[0], SPOT_DOMAIN[1])
        r = np.random.uniform(RATE_DOMAIN[0], RATE_DOMAIN[1])
        v = np.random.uniform(VOL_DOMAIN[0], VOL_DOMAIN[1])
        test_pts.append([s, r, v])

    py_vals = [py_portfolio.vectorized_eval(p, [0, 0, 0]) for p in test_pts]

    if HAS_MOCAX:
        t0 = time.perf_counter()
        mx_a = _build_mocax(func_trade_a, 2, [SPOT_DOMAIN, RATE_DOMAIN], [spot_ns, rate_ns])
        mx_b = _build_mocax(func_trade_b, 2, [SPOT_DOMAIN, VOL_DOMAIN], [spot_ns, vol_ns])
        mx_a_3d = mx_a.extrude((2, tuple(VOL_DOMAIN)))
        mx_b_3d = mx_b.extrude((1, tuple(RATE_DOMAIN)))
        mx_portfolio = mx_a_3d + mx_b_3d
        mx_time = time.perf_counter() - t0
        mx_vals = [_eval_mocax(mx_portfolio, p, [0, 0, 0]) for p in test_pts]
        _report("Portfolio price", py_vals, mx_vals)
        print(f"  PyChebyshev: {py_time:.4f}s, MoCaX: {mx_time:.4f}s")
    else:
        exact_vals = [func_trade_a([p[0], p[1]], None) + func_trade_b([p[0], p[2]], None)
                      for p in test_pts]
        _report("Portfolio price (vs exact)", py_vals, exact_vals)
        print(f"  PyChebyshev: {py_time:.4f}s")

    # Also check derivatives
    py_d_spot = [py_portfolio.vectorized_eval(p, [1, 0, 0]) for p in test_pts[:10]]
    exact_d_spot = [math.cos(p[0] / 20.0) / 20.0 + (-math.sin(p[0] / 20.0) / 20.0) * p[2]
                    for p in test_pts[:10]]
    _report("Portfolio d/d(spot) (vs exact)", py_d_spot, exact_d_spot)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  PyChebyshev vs MoCaX -- Extrusion & Slicing Comparison")
    print(f"  MoCaX available: {HAS_MOCAX}")
    print("=" * 70)

    test_1_extrude_2d_to_3d()
    test_2_slice_3d_to_2d()
    test_3_multi_extrude()
    test_4_multi_slice()
    test_5_extrude_then_slice()
    test_6_portfolio_via_extrude()

    print("\n" + "=" * 70)
    print("  All tests complete.")
    print("=" * 70)
