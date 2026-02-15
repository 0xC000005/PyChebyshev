"""
compare_algebra.py -- PyChebyshev vs MoCaX algebra comparison (local only).

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
from pychebyshev import ChebyshevApproximation, ChebyshevSlider

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

def func_a_2d(x, _):
    """sin(x) + sin(y)"""
    return math.sin(x[0]) + math.sin(x[1])

def func_b_2d(x, _):
    """cos(x) * cos(y)"""
    return math.cos(x[0]) * math.cos(x[1])

def func_a_3d(x, _):
    """sin(x) + sin(y) + sin(z)"""
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

def func_b_3d(x, _):
    """cos(x) + cos(y) + cos(z)"""
    return math.cos(x[0]) + math.cos(x[1]) + math.cos(x[2])


DOMAIN_2D = [[-1, 1], [-1, 1]]
NS_2D = [11, 11]
DOMAIN_3D = [[-1, 1], [-1, 1], [-1, 1]]
NS_3D = [11, 11, 11]

TEST_POINTS_2D = [
    [0.5, 0.3], [-0.7, 0.8], [0.0, 0.0], [0.9, -0.9], [-0.2, 0.6],
    [0.1, -0.4], [-0.9, 0.1], [0.3, 0.7], [-0.5, -0.5], [0.8, 0.2],
]
TEST_POINTS_3D = [
    [0.5, 0.3, 0.7], [-0.5, 0.8, -0.2], [0.1, -0.3, 0.9],
    [0.0, 0.0, 0.0], [-0.7, 0.4, -0.6], [0.3, -0.8, 0.1],
]


def _build_mocax(func, ndim, domain, ns):
    """Build a MoCaX Mocax object."""
    mx = mocaxpy.Mocax(func, ndim, domain, None, ns, max_derivative_order=2)
    return mx


def _build_mocax_sliding(func, ndim, domain, ns, partition, pivot):
    """Build a MoCaX MocaxSliding object."""
    mx = mocaxpy.MocaxSliding(func, partition, domain, ns, pivot,
                              max_derivative_order=2)
    return mx


def _eval_mocax(mx, point, deriv):
    """Evaluate MoCaX object."""
    return mx.eval(point, deriv)


def _report(test_name, py_vals, mx_vals, test_points):
    """Print comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {test_name}")
    print(f"{'=' * 70}")
    ref_label = "MoCaX" if HAS_MOCAX else "Exact"
    print(f"  {'Point':<30} {'PyChebyshev':>15} {ref_label:>15} {'Diff':>12}")
    print(f"  {'-' * 72}")
    max_diff = 0.0
    for i, p in enumerate(test_points):
        diff = abs(py_vals[i] - mx_vals[i])
        max_diff = max(max_diff, diff)
        p_str = str([round(x, 2) for x in p])
        print(f"  {p_str:<30} {py_vals[i]:>15.10f} {mx_vals[i]:>15.10f} {diff:>12.2e}")
    print(f"  {'-' * 72}")
    # When comparing against MoCaX, expect machine precision (~1e-14).
    # When comparing against exact values, allow for interpolation error (~1e-9).
    tol = 1e-10 if HAS_MOCAX else 1e-8
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"  Max diff: {max_diff:.2e}  [{status}]")
    return max_diff


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_addition():
    """Test 1: 2D addition -- (a + b) accuracy."""
    print("\n" + "=" * 70)
    print("  TEST 1: 2D Addition  f + g")
    print("=" * 70)

    # PyChebyshev
    py_a = ChebyshevApproximation(func_a_2d, 2, DOMAIN_2D, NS_2D)
    py_b = ChebyshevApproximation(func_b_2d, 2, DOMAIN_2D, NS_2D)
    py_a.build(verbose=False); py_b.build(verbose=False)
    py_sum = py_a + py_b
    py_vals = [py_sum.vectorized_eval(p, [0, 0]) for p in TEST_POINTS_2D]

    if HAS_MOCAX:
        mx_a = _build_mocax(func_a_2d, 2, DOMAIN_2D, NS_2D)
        mx_b = _build_mocax(func_b_2d, 2, DOMAIN_2D, NS_2D)
        mx_sum = mx_a + mx_b
        mx_vals = [_eval_mocax(mx_sum, p, [0, 0]) for p in TEST_POINTS_2D]
        _report("Addition: f + g", py_vals, mx_vals, TEST_POINTS_2D)
    else:
        # Compare against exact
        exact_vals = [func_a_2d(p, None) + func_b_2d(p, None) for p in TEST_POINTS_2D]
        _report("Addition: f + g (vs exact)", py_vals, exact_vals, TEST_POINTS_2D)


def test_2_scalar_multiply():
    """Test 2: Scalar multiplication -- 3.14159 * f."""
    print("\n" + "=" * 70)
    print("  TEST 2: Scalar Multiplication  pi * f")
    print("=" * 70)

    scale = 3.14159
    py_a = ChebyshevApproximation(func_a_2d, 2, DOMAIN_2D, NS_2D)
    py_a.build(verbose=False)
    py_scaled = scale * py_a
    py_vals = [py_scaled.vectorized_eval(p, [0, 0]) for p in TEST_POINTS_2D]

    if HAS_MOCAX:
        mx_a = _build_mocax(func_a_2d, 2, DOMAIN_2D, NS_2D)
        mx_scaled = mx_a * scale
        mx_vals = [_eval_mocax(mx_scaled, p, [0, 0]) for p in TEST_POINTS_2D]
        _report("Scalar multiply: pi * f", py_vals, mx_vals, TEST_POINTS_2D)
    else:
        exact_vals = [scale * func_a_2d(p, None) for p in TEST_POINTS_2D]
        _report("Scalar multiply: pi * f (vs exact)", py_vals, exact_vals, TEST_POINTS_2D)


def test_3_portfolio():
    """Test 3: Portfolio -- 0.6*a + 0.4*b."""
    print("\n" + "=" * 70)
    print("  TEST 3: Portfolio  0.6*f + 0.4*g")
    print("=" * 70)

    py_a = ChebyshevApproximation(func_a_2d, 2, DOMAIN_2D, NS_2D)
    py_b = ChebyshevApproximation(func_b_2d, 2, DOMAIN_2D, NS_2D)
    py_a.build(verbose=False); py_b.build(verbose=False)
    py_port = 0.6 * py_a + 0.4 * py_b
    py_vals = [py_port.vectorized_eval(p, [0, 0]) for p in TEST_POINTS_2D]
    py_derivs = [py_port.vectorized_eval(p, [1, 0]) for p in TEST_POINTS_2D]

    if HAS_MOCAX:
        mx_a = _build_mocax(func_a_2d, 2, DOMAIN_2D, NS_2D)
        mx_b = _build_mocax(func_b_2d, 2, DOMAIN_2D, NS_2D)
        mx_port = mx_a * 0.6 + mx_b * 0.4
        mx_vals = [_eval_mocax(mx_port, p, [0, 0]) for p in TEST_POINTS_2D]
        mx_derivs = [_eval_mocax(mx_port, p, [1, 0]) for p in TEST_POINTS_2D]
        _report("Portfolio price: 0.6*f + 0.4*g", py_vals, mx_vals, TEST_POINTS_2D)
        _report("Portfolio d/dx0: 0.6*f + 0.4*g", py_derivs, mx_derivs, TEST_POINTS_2D)
    else:
        exact_vals = [0.6 * func_a_2d(p, None) + 0.4 * func_b_2d(p, None) for p in TEST_POINTS_2D]
        _report("Portfolio: 0.6*f + 0.4*g (vs exact)", py_vals, exact_vals, TEST_POINTS_2D)


def test_4_self_subtraction():
    """Test 4: Self-cancellation -- f - f = 0."""
    print("\n" + "=" * 70)
    print("  TEST 4: Self-Subtraction  f - f = 0")
    print("=" * 70)

    py_a = ChebyshevApproximation(func_a_2d, 2, DOMAIN_2D, NS_2D)
    py_a.build(verbose=False)
    py_zero = py_a - py_a
    py_vals = [py_zero.vectorized_eval(p, [0, 0]) for p in TEST_POINTS_2D]

    if HAS_MOCAX:
        mx_a = _build_mocax(func_a_2d, 2, DOMAIN_2D, NS_2D)
        mx_zero = mx_a - mx_a
        mx_vals = [_eval_mocax(mx_zero, p, [0, 0]) for p in TEST_POINTS_2D]
        _report("Self-subtraction: f - f", py_vals, mx_vals, TEST_POINTS_2D)
    else:
        zeros = [0.0] * len(TEST_POINTS_2D)
        _report("Self-subtraction: f - f (vs 0)", py_vals, zeros, TEST_POINTS_2D)

    max_abs = max(abs(v) for v in py_vals)
    status = "PASS" if max_abs < 1e-14 else "FAIL"
    print(f"  PyChebyshev max |f-f|: {max_abs:.2e}  [{status}]")


def test_5_slider_algebra():
    """Test 5: Slider addition."""
    print("\n" + "=" * 70)
    print("  TEST 5: Slider Algebra  slider_a + slider_b")
    print("=" * 70)

    partition = [[0], [1], [2]]
    pivot = [0.0, 0.0, 0.0]

    py_a = ChebyshevSlider(func_a_3d, 3, DOMAIN_3D, NS_3D,
                           partition=partition, pivot_point=pivot)
    py_b = ChebyshevSlider(func_b_3d, 3, DOMAIN_3D, NS_3D,
                           partition=partition, pivot_point=pivot)
    py_a.build(verbose=False); py_b.build(verbose=False)
    py_sum = py_a + py_b
    py_vals = [py_sum.eval(p, [0, 0, 0]) for p in TEST_POINTS_3D]

    if HAS_MOCAX:
        mx_a = _build_mocax_sliding(func_a_3d, 3, DOMAIN_3D, NS_3D, partition, pivot)
        mx_b = _build_mocax_sliding(func_b_3d, 3, DOMAIN_3D, NS_3D, partition, pivot)
        mx_sum = mx_a + mx_b
        mx_vals = [mx_sum.eval(p, [0, 0, 0]) for p in TEST_POINTS_3D]
        _report("Slider addition: slider_a + slider_b", py_vals, mx_vals, TEST_POINTS_3D)
    else:
        exact_vals = [func_a_3d(p, None) + func_b_3d(p, None) for p in TEST_POINTS_3D]
        _report("Slider addition (vs exact)", py_vals, exact_vals, TEST_POINTS_3D)


def test_6_chained_ops():
    """Test 6: Chained operations -- 0.5*a + 0.3*b - 0.2*a = 0.3*a + 0.3*b."""
    print("\n" + "=" * 70)
    print("  TEST 6: Chained  0.5*f + 0.3*g - 0.2*f = 0.3*f + 0.3*g")
    print("=" * 70)

    py_a = ChebyshevApproximation(func_a_2d, 2, DOMAIN_2D, NS_2D)
    py_b = ChebyshevApproximation(func_b_2d, 2, DOMAIN_2D, NS_2D)
    py_a.build(verbose=False); py_b.build(verbose=False)
    py_chained = 0.5 * py_a + 0.3 * py_b - 0.2 * py_a
    py_vals = [py_chained.vectorized_eval(p, [0, 0]) for p in TEST_POINTS_2D]

    if HAS_MOCAX:
        mx_a = _build_mocax(func_a_2d, 2, DOMAIN_2D, NS_2D)
        mx_b = _build_mocax(func_b_2d, 2, DOMAIN_2D, NS_2D)
        mx_chained = mx_a * 0.5 + mx_b * 0.3 - mx_a * 0.2
        mx_vals = [_eval_mocax(mx_chained, p, [0, 0]) for p in TEST_POINTS_2D]
        _report("Chained: 0.5*f + 0.3*g - 0.2*f", py_vals, mx_vals, TEST_POINTS_2D)
    else:
        exact_vals = [0.3 * func_a_2d(p, None) + 0.3 * func_b_2d(p, None) for p in TEST_POINTS_2D]
        _report("Chained (vs exact)", py_vals, exact_vals, TEST_POINTS_2D)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  PyChebyshev vs MoCaX -- Chebyshev Algebra Comparison")
    print(f"  MoCaX available: {HAS_MOCAX}")
    print("=" * 70)

    test_1_addition()
    test_2_scalar_multiply()
    test_3_portfolio()
    test_4_self_subtraction()
    test_5_slider_algebra()
    test_6_chained_ops()

    print("\n" + "=" * 70)
    print("  All tests complete.")
    print("=" * 70)
