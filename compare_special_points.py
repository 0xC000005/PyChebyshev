"""
Compare PyChebyshev special_points vs MoCaX MocaxSpecialPoints + MocaxNs.

Tests:
1. 1D |x| with kink at 0.0 and per-piece N = 11
2. 1D |x - 0.3| with kink at 0.3 and different per-piece Ns (7, 13)
3. 2D abs(x) + abs(y) with kinks at origin on both dims (4 pieces), N=7
4. 2D single-dim kink: f(x, y) = abs(x) * (1 + y^2), kink on dim 0

For each: PyChebyshev value, MoCaX value, abs difference at several test
points. PASS if max difference < 1e-12 across all tests.

Requires: mocaxpy + libmocaxc.so (skipped cleanly if unavailable)

Usage:
    uv run python compare_special_points.py

NOTE: Local benchmarking only. Not part of the test suite; not in CI.
"""

import math
import os
import sys
import time

import numpy as np

from pychebyshev import ChebyshevApproximation

# ============================================================================
# MoCaX setup (module-level, matching working compare_spline.py pattern)
# ============================================================================

mocax_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocax_lib')
sys.path.insert(0, mocax_lib_dir)

from ctypes import CDLL
import ctypes

_original_cdll_init = CDLL.__init__

def _patched_cdll_init(self, name, mode=ctypes.DEFAULT_MODE, handle=None,
                       use_errno=False, use_last_error=False, winmode=None):
    if isinstance(name, str) and ('libmocaxc.so' in name or name == 'libmocaxc.so'):
        name = os.path.join(mocax_lib_dir, 'libmocaxc.so')
    _original_cdll_init(self, name, mode, handle, use_errno, use_last_error, winmode)

CDLL.__init__ = _patched_cdll_init

HAS_MOCAX = False
try:
    import mocaxpy
    HAS_MOCAX = True
    print(f"MoCaX version: {mocaxpy.get_version_id()}")
except ImportError as e:
    print(f"MoCaX not available: {e}")
    print("  (MoCaX comparison will be skipped, PyChebyshev results shown)\n")


# ============================================================================
# Helpers
# ============================================================================

def _mocax_ns_from_nested(n_nodes):
    """Convert PyChebyshev nested node-count n_nodes into MoCaX nested degree list.

    PyChebyshev uses node count per piece (n). MoCaX MocaxNs takes polynomial
    degree N = n - 1. Any flat entry is converted element-wise; nested entries
    are converted sub-list by sub-list.
    """
    out = []
    for entry in n_nodes:
        if isinstance(entry, (list, tuple)):
            out.append([int(x) - 1 for x in entry])
        else:
            out.append(int(entry) - 1)
    return out


def _build_py_and_mx(f_py, f_mx, num_dim, domain, n_nodes, special_points):
    """Returns (py_obj, mx_obj) both built over the same kinks + Ns."""
    py = ChebyshevApproximation(
        f_py, num_dim, domain,
        n_nodes=n_nodes,
        special_points=special_points,
    )
    py.build(verbose=False)

    if not HAS_MOCAX:
        return py, None

    mx_special = mocaxpy.MocaxSpecialPoints(special_points)
    mx_ns = mocaxpy.MocaxNs(_mocax_ns_from_nested(n_nodes))
    mx_domain = mocaxpy.MocaxDomain([list(d) for d in domain])
    # Mocax ctor signature:
    #   Mocax(function, num_dimensions, domain, error_threshold, n,
    #         special_points=None, max_derivative_order=0)
    mx = mocaxpy.Mocax(
        f_mx, num_dim, mx_domain, None, mx_ns,
        special_points=mx_special, max_derivative_order=0,
    )
    return py, mx


def _diff_1d(py, mx, xs, dim_count=1):
    diffs = []
    print(f"    {'x':>8s} {'PyChebyshev':>16s} {'MoCaX':>16s} {'|diff|':>12s}")
    for x in xs:
        pt = [float(x)]
        py_v = py.eval(pt, [0] * dim_count)
        mx_v = mx.eval(pt)
        d = abs(py_v - mx_v)
        diffs.append(d)
        print(f"    {x:>8.4f} {py_v:>16.12f} {mx_v:>16.12f} {d:>12.2e}")
    return max(diffs) if diffs else float('nan')


def _diff_2d(py, mx, points):
    diffs = []
    print(f"    {'(x,y)':>16s} {'PyChebyshev':>16s} {'MoCaX':>16s} {'|diff|':>12s}")
    for pt in points:
        py_v = py.eval(list(pt), [0, 0])
        mx_v = mx.eval(list(pt))
        d = abs(py_v - mx_v)
        diffs.append(d)
        pt_str = f"({pt[0]:.2f},{pt[1]:.2f})"
        print(f"    {pt_str:>16s} {py_v:>16.12f} {mx_v:>16.12f} {d:>12.2e}")
    return max(diffs)


# ============================================================================
# Test functions
# ============================================================================

def _abs1d(x, _):
    return abs(x[0])


def compare_abs_1d():
    print("\n[1] 1D |x| kink at 0, n_nodes=[[11, 11]]")
    py, mx = _build_py_and_mx(
        _abs1d,
        _abs1d,
        1, [[-1, 1]],
        n_nodes=[[11, 11]],
        special_points=[[0.0]],
    )
    if mx is None:
        return None
    xs = [float(v) for v in np.linspace(-0.95, 0.95, 11) if abs(v) > 1e-8]
    return _diff_1d(py, mx, xs)


def compare_abs_off_origin_different_ns():
    print("\n[2] 1D |x - 0.3| kink at 0.3, n_nodes=[[7, 13]]")
    def f_py(x, _): return abs(x[0] - 0.3)
    def f_mx(x, _): return abs(x[0] - 0.3)
    py, mx = _build_py_and_mx(
        f_py, f_mx,
        1, [[-1, 1]],
        n_nodes=[[7, 13]],
        special_points=[[0.3]],
    )
    if mx is None:
        return None
    xs = [-0.7, -0.2, 0.1, 0.5, 0.8]
    return _diff_1d(py, mx, xs)


def compare_2d_kink_one_dim():
    print("\n[3] 2D |x| * (1 + y^2), 1-dim kink on dim 0, n_nodes=[[9, 9], [11]]")
    def f_py(x, _): return abs(x[0]) * (1.0 + x[1] ** 2)
    def f_mx(x, _): return abs(x[0]) * (1.0 + x[1] ** 2)
    py, mx = _build_py_and_mx(
        f_py, f_mx,
        2, [[-1, 1], [-1, 1]],
        n_nodes=[[9, 9], [11]],
        special_points=[[0.0], []],
    )
    if mx is None:
        return None
    points = [(x, y) for x in [-0.7, 0.3] for y in [-0.8, 0.5]]
    return _diff_2d(py, mx, points)


def compare_2d_both_dim_kinks():
    print("\n[4] 2D |x| + |y|, kinks on both dims (4 pieces), n_nodes=[[7, 7], [7, 7]]")
    def f_py(x, _): return abs(x[0]) + abs(x[1])
    def f_mx(x, _): return abs(x[0]) + abs(x[1])
    py, mx = _build_py_and_mx(
        f_py, f_mx,
        2, [[-1, 1], [-1, 1]],
        n_nodes=[[7, 7], [7, 7]],
        special_points=[[0.0], [0.0]],
    )
    if mx is None:
        return None
    points = [(x, y) for x in [-0.6, 0.4] for y in [-0.8, 0.2]]
    return _diff_2d(py, mx, points)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 66)
    print("  PyChebyshev vs MoCaX — Special Points value-match benchmark")
    print("=" * 66)
    if not HAS_MOCAX:
        print("MoCaX unavailable; only PyChebyshev was constructed.")
        sys.exit(0)

    results = [
        ("1D |x| kink @ 0, N=[[11,11]]",      compare_abs_1d()),
        ("1D |x-0.3| kink @ 0.3, N=[[7,13]]", compare_abs_off_origin_different_ns()),
        ("2D |x|*(1+y^2), 1-dim kink",        compare_2d_kink_one_dim()),
        ("2D |x|+|y|, both-dim kinks (4 pc)", compare_2d_both_dim_kinks()),
    ]

    print("\n" + "=" * 66)
    print("  Summary")
    print("=" * 66)
    max_diff = 0.0
    for label, d in results:
        print(f"  {label:<42s}  max|py - mx| = {d:.3e}")
        max_diff = max(max_diff, d)

    print("=" * 66)
    tol = 1e-12
    if max_diff < tol:
        print(f"PASS — max diff {max_diff:.3e} < {tol:.0e}")
        sys.exit(0)
    else:
        print(f"DIVERGENCE — max diff {max_diff:.3e} >= {tol:.0e}")
        sys.exit(1)
