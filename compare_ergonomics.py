"""
compare_ergonomics.py -- v0.15 ergonomics features vs MoCaX (local-only).

Tests v0.15 ergonomics surface: additional_data, descriptor get/set,
derivative_id registry, getConstructorType, getUsedNs, isConstructionFinished.

Requires: mocax_lib/ with libmocaxc.so + mocaxpy (gitignored; MoCaX portion is
skipped cleanly if unavailable).

Usage:
    uv run python compare_ergonomics.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import time

import numpy as np

from pychebyshev import ChebyshevApproximation

# ============================================================================
# MoCaX setup (matching compare_special_points.py pattern)
# ============================================================================

mocax_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mocax_lib")
sys.path.insert(0, mocax_lib_dir)

from ctypes import CDLL

_original_cdll_init = CDLL.__init__


def _patched_cdll_init(
    self, name, mode=ctypes.DEFAULT_MODE, handle=None,
    use_errno=False, use_last_error=False, winmode=None
):
    if isinstance(name, str) and "libmocaxc.so" in name:
        name = os.path.join(mocax_lib_dir, "libmocaxc.so")
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
# Shared test function
# ============================================================================

def f(point, data):
    """f(x, y) = sin(x) * exp(strike / 100) — uses additional_data."""
    x, y = point
    strike = data["strike"]
    return math.sin(x) * math.exp(strike / 100.0)


# ============================================================================
# Build helpers
# ============================================================================

def _build_pychebyshev(additional_data):
    """Build ChebyshevApproximation with additional_data."""
    cheb = ChebyshevApproximation(
        f, 2,
        [(-1.0, 1.0), (-1.0, 1.0)],
        [11, 11],
        max_derivative_order=2,
        additional_data=additional_data,
    )
    cheb.build(verbose=False)
    return cheb


def _build_mocax(additional_data):
    """Build MoCaX Mocax object with equivalent parameters.

    MoCaX Python binding (mocaxpy) exposes:
      Mocax(function, num_dimensions, domain, error_threshold, n,
            max_derivative_order=2, additional_data=None)

    Ergonomics methods:
      mx.setDescriptor(s)           mx.getDescriptor()
      mx.getDerivativeId(orders)    (returns stable int)
      mx.eval(point, derivativeId)  (derivativeId from getDerivativeId)
      mx.getConstructorType()
      mx.getUsedNs()
      mx.isConstructionFinished()
    """
    if not HAS_MOCAX:
        return None

    # MoCaX uses degree N = n_nodes - 1
    ns = [10, 10]
    domain = [[-1.0, 1.0], [-1.0, 1.0]]

    # Wrap the function so it accepts (point, data) like PyChebyshev.
    # MoCaX may call the function with only the point, so close over data.
    _data = additional_data

    def f_mx(point, _=None):
        return f(point, _data)

    mx = mocaxpy.Mocax(
        f_mx, 2, domain, None, ns,
        max_derivative_order=2,
        additional_data=additional_data,
    )
    return mx


# ============================================================================
# Comparison helpers
# ============================================================================

def _compare_introspection(cheb, mx):
    """Print side-by-side introspection values."""
    py_descriptor = cheb.get_descriptor()
    py_ctor_type  = cheb.get_constructor_type()
    py_used_ns    = cheb.get_used_ns()
    py_finished   = cheb.is_construction_finished()
    py_deriv_id   = cheb.get_derivative_id([1, 0])

    print("  PyChebyshev:")
    print(f"    get_descriptor()         = {py_descriptor!r}")
    print(f"    get_constructor_type()   = {py_ctor_type!r}")
    print(f"    get_used_ns()            = {py_used_ns}")
    print(f"    is_construction_finished()= {py_finished}")
    print(f"    get_derivative_id([1,0]) = {py_deriv_id}")

    if mx is None:
        print("  MoCaX: (unavailable)")
        return

    mx_descriptor = mx.getDescriptor()
    mx_ctor_type  = mx.getConstructorType()
    mx_used_ns    = mx.getUsedNs()
    mx_finished   = mx.isConstructionFinished()
    mx_deriv_id   = mx.getDerivativeId([1, 0])

    print("  MoCaX:")
    print(f"    getDescriptor()          = {mx_descriptor!r}")
    print(f"    getConstructorType()     = {mx_ctor_type!r}")
    print(f"    getUsedNs()              = {mx_used_ns}")
    print(f"    isConstructionFinished() = {mx_finished}")
    print(f"    getDerivativeId([1,0])   = {mx_deriv_id}")


def _compare_values(cheb, mx, test_points, label):
    """Evaluate at test_points and return max absolute error vs MoCaX."""
    if mx is None:
        return float("nan")

    # Register derivative ids for price and d/dx
    price_id  = cheb.get_derivative_id([0, 0])
    mx_price_id = mx.getDerivativeId([0, 0])

    max_err = 0.0
    for pt in test_points:
        pt_list = pt.tolist()

        py_val = cheb.eval(pt_list, derivative_id=price_id)
        mx_val = mx.eval(pt_list, mx_price_id)
        err = abs(py_val - mx_val)
        max_err = max(max_err, err)

    print(f"  [{label}] max |PyChebyshev - MoCaX| over "
          f"{len(test_points)} points: {max_err:.2e}")
    return max_err


# ============================================================================
# Test: additional_data payload is threaded into function at build time
# ============================================================================

def test_additional_data_basic():
    """Build with additional_data and verify values match direct computation."""
    print("\n[1] additional_data basic: strike=100")
    additional_data = {"strike": 100.0}
    cheb = _build_pychebyshev(additional_data)
    assert cheb.additional_data is additional_data, "additional_data not stored"

    # Spot-check: eval at (0.5, 0.3) should match direct call
    pt = [0.5, 0.3]
    expected = f(pt, additional_data)
    got = cheb.eval(pt, [0, 0])
    err = abs(got - expected)
    print(f"  f(0.5, 0.3) expected={expected:.12f}  got={got:.12f}  err={err:.2e}")
    assert err < 1e-10, f"additional_data not wired correctly: err={err:.2e}"
    print("  PASS")
    return cheb


# ============================================================================
# Test: descriptor get/set
# ============================================================================

def test_descriptor(cheb, mx):
    """set_descriptor / get_descriptor round-trip on both sides."""
    print("\n[2] descriptor get/set")
    label = "v0.15-ergonomics-benchmark"
    cheb.set_descriptor(label)
    assert cheb.get_descriptor() == label, "PyChebyshev descriptor mismatch"

    if mx is not None:
        mx.setDescriptor(label)
        mx_label = mx.getDescriptor()
        match = (mx_label == label)
        print(f"  PyChebyshev: {cheb.get_descriptor()!r}")
        print(f"  MoCaX:       {mx_label!r}  {'OK' if match else 'MISMATCH'}")
        assert match, f"MoCaX descriptor mismatch: {mx_label!r} != {label!r}"
    else:
        print(f"  PyChebyshev: {cheb.get_descriptor()!r}  OK")
    print("  PASS")


# ============================================================================
# Test: derivative_id registry
# ============================================================================

def test_derivative_id(cheb, mx):
    """get_derivative_id returns stable ints; same orders → same id."""
    print("\n[3] derivative_id registry")

    price_id  = cheb.get_derivative_id([0, 0])
    delta_id  = cheb.get_derivative_id([1, 0])
    gamma_id  = cheb.get_derivative_id([2, 0])
    price_id2 = cheb.get_derivative_id([0, 0])  # must equal price_id

    assert price_id == price_id2, "Same orders must give same id"
    assert len({price_id, delta_id, gamma_id}) == 3, "Distinct orders must give distinct ids"

    print(f"  PyChebyshev ids: price={price_id} delta={delta_id} gamma={gamma_id}")
    print(f"  get_derivative_id([0,0]) twice: {price_id} == {price_id2}  OK")

    if mx is not None:
        mx_price_id  = mx.getDerivativeId([0, 0])
        mx_delta_id  = mx.getDerivativeId([1, 0])
        mx_gamma_id  = mx.getDerivativeId([2, 0])
        mx_price_id2 = mx.getDerivativeId([0, 0])
        print(f"  MoCaX     ids: price={mx_price_id} delta={mx_delta_id} gamma={mx_gamma_id}")
        assert mx_price_id == mx_price_id2, "MoCaX same-orders id mismatch"

    print("  PASS")


# ============================================================================
# Test: introspection trio (getConstructorType, getUsedNs, isConstructionFinished)
# ============================================================================

def test_introspection(cheb, mx):
    """getConstructorType / getUsedNs / isConstructionFinished."""
    print("\n[4] introspection trio")
    _compare_introspection(cheb, mx)

    assert cheb.is_construction_finished(), "Should be True after build()"
    assert cheb.get_constructor_type() == "ChebyshevApproximation"
    assert cheb.get_used_ns() == [11, 11]

    print("  PASS")


# ============================================================================
# Test: eval via derivative_id agrees with eval via derivative_order
# ============================================================================

def test_eval_derivative_id(cheb):
    """eval via derivative_id must match eval via derivative_order exactly."""
    print("\n[5] eval via derivative_id vs derivative_order")
    rng = np.random.default_rng(0)
    test_points = rng.uniform(-1, 1, size=(30, 2))
    max_err = 0.0
    for pt in test_points:
        pt_list = pt.tolist()
        price_id = cheb.get_derivative_id([0, 0])
        delta_id = cheb.get_derivative_id([1, 0])
        v_order  = cheb.eval(pt_list, derivative_order=[0, 0])
        v_id     = cheb.eval(pt_list, derivative_id=price_id)
        dv_order = cheb.eval(pt_list, derivative_order=[1, 0])
        dv_id    = cheb.eval(pt_list, derivative_id=delta_id)
        max_err  = max(max_err, abs(v_order - v_id), abs(dv_order - dv_id))
    print(f"  max |eval(order) - eval(id)| = {max_err:.2e}")
    assert max_err == 0.0, "derivative_id and derivative_order must agree exactly"
    print("  PASS")


# ============================================================================
# Test: numerical agreement vs MoCaX
# ============================================================================

def test_numerical_agreement(cheb, mx):
    """Values must agree with MoCaX to within 1e-12 absolute."""
    print("\n[6] numerical agreement vs MoCaX (50 random points)")
    rng = np.random.default_rng(42)
    test_points = rng.uniform(-1, 1, size=(50, 2))

    if mx is None:
        print("  MoCaX unavailable — skipped")
        return float("nan")

    max_err = _compare_values(cheb, mx, test_points, "price")
    assert max_err < 1e-12, f"Disagreement: max_err={max_err:.2e}"
    print("  PASS")
    return max_err


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 66)
    print("  PyChebyshev v0.15 ergonomics vs MoCaX — benchmark")
    print("=" * 66)

    # Build PyChebyshev
    t0 = time.perf_counter()
    cheb = test_additional_data_basic()
    t_py = time.perf_counter() - t0
    print(f"  PyChebyshev build time: {t_py*1e3:.1f} ms")

    # Build MoCaX (None if unavailable)
    t0 = time.perf_counter()
    additional_data = {"strike": 100.0}
    mx = _build_mocax(additional_data)
    t_mx = time.perf_counter() - t0
    if mx is not None:
        print(f"  MoCaX      build time: {t_mx*1e3:.1f} ms")

    # Run tests
    test_descriptor(cheb, mx)
    test_derivative_id(cheb, mx)
    test_introspection(cheb, mx)
    test_eval_derivative_id(cheb)
    err = test_numerical_agreement(cheb, mx)

    # Summary
    print("\n" + "=" * 66)
    print("  Summary")
    print("=" * 66)
    print(f"  additional_data threading     OK")
    print(f"  descriptor get/set            OK")
    print(f"  derivative_id registry        OK")
    print(f"  introspection trio            OK")
    print(f"  eval(derivative_id=...)       OK")
    if not math.isnan(err):
        print(f"  numerical agreement vs MoCaX  max_err={err:.2e}  OK")
    else:
        print(f"  numerical agreement vs MoCaX  (MoCaX unavailable — skipped)")
    print("=" * 66)
    print("PASS")


if __name__ == "__main__":
    main()
