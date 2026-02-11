"""
Compare PyChebyshev error_estimate() vs MoCaX get_error_threshold().

Both libraries provide an ex ante error estimate based on Chebyshev expansion
coefficients. The mathematical basis is the same (magnitude of highest-order
Chebyshev coefficients), but the implementations differ:

  - PyChebyshev uses Type I nodes (interior, via DCT-II)
  - MoCaX uses Type II nodes (endpoints, via mirror+FFT / DCT-I)

Tests:
1. 1D Black-Scholes: compare error estimates against empirical max error
2. 3D Black-Scholes: compare error estimates in higher dimensions
3. Convergence: error estimates vs empirical error as n increases

Requires: mocaxpy + libmocaxc.so (MoCaX portion is skipped if unavailable)

Usage:
    uv run python compare_error_estimation.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

import math
import os
import sys
import time

import numpy as np

from pychebyshev import ChebyshevApproximation

# ============================================================================
# MoCaX setup (module-level, matching working compare_sliding.py pattern)
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
# Test functions
# ============================================================================

def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_1d(x, _):
    """1D Black-Scholes wrapper: V(S) with fixed K=100, T=1, sigma=0.25, r=0.05."""
    return black_scholes_call(S=x[0], K=100.0, T=1.0, sigma=0.25, r=0.05)


def bs_3d(x, _):
    """3D Black-Scholes wrapper: V(S, T, sigma) with fixed K=100, r=0.05."""
    return black_scholes_call(S=x[0], K=100.0, T=x[1], sigma=x[2], r=0.05)


# ============================================================================
# Helpers
# ============================================================================

def compute_empirical_max_error(cheb, func, domain, n_samples=1000, seed=42):
    """Compute empirical max absolute error over random test points."""
    rng = np.random.default_rng(seed)
    ndim = len(domain)
    max_err = 0.0
    deriv_order = [0] * ndim
    for _ in range(n_samples):
        pt = [rng.uniform(lo, hi) for lo, hi in domain]
        exact = func(pt, None)
        approx = cheb.vectorized_eval(pt, deriv_order)
        max_err = max(max_err, abs(approx - exact))
    return max_err


# ============================================================================
# Test 1: 1D Black-Scholes comparison
# ============================================================================

def test_1d_comparison():
    """Compare error estimates for 1D BS(S) with n=11 nodes."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 1: 1D Black-Scholes — Error Estimate Comparison")
    print(f"{'=' * 78}")
    print(f"  BS call price as V(S), S in [50, 150]")
    print(f"  Fixed: K=100, T=1, sigma=0.25, r=0.05")
    print(f"  PyChebyshev: n=11 nodes (Type I, interior)")
    if HAS_MOCAX:
        print(f"  MoCaX:       n=11 nodes (Type II, endpoints)")

    n = 11
    domain = [[50.0, 150.0]]

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    cheb = ChebyshevApproximation(bs_1d, 1, domain, [n])
    cheb.build(verbose=False)
    pyc_build = time.perf_counter() - start

    pyc_error_est = cheb.error_estimate()
    empirical_err = compute_empirical_max_error(cheb, bs_1d, domain)

    print(f"\n  PyChebyshev:")
    print(f"    Build time:      {pyc_build:.4f}s")
    print(f"    error_estimate(): {pyc_error_est:.6e}")
    print(f"    Empirical max:    {empirical_err:.6e} (1000 random points)")

    # --- Build MoCaX ---
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1])  # MoCaX uses degree, not count
            start = time.perf_counter()
            mcx_obj = mocaxpy.Mocax(bs_1d, 1, mx_domain, None, mx_ns)
            mx_build = time.perf_counter() - start

            mx_error_est = mcx_obj.get_error_threshold()

            # Compute MoCaX empirical error
            rng = np.random.default_rng(42)
            mx_max_err = 0.0
            for _ in range(1000):
                pt = [rng.uniform(50.0, 150.0)]
                exact = bs_1d(pt, None)
                approx = mcx_obj.eval(pt)
                mx_max_err = max(mx_max_err, abs(approx - exact))

            print(f"\n  MoCaX:")
            print(f"    Build time:           {mx_build:.4f}s")
            print(f"    get_error_threshold(): {mx_error_est:.6e}")
            print(f"    Empirical max:         {mx_max_err:.6e} (1000 random points)")
        except Exception as e:
            print(f"\n  MoCaX build failed: {e}")

    # --- Summary ---
    print(f"\n  Summary:")
    print(f"    PyChebyshev estimate / empirical = {pyc_error_est / empirical_err:.2f}x")
    if HAS_MOCAX:
        try:
            print(f"    MoCaX estimate / empirical      = {mx_error_est / mx_max_err:.2f}x")
        except Exception:
            pass


# ============================================================================
# Test 2: 3D Black-Scholes comparison
# ============================================================================

def test_3d_comparison():
    """Compare error estimates for 3D BS(S, T, sigma)."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 2: 3D Black-Scholes — Error Estimate Comparison")
    print(f"{'=' * 78}")
    print(f"  BS call price as V(S, T, sigma)")
    print(f"  Domain: S=[50,150] x T=[0.1,2.0] x sigma=[0.1,0.5]")
    print(f"  Fixed: K=100, r=0.05")

    n_nodes = [15, 12, 10]
    domain = [[50.0, 150.0], [0.1, 2.0], [0.1, 0.5]]

    print(f"  PyChebyshev nodes: {n_nodes}")
    if HAS_MOCAX:
        print(f"  MoCaX nodes:       {n_nodes} (degree = {[n-1 for n in n_nodes]})")

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    cheb = ChebyshevApproximation(bs_3d, 3, domain, n_nodes)
    cheb.build(verbose=False)
    pyc_build = time.perf_counter() - start

    pyc_error_est = cheb.error_estimate()
    empirical_err = compute_empirical_max_error(cheb, bs_3d, domain)

    print(f"\n  PyChebyshev:")
    print(f"    Build time:      {pyc_build:.4f}s ({int(np.prod(n_nodes)):,} evaluations)")
    print(f"    error_estimate(): {pyc_error_est:.6e}")
    print(f"    Empirical max:    {empirical_err:.6e} (1000 random points)")

    # --- Build MoCaX ---
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1 for n in n_nodes])
            start = time.perf_counter()
            mcx_obj = mocaxpy.Mocax(bs_3d, 3, mx_domain, None, mx_ns)
            mx_build = time.perf_counter() - start

            mx_error_est = mcx_obj.get_error_threshold()

            # Compute MoCaX empirical error
            rng = np.random.default_rng(42)
            mx_max_err = 0.0
            for _ in range(1000):
                pt = [
                    rng.uniform(50.0, 150.0),
                    rng.uniform(0.1, 2.0),
                    rng.uniform(0.1, 0.5),
                ]
                exact = bs_3d(pt, None)
                approx = mcx_obj.eval(pt)
                mx_max_err = max(mx_max_err, abs(approx - exact))

            print(f"\n  MoCaX:")
            print(f"    Build time:           {mx_build:.4f}s")
            print(f"    get_error_threshold(): {mx_error_est:.6e}")
            print(f"    Empirical max:         {mx_max_err:.6e} (1000 random points)")
        except Exception as e:
            print(f"\n  MoCaX build failed: {e}")

    # --- Summary ---
    print(f"\n  Summary:")
    ratio_pyc = pyc_error_est / empirical_err if empirical_err > 0 else float('inf')
    print(f"    PyChebyshev estimate / empirical = {ratio_pyc:.2f}x")
    if HAS_MOCAX:
        try:
            ratio_mx = mx_error_est / mx_max_err if mx_max_err > 0 else float('inf')
            print(f"    MoCaX estimate / empirical      = {ratio_mx:.2f}x")
        except Exception:
            pass


# ============================================================================
# Test 3: Convergence comparison
# ============================================================================

def test_convergence():
    """Compare error estimates as n increases for 1D BS."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 3: Convergence — Error Estimates vs Empirical (1D BS)")
    print(f"{'=' * 78}")
    print(f"  BS call price as V(S), S in [50, 150]")
    print(f"  Fixed: K=100, T=1, sigma=0.25, r=0.05")
    print(f"  Increasing number of nodes: both estimates should track empirical error")

    n_values = [5, 8, 11, 15, 20, 25, 30]
    domain = [[50.0, 150.0]]

    # Table header
    header = f"\n  {'n':>4s} | {'PyChebyshev est':>16s} | {'Empirical max':>14s} | {'Ratio':>6s}"
    if HAS_MOCAX:
        header += f" | {'MoCaX est':>16s} | {'MoCaX empirical':>16s} | {'Ratio':>6s}"
    print(header)
    sep = f"  {'─' * 4}-+-{'─' * 16}-+-{'─' * 14}-+-{'─' * 6}"
    if HAS_MOCAX:
        sep += f"-+-{'─' * 16}-+-{'─' * 16}-+-{'─' * 6}"
    print(sep)

    for n in n_values:
        # --- PyChebyshev ---
        cheb = ChebyshevApproximation(bs_1d, 1, domain, [n])
        cheb.build(verbose=False)
        pyc_est = cheb.error_estimate()
        pyc_emp = compute_empirical_max_error(cheb, bs_1d, domain)
        pyc_ratio = pyc_est / pyc_emp if pyc_emp > 0 else float('inf')

        line = f"  {n:>4d} | {pyc_est:>16.6e} | {pyc_emp:>14.6e} | {pyc_ratio:>6.2f}"

        # --- MoCaX ---
        if HAS_MOCAX:
            try:
                mx_domain = mocaxpy.MocaxDomain(domain)
                mx_ns = mocaxpy.MocaxNs([n - 1])
                mcx_obj = mocaxpy.Mocax(bs_1d, 1, mx_domain, None, mx_ns)
                mx_est = mcx_obj.get_error_threshold()

                # MoCaX empirical error
                rng = np.random.default_rng(42)
                mx_emp = 0.0
                for _ in range(1000):
                    pt = [rng.uniform(50.0, 150.0)]
                    exact = bs_1d(pt, None)
                    approx = mcx_obj.eval(pt)
                    mx_emp = max(mx_emp, abs(approx - exact))

                mx_ratio = mx_est / mx_emp if mx_emp > 0 else float('inf')
                line += f" | {mx_est:>16.6e} | {mx_emp:>16.6e} | {mx_ratio:>6.2f}"
            except Exception as e:
                line += f" |     (failed: {e})"

        print(line)

    print(f"\n  Notes:")
    print(f"    - Ratio > 1 means the estimate is conservative (overestimates error)")
    print(f"    - Both estimates should decrease as n increases (spectral convergence)")
    print(f"    - Ideal ratio is close to 1 (tight estimate)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("  COMPARISON: PyChebyshev error_estimate() vs MoCaX get_error_threshold()")
    print("=" * 78)
    print()
    print("  Both methods estimate interpolation error from Chebyshev coefficients.")
    print("  PyChebyshev: Type I nodes (interior) + DCT-II for coefficients")
    if HAS_MOCAX:
        print("  MoCaX:       Type II nodes (endpoints) + mirror+FFT / DCT-I")
    else:
        print("\n  NOTE: MoCaX not available. Showing PyChebyshev results only.")

    test_1d_comparison()
    test_3d_comparison()
    test_convergence()

    print(f"\n{'=' * 78}")
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
