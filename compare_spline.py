"""
Compare PyChebyshev ChebyshevSpline vs MoCaX Spine (Special Points) on identical tests.

Tests:
1. 1D |x|: knot at x=0, compare accuracy + derivatives
2. 2D BS payoff: knot at K=100, compare price accuracy at 20 points
3. 3D BS call: knot at S=K=100, compare price + Delta + Gamma
4. Convergence: nodes 5->25, show algebraic (global) vs exponential (spline)
5. Summary table: side-by-side

PyChebyshev Spline provides analytical derivatives via spectral diff matrices
on each piece.  MoCaX Spine uses the same Chebyshev spline idea with its C++
implementation.

Requires: mocaxpy + libmocaxc.so (MoCaX portion is skipped if unavailable)

Usage:
    uv run python compare_spline.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

import math
import os
import sys
import time

import numpy as np

from pychebyshev import ChebyshevApproximation, ChebyshevSpline

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

def abs_1d(x, _):
    """1D |x|."""
    return abs(x[0])


def call_payoff_2d(x, _):
    """Discounted call payoff: max(S - K, 0) * exp(-r * T)."""
    return max(x[0] - 100.0, 0.0) * math.exp(-0.05 * x[1])


def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_3d(x, _):
    """3D Black-Scholes wrapper: C(S, T, sigma) with K=100, r=0.05."""
    return black_scholes_call(S=x[0], K=100.0, T=x[1], r=0.05, sigma=x[2])


def bs_greeks_3d(S, T, sigma, K=100.0, r=0.05, q=0.0):
    """Return dict of analytical BS Greeks for the 3D test."""
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    npd1 = norm.pdf(d1)
    return {
        "delta": math.exp(-q * T) * norm.cdf(d1),
        "gamma": math.exp(-q * T) * npd1 / (S * sigma * math.sqrt(T)),
    }


# ============================================================================
# Helpers
# ============================================================================

def generate_samples(domain, n, seed=42):
    """Generate random test points within domain."""
    rng = np.random.default_rng(seed)
    samples = np.empty((n, len(domain)))
    for d, (lo, hi) in enumerate(domain):
        samples[:, d] = rng.uniform(lo, hi, n)
    return samples


def fd_central(eval_fn, point, dim, bump):
    """Central finite difference: (f(x+h) - f(x-h)) / (2h)."""
    pt_up = list(point)
    pt_dn = list(point)
    pt_up[dim] += bump
    pt_dn[dim] -= bump
    return (eval_fn(pt_up) - eval_fn(pt_dn)) / (2 * bump)


def fd_second(eval_fn, point, dim, bump):
    """Central second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2."""
    pt_up = list(point)
    pt_dn = list(point)
    pt_up[dim] += bump
    pt_dn[dim] -= bump
    return (eval_fn(pt_up) - 2 * eval_fn(point) + eval_fn(pt_dn)) / (bump ** 2)


# ============================================================================
# Test 1: 1D |x| comparison
# ============================================================================

def test_abs_1d():
    """Compare PyChebyshev Spline vs MoCaX Spine on 1D |x| with knot at x=0."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 1: 1D |x| — Spline with knot at x=0")
    print(f"{'=' * 78}")

    domain = [[-1.0, 1.0]]
    n_nodes = [15]
    knots = [[0.0]]

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    sp = ChebyshevSpline(abs_1d, 1, domain, n_nodes, knots)
    sp.build(verbose=False)
    pyc_build = time.perf_counter() - start
    print(f"\n  PyChebyshev: built in {pyc_build:.4f}s ({sp.num_pieces} pieces)")

    # --- Build MoCaX ---
    mocax_obj = None
    mocax_build = None
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1 for n in n_nodes])
            mx_sp = mocaxpy.MocaxSpecialPoints(knots)
            start = time.perf_counter()
            mocax_obj = mocaxpy.Mocax(
                abs_1d, 1, mx_domain, None, mx_ns,
                special_points=mx_sp, max_derivative_order=2,
            )
            mocax_build = time.perf_counter() - start
            print(f"  MoCaX:       built in {mocax_build:.4f}s")
        except Exception as e:
            print(f"  MoCaX build failed: {e}")

    # --- Value accuracy ---
    test_pts = [-0.9, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.9]

    header = f"\n  {'x':>8s} {'Exact':>12s} {'PyChebyshev':>14s} {'Err':>12s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err':>12s}"
    print(header)
    print(f"  {'─' * (46 + (26 if mocax_obj else 0))}")

    for x in test_pts:
        exact = abs(x)
        pyc_val = sp.eval([x], [0])
        pyc_err = abs(pyc_val - exact)

        line = f"  {x:>8.2f} {exact:>12.8f} {pyc_val:>14.8f} {pyc_err:>12.2e}"
        if mocax_obj is not None:
            mx_val = mocax_obj.eval([x])
            mx_err = abs(mx_val - exact)
            line += f" {mx_val:>14.8f} {mx_err:>12.2e}"
        print(line)

    # --- Derivative accuracy ---
    print(f"\n  Derivatives (d|x|/dx):")
    header = f"  {'x':>8s} {'Exact':>12s} {'PyChebyshev':>14s} {'Err':>12s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err':>12s}"
    print(header)
    print(f"  {'─' * (46 + (26 if mocax_obj else 0))}")

    for x in [-0.5, -0.3, 0.3, 0.5]:
        exact_deriv = -1.0 if x < 0 else 1.0
        pyc_deriv = sp.eval([x], [1])
        pyc_err = abs(pyc_deriv - exact_deriv)

        line = f"  {x:>8.2f} {exact_deriv:>12.4f} {pyc_deriv:>14.8f} {pyc_err:>12.2e}"
        if mocax_obj is not None:
            deriv_id = mocax_obj.get_derivative_id([1])
            mx_deriv = mocax_obj.eval([x], deriv_id)
            mx_err = abs(mx_deriv - exact_deriv)
            line += f" {mx_deriv:>14.8f} {mx_err:>12.2e}"
        print(line)


# ============================================================================
# Test 2: 2D BS payoff comparison
# ============================================================================

def test_bs_payoff_2d():
    """Compare on 2D discounted call payoff max(S-K,0)*exp(-rT)."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 2: 2D Discounted Call Payoff — knot at K=100")
    print(f"{'=' * 78}")

    domain = [[80.0, 120.0], [0.25, 1.0]]
    n_nodes = [15, 15]
    knots = [[100.0], []]

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    sp = ChebyshevSpline(call_payoff_2d, 2, domain, n_nodes, knots)
    sp.build(verbose=False)
    pyc_build = time.perf_counter() - start
    print(f"\n  PyChebyshev: built in {pyc_build:.4f}s ({sp.num_pieces} pieces)")

    # --- Build MoCaX ---
    mocax_obj = None
    mocax_build = None
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1 for n in n_nodes])
            mx_sp = mocaxpy.MocaxSpecialPoints(knots)
            start = time.perf_counter()
            mocax_obj = mocaxpy.Mocax(
                call_payoff_2d, 2, mx_domain, None, mx_ns,
                special_points=mx_sp, max_derivative_order=2,
            )
            mocax_build = time.perf_counter() - start
            print(f"  MoCaX:       built in {mocax_build:.4f}s")
        except Exception as e:
            print(f"  MoCaX build failed: {e}")

    # --- Accuracy at 20 test points ---
    n_test = 20
    samples = generate_samples(domain, n_test, seed=42)

    header = f"\n  {'S':>8s} {'T':>6s} {'Exact':>12s} {'PyChebyshev':>14s} {'Err':>12s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err':>12s}"
    print(header)
    print(f"  {'─' * (52 + (26 if mocax_obj else 0))}")

    pyc_errs = []
    mx_errs = []
    for i in range(n_test):
        S, T = samples[i]
        exact = max(S - 100.0, 0.0) * math.exp(-0.05 * T)
        pyc_val = sp.eval([S, T], [0, 0])
        pyc_err = abs(pyc_val - exact)
        pyc_errs.append(pyc_err)

        line = f"  {S:>8.2f} {T:>6.3f} {exact:>12.6f} {pyc_val:>14.6f} {pyc_err:>12.2e}"
        if mocax_obj is not None:
            mx_val = mocax_obj.eval([S, T])
            mx_err = abs(mx_val - exact)
            mx_errs.append(mx_err)
            line += f" {mx_val:>14.6f} {mx_err:>12.2e}"
        print(line)

    print(f"\n  Max error: PyChebyshev={max(pyc_errs):.2e}", end="")
    if mx_errs:
        print(f"  MoCaX={max(mx_errs):.2e}")
    else:
        print()


# ============================================================================
# Test 3: 3D BS call with spine
# ============================================================================

def test_bs_3d_spine():
    """3D Black-Scholes C(S, T, sigma) with knot at S=100."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 3: 3D BS Call C(S, T, sigma) — knot at S=K=100")
    print(f"{'=' * 78}")

    domain = [[80.0, 120.0], [0.25, 2.0], [0.1, 0.5]]
    n_nodes = [15, 12, 10]
    knots = [[100.0], [], []]

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    sp = ChebyshevSpline(bs_3d, 3, domain, n_nodes, knots)
    sp.build(verbose=False)
    pyc_build = time.perf_counter() - start
    print(f"\n  PyChebyshev: built in {pyc_build:.4f}s ({sp.num_pieces} pieces)")
    print(f"    Nodes per piece: {n_nodes}")
    print(f"    Total build evals: {sp.total_build_evals:,}")

    # --- Build MoCaX ---
    mocax_obj = None
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1 for n in n_nodes])
            mx_sp = mocaxpy.MocaxSpecialPoints(knots)
            start = time.perf_counter()
            mocax_obj = mocaxpy.Mocax(
                bs_3d, 3, mx_domain, None, mx_ns,
                special_points=mx_sp, max_derivative_order=2,
            )
            mocax_build = time.perf_counter() - start
            print(f"  MoCaX:       built in {mocax_build:.4f}s")
        except Exception as e:
            print(f"  MoCaX build failed: {e}")

    # --- Price accuracy ---
    test_cases = [
        ([100.0, 1.0, 0.25], "ATM"),
        ([110.0, 1.0, 0.25], "ITM"),
        ([90.0, 1.0, 0.25], "OTM"),
        ([100.0, 0.5, 0.25], "Short T"),
        ([100.0, 1.0, 0.40], "High vol"),
    ]

    header = f"\n  {'Case':<12s} {'Analytical':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err%':>8s}"
    print(header)
    print(f"  {'─' * (44 + (22 if mocax_obj else 0))}")

    for params, label in test_cases:
        S, T, sigma = params
        exact = black_scholes_call(S, K=100.0, T=T, sigma=sigma, r=0.05)
        pyc_val = sp.eval(params, [0, 0, 0])
        pyc_err = abs(pyc_val - exact) / max(abs(exact), 1e-15) * 100

        line = f"  {label:<12s} {exact:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"
        if mocax_obj is not None:
            mx_val = mocax_obj.eval(params)
            mx_err = abs(mx_val - exact) / max(abs(exact), 1e-15) * 100
            line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"
        print(line)

    # --- Delta + Gamma ---
    # Evaluate slightly off the knot (S=100 is a knot — derivatives undefined there)
    print(f"\n  Greeks at S=100.01, T=1, sigma=0.25:")
    pt = [100.01, 1.0, 0.25]
    greeks = bs_greeks_3d(S=100.01, T=1.0, sigma=0.25)

    # Delta
    pyc_delta = sp.eval(pt, [1, 0, 0])
    exact_delta = greeks["delta"]
    pyc_delta_err = abs(pyc_delta - exact_delta) / abs(exact_delta) * 100
    line = f"    Delta:  exact={exact_delta:.6f}  PyChebyshev={pyc_delta:.6f} (err={pyc_delta_err:.3f}%)"
    if mocax_obj is not None:
        deriv_id = mocax_obj.get_derivative_id([1, 0, 0])
        mx_delta = mocax_obj.eval(pt, deriv_id)
        mx_delta_err = abs(mx_delta - exact_delta) / abs(exact_delta) * 100
        line += f"  MoCaX={mx_delta:.6f} (err={mx_delta_err:.3f}%)"
    print(line)

    # Gamma
    pyc_gamma = sp.eval(pt, [2, 0, 0])
    exact_gamma = greeks["gamma"]
    pyc_gamma_err = abs(pyc_gamma - exact_gamma) / abs(exact_gamma) * 100
    line = f"    Gamma:  exact={exact_gamma:.6f}  PyChebyshev={pyc_gamma:.6f} (err={pyc_gamma_err:.3f}%)"
    if mocax_obj is not None:
        deriv_id = mocax_obj.get_derivative_id([2, 0, 0])
        mx_gamma = mocax_obj.eval(pt, deriv_id)
        mx_gamma_err = abs(mx_gamma - exact_gamma) / abs(exact_gamma) * 100
        line += f"  MoCaX={mx_gamma:.6f} (err={mx_gamma_err:.3f}%)"
    print(line)


# ============================================================================
# Test 4: Convergence comparison
# ============================================================================

def test_convergence():
    """Show algebraic (global) vs exponential (spline) convergence for kinked function."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 4: Convergence — Global vs Spline for |x| on [-1,1]")
    print(f"{'=' * 78}")
    print(f"  Global: O(1/n^2) algebraic convergence (kink at x=0)")
    print(f"  Spline: exponential (spectral) convergence on each piece")

    n_values = [5, 7, 9, 11, 15, 19, 25]
    test_pts = np.linspace(-0.95, 0.95, 100)

    header = f"\n  {'n':>4s} {'Global max err':>16s} {'Spline max err':>16s}"
    if HAS_MOCAX:
        header += f" {'MoCaX max err':>16s}"
    print(header)
    print(f"  {'─' * (36 + (18 if HAS_MOCAX else 0))}")

    for n in n_values:
        # Global
        global_cheb = ChebyshevApproximation(abs_1d, 1, [[-1, 1]], [n])
        global_cheb.build(verbose=False)

        # PyChebyshev Spline
        sp = ChebyshevSpline(abs_1d, 1, [[-1, 1]], [n], [[0.0]])
        sp.build(verbose=False)

        # MoCaX Spine
        mocax_obj = None
        if HAS_MOCAX:
            try:
                mx_domain = mocaxpy.MocaxDomain([[-1.0, 1.0]])
                mx_ns = mocaxpy.MocaxNs([n - 1])
                mx_sp = mocaxpy.MocaxSpecialPoints([[0.0]])
                mocax_obj = mocaxpy.Mocax(
                    abs_1d, 1, mx_domain, None, mx_ns,
                    special_points=mx_sp, max_derivative_order=0,
                )
            except Exception:
                pass

        # Compute errors
        global_max_err = max(
            abs(global_cheb.vectorized_eval([x], [0]) - abs(x))
            for x in test_pts
        )
        spline_max_err = max(
            abs(sp.eval([x], [0]) - abs(x))
            for x in test_pts
        )

        line = f"  {n:>4d} {global_max_err:>16.8e} {spline_max_err:>16.8e}"
        if mocax_obj is not None:
            mx_max_err = max(
                abs(mocax_obj.eval([x]) - abs(x))
                for x in test_pts
            )
            line += f" {mx_max_err:>16.8e}"
        print(line)


# ============================================================================
# Summary table
# ============================================================================

def print_summary():
    """Print final summary table."""
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY: PyChebyshev ChebyshevSpline vs MoCaX Spine")
    print(f"{'=' * 78}")

    col_w = 22
    header = f"  {'Feature':<30s} {'PyChebyshev':>{col_w}s}"
    if HAS_MOCAX:
        header += f" {'MoCaX':>{col_w}s}"
    print(header)
    print(f"  {'─' * (30 + col_w + (col_w + 1 if HAS_MOCAX else 0))}")

    features = [
        ("Algorithm", "Chebyshev Splines", "Spine (Special Points)"),
        ("Derivatives", "Analytical (spectral)", "Analytical (spectral)"),
        ("Max derivative order", "User-specified", "User-specified"),
        ("Knot placement", "User-specified", "User-specified"),
        ("Multi-D knots", "Cartesian product", "Cartesian product"),
        ("Convergence (kinked)", "Exponential per piece", "Exponential per piece"),
        ("Implementation", "Pure Python/NumPy", "C++ with Python wrapper"),
        ("Error estimation", "Chebyshev coefficients", "Built-in threshold"),
        ("Serialization", "pickle", "Binary / pickle"),
    ]

    for name, pyc, mx in features:
        line = f"  {name:<30s} {pyc:>{col_w}s}"
        if HAS_MOCAX:
            line += f" {mx:>{col_w}s}"
        print(line)

    print(f"\n  Notes:")
    print(f"    - Both methods implement the same mathematical idea: piecewise")
    print(f"      Chebyshev interpolation with knots at known singularities.")
    print(f"    - PyChebyshev uses Python + BLAS; MoCaX uses optimized C++.")
    print(f"    - PyChebyshev ChebyshevSpline composes with ChebyshevApproximation,")
    print(f"      inheriting all its features (vectorized eval, eval_multi, batch).")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("  COMPARISON: PyChebyshev ChebyshevSpline vs MoCaX Spine")
    print("=" * 78)
    print()
    print("  Both methods implement piecewise Chebyshev interpolation with")
    print("  user-specified knots at singularity locations (kinks, discontinuities).")
    if not HAS_MOCAX:
        print("\n  NOTE: MoCaX not available. Showing PyChebyshev results only.")

    test_abs_1d()
    test_bs_payoff_2d()
    test_bs_3d_spine()
    test_convergence()
    print_summary()

    print(f"{'=' * 78}")
    print("  DONE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
