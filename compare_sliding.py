"""
Compare PyChebyshev Slider vs MoCaX Sliding on identical test conditions.

Tests:
1. 5D Black-Scholes: head-to-head comparison (same domain, nodes, pivot)
2. 5D Black-Scholes with grouped partition: shows accuracy vs decomposition
3. 5D additively separable sin: PyChebyshev only (MoCaX segfaults on this)

PyChebyshev Slider additionally provides analytical derivatives per slide
(MoCaX Sliding only supports function values via finite differences).

Requires: mocaxpy + libmocaxc.so (MoCaX portion is skipped if unavailable)

Usage:
    uv run python compare_sliding.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

import math
import os
import sys
import time

import numpy as np

from pychebyshev import ChebyshevSlider

# ============================================================================
# MoCaX setup (module-level, matching working mocax_sliding.py pattern)
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

def sin_sum_5d(x, _):
    """sin(x0) + sin(x1) + sin(x2) + sin(x3) + sin(x4)."""
    return sum(math.sin(xi) for xi in x)


def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_5d(x, _):
    """5D Black-Scholes wrapper: V(S, K, T, sigma, r)."""
    return black_scholes_call(S=x[0], K=x[1], T=x[2], sigma=x[3], r=x[4])


# ============================================================================
# Analytical Black-Scholes Greeks
# ============================================================================

def bs_greeks(S, K, T, sigma, r, q=0.0):
    """Return dict of analytical BS Greeks: delta, gamma, vega, rho, theta."""
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1 = norm.cdf(d1)
    npd1 = norm.pdf(d1)
    return {
        "delta": math.exp(-q * T) * nd1,
        "gamma": math.exp(-q * T) * npd1 / (S * sigma * math.sqrt(T)),
        "vega": S * math.exp(-q * T) * npd1 * math.sqrt(T),
        "rho": K * T * math.exp(-r * T) * norm.cdf(d2),
        "theta": (
            -S * math.exp(-q * T) * npd1 * sigma / (2 * math.sqrt(T))
            + q * S * math.exp(-q * T) * nd1
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ),
    }


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
# Helpers
# ============================================================================

def generate_samples(domain, n, seed=42):
    """Generate random test points within domain."""
    rng = np.random.default_rng(seed)
    samples = np.empty((n, len(domain)))
    for d, (lo, hi) in enumerate(domain):
        samples[:, d] = rng.uniform(lo, hi, n)
    return samples


# ============================================================================
# Test 1: Black-Scholes head-to-head
# ============================================================================

def test_bs_headtohead():
    """Compare PyChebyshev Slider vs MoCaX Sliding on 5D Black-Scholes."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 1: Black-Scholes 5D — PyChebyshev vs MoCaX [1,1,1,1,1] partition")
    print(f"{'=' * 78}")

    # Match mocax_sliding.py configuration exactly
    domain = [[50.0, 150.0], [50.0, 150.0], [0.1, 3.0], [0.1, 0.5], [0.01, 0.15]]
    n_nodes = [11] * 5
    partition = [[0], [1], [2], [3], [4]]
    pivot = [100.0, 100.0, 1.0, 0.25, 0.05]

    full_evals = int(np.prod(n_nodes))
    slide_evals = sum(n_nodes)
    print(f"  Domain: S=[50,150] K=[50,150] T=[0.1,3] sigma=[0.1,0.5] r=[0.01,0.15]")
    print(f"  Nodes: {n_nodes}  Pivot: {pivot}")
    print(f"  Build evals: {slide_evals} (sliding) vs {full_evals:,} (full tensor)")

    # --- Build PyChebyshev ---
    start = time.perf_counter()
    slider = ChebyshevSlider(
        bs_5d, 5, domain, n_nodes,
        partition=partition, pivot_point=pivot,
    )
    slider.build(verbose=False)
    pyc_build = time.perf_counter() - start

    # --- Build MoCaX ---
    mocax_obj = None
    mocax_build = None
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([n - 1 for n in n_nodes])
            start = time.perf_counter()
            mocax_obj = mocaxpy.MocaxSliding(
                bs_5d, [1, 1, 1, 1, 1], mx_domain, mx_ns, pivot
            )
            mocax_build = time.perf_counter() - start
        except Exception as e:
            print(f"  MoCaX build failed: {e}")

    # --- Print build times ---
    print(f"\n  Build time:")
    print(f"    PyChebyshev: {pyc_build:.4f}s")
    if mocax_build is not None:
        ratio = pyc_build / mocax_build
        print(f"    MoCaX:       {mocax_build:.4f}s  ({ratio:.1f}x)")

    # --- Accuracy on test cases from mocax_sliding.py ---
    test_cases = [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ]

    header = f"\n  {'Case':<12s} {'Analytical':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err%':>8s}"
    print(header)
    print(f"  {'─' * (44 + (22 if mocax_obj else 0))}")

    for params, label in test_cases:
        S, K, T, sigma, r = params
        exact = black_scholes_call(S, K, T, sigma, r)
        pyc_val = slider.eval(params, [0, 0, 0, 0, 0])
        pyc_err = abs(pyc_val - exact) / exact * 100

        line = f"  {label:<12s} {exact:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"
        if mocax_obj is not None:
            mx_val = mocax_obj.eval(params)
            mx_err = abs(mx_val - exact) / exact * 100
            line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"
        print(line)

    # --- Timing ---
    n_timing = 1000
    samples = generate_samples(
        [(50, 150), (50, 150), (0.1, 3), (0.1, 0.5), (0.01, 0.15)],
        n_timing, seed=99,
    )
    zero_d = [0, 0, 0, 0, 0]

    slider.eval(samples[0].tolist(), zero_d)  # warmup
    start = time.perf_counter()
    for i in range(n_timing):
        slider.eval(samples[i].tolist(), zero_d)
    pyc_ms = (time.perf_counter() - start) / n_timing * 1000

    print(f"\n  Query time ({n_timing} samples, price only):")
    print(f"    PyChebyshev: {pyc_ms:.4f} ms/query")

    if mocax_obj is not None:
        mocax_obj.eval(samples[0].tolist())
        start = time.perf_counter()
        for i in range(n_timing):
            mocax_obj.eval(samples[i].tolist())
        mx_ms = (time.perf_counter() - start) / n_timing * 1000
        ratio = pyc_ms / mx_ms
        print(f"    MoCaX:       {mx_ms:.4f} ms/query  ({ratio:.1f}x)")

    return slider, mocax_obj


def test_derivative_accuracy(slider, mocax_obj):
    """Compare derivative accuracy: PyChebyshev analytical vs MoCaX 1bp bump vs exact."""
    print(f"\n{'=' * 78}")
    print(f"  DERIVATIVE ACCURACY: PyChebyshev analytical vs MoCaX 1bp bump")
    print(f"{'=' * 78}")
    print(f"  PyChebyshev: analytical derivatives via spectral diff matrices")
    if mocax_obj is not None:
        print(f"  MoCaX:       central finite differences with 1bp bump")
    print(f"  Reference:   exact Black-Scholes Greeks")

    # Greeks config: (name, deriv_order, dim_to_bump, bump_size, exact_key)
    # 1bp bumps: S bump = 0.01 (1bp of S=100), sigma = 0.0001, r = 0.0001
    greeks_config = [
        ("Delta",  [1, 0, 0, 0, 0], 0, 0.01,   "delta"),
        ("Gamma",  [2, 0, 0, 0, 0], 0, 0.01,   "gamma"),
        ("Vega",   [0, 0, 0, 1, 0], 3, 0.0001,  "vega"),
        ("Rho",    [0, 0, 0, 0, 1], 4, 0.0001,  "rho"),
        ("Theta",  [0, 0, 1, 0, 0], 2, 0.0001,  "theta"),
    ]

    test_points = [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ]

    for greek_name, deriv_order, bump_dim, bump_h, exact_key in greeks_config:
        is_second = sum(deriv_order) == 2
        # Theta: dV/dT with sign convention (theta = -dV/dT)
        theta_sign = -1.0 if greek_name == "Theta" else 1.0

        header = f"\n  {greek_name}"
        if greek_name == "Theta":
            header += " (= -dV/dT)"
        print(header)

        col_header = f"  {'Case':<12s} {'Exact':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
        if mocax_obj is not None:
            col_header += f" {'MoCaX 1bp':>14s} {'Err%':>8s}"
        print(col_header)
        print(f"  {'─' * (44 + (22 if mocax_obj else 0))}")

        for pt, label in test_points:
            S, K, T, sigma, r = pt
            exact_greeks = bs_greeks(S, K, T, sigma, r)
            exact_val = exact_greeks[exact_key]

            # PyChebyshev analytical
            pyc_val = slider.eval(pt, deriv_order) * theta_sign
            pyc_err = abs(pyc_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0

            line = f"  {label:<12s} {exact_val:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"

            if mocax_obj is not None:
                # MoCaX bump derivative
                eval_fn = lambda p: mocax_obj.eval(p)
                if is_second:
                    mx_val = fd_second(eval_fn, pt, bump_dim, bump_h) * theta_sign
                else:
                    mx_val = fd_central(eval_fn, pt, bump_dim, bump_h) * theta_sign
                mx_err = abs(mx_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0
                line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"

            print(line)

    # --- Summary: average errors across all Greeks and test points ---
    print(f"\n  {'─' * 78}")
    print(f"  DERIVATIVE SUMMARY (avg |error%| across all test points)")
    print(f"  {'─' * 78}")

    sum_header = f"  {'Greek':<12s} {'PyChebyshev':>14s}"
    if mocax_obj is not None:
        sum_header += f" {'MoCaX 1bp':>14s}"
    print(sum_header)
    print(f"  {'─' * (26 + (16 if mocax_obj else 0))}")

    for greek_name, deriv_order, bump_dim, bump_h, exact_key in greeks_config:
        is_second = sum(deriv_order) == 2
        theta_sign = -1.0 if greek_name == "Theta" else 1.0

        pyc_errs = []
        mx_errs = []
        for pt, _ in test_points:
            S, K, T, sigma, r = pt
            exact_val = bs_greeks(S, K, T, sigma, r)[exact_key]

            pyc_val = slider.eval(pt, deriv_order) * theta_sign
            pyc_errs.append(abs(pyc_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0)

            if mocax_obj is not None:
                eval_fn = lambda p: mocax_obj.eval(p)
                if is_second:
                    mx_val = fd_second(eval_fn, pt, bump_dim, bump_h) * theta_sign
                else:
                    mx_val = fd_central(eval_fn, pt, bump_dim, bump_h) * theta_sign
                mx_errs.append(abs(mx_val - exact_val) / abs(exact_val) * 100 if exact_val != 0 else 0.0)

        line = f"  {greek_name:<12s} {np.mean(pyc_errs):>13.4f}%"
        if mx_errs:
            line += f" {np.mean(mx_errs):>13.4f}%"
        print(line)

    print()


# ============================================================================
# Test 2: BS with grouped partition
# ============================================================================

def test_bs_grouped():
    """BS with [[0,1],[2,3],[4]] — captures more cross-terms."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 2: Black-Scholes 5D — grouped [[0,1],[2,3],[4]]")
    print(f"{'=' * 78}")

    domain = [[50.0, 150.0], [50.0, 150.0], [0.1, 3.0], [0.1, 0.5], [0.01, 0.15]]
    n_nodes = [11] * 5
    partition = [[0, 1], [2, 3], [4]]
    pivot = [100.0, 100.0, 1.0, 0.25, 0.05]

    slide_evals = 11 * 11 + 11 * 11 + 11
    print(f"  Partition: {partition}")
    print(f"  Build evals: {slide_evals} (grouped) vs 55 (fully separated) vs {11**5:,} (full)")

    start = time.perf_counter()
    slider = ChebyshevSlider(
        bs_5d, 5, domain, n_nodes,
        partition=partition, pivot_point=pivot,
    )
    slider.build(verbose=False)
    build_time = time.perf_counter() - start
    print(f"  Build time: {build_time:.4f}s")

    # MoCaX with grouped partition
    mocax_obj = None
    if HAS_MOCAX:
        try:
            mx_domain = mocaxpy.MocaxDomain(domain)
            mx_ns = mocaxpy.MocaxNs([10] * 5)
            start = time.perf_counter()
            mocax_obj = mocaxpy.MocaxSliding(
                bs_5d, [2, 2, 1], mx_domain, mx_ns, pivot
            )
            mx_build = time.perf_counter() - start
            print(f"  MoCaX build: {mx_build:.4f}s")
        except Exception as e:
            print(f"  MoCaX build failed: {e}")

    test_cases = [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ]

    header = f"\n  {'Case':<12s} {'Analytical':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err%':>8s}"
    print(header)
    print(f"  {'─' * (44 + (22 if mocax_obj else 0))}")

    for params, label in test_cases:
        S, K, T, sigma, r = params
        exact = black_scholes_call(S, K, T, sigma, r)
        pyc_val = slider.eval(params, [0, 0, 0, 0, 0])
        pyc_err = abs(pyc_val - exact) / exact * 100

        line = f"  {label:<12s} {exact:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"
        if mocax_obj is not None:
            mx_val = mocax_obj.eval(params)
            mx_err = abs(mx_val - exact) / exact * 100
            line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"
        print(line)


# ============================================================================
# Test 3: Additively separable (PyChebyshev only)
# ============================================================================

def test_sin_separable():
    """sin sum — ideal case for sliding. PyChebyshev only (MoCaX segfaults)."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 3: Additively Separable sin(x0)+...+sin(x4) [PyChebyshev only]")
    print(f"{'=' * 78}")
    print(f"  (MoCaX Sliding segfaults on non-BS domains — skipped)")

    domain = [[-1.0, 1.0]] * 5
    n_nodes = [11] * 5
    partition = [[0], [1], [2], [3], [4]]
    pivot = [0.0] * 5

    slide_evals = sum(n_nodes)
    print(f"  Nodes: {n_nodes}  Pivot: {pivot}")
    print(f"  Build evals: {slide_evals} (sliding) vs {11**5:,} (full tensor)")

    start = time.perf_counter()
    slider = ChebyshevSlider(
        sin_sum_5d, 5, domain, n_nodes,
        partition=partition, pivot_point=pivot,
    )
    slider.build(verbose=False)
    build_time = time.perf_counter() - start
    print(f"  Build time: {build_time:.4f}s")

    # Accuracy on 200 random points
    n_samples = 200
    samples = generate_samples([(-1, 1)] * 5, n_samples)
    errors = []
    for i in range(n_samples):
        pt = samples[i].tolist()
        exact = sum(math.sin(xi) for xi in pt)
        approx = slider.eval(pt, [0, 0, 0, 0, 0])
        errors.append(abs(approx - exact) / max(abs(exact), 1e-15) * 100)

    print(f"\n  Accuracy ({n_samples} random samples):")
    print(f"    Avg error: {np.mean(errors):.2e}%")
    print(f"    Max error: {np.max(errors):.2e}%")
    print(f"    (Should be near-zero — function is additively separable)")

    # Derivatives
    pt = [0.5, -0.3, 0.7, -0.1, 0.9]
    print(f"\n  Derivatives at {pt}:")
    for d in range(5):
        do = [0] * 5
        do[d] = 1
        val = slider.eval(pt, do)
        exact = math.cos(pt[d])
        err = abs(val - exact)
        print(f"    df/dx{d} = {val:>12.8f}  (exact: {exact:>12.8f}, err: {err:.2e})")

    # Timing
    n_timing = 1000
    timing_samples = generate_samples([(-1, 1)] * 5, n_timing, seed=99)
    slider.eval(timing_samples[0].tolist(), [0] * 5)
    start = time.perf_counter()
    for i in range(n_timing):
        slider.eval(timing_samples[i].tolist(), [0] * 5)
    ms = (time.perf_counter() - start) / n_timing * 1000
    print(f"\n  Query time: {ms:.4f} ms/query (price only)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("  COMPARISON: PyChebyshev Slider vs MoCaX Sliding")
    print("=" * 78)
    print()
    print("  Both methods implement the same additive decomposition (sliding).")
    print("  PyChebyshev additionally provides analytical derivatives per slide.")
    if not HAS_MOCAX:
        print("\n  NOTE: MoCaX not available. Showing PyChebyshev results only.")

    slider, mocax_obj = test_bs_headtohead()
    test_derivative_accuracy(slider, mocax_obj)
    test_bs_grouped()
    test_sin_separable()

    print(f"\n{'=' * 78}")
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
