"""
Compare PyChebyshev ChebyshevTT vs MoCaX Extend (Tensor Train) on 5D Black-Scholes.

Tests:
1. Head-to-head: build time, num evals, accuracy at 50 random test points
2. Eval speed: 1000 point timing comparison
3. FD Greeks (Delta, Gamma) at 10 scenarios vs analytical

PyChebyshev TT uses TT-Cross approximation with Chebyshev coefficient cores.
MoCaX Extend uses rank-adaptive ALS on a random subgrid of the Chebyshev grid.

Both methods compute derivatives via finite differences (neither TT format
supports analytical spectral derivatives).

Requires: mocaxextend_lib/ with shared_libs/ (MoCaX portion is skipped if unavailable)

Usage:
    uv run python compare_tensor_train.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

from pychebyshev import ChebyshevTT


# ============================================================================
# MoCaX Extend setup
# ============================================================================

mocaxextend_lib_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mocaxextend_lib"
)
sys.path.insert(0, mocaxextend_lib_dir)

HAS_MOCAX = False
me = None
try:
    import mocaxextendpy.mocax_extend as me

    HAS_MOCAX = True
    print("MoCaX Extend: available")
except ImportError as e:
    print(f"MoCaX Extend not available: {e}")
    print("  (MoCaX comparison will be skipped, PyChebyshev results shown)\n")


# ============================================================================
# Domain and configuration
# ============================================================================

DIMENSION = 5
DOMAIN = [
    [80.0, 120.0],   # S: spot price
    [90.0, 110.0],   # K: strike price
    [0.25, 1.0],     # T: time to maturity
    [0.15, 0.35],    # sigma: volatility
    [0.01, 0.08],    # r: risk-free rate
]
N_NODES = [11, 11, 11, 11, 11]
Q = 0.02  # dividend yield


# ============================================================================
# Black-Scholes functions
# ============================================================================

def black_scholes_call(S, K, T, sigma, r, q=Q):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_vectorized(x):
    """Vectorized Black-Scholes call price for MoCaX subgrid evaluation."""
    S, K, T, sigma, r = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    results = np.zeros(len(S))
    valid = T > 0
    if np.any(valid):
        Sv, Kv, Tv, sv, rv = S[valid], K[valid], T[valid], sigma[valid], r[valid]
        d1 = (np.log(Sv / Kv) + (rv - Q + 0.5 * sv**2) * Tv) / (sv * np.sqrt(Tv))
        d2 = d1 - sv * np.sqrt(Tv)
        results[valid] = (
            Sv * np.exp(-Q * Tv) * norm.cdf(d1)
            - Kv * np.exp(-rv * Tv) * norm.cdf(d2)
        )
    if np.any(~valid):
        results[~valid] = np.maximum(S[~valid] - K[~valid], 0)
    return results


def bs_5d(x, _):
    """5D Black-Scholes wrapper: V(S, K, T, sigma, r)."""
    return black_scholes_call(S=x[0], K=x[1], T=x[2], sigma=x[3], r=x[4])


# ============================================================================
# Analytical Greeks
# ============================================================================

def bs_greeks(S, K, T, sigma, r, q=Q):
    """Return dict of analytical BS Greeks: delta, gamma."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
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
    return (eval_fn(pt_up) - 2 * eval_fn(point) + eval_fn(pt_dn)) / (bump**2)


# ============================================================================
# Build MoCaX Extend TT
# ============================================================================

def build_mocax_tt(num_scenarios=8000):
    """Build MoCaX Extend TT approximation. Returns (obj, build_time, n_evals)."""
    obj = me.MocaxExtend(DIMENSION, N_NODES, DOMAIN)

    start = time.perf_counter()
    random_cheb_pts = obj.subgrid_by_number(num_scenarios)
    vals_subgrid = black_scholes_call_vectorized(random_cheb_pts)
    obj.set_subgrid_values(vals_subgrid)
    obj.gen_train_val_data()

    rank_adaptive_params = {
        "tolerance": 1e-3,
        "rel_tolerance": 1e-8,
        "max_iters": 100,
        "max_rank": 20,
        "print_progress": False,
        "max_rounds": 5,
    }
    obj.run_rank_adaptive_algo(**rank_adaptive_params)
    build_time = time.perf_counter() - start

    n_evals = num_scenarios  # subgrid evaluations
    return obj, build_time, n_evals


# ============================================================================
# Test 1: Head-to-head accuracy
# ============================================================================

def test_accuracy_headtohead():
    """Compare PyChebyshev TT vs MoCaX TT on accuracy at 50 random test points."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 1: Black-Scholes 5D — Accuracy at 50 Random Test Points")
    print(f"{'=' * 78}")
    print(f"  Domain: S=[80,120] K=[90,110] T=[0.25,1] sigma=[0.15,0.35] r=[0.01,0.08]")
    print(f"  Nodes: {N_NODES}  q={Q} (dividend yield)")
    print(f"  Full tensor: {int(np.prod(N_NODES)):,} evaluations")

    # --- Build PyChebyshev TT ---
    print(f"\n  Building PyChebyshev TT (max_rank=15, cross)...")
    start = time.perf_counter()
    pyc_tt = ChebyshevTT(
        bs_5d, DIMENSION, DOMAIN, N_NODES,
        max_rank=15, max_sweeps=10, tolerance=1e-6,
    )
    pyc_tt.build(verbose=False, seed=42)
    pyc_build = time.perf_counter() - start
    print(f"    Built in {pyc_build:.3f}s ({pyc_tt.total_build_evals:,} func evals)")
    print(f"    TT ranks: {pyc_tt.tt_ranks}")
    print(f"    Compression: {pyc_tt.compression_ratio:.1f}x")

    # --- Build MoCaX TT ---
    mocax_obj = None
    mocax_build = None
    mocax_evals = None
    if HAS_MOCAX:
        try:
            print(f"\n  Building MoCaX TT (8000 subgrid, rank-adaptive ALS)...")
            mocax_obj, mocax_build, mocax_evals = build_mocax_tt(num_scenarios=8000)
            print(f"    Built in {mocax_build:.3f}s ({mocax_evals:,} func evals)")
        except Exception as e:
            print(f"    MoCaX build failed: {e}")

    # --- Build time comparison ---
    print(f"\n  Build Comparison:")
    print(f"    {'Method':<20s} {'Time (s)':>10s} {'Func Evals':>12s}")
    print(f"    {'─' * 44}")
    print(f"    {'PyChebyshev TT':<20s} {pyc_build:>10.3f} {pyc_tt.total_build_evals:>12,}")
    if mocax_build is not None:
        print(f"    {'MoCaX TT':<20s} {mocax_build:>10.3f} {mocax_evals:>12,}")
        ratio = pyc_build / mocax_build if mocax_build > 0 else float("inf")
        print(f"    PyChebyshev is {ratio:.2f}x {'faster' if ratio < 1 else 'slower'}")

    # --- Accuracy on 50 random test points ---
    n_test = 50
    samples = generate_samples(DOMAIN, n_test, seed=42)

    print(f"\n  Accuracy ({n_test} random test points, price > $0.50):")

    pyc_errors = []
    mx_errors = []
    for i in range(n_test):
        pt = samples[i].tolist()
        S, K, T, sigma, r = pt
        exact = black_scholes_call(S, K, T, sigma, r)
        if abs(exact) < 0.50:
            continue

        pyc_val = pyc_tt.eval(pt)
        pyc_err = abs(pyc_val - exact) / abs(exact) * 100
        pyc_errors.append(pyc_err)

        if mocax_obj is not None:
            mx_val = mocax_obj.cheb_tensor_evals(samples[i:i + 1])[0]
            mx_err = abs(mx_val - exact) / abs(exact) * 100
            mx_errors.append(mx_err)

    header = f"    {'Metric':<24s} {'PyChebyshev':>14s}"
    if mx_errors:
        header += f" {'MoCaX':>14s}"
    print(header)
    print(f"    {'─' * (38 + (16 if mx_errors else 0))}")

    print(f"    {'Mean error':<24s} {np.mean(pyc_errors):>13.4f}%", end="")
    if mx_errors:
        print(f" {np.mean(mx_errors):>13.4f}%")
    else:
        print()

    print(f"    {'Max error':<24s} {np.max(pyc_errors):>13.4f}%", end="")
    if mx_errors:
        print(f" {np.max(mx_errors):>13.4f}%")
    else:
        print()

    print(f"    {'Median error':<24s} {np.median(pyc_errors):>13.4f}%", end="")
    if mx_errors:
        print(f" {np.median(mx_errors):>13.4f}%")
    else:
        print()

    print(f"    {'95th percentile':<24s} {np.percentile(pyc_errors, 95):>13.4f}%", end="")
    if mx_errors:
        print(f" {np.percentile(mx_errors, 95):>13.4f}%")
    else:
        print()

    print(f"    {'Points tested':<24s} {len(pyc_errors):>14d}", end="")
    if mx_errors:
        print(f" {len(mx_errors):>14d}")
    else:
        print()

    # --- Named scenarios ---
    test_cases = [
        ([100.0, 100.0, 1.0, 0.25, 0.05], "ATM"),
        ([110.0, 100.0, 1.0, 0.25, 0.05], "ITM"),
        ([90.0, 100.0, 1.0, 0.25, 0.05], "OTM"),
        ([100.0, 100.0, 0.5, 0.25, 0.05], "Short T"),
        ([100.0, 100.0, 0.25, 0.25, 0.05], "Very Short T"),
        ([100.0, 100.0, 1.0, 0.15, 0.05], "Low vol"),
        ([100.0, 100.0, 1.0, 0.35, 0.05], "High vol"),
        ([100.0, 100.0, 1.0, 0.25, 0.01], "Low r"),
        ([100.0, 100.0, 1.0, 0.25, 0.08], "High r"),
        ([85.0, 105.0, 0.5, 0.20, 0.03], "Corner1"),
        ([115.0, 95.0, 0.75, 0.30, 0.07], "Corner2"),
    ]

    header = f"\n  {'Case':<14s} {'Analytical':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>14s} {'Err%':>8s}"
    print(header)
    print(f"  {'─' * (48 + (22 if mocax_obj else 0))}")

    for params, label in test_cases:
        S, K, T, sigma, r = params
        exact = black_scholes_call(S, K, T, sigma, r)
        pyc_val = pyc_tt.eval(params)
        pyc_err = abs(pyc_val - exact) / abs(exact) * 100 if abs(exact) > 1e-10 else 0.0

        line = f"  {label:<14s} {exact:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"
        if mocax_obj is not None:
            mx_val = mocax_obj.cheb_tensor_evals(np.array([params]))[0]
            mx_err = abs(mx_val - exact) / abs(exact) * 100 if abs(exact) > 1e-10 else 0.0
            line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"
        print(line)

    return pyc_tt, mocax_obj, pyc_build, mocax_build


# ============================================================================
# Test 2: Eval speed
# ============================================================================

def test_eval_speed(pyc_tt, mocax_obj):
    """Compare evaluation speed: PyChebyshev vs MoCaX on 1000 points."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 2: Evaluation Speed — 1000 Random Points")
    print(f"{'=' * 78}")

    n_timing = 1000
    samples = generate_samples(DOMAIN, n_timing, seed=99)

    # --- PyChebyshev: single-point eval ---
    pyc_tt.eval(samples[0].tolist())  # warmup
    start = time.perf_counter()
    for i in range(n_timing):
        pyc_tt.eval(samples[i].tolist())
    pyc_single_ms = (time.perf_counter() - start) / n_timing * 1000

    # --- PyChebyshev: batch eval ---
    pyc_tt.eval_batch(samples[:10])  # warmup
    start = time.perf_counter()
    pyc_tt.eval_batch(samples)
    pyc_batch_ms = (time.perf_counter() - start) / n_timing * 1000

    print(f"\n  PyChebyshev TT:")
    print(f"    Single eval:  {pyc_single_ms:.4f} ms/query")
    print(f"    Batch eval:   {pyc_batch_ms:.4f} ms/query ({pyc_single_ms / pyc_batch_ms:.1f}x speedup)")

    # --- MoCaX: batch eval (only mode) ---
    if mocax_obj is not None:
        mocax_obj.cheb_tensor_evals(samples[:10])  # warmup
        start = time.perf_counter()
        mocax_obj.cheb_tensor_evals(samples)
        mx_ms = (time.perf_counter() - start) / n_timing * 1000

        print(f"\n  MoCaX TT:")
        print(f"    Batch eval:   {mx_ms:.4f} ms/query")

        ratio = pyc_batch_ms / mx_ms if mx_ms > 0 else float("inf")
        print(f"\n  PyChebyshev batch is {ratio:.2f}x {'faster' if ratio < 1 else 'slower'} than MoCaX")


# ============================================================================
# Test 3: FD Greeks comparison
# ============================================================================

def test_greeks_comparison(pyc_tt, mocax_obj):
    """Compare FD Greeks (Delta, Gamma) at 10 scenarios vs analytical."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 3: FD Greeks — Delta & Gamma at 10 Scenarios")
    print(f"{'=' * 78}")
    print(f"  Both methods use central finite differences.")
    print(f"  PyChebyshev: built-in eval_multi() with automatic FD")
    if mocax_obj is not None:
        print(f"  MoCaX: manual central FD on cheb_tensor_evals()")
    print(f"  Reference: exact Black-Scholes Greeks (q={Q})")

    test_points = [
        ([100.0, 100.0, 1.0, 0.25, 0.05],   "ATM"),
        ([110.0, 100.0, 1.0, 0.25, 0.05],   "ITM"),
        ([90.0, 100.0, 1.0, 0.25, 0.05],    "OTM"),
        ([100.0, 100.0, 0.5, 0.25, 0.05],   "Short T"),
        ([100.0, 100.0, 0.25, 0.25, 0.05],  "Very Short T"),
        ([100.0, 100.0, 1.0, 0.15, 0.05],   "Low vol"),
        ([100.0, 100.0, 1.0, 0.35, 0.05],   "High vol"),
        ([100.0, 100.0, 1.0, 0.25, 0.01],   "Low r"),
        ([85.0, 105.0, 0.5, 0.20, 0.03],    "Corner1"),
        ([115.0, 95.0, 0.75, 0.30, 0.07],   "Corner2"),
    ]

    # MoCaX eval wrapper (single point)
    def mocax_eval_single(pt):
        return mocax_obj.cheb_tensor_evals(np.array([pt]))[0]

    # Greeks config: (name, pyc_deriv_order, dim_to_bump, bump_size, exact_key, is_second)
    greeks_config = [
        ("Delta", [1, 0, 0, 0, 0], 0, 0.5,   "delta", False),
        ("Gamma", [2, 0, 0, 0, 0], 0, 0.5,   "gamma", True),
    ]

    for greek_name, deriv_order, bump_dim, bump_h, exact_key, is_second in greeks_config:
        header = f"\n  {greek_name}"
        print(header)

        col_header = f"  {'Case':<14s} {'Exact':>12s} {'PyChebyshev':>14s} {'Err%':>8s}"
        if mocax_obj is not None:
            col_header += f" {'MoCaX FD':>14s} {'Err%':>8s}"
        print(col_header)
        print(f"  {'─' * (48 + (22 if mocax_obj else 0))}")

        pyc_errs = []
        mx_errs = []

        for pt, label in test_points:
            S, K, T, sigma, r = pt
            exact_val = bs_greeks(S, K, T, sigma, r)[exact_key]

            # PyChebyshev: built-in FD via eval_multi
            pyc_results = pyc_tt.eval_multi(pt, [[0] * 5, deriv_order])
            pyc_val = pyc_results[1]
            pyc_err = abs(pyc_val - exact_val) / abs(exact_val) * 100 if abs(exact_val) > 1e-10 else 0.0
            pyc_errs.append(pyc_err)

            line = f"  {label:<14s} {exact_val:>12.6f} {pyc_val:>14.6f} {pyc_err:>7.3f}%"

            if mocax_obj is not None:
                if is_second:
                    mx_val = fd_second(mocax_eval_single, pt, bump_dim, bump_h)
                else:
                    mx_val = fd_central(mocax_eval_single, pt, bump_dim, bump_h)
                mx_err = abs(mx_val - exact_val) / abs(exact_val) * 100 if abs(exact_val) > 1e-10 else 0.0
                mx_errs.append(mx_err)
                line += f" {mx_val:>14.6f} {mx_err:>7.3f}%"

            print(line)

        # Per-Greek summary
        print(f"  {'Average':<14s} {'':>12s} {'':>14s} {np.mean(pyc_errs):>7.3f}%", end="")
        if mx_errs:
            print(f" {'':>14s} {np.mean(mx_errs):>7.3f}%")
        else:
            print()

    # --- Overall summary ---
    print(f"\n  {'─' * 78}")
    print(f"  GREEKS SUMMARY (avg |error%|)")
    print(f"  {'─' * 78}")

    sum_header = f"  {'Greek':<14s} {'PyChebyshev':>14s}"
    if mocax_obj is not None:
        sum_header += f" {'MoCaX FD':>14s}"
    print(sum_header)
    print(f"  {'─' * (28 + (16 if mocax_obj else 0))}")

    for greek_name, deriv_order, bump_dim, bump_h, exact_key, is_second in greeks_config:
        pyc_errs = []
        mx_errs = []
        for pt, _ in test_points:
            S, K, T, sigma, r = pt
            exact_val = bs_greeks(S, K, T, sigma, r)[exact_key]

            pyc_val = pyc_tt.eval_multi(pt, [deriv_order])[0]
            pyc_errs.append(
                abs(pyc_val - exact_val) / abs(exact_val) * 100
                if abs(exact_val) > 1e-10
                else 0.0
            )

            if mocax_obj is not None:
                if is_second:
                    mx_val = fd_second(mocax_eval_single, pt, bump_dim, bump_h)
                else:
                    mx_val = fd_central(mocax_eval_single, pt, bump_dim, bump_h)
                mx_errs.append(
                    abs(mx_val - exact_val) / abs(exact_val) * 100
                    if abs(exact_val) > 1e-10
                    else 0.0
                )

        line = f"  {greek_name:<14s} {np.mean(pyc_errs):>13.4f}%"
        if mx_errs:
            line += f" {np.mean(mx_errs):>13.4f}%"
        print(line)


# ============================================================================
# Summary table
# ============================================================================

def print_summary(pyc_tt, mocax_obj, pyc_build_time, mocax_build_time):
    """Print final summary table."""
    print(f"\n{'=' * 78}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 78}")

    # Recompute accuracy stats for summary
    n_test = 50
    samples = generate_samples(DOMAIN, n_test, seed=42)

    pyc_errors = []
    mx_errors = []
    for i in range(n_test):
        pt = samples[i].tolist()
        S, K, T, sigma, r = pt
        exact = black_scholes_call(S, K, T, sigma, r)
        if abs(exact) < 0.50:
            continue
        pyc_val = pyc_tt.eval(pt)
        pyc_errors.append(abs(pyc_val - exact) / abs(exact) * 100)
        if mocax_obj is not None:
            mx_val = mocax_obj.cheb_tensor_evals(samples[i:i + 1])[0]
            mx_errors.append(abs(mx_val - exact) / abs(exact) * 100)

    # Timing
    n_timing = 1000
    timing_samples = generate_samples(DOMAIN, n_timing, seed=99)

    pyc_tt.eval_batch(timing_samples[:10])
    start = time.perf_counter()
    pyc_tt.eval_batch(timing_samples)
    pyc_ms = (time.perf_counter() - start) / n_timing * 1000

    mx_ms = None
    if mocax_obj is not None:
        mocax_obj.cheb_tensor_evals(timing_samples[:10])
        start = time.perf_counter()
        mocax_obj.cheb_tensor_evals(timing_samples)
        mx_ms = (time.perf_counter() - start) / n_timing * 1000

    col_w = 18
    header = f"  {'Metric':<30s} {'PyChebyshev':>{col_w}s}"
    if mocax_obj is not None:
        header += f" {'MoCaX':>{col_w}s}"
    print(header)
    print(f"  {'─' * (30 + col_w + (col_w + 1 if mocax_obj else 0))}")

    # Method
    line = f"  {'TT algorithm':<30s} {'TT-Cross':>{col_w}s}"
    if mocax_obj is not None:
        line += f" {'Rank-adaptive ALS':>{col_w}s}"
    print(line)

    # Build time
    line = f"  {'Build time (s)':<30s} {pyc_build_time:>{col_w}.3f}"
    if mocax_build_time is not None:
        line += f" {mocax_build_time:>{col_w}.3f}"
    print(line)

    # Func evals
    line = f"  {'Function evaluations':<30s} {pyc_tt.total_build_evals:>{col_w},}"
    if mocax_obj is not None:
        line += f" {8000:>{col_w},}"  # subgrid size
    print(line)

    # TT ranks
    line = f"  {'TT ranks':<30s} {str(pyc_tt.tt_ranks):>{col_w}s}"
    if mocax_obj is not None:
        line += f" {'(not exposed)':>{col_w}s}"
    print(line)

    # Compression
    line = f"  {'Compression ratio':<30s} {pyc_tt.compression_ratio:>{col_w}.1f}x"
    if mocax_obj is not None:
        line += f" {'N/A':>{col_w}s}"
    print(line)

    # Price accuracy
    line = f"  {'Mean price error':<30s} {np.mean(pyc_errors):>{col_w - 1}.4f}%"
    if mx_errors:
        line += f" {np.mean(mx_errors):>{col_w - 1}.4f}%"
    print(line)

    line = f"  {'Max price error':<30s} {np.max(pyc_errors):>{col_w - 1}.4f}%"
    if mx_errors:
        line += f" {np.max(mx_errors):>{col_w - 1}.4f}%"
    print(line)

    # Eval speed
    line = f"  {'Batch eval (ms/query)':<30s} {pyc_ms:>{col_w}.4f}"
    if mx_ms is not None:
        line += f" {mx_ms:>{col_w}.4f}"
    print(line)

    # Derivatives
    line = f"  {'Derivatives':<30s} {'FD (built-in)':>{col_w}s}"
    if mocax_obj is not None:
        line += f" {'FD (manual)':>{col_w}s}"
    print(line)

    print(f"\n  Notes:")
    print(f"    - Both methods use finite differences for Greeks (no analytical derivs in TT)")
    print(f"    - PyChebyshev TT-Cross selects grid points adaptively via maxvol pivoting")
    print(f"    - MoCaX uses random subgrid + alternating least squares optimization")
    print(f"    - Fixed dividend yield q={Q}")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("  COMPARISON: PyChebyshev ChebyshevTT vs MoCaX Extend (Tensor Train)")
    print("=" * 78)
    print()
    print("  Both methods build Chebyshev interpolation in Tensor Train format")
    print("  for 5D Black-Scholes option pricing V(S, K, T, sigma, r).")
    print(f"  Domain: S=[80,120] K=[90,110] T=[0.25,1] sigma=[0.15,0.35] r=[0.01,0.08]")
    if not HAS_MOCAX:
        print("\n  NOTE: MoCaX not available. Showing PyChebyshev results only.")

    pyc_tt, mocax_obj, pyc_build_time, mocax_build_time = test_accuracy_headtohead()

    test_eval_speed(pyc_tt, mocax_obj)
    test_greeks_comparison(pyc_tt, mocax_obj)
    print_summary(pyc_tt, mocax_obj, pyc_build_time, mocax_build_time)

    print(f"{'=' * 78}")
    print("  DONE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
