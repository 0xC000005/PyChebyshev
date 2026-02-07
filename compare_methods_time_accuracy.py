"""
Fair Comparison: Chebyshev Barycentric vs MoCaX Standard

Compares speed and accuracy on identical 5D Black-Scholes test conditions:
- Same domain, same Chebyshev nodes (11 per dim), same random test points
- Clean timing: query time measured separately from error computation
- Both methods use analytical derivatives (no finite differences)

Ground truth: blackscholes library analytical formulas

Usage:
    uv run python compare_methods_time_accuracy.py
    LD_LIBRARY_PATH=mocax_lib:$LD_LIBRARY_PATH uv run python compare_methods_time_accuracy.py
    N_SAMPLES=500 uv run python compare_methods_time_accuracy.py
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from blackscholes import BlackScholesCall
from chebyshev_barycentric import ChebyshevApproximation as ChebyshevBarycentric


# ============================================================================
# Data structures
# ============================================================================

METRICS = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']

# Derivative orders: price + 5 Greeks
# Note: sigma is dim 3, T is dim 2 in our ordering [S, K, T, sigma, r]
ALL_DERIV_ORDERS = [
    [0, 0, 0, 0, 0],  # Price
    [1, 0, 0, 0, 0],  # Delta (dV/dS)
    [2, 0, 0, 0, 0],  # Gamma (d²V/dS²)
    [0, 0, 0, 1, 0],  # Vega (dV/dσ)
    [0, 0, 1, 0, 0],  # dV/dT (negated for Theta)
    [0, 0, 0, 0, 1],  # Rho (dV/dr)
]


@dataclass
class MethodErrors:
    """Error statistics per metric."""
    data: Dict[str, List[float]] = field(default_factory=lambda: {m: [] for m in METRICS})

    def add(self, metric: str, error: float):
        self.data[metric].append(error)

    def stats(self, metric: str) -> Tuple[float, float]:
        """Return (mean, max) error for a metric."""
        errs = self.data[metric]
        return float(np.mean(errs)), float(np.max(errs))


# ============================================================================
# Domain & ground truth
# ============================================================================

DOMAIN = [
    (80.0, 120.0),   # S
    (90.0, 110.0),   # K
    (0.25, 1.0),     # T
    (0.15, 0.35),    # sigma
    (0.01, 0.08),    # r
]
Q = 0.02  # fixed dividend yield


def generate_samples(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = np.empty((n, 5))
    for d, (lo, hi) in enumerate(DOMAIN):
        samples[:, d] = rng.uniform(lo, hi, n)
    return samples


def compute_ground_truth(samples: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = len(samples)
    prices = np.empty(n)
    greeks = {m: np.empty(n) for m in METRICS if m != 'price'}

    for i in range(n):
        S, K, T, sigma, r = samples[i]
        c = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=Q)
        prices[i] = c.price()
        core = c.get_core_greeks()
        for m in greeks:
            greeks[m][i] = core[m]

    return prices, greeks


def compute_errors(approx_values: np.ndarray, gt_prices: np.ndarray,
                   gt_greeks: Dict[str, np.ndarray]) -> MethodErrors:
    """Compute relative errors from pre-computed approximation values.

    approx_values: shape (n, 6) — columns: price, delta, gamma, vega, dV/dT, rho
    """
    errors = MethodErrors()
    n = len(approx_values)

    for i in range(n):
        vals = approx_values[i]

        # Price
        err = abs(vals[0] - gt_prices[i]) / gt_prices[i] * 100
        errors.add('price', err)

        # Delta
        err = abs(vals[1] - gt_greeks['delta'][i]) / abs(gt_greeks['delta'][i]) * 100
        errors.add('delta', err)

        # Gamma
        err = abs(vals[2] - gt_greeks['gamma'][i]) / abs(gt_greeks['gamma'][i]) * 100
        errors.add('gamma', err)

        # Vega
        err = abs(vals[3] - gt_greeks['vega'][i]) / abs(gt_greeks['vega'][i]) * 100
        errors.add('vega', err)

        # Theta (negate dV/dT)
        theta_approx = -vals[4]
        err = abs(theta_approx - gt_greeks['theta'][i]) / abs(gt_greeks['theta'][i]) * 100
        errors.add('theta', err)

        # Rho
        err = abs(vals[5] - gt_greeks['rho'][i]) / abs(gt_greeks['rho'][i]) * 100
        errors.add('rho', err)

    return errors


# ============================================================================
# Build methods
# ============================================================================

def build_barycentric():
    """Build Chebyshev Barycentric approximation."""
    def bs_5d(x, _):
        return BlackScholesCall(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=Q).price()

    cheb = ChebyshevBarycentric(bs_5d, 5, DOMAIN, [11]*5, max_derivative_order=2)
    start = time.perf_counter()
    cheb.build()
    build_time = time.perf_counter() - start
    return cheb, build_time


def build_mocax():
    """Build MoCaX Standard approximation. Returns None if unavailable."""
    try:
        mocax_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocax_lib')
        sys.path.insert(0, mocax_lib_dir)

        from ctypes import CDLL
        import ctypes
        _orig = CDLL.__init__

        def _patched(self, name, mode=ctypes.DEFAULT_MODE, handle=None,
                     use_errno=False, use_last_error=False, winmode=None):
            if isinstance(name, str) and 'libmocaxc.so' in name:
                name = os.path.join(mocax_lib_dir, 'libmocaxc.so')
            _orig(self, name, mode, handle, use_errno, use_last_error, winmode)
        CDLL.__init__ = _patched

        import mocaxpy
        print(f"  MoCaX version: {mocaxpy.get_version_id()}")

        def bs_wrapper(x, _):
            return BlackScholesCall(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=Q).price()

        domain_lists = [list(d) for d in DOMAIN]
        mocax_domain = mocaxpy.MocaxDomain(domain_lists)
        ns = mocaxpy.MocaxNs([11]*5)

        start = time.perf_counter()
        obj = mocaxpy.Mocax(bs_wrapper, 5, mocax_domain, None, ns, max_derivative_order=2)
        build_time = time.perf_counter() - start

        return obj, build_time

    except Exception as e:
        print(f"  MoCaX not available: {e}")
        return None, None


# ============================================================================
# Evaluate methods (accuracy)
# ============================================================================

def evaluate_barycentric(cheb, samples):
    """Evaluate barycentric on all samples. Returns (query_time, approx_values)."""
    n = len(samples)
    results = np.empty((n, 6))

    start = time.perf_counter()
    for i in range(n):
        point = samples[i].tolist()
        vals = cheb.vectorized_eval_multi(point, ALL_DERIV_ORDERS)
        results[i] = vals
    query_time = time.perf_counter() - start

    return query_time, results


def evaluate_mocax(mocax_obj, samples):
    """Evaluate MoCaX on all samples. Returns (query_time, approx_values)."""
    n = len(samples)
    results = np.empty((n, 6))

    # Pre-compute derivative IDs
    deriv_ids = [mocax_obj.get_derivative_id(d) for d in ALL_DERIV_ORDERS]

    start = time.perf_counter()
    for i in range(n):
        point = samples[i].tolist()
        for j, did in enumerate(deriv_ids):
            results[i, j] = mocax_obj.eval(point, did)
    query_time = time.perf_counter() - start

    return query_time, results


# ============================================================================
# Dedicated timing benchmark
# ============================================================================

def benchmark_timing(cheb, mocax_obj, n_timing=1000):
    """Dedicated timing with no error computation overhead."""
    print(f"\n{'='*80}")
    print(f"TIMING BENCHMARK ({n_timing} iterations, no overhead)")
    print(f"{'='*80}")

    samples = generate_samples(n_timing, seed=99)
    zero_deriv = [0, 0, 0, 0, 0]

    # --- Barycentric: price only ---
    # Warmup
    cheb.vectorized_eval(samples[0].tolist(), zero_deriv)
    start = time.perf_counter()
    for i in range(n_timing):
        cheb.vectorized_eval(samples[i].tolist(), zero_deriv)
    bary_price = (time.perf_counter() - start) / n_timing * 1000

    # --- Barycentric: price + 5 Greeks ---
    start = time.perf_counter()
    for i in range(n_timing):
        cheb.vectorized_eval_multi(samples[i].tolist(), ALL_DERIV_ORDERS)
    bary_greeks = (time.perf_counter() - start) / n_timing * 1000

    mocax_price = None
    mocax_greeks = None

    if mocax_obj is not None:
        deriv_ids = [mocax_obj.get_derivative_id(d) for d in ALL_DERIV_ORDERS]
        deriv_price_id = deriv_ids[0]

        # --- MoCaX: price only ---
        # Warmup
        mocax_obj.eval(samples[0].tolist(), deriv_price_id)
        start = time.perf_counter()
        for i in range(n_timing):
            mocax_obj.eval(samples[i].tolist(), deriv_price_id)
        mocax_price = (time.perf_counter() - start) / n_timing * 1000

        # --- MoCaX: price + 5 Greeks ---
        start = time.perf_counter()
        for i in range(n_timing):
            pt = samples[i].tolist()
            for did in deriv_ids:
                mocax_obj.eval(pt, did)
        mocax_greeks = (time.perf_counter() - start) / n_timing * 1000

    # Print results
    print(f"\n{'':24s} {'Barycentric':>14s}", end="")
    if mocax_obj is not None:
        print(f" {'MoCaX Std':>14s} {'Ratio':>8s}", end="")
    print()

    print(f"{'':24s} {'(ms/query)':>14s}", end="")
    if mocax_obj is not None:
        print(f" {'(ms/query)':>14s} {'(B/M)':>8s}", end="")
    print()
    print("-" * (24 + 14 + (24 if mocax_obj else 0)))

    # Price only
    print(f"{'Price only':<24s} {bary_price:>14.4f}", end="")
    if mocax_price is not None:
        ratio = bary_price / mocax_price
        print(f" {mocax_price:>14.4f} {ratio:>7.1f}x", end="")
    print()

    # Price + 5 Greeks
    print(f"{'Price + 5 Greeks':<24s} {bary_greeks:>14.4f}", end="")
    if mocax_greeks is not None:
        ratio = bary_greeks / mocax_greeks
        print(f" {mocax_greeks:>14.4f} {ratio:>7.1f}x", end="")
    print()

    return bary_price, bary_greeks, mocax_price, mocax_greeks


# ============================================================================
# Output tables
# ============================================================================

def print_build_table(bary_time, mocax_time):
    print(f"\n{'='*80}")
    print("BUILD TIME")
    print(f"{'='*80}")
    print(f"{'Method':<24s} {'Time (s)':>10s} {'Grid Points':>14s} {'Notes':<24s}")
    print("-" * 80)
    print(f"{'Chebyshev Barycentric':<24s} {bary_time:>10.3f} {'161,051':>14s} {'Pure Python + NumPy':<24s}")
    if mocax_time is not None:
        print(f"{'MoCaX Standard':<24s} {mocax_time:>10.3f} {'161,051':>14s} {'C++ library':<24s}")
    print(f"{'='*80}")


def print_accuracy_table(bary_errors, mocax_errors):
    print(f"\n{'='*80}")
    print("ACCURACY (relative % error vs analytical Black-Scholes)")
    print(f"{'='*80}")

    print(f"{'Metric':<10s}", end="")
    print(f" {'Barycentric':>24s}", end="")
    if mocax_errors is not None:
        print(f" {'MoCaX Standard':>24s}", end="")
    print()

    print(f"{'':10s}", end="")
    print(f" {'Avg% / Max%':>24s}", end="")
    if mocax_errors is not None:
        print(f" {'Avg% / Max%':>24s}", end="")
    print()
    print("-" * (10 + 24 + (26 if mocax_errors else 0)))

    for metric in METRICS:
        b_avg, b_max = bary_errors.stats(metric)
        print(f"{metric.capitalize():<10s}", end="")
        print(f" {b_avg:>10.4f}% / {b_max:>8.4f}%", end="")

        if mocax_errors is not None:
            m_avg, m_max = mocax_errors.stats(metric)
            print(f" {m_avg:>10.4f}% / {m_max:>8.4f}%", end="")

        print()

    print(f"{'='*80}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("FAIR COMPARISON: Chebyshev Barycentric vs MoCaX Standard")
    print("=" * 80)
    print()
    print("Both methods use:")
    print("  - Same domain: S=[80,120], K=[90,110], T=[0.25,1], sigma=[0.15,0.35], r=[0.01,0.08]")
    print("  - Same Chebyshev grid: 11 nodes per dim (161,051 total)")
    print("  - Same function: Black-Scholes call with q=0.02")
    print("  - Analytical derivatives (no finite differences)")
    print("  - Same random test points")

    n_samples = int(os.environ.get('N_SAMPLES', 200))
    print(f"\nAccuracy test: {n_samples} random samples (set N_SAMPLES to change)")

    # --- Phase 1: Build ---
    print(f"\n{'='*80}")
    print("PHASE 1: Build")
    print(f"{'='*80}")

    print("\nBuilding Chebyshev Barycentric...")
    cheb, bary_build_time = build_barycentric()

    print("\nBuilding MoCaX Standard...")
    mocax_obj, mocax_build_time = build_mocax()

    print_build_table(bary_build_time, mocax_build_time)

    # --- Phase 2: Accuracy ---
    print(f"\n{'='*80}")
    print(f"PHASE 2: Accuracy ({n_samples} random samples)")
    print(f"{'='*80}")

    print("\nGenerating samples and ground truth...")
    samples = generate_samples(n_samples)
    gt_prices, gt_greeks = compute_ground_truth(samples)
    print(f"  Ground truth computed for {n_samples} points")

    print("\nEvaluating Chebyshev Barycentric...")
    bary_time, bary_vals = evaluate_barycentric(cheb, samples)
    bary_errors = compute_errors(bary_vals, gt_prices, gt_greeks)
    print(f"  Done in {bary_time:.3f}s ({bary_time/n_samples*1000:.3f} ms/sample)")

    mocax_errors = None
    if mocax_obj is not None:
        print("\nEvaluating MoCaX Standard...")
        mx_time, mx_vals = evaluate_mocax(mocax_obj, samples)
        mocax_errors = compute_errors(mx_vals, gt_prices, gt_greeks)
        print(f"  Done in {mx_time:.3f}s ({mx_time/n_samples*1000:.3f} ms/sample)")

    print_accuracy_table(bary_errors, mocax_errors)

    # --- Phase 3: Timing ---
    bary_p, bary_g, mocax_p, mocax_g = benchmark_timing(cheb, mocax_obj, n_timing=1000)

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<28s} {'Barycentric':>14s}", end="")
    if mocax_obj is not None:
        print(f" {'MoCaX':>14s} {'Ratio':>8s}", end="")
    print()
    print(f"  {'-'*28} {'-'*14}", end="")
    if mocax_obj is not None:
        print(f" {'-'*14} {'-'*8}", end="")
    print()

    print(f"  {'Build time':<28s} {bary_build_time:>13.3f}s", end="")
    if mocax_build_time is not None:
        r = bary_build_time / mocax_build_time
        print(f" {mocax_build_time:>13.3f}s {r:>7.2f}x", end="")
    print()

    print(f"  {'Query: price only':<28s} {bary_p:>12.4f}ms", end="")
    if mocax_p is not None:
        r = bary_p / mocax_p
        print(f" {mocax_p:>12.4f}ms {r:>7.1f}x", end="")
    print()

    print(f"  {'Query: price + 5 Greeks':<28s} {bary_g:>12.4f}ms", end="")
    if mocax_g is not None:
        r = bary_g / mocax_g
        print(f" {mocax_g:>12.4f}ms {r:>7.1f}x", end="")
    print()

    b_price_avg, b_price_max = bary_errors.stats('price')
    print(f"  {'Price max error':<28s} {b_price_max:>12.4f}%", end="")
    if mocax_errors is not None:
        m_price_avg, m_price_max = mocax_errors.stats('price')
        print(f" {m_price_max:>12.4f}%", end="")
    print()

    # Max Greek error across all Greeks
    b_greek_max = max(bary_errors.stats(m)[1] for m in METRICS if m != 'price')
    print(f"  {'Greek max error':<28s} {b_greek_max:>12.4f}%", end="")
    if mocax_errors is not None:
        m_greek_max = max(mocax_errors.stats(m)[1] for m in METRICS if m != 'price')
        print(f" {m_greek_max:>12.4f}%", end="")
    print()

    print(f"\n  Note: Both methods achieve identical accuracy (same mathematical algorithm)")
    print(f"  Note: Speed difference is Python (NumPy) vs C++ implementation")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
