"""
compare_from_values.py -- PyChebyshev nodes()/from_values() vs MoCaX Extend comparison.

Tests the "nodes first, values later" workflow:
1. PyChebyshev: nodes() -> external eval -> from_values() -> evaluate
2. MoCaX Extend: cheb_pts_per_dimension -> set_subgrid_values() -> run_rank_adaptive_algo()
3. Head-to-head accuracy comparison at 50 random test points + derivatives

Note: MoCaX uses TT compression on a random subgrid, while PyChebyshev uses
the full tensor grid. Expect MoCaX to have slightly larger errors but far fewer
evaluations.

Requires: mocaxextend_lib/ with shared_libs/ (MoCaX portion is skipped if unavailable)

Usage:
    uv run python compare_from_values.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

import math
import os
import sys
import time

import numpy as np
from scipy.stats import norm

from pychebyshev import ChebyshevApproximation

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

DIMENSION = 3
DOMAIN = [
    [80.0, 120.0],   # S: spot price
    [0.25, 1.0],     # T: time to maturity
    [0.15, 0.35],    # sigma: volatility
]
N_NODES = [15, 12, 10]
K, R, Q = 100.0, 0.05, 0.02


# ============================================================================
# Black-Scholes functions
# ============================================================================

def black_scholes_call(S, T, sigma, K=K, r=R, q=Q):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_vectorized(pts):
    """Vectorized BS call for numpy arrays of shape (n, 3)."""
    S, T, sigma = pts[:, 0], pts[:, 1], pts[:, 2]
    results = np.zeros(len(S))
    valid = T > 0
    if np.any(valid):
        Sv, Tv, sv = S[valid], T[valid], sigma[valid]
        d1 = (np.log(Sv / K) + (R - Q + 0.5 * sv**2) * Tv) / (sv * np.sqrt(Tv))
        d2 = d1 - sv * np.sqrt(Tv)
        results[valid] = (
            Sv * np.exp(-Q * Tv) * norm.cdf(d1)
            - K * np.exp(-R * Tv) * norm.cdf(d2)
        )
    return results


def bs_3d(x, _):
    """3D BS wrapper: V(S, T, sigma)."""
    return black_scholes_call(S=x[0], T=x[1], sigma=x[2])


def bs_delta_analytical(S, T, sigma, K=K, r=R, q=Q):
    """Analytical BS Delta."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.cdf(d1)


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


def report(title, labels, py_vals, ref_vals, ref_label="Exact"):
    """Print comparison table."""
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Point':<30} {'PyChebyshev':>15} {ref_label:>15} {'Diff':>12}")
    print(f"  {'-' * 72}")
    max_diff = 0.0
    for i in range(len(py_vals)):
        diff = abs(py_vals[i] - ref_vals[i])
        max_diff = max(max_diff, diff)
        print(f"  {labels[i]:<30} {py_vals[i]:>15.8f} {ref_vals[i]:>15.8f} {diff:>12.2e}")
    print(f"  {'-' * 72}")
    status = "PASS" if max_diff < 1e-6 else "WARN"
    print(f"  Max diff: {max_diff:.2e}  [{status}]")
    return max_diff


# ============================================================================
# Test 1: PyChebyshev from_values() vs build()
# ============================================================================

def test_from_values_vs_build():
    """Verify nodes()+from_values() is bit-identical to build()."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 1: from_values() vs build() — Bit-Identical Check")
    print(f"{'=' * 78}")

    # Path A: traditional build()
    start_a = time.perf_counter()
    cheb_a = ChebyshevApproximation(bs_3d, DIMENSION, DOMAIN, N_NODES)
    cheb_a.build(verbose=False)
    time_a = time.perf_counter() - start_a

    # Path B: nodes() + from_values()
    start_b = time.perf_counter()
    info = ChebyshevApproximation.nodes(DIMENSION, DOMAIN, N_NODES)
    values = black_scholes_call_vectorized(info["full_grid"]).reshape(info["shape"])
    cheb_b = ChebyshevApproximation.from_values(values, DIMENSION, DOMAIN, N_NODES)
    time_b = time.perf_counter() - start_b

    print(f"  build() time:       {time_a:.4f}s")
    print(f"  from_values() time: {time_b:.4f}s (incl. vectorized eval)")
    print(f"  Full grid points:   {info['full_grid'].shape[0]:,}")

    # Compare at 50 random points
    test_pts = generate_samples(DOMAIN, 50)
    max_diff = 0.0
    for pt in test_pts:
        va = cheb_a.vectorized_eval(list(pt), [0, 0, 0])
        vb = cheb_b.vectorized_eval(list(pt), [0, 0, 0])
        max_diff = max(max_diff, abs(va - vb))

    print(f"  Max diff (50 pts):  {max_diff:.2e}")
    status = "PASS" if max_diff == 0.0 else "FAIL"
    print(f"  Bit-identical:      {status}")

    # Check derivative
    deriv_a = cheb_a.vectorized_eval([100.0, 0.5, 0.25], [1, 0, 0])
    deriv_b = cheb_b.vectorized_eval([100.0, 0.5, 0.25], [1, 0, 0])
    print(f"  Delta diff:         {abs(deriv_a - deriv_b):.2e}")

    return cheb_b


# ============================================================================
# Test 2: PyChebyshev from_values() vs exact
# ============================================================================

def test_from_values_accuracy(cheb):
    """Check from_values() accuracy vs analytical Black-Scholes."""
    print(f"\n{'=' * 78}")
    print(f"  TEST 2: from_values() Accuracy vs Analytical BS")
    print(f"{'=' * 78}")

    test_pts = generate_samples(DOMAIN, 20, seed=99)
    py_vals = []
    exact_vals = []
    labels = []

    for pt in test_pts:
        py_vals.append(cheb.vectorized_eval(list(pt), [0, 0, 0]))
        exact_vals.append(black_scholes_call(pt[0], pt[1], pt[2]))
        labels.append(f"S={pt[0]:.1f} T={pt[1]:.2f} s={pt[2]:.3f}")

    report("Price: from_values() vs Analytical", labels, py_vals, exact_vals)

    # Delta
    print(f"\n  --- Delta (dV/dS) at 10 points ---")
    py_deltas = []
    exact_deltas = []
    for pt in test_pts[:10]:
        py_deltas.append(cheb.vectorized_eval(list(pt), [1, 0, 0]))
        exact_deltas.append(bs_delta_analytical(pt[0], pt[1], pt[2]))

    report(
        "Delta: from_values() vs Analytical",
        labels[:10], py_deltas, exact_deltas,
    )


# ============================================================================
# Test 3: PyChebyshev from_values() vs MoCaX Extend
# ============================================================================

def test_vs_mocax():
    """Compare PyChebyshev from_values() vs MoCaX Extend."""
    if not HAS_MOCAX:
        print(f"\n{'=' * 78}")
        print("  TEST 3: SKIPPED (MoCaX Extend not available)")
        print(f"{'=' * 78}")
        return

    print(f"\n{'=' * 78}")
    print(f"  TEST 3: PyChebyshev from_values() vs MoCaX Extend")
    print(f"{'=' * 78}")

    # --- PyChebyshev: nodes() + vectorized eval + from_values() ---
    start_py = time.perf_counter()
    info = ChebyshevApproximation.nodes(DIMENSION, DOMAIN, N_NODES)
    values = black_scholes_call_vectorized(info["full_grid"]).reshape(info["shape"])
    cheb = ChebyshevApproximation.from_values(values, DIMENSION, DOMAIN, N_NODES)
    time_py = time.perf_counter() - start_py
    n_evals_py = info["full_grid"].shape[0]

    # --- MoCaX Extend: TT decomposition ---
    start_mx = time.perf_counter()
    mx_obj = me.MocaxExtend(DIMENSION, N_NODES, DOMAIN)
    random_pts = mx_obj.subgrid_by_number(5000)
    mx_vals = black_scholes_call_vectorized(random_pts)
    mx_obj.set_subgrid_values(mx_vals)
    mx_obj.gen_train_val_data()
    mx_obj.run_rank_adaptive_algo(
        tolerance=1e-3, rel_tolerance=1e-8, max_iters=100,
        max_rank=20, print_progress=False, max_rounds=5,
    )
    time_mx = time.perf_counter() - start_mx
    n_evals_mx = len(random_pts)

    print(f"  PyChebyshev: {n_evals_py:,} evals, {time_py:.4f}s")
    print(f"  MoCaX:       {n_evals_mx:,} evals, {time_mx:.4f}s")
    print(f"  Grid type:   PyChebyshev=full tensor, MoCaX=random subgrid")

    # Compare at 50 random test points
    test_pts = generate_samples(DOMAIN, 50, seed=123)
    py_vals = []
    mx_vals_list = []
    exact_vals = []
    labels = []

    for pt in test_pts:
        py_vals.append(cheb.vectorized_eval(list(pt), [0, 0, 0]))
        mx_vals_list.append(mx_obj.eval(list(pt), [0, 0, 0]))
        exact_vals.append(black_scholes_call(pt[0], pt[1], pt[2]))
        labels.append(f"S={pt[0]:.1f} T={pt[1]:.2f} s={pt[2]:.3f}")

    py_errs = [abs(py_vals[i] - exact_vals[i]) for i in range(50)]
    mx_errs = [abs(mx_vals_list[i] - exact_vals[i]) for i in range(50)]

    print(f"\n  PyChebyshev max error: {max(py_errs):.2e}")
    print(f"  MoCaX max error:      {max(mx_errs):.2e}")
    print(f"  PyChebyshev mean err: {np.mean(py_errs):.2e}")
    print(f"  MoCaX mean error:     {np.mean(mx_errs):.2e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  compare_from_values.py")
    print("  PyChebyshev nodes()/from_values() vs MoCaX Extend")
    print("  3D Black-Scholes: V(S, T, sigma),  K=100, r=0.05, q=0.02")
    print(f"  Nodes: {N_NODES}")
    print("=" * 78)

    cheb = test_from_values_vs_build()
    test_from_values_accuracy(cheb)
    test_vs_mocax()

    print(f"\n{'=' * 78}")
    print("  Done.")
    print(f"{'=' * 78}")
