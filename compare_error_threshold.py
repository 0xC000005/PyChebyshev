"""Benchmark: PyChebyshev error-threshold ctor vs MoCaX variable-size ctor.

Builds both sides at several target error thresholds and reports:

- Resolved Ns (node counts per dimension) chosen by each library.
- Wall-clock build time.
- Point-wise value agreement on 200 random 5D Black-Scholes samples.

Not part of the CI test suite (imports proprietary MoCaX libs that
are gitignored). Run locally after installing MoCaX under
``mocax_lib/``.

Usage
-----
    uv run python compare_error_threshold.py

NOTE: This script is for local benchmarking only. It is NOT part of the
test suite and is NOT run in CI (requires MoCaX C++ library).
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import norm

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

try:
    import mocaxpy
    print(f"MoCaX version: {mocaxpy.get_version_id()}")
except ImportError as e:
    print(f"ERROR: mocaxpy not importable: {e}")
    print("  Install MoCaX under mocax_lib/ before running this script.")
    sys.exit(1)


# ============================================================================
# Test function: 5D Black-Scholes call price
# ============================================================================

def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """Analytical Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_5d(x: list[float], _: Any) -> float:
    """5D Black-Scholes wrapper: V(S, K, T, sigma, r)."""
    return black_scholes_call(S=x[0], K=x[1], T=x[2], sigma=x[3], r=x[4])


DOMAIN_5D = [
    [80.0, 120.0],   # S: spot price
    [90.0, 110.0],   # K: strike
    [0.25, 1.0],     # T: time to maturity
    [0.15, 0.35],    # sigma: volatility
    [0.01, 0.08],    # r: risk-free rate
]
NUM_SAMPLES = 200


# ============================================================================
# Core comparison
# ============================================================================

def compare_for_threshold(eps: float) -> dict:
    """Build both sides at ``error_threshold=eps``; report resolved Ns,
    runtime, and value diff on NUM_SAMPLES random points.
    """
    # --- PyChebyshev auto-N build ---
    t0 = time.perf_counter()
    py = ChebyshevApproximation(
        bs_5d, 5, DOMAIN_5D, error_threshold=eps, max_n=32,
    )
    py.build(verbose=False)
    py_build_s = time.perf_counter() - t0
    py_ns = list(py.n_nodes)
    py_err = py.error_estimate()

    # --- MoCaX variable-size build ---
    # Mocax(function, num_dims, domain, error_threshold, n, ...).
    # Passing n=None with error_threshold triggers full variable-size
    # auto-calibration (constructor_type 2 in MoCaX internals).
    mocax_domain = mocaxpy.MocaxDomain(DOMAIN_5D)
    t0 = time.perf_counter()
    mocax_obj = mocaxpy.Mocax(bs_5d, 5, mocax_domain, eps, None)
    mocax_build_s = time.perf_counter() - t0

    # get_used_ns() returns a MocaxNs object; export_as_list() gives a
    # Python list of per-dimension node counts.
    mocax_ns_obj = mocax_obj.get_used_ns()
    try:
        mocax_ns = mocax_ns_obj.export_as_list()
    except AttributeError:
        mocax_ns = str(mocax_ns_obj)

    # --- Random-sample value-match on NUM_SAMPLES points ---
    rng = np.random.default_rng(42)
    pts = rng.uniform(
        low=[d[0] for d in DOMAIN_5D],
        high=[d[1] for d in DOMAIN_5D],
        size=(NUM_SAMPLES, 5),
    )
    zero_deriv = [0] * 5
    py_vals = np.array(
        [py.vectorized_eval(p.tolist(), zero_deriv) for p in pts]
    )
    mocax_deriv_id = mocax_obj.get_derivative_id(zero_deriv)
    mocax_vals = np.array(
        [mocax_obj.eval(p.tolist(), mocax_deriv_id) for p in pts]
    )

    max_abs = float(np.max(np.abs(py_vals - mocax_vals)))
    denom = np.abs(py_vals) + 1e-16
    max_rel = float(np.max(np.abs((py_vals - mocax_vals) / denom)))

    return {
        "eps": eps,
        "py_ns": py_ns,
        "mocax_ns": mocax_ns,
        "py_err": py_err,
        "py_build_s": py_build_s,
        "mocax_build_s": mocax_build_s,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    print(f"\n{'=' * 110}")
    print("  PyChebyshev error-threshold ctor vs MoCaX variable-size ctor")
    print(f"{'=' * 110}")
    print(f"  5D Black-Scholes call price; domain = {DOMAIN_5D}")
    print(f"  {NUM_SAMPLES} random sample points per threshold.\n")

    thresholds = [1e-3, 1e-4, 1e-5, 1e-6]
    header = (
        f"{'eps':>8}  {'PyChebyshev Ns':>22}  {'MoCaX Ns':>22}  "
        f"{'max|Δ|':>10}  {'max|Δ|/|val|':>14}  "
        f"{'PY build':>10}  {'MX build':>10}"
    )
    print(header)
    print("-" * len(header))

    any_fail = False
    for eps in thresholds:
        try:
            r = compare_for_threshold(eps)
        except Exception as exc:
            print(f"{eps:8.1e}  FAILED: {exc}")
            any_fail = True
            continue
        print(
            f"{r['eps']:8.1e}  {str(r['py_ns']):>22}  {str(r['mocax_ns']):>22}  "
            f"{r['max_abs_diff']:10.3e}  {r['max_rel_diff']:14.3e}  "
            f"{r['py_build_s']:10.3f}  {r['mocax_build_s']:10.3f}"
        )
        # Tolerance: both sides target `eps`, so disagreement up to
        # ~10*eps is plausible (each side's own error can be ~eps).
        if r["max_abs_diff"] > 10 * eps:
            print(
                f"  WARNING: max_abs_diff {r['max_abs_diff']:.3e} "
                f"exceeds 10 * eps ({10 * eps:.1e}) — investigate."
            )
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
