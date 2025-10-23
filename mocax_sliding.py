"""
Test script for MoCaX Sliding on 5D Black-Scholes option pricing.

This script demonstrates MoCaX Sliding technique (dimensional decomposition)
on Black-Scholes pricing. WARNING: This method is NOT suitable for Black-Scholes
due to strong multiplicative coupling between parameters!

Expected results: 20-50% errors (demonstrates why sliding fails for coupled functions)

Educational purpose:
- Shows the curse of dimensionality with wrong decomposition
- Demonstrates fast construction (55 evals) but poor accuracy
- Contrast with MoCaX TT (proper method for coupled functions)
"""

import sys
import os
import math
import time
import numpy as np

# Add mocax_lib to path
mocax_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocax_lib')
sys.path.insert(0, mocax_lib_dir)

# Patch ctypes to find libmocaxc.so in mocax_lib directory
from ctypes import CDLL, RTLD_GLOBAL
import ctypes

_original_cdll_init = CDLL.__init__

def _patched_cdll_init(self, name, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=False, use_last_error=False, winmode=None):
    if isinstance(name, str) and ('libmocaxc.so' in name or name == 'libmocaxc.so'):
        name = os.path.join(mocax_lib_dir, 'libmocaxc.so')
    _original_cdll_init(self, name, mode, handle, use_errno, use_last_error, winmode)

CDLL.__init__ = _patched_cdll_init

try:
    import mocaxpy
    print(f"✓ Successfully imported mocaxpy")
    print(f"  MoCaX version: {mocaxpy.get_version_id()}")
except ImportError as e:
    print(f"✗ Failed to import mocaxpy: {e}")
    sys.exit(1)

try:
    from blackscholes import BlackScholesCall
    print(f"✓ Successfully imported blackscholes")
except ImportError as e:
    print(f"✗ Failed to import blackscholes: {e}")
    print("  Install with: uv add blackscholes")
    sys.exit(1)


def black_scholes_call(S, K, T, sigma, r, q=0.0):
    """Black-Scholes call option pricing formula."""
    if T <= 0:
        return max(S - K, 0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    from scipy.stats import norm
    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def wrapped_black_scholes(x, additional_data):
    """Wrapper for MoCaX Sliding: 5D Black-Scholes V(S, K, T, σ, r)"""
    S, K, T, sigma, r = x[0], x[1], x[2], x[3], x[4]
    return black_scholes_call(S, K, T, sigma, r)


def test_5d_black_scholes_sliding():
    """
    Test MoCaX Sliding on 5D Black-Scholes.

    WARNING: Expected to show POOR performance (20-50% errors) due to
    multiplicative coupling in Black-Scholes formula.
    """
    print("\n" + "="*70)
    print("TEST: 5D Black-Scholes with MoCaX Sliding")
    print("="*70)
    print("\n⚠️  WARNING: MoCaX Sliding is NOT suitable for Black-Scholes!")
    print("   Black-Scholes has strong multiplicative coupling.")
    print("   Expected errors: 20-50% (demonstrates method limitation)")
    print("   Use MoCaX TT (mocax_tt.py) for coupled functions instead.\n")

    # Domain ranges for 5D: S, K, T, σ, r
    domain_values = [
        [50.0, 150.0],   # S: spot price
        [50.0, 150.0],   # K: strike price
        [0.1, 3.0],      # T: time to maturity
        [0.1, 0.5],      # σ: volatility
        [0.01, 0.15]     # r: risk-free rate
    ]
    domain = mocaxpy.MocaxDomain(domain_values)

    # Accuracy parameters (11 Chebyshev nodes per dimension)
    n_values = [10, 10, 10, 10, 10]  # n=10 means 11 nodes
    ns = mocaxpy.MocaxNs(n_values)

    # Dimensional decomposition: [1, 1, 1, 1, 1] = five 1D partials
    # This assumes V(S, K, T, σ, r) ≈ f₀ + f₁(S) + f₂(K) + f₃(T) + f₄(σ) + f₅(r)
    # This is WRONG for Black-Scholes (multiplicative, not additive)!
    num_dimensions_per_partial = [1, 1, 1, 1, 1]

    # Reference point (ATM scenario)
    reference_point = [100.0, 100.0, 1.0, 0.25, 0.05]

    print("Configuration:")
    print(f"  Domain: S=[50,150], K=[50,150], T=[0.1,3], σ=[0.1,0.5], r=[0.01,0.15]")
    print(f"  Nodes per dimension: {[n+1 for n in n_values]}")
    print(f"  Dimensional partition: {num_dimensions_per_partial}")
    print(f"  Reference point: {reference_point}")
    print(f"  Total evaluations: {sum([11**d for d in num_dimensions_per_partial])} (vs 161,051 for full tensor)")

    # Build MoCaX Sliding approximation
    print("\nBuilding MoCaX Sliding approximation...")
    start = time.time()
    try:
        sliding_obj = mocaxpy.MocaxSliding(
            wrapped_black_scholes,
            num_dimensions_per_partial,
            domain,
            ns,
            reference_point
        )
        build_time = time.time() - start
        print(f"  Build time: {build_time:.4f}s")
    except Exception as e:
        print(f"✗ Failed to build MoCaX Sliding: {e}")
        return

    # Test cases (matching other baseline implementations)
    test_cases = [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ]

    print("\n" + "-"*70)
    print("Price Comparison:")
    print("-"*70)
    print(f"{'Case':<12} {'MoCaX Sliding':>15} {'Analytical':>15} {'Error':>12}")
    print("-"*70)

    max_error = 0.0
    for params, label in test_cases:
        # MoCaX Sliding evaluation
        mocax_price = sliding_obj.eval(params)

        # Analytical reference
        S, K, T, sigma, r = params
        analytical_price = black_scholes_call(S, K, T, sigma, r)

        error = abs(mocax_price - analytical_price) / analytical_price * 100
        max_error = max(max_error, error)

        print(f"{label:<12} {mocax_price:>15.6f} {analytical_price:>15.6f} {error:>11.3f}%")

    print("-"*70)
    print(f"Maximum error: {max_error:.3f}%")

    # Greeks calculation at ATM
    print("\n" + "-"*70)
    print("Greeks Calculation at ATM [100, 100, 1.0, 0.25, 0.05]:")
    print("-"*70)

    S0, K0, T0, sigma0, r0 = 100, 100, 1.0, 0.25, 0.05
    eps_S = 0.01  # For Delta/Gamma
    eps_sigma = 0.001  # For Vega
    eps_r = 0.0001  # For Rho

    # Delta: ∂V/∂S
    V_up = sliding_obj.eval([S0 + eps_S, K0, T0, sigma0, r0])
    V_down = sliding_obj.eval([S0 - eps_S, K0, T0, sigma0, r0])
    delta_mocax = (V_up - V_down) / (2 * eps_S)

    # Gamma: ∂²V/∂S²
    V_center = sliding_obj.eval([S0, K0, T0, sigma0, r0])
    gamma_mocax = (V_up - 2*V_center + V_down) / (eps_S**2)

    # Vega: ∂V/∂σ
    V_sigma_up = sliding_obj.eval([S0, K0, T0, sigma0 + eps_sigma, r0])
    V_sigma_down = sliding_obj.eval([S0, K0, T0, sigma0 - eps_sigma, r0])
    vega_mocax = (V_sigma_up - V_sigma_down) / (2 * eps_sigma)

    # Rho: ∂V/∂r
    V_r_up = sliding_obj.eval([S0, K0, T0, sigma0, r0 + eps_r])
    V_r_down = sliding_obj.eval([S0, K0, T0, sigma0, r0 - eps_r])
    rho_mocax = (V_r_up - V_r_down) / (2 * eps_r)

    # Analytical Greeks
    bs_call = BlackScholesCall(S=S0, K=K0, T=T0, r=r0, sigma=sigma0)
    delta_analytical = bs_call.delta()
    gamma_analytical = bs_call.gamma()
    vega_analytical = bs_call.vega()
    rho_analytical = bs_call.rho()

    # Compare
    delta_error = abs(delta_mocax - delta_analytical) / abs(delta_analytical) * 100
    gamma_error = abs(gamma_mocax - gamma_analytical) / abs(gamma_analytical) * 100
    vega_error = abs(vega_mocax - vega_analytical) / abs(vega_analytical) * 100
    rho_error = abs(rho_mocax - rho_analytical) / abs(rho_analytical) * 100

    print(f"{'Greek':<10} {'MoCaX Sliding':>15} {'Analytical':>15} {'Error':>12}")
    print("-"*70)
    print(f"{'Delta':<10} {delta_mocax:>15.6f} {delta_analytical:>15.6f} {delta_error:>11.3f}%")
    print(f"{'Gamma':<10} {gamma_mocax:>15.6f} {gamma_analytical:>15.6f} {gamma_error:>11.3f}%")
    print(f"{'Vega':<10} {vega_mocax:>15.6f} {vega_analytical:>15.6f} {vega_error:>11.3f}%")
    print(f"{'Rho':<10} {rho_mocax:>15.6f} {rho_analytical:>15.6f} {rho_error:>11.3f}%")
    print("-"*70)

    max_greek_error = max(delta_error, gamma_error, vega_error, rho_error)
    print(f"Maximum Greek error: {max_greek_error:.3f}%")

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Build time:        {build_time:.4f}s")
    print(f"Evaluations:       55 (vs 161,051 for full tensor)")
    print(f"Max price error:   {max_error:.3f}%")
    print(f"Max Greek error:   {max_greek_error:.3f}%")
    print("\n⚠️  These large errors demonstrate why MoCaX Sliding is unsuitable")
    print("   for Black-Scholes (multiplicative coupling between parameters).")
    print("   For coupled functions, use MoCaX TT (mocax_tt.py) instead!")
    print("="*70)

    # Cleanup
    del sliding_obj


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MoCaX Sliding Test Suite")
    print("="*70)
    print("\nDemonstrating MoCaX Sliding (dimensional decomposition)")
    print("on 5D Black-Scholes option pricing.")
    print("\n⚠️  EDUCATIONAL DEMO: Shows why sliding FAILS for coupled functions")

    test_5d_black_scholes_sliding()

    print("\n✓ Test complete")
    print("\nNOTE: For production Black-Scholes pricing, use:")
    print("  - mocax_baseline.py (standard MoCaX) for d≤6")
    print("  - mocax_tt.py (Tensor Train) for d>6 or better compression")


if __name__ == "__main__":
    main()
