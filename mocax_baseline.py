"""
Test script for MoCaX installation with Black-Scholes option pricing.

This script demonstrates:
1. Basic MoCaX functionality with a simple function
2. Using MoCaX to accelerate Black-Scholes option pricing
3. Comparison with analytical formulas
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
# This must be done before importing mocaxpy
from ctypes import CDLL, RTLD_GLOBAL
import ctypes

# Save original LoadLibrary
_original_cdll_init = CDLL.__init__

def _patched_cdll_init(self, name, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=False, use_last_error=False, winmode=None):
    # If loading libmocaxc.so without a path, use our local copy
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


def test_simple_3d_function():
    """Test MoCaX with a simple 3D function."""
    print("\n" + "="*70)
    print("TEST 1: Simple 3D Function (sin(x) + sin(y) + sin(z))")
    print("="*70)

    # Define simple test function
    def test_function(x, additional_data):
        return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

    # MoCaX setup
    num_dimensions = 3
    domain_values = [[-1.0, 1.0], [-1.0, 1.0], [1.0, 3.0]]
    domain = mocaxpy.MocaxDomain(domain_values)

    # Accuracy parameters
    n_values = [10, 8, 4]
    ns = mocaxpy.MocaxNs(n_values)
    max_derivative_order = 2

    # Build MoCaX approximation
    print("\nBuilding MoCaX approximation...")
    start = time.time()
    mocax_obj = mocaxpy.Mocax(test_function, num_dimensions, domain,
                              None, ns, max_derivative_order=max_derivative_order)
    build_time = time.time() - start
    print(f"  Build time: {build_time*1000:.2f} ms")

    # Test evaluation
    test_point = [0.1, 0.3, 1.7]

    # Original function
    original_value = test_function(test_point, None)

    # MoCaX evaluation
    derivative_id = mocax_obj.get_derivative_id([0, 0, 0])
    mocax_value = mocax_obj.eval(test_point, derivative_id)

    error = abs(mocax_value - original_value)
    rel_error = error / abs(original_value) * 100

    print(f"\nResults at point {test_point}:")
    print(f"  Original function:  {original_value:.10f}")
    print(f"  MoCaX approximation: {mocax_value:.10f}")
    print(f"  Absolute error:     {error:.2e}")
    print(f"  Relative error:     {rel_error:.4f}%")

    # Test derivative
    derivative_id = mocax_obj.get_derivative_id([0, 1, 0])
    derivative_value = mocax_obj.eval(test_point, derivative_id)
    expected_derivative = math.cos(test_point[1])
    deriv_error = abs(derivative_value - expected_derivative)

    print(f"\nDerivative df/dy:")
    print(f"  Expected: {expected_derivative:.10f}")
    print(f"  MoCaX:    {derivative_value:.10f}")
    print(f"  Error:    {deriv_error:.2e}")

    del mocax_obj

    if rel_error < 1.0:
        print("\n✓ Test PASSED: MoCaX approximation is accurate")
        return True
    else:
        print(f"\n✗ Test FAILED: Error {rel_error:.4f}% exceeds 1% threshold")
        return False


def test_black_scholes_call():
    """Test MoCaX with Black-Scholes call option pricing."""
    print("\n" + "="*70)
    print("TEST 2: Black-Scholes Call Option with MoCaX")
    print("="*70)

    # Fixed parameters
    K = 100.0      # Strike price
    r = 0.05       # Risk-free rate
    q = 0.02       # Dividend yield
    sigma = 0.25   # Volatility
    T = 1.0        # Time to maturity

    # Function to approximate: C(S, T, sigma) for varying S, T, sigma
    def bs_call_wrapper(x, additional_data):
        """
        Wrapper for Black-Scholes call option.
        x[0] = S (spot price)
        x[1] = T (time to maturity)
        x[2] = sigma (volatility)
        """
        S, T, sigma = x[0], x[1], x[2]
        # Use blackscholes library for exact calculation
        option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        return option.price()

    # Define domain for (S, T, sigma)
    num_dimensions = 3
    domain_values = [
        [50.0, 150.0],    # S: spot price range
        [0.1, 2.0],       # T: time to maturity
        [0.1, 0.5]        # sigma: volatility
    ]
    domain = mocaxpy.MocaxDomain(domain_values)

    # Accuracy parameters (higher for financial accuracy)
    n_values = [15, 12, 10]  # Higher accuracy
    ns = mocaxpy.MocaxNs(n_values)
    max_derivative_order = 2  # Need up to 2nd derivatives for Greeks

    print("\nBuilding MoCaX approximation for Black-Scholes...")
    print(f"  Domain: S ∈ [50, 150], T ∈ [0.1, 2.0], σ ∈ [0.1, 0.5]")
    print(f"  Chebyshev nodes: {n_values}")

    start = time.time()
    mocax_bs = mocaxpy.Mocax(bs_call_wrapper, num_dimensions, domain,
                             None, ns, max_derivative_order=max_derivative_order)
    build_time = time.time() - start
    print(f"  Build time: {build_time*1000:.2f} ms")

    # Test cases (matching chebyshev_barycentric.py)
    test_cases = [
        ([100.0, 1.0, 0.25], "ATM"),
        ([120.0, 1.0, 0.25], "ITM"),
        ([80.0, 1.0, 0.25], "OTM"),
    ]

    print(f"\n{'Case':<6} {'Price (Exact)':>13} {'Price (MoCaX)':>13} {'Error':>8}")
    print("-" * 50)

    max_error = 0.0
    for test_point, case_name in test_cases:
        S, T_test, sigma_test = test_point

        # Analytical value
        option = BlackScholesCall(S=S, K=K, T=T_test, r=r, sigma=sigma_test, q=q)
        analytical = option.price()

        # MoCaX evaluation
        derivative_id = mocax_bs.get_derivative_id([0, 0, 0])
        mocax_value = mocax_bs.eval(test_point, derivative_id)

        error = abs(mocax_value - analytical)
        rel_error = error / analytical * 100
        max_error = max(max_error, rel_error)

        print(f"{case_name:<6} {analytical:>13.6f} {mocax_value:>13.6f} {rel_error:>7.3f}%")

    # Delta at ATM
    p = [100.0, 1.0, 0.25]
    opt = BlackScholesCall(S=p[0], K=K, T=p[1], r=r, sigma=p[2], q=q)
    delta_exact = opt.delta()
    delta_approx = mocax_bs.eval(p, mocax_bs.get_derivative_id([1, 0, 0]))
    delta_err = abs(delta_approx - delta_exact) / delta_exact * 100

    print(f"\nDelta at ATM:")
    print(f"  Exact:     {delta_exact:.6f}")
    print(f"  MoCaX:     {delta_approx:.6f}")
    print(f"  Error:     {delta_err:.3f}%")

    del mocax_bs

    if max_error < 0.5:  # Less than 0.5% error
        print(f"\n✓ Test PASSED: Maximum error {max_error:.3f}% is acceptable")
        return True
    else:
        print(f"\n✗ Test FAILED: Maximum error {max_error:.3f}% exceeds 0.5% threshold")
        return False


def test_5d_parametric_black_scholes():
    """
    Test MoCaX with 5D parametric Black-Scholes: V(S, K, T, sigma, r).

    This is the challenging case where our tensor interpolation demo
    was forced to use linear interpolation on Chebyshev nodes.
    MoCaX uses true Chebyshev polynomial evaluation with analytical derivatives.
    """
    print("\n" + "="*70)
    print("TEST 3: 5D Parametric Black-Scholes (S, K, T, σ, r)")
    print("="*70)
    print("\nThis test demonstrates MoCaX on the difficult multi-dimensional case")
    print("where we previously had to fall back to linear interpolation.")

    # Fixed dividend yield
    q = 0.02

    # 5D wrapper function: V(S, K, T, sigma, r)
    def bs_5d_wrapper(x, additional_data):
        """
        Full parametric Black-Scholes call option.
        x[0] = S (spot price)
        x[1] = K (strike price)
        x[2] = T (time to maturity)
        x[3] = sigma (volatility)
        x[4] = r (risk-free rate)
        """
        S, K, T, sigma, r = x[0], x[1], x[2], x[3], x[4]
        option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        return option.price()

    # Define 5D domain matching our tensor interpolation demo
    num_dimensions = 5
    domain_values = [
        [80.0, 120.0],    # S: spot price range
        [90.0, 110.0],    # K: strike price range
        [0.25, 1.0],      # T: time to maturity
        [0.15, 0.35],     # sigma: volatility
        [0.01, 0.08]      # r: risk-free rate
    ]
    domain = mocaxpy.MocaxDomain(domain_values)

    # Accuracy parameters for 5D
    # Use fewer nodes per dimension due to curse of dimensionality
    # Total nodes = 11^5 = 161,051 function evaluations
    n_values = [11, 11, 11, 11, 11]
    ns = mocaxpy.MocaxNs(n_values)
    max_derivative_order = 2  # Need 2nd derivatives for Gamma

    print(f"\nBuilding 5D MoCaX approximation...")
    print(f"  Dimensions: 5 (S, K, T, σ, r)")
    print(f"  Domain: S ∈ {domain_values[0]}, K ∈ {domain_values[1]}, T ∈ {domain_values[2]}")
    print(f"          σ ∈ {domain_values[3]}, r ∈ {domain_values[4]}")
    print(f"  Chebyshev nodes per dimension: {n_values}")
    print(f"  Total function evaluations: {np.prod(n_values):,}")

    start = time.time()
    mocax_5d = mocaxpy.Mocax(bs_5d_wrapper, num_dimensions, domain,
                             None, ns, max_derivative_order=max_derivative_order)
    build_time = time.time() - start
    print(f"  Build time: {build_time:.3f} s ({build_time*1000:.1f} ms)")
    print(f"  Evaluations per second: {np.prod(n_values)/build_time:,.0f}")

    # Test cases (matching chebyshev_barycentric.py)
    test_cases = [
        ([100.0, 100.0, 1.0, 0.25, 0.05], "ATM"),
        ([110.0, 100.0, 1.0, 0.25, 0.05], "ITM"),
        ([90.0, 100.0, 1.0, 0.25, 0.05], "OTM"),
        ([100.0, 100.0, 0.5, 0.25, 0.05], "Short T"),
        ([100.0, 100.0, 1.0, 0.35, 0.05], "High vol"),
    ]

    print(f"\n{'Case':<10} {'Price (Exact)':>13} {'Price (MoCaX)':>13} {'Error':>8}")
    print("-" * 50)

    errors = []
    for test_point, case_name in test_cases:
        S, K, T, sigma, r = test_point

        # Analytical value
        option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        analytical = option.price()

        # MoCaX evaluation
        derivative_id = mocax_5d.get_derivative_id([0, 0, 0, 0, 0])
        mocax_value = mocax_5d.eval(test_point, derivative_id)

        error = abs(mocax_value - analytical)
        rel_error = error / analytical * 100
        errors.append(rel_error)

        print(f"{case_name:<10} {analytical:>13.6f} {mocax_value:>13.6f} {rel_error:>7.3f}%")

    max_price_err = max(errors)

    # Greeks at ATM
    p = [100.0, 100.0, 1.0, 0.25, 0.05]
    opt = BlackScholesCall(S=p[0], K=p[1], T=p[2], r=p[4], sigma=p[3], q=q)

    greeks = {
        'Delta': ([1, 0, 0, 0, 0], opt.delta()),
        'Gamma': ([2, 0, 0, 0, 0], opt.gamma()),
        'Vega': ([0, 0, 0, 1, 0], opt.vega()),
        'Rho': ([0, 0, 0, 0, 1], opt.rho()),
    }

    print(f"\nGreeks at ATM:")
    print(f"{'Greek':<8} {'Exact':>12} {'MoCaX':>12} {'Error':>8}")
    print("-" * 50)

    greek_errors = []
    for name, (deriv, exact) in greeks.items():
        derivative_id = mocax_5d.get_derivative_id(deriv)
        approx = mocax_5d.eval(p, derivative_id)
        err = abs(approx - exact) / exact * 100
        greek_errors.append(err)
        print(f"{name:<8} {exact:>12.6f} {approx:>12.6f} {err:>7.3f}%")

    max_greek_err = max(greek_errors)

    print(f"\nMax errors: Price {max_price_err:.3f}%, Greeks {max_greek_err:.3f}%")

    del mocax_5d

    if max_price_err < 1.0 and max_greek_err < 10.0:
        print(f"\n✓ Test PASSED")
        return True
    else:
        print(f"\n✗ Test FAILED")
        return False


def main():
    """Run all MoCaX tests."""
    print("="*70)
    print("MoCaX Installation and Integration Test Suite")
    print("="*70)

    results = []

    # Test 1: Simple function
    try:
        results.append(("Simple 3D Function", test_simple_3d_function()))
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Simple 3D Function", False))

    # Test 2: Black-Scholes 3D
    try:
        results.append(("Black-Scholes Call (3D)", test_black_scholes_call()))
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Black-Scholes Call (3D)", False))

    # Test 3: 5D Parametric Black-Scholes
    try:
        results.append(("5D Parametric Black-Scholes", test_5d_parametric_black_scholes()))
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("5D Parametric Black-Scholes", False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<30} {status}")

    all_passed = all(result[1] for result in results)
    print("="*70)
    if all_passed:
        print("✓ All tests PASSED - MoCaX is ready to use!")
        return 0
    else:
        print("✗ Some tests FAILED - please review the output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
