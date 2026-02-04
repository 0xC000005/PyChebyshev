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

from tqdm import tqdm


def generate_5d_grid_samples(domain_values, n_points_per_dim):
    """
    Generate regular grid samples in 5D parameter space (S, K, T, σ, r).

    Args:
        domain_values: List of [min, max] for each dimension
        n_points_per_dim: Number of points per dimension

    Returns:
        samples: Array of shape (n_points_per_dim^5, 5)
    """
    grids_1d = []
    for d in range(5):
        min_val, max_val = domain_values[d]
        grids_1d.append(np.linspace(min_val, max_val, n_points_per_dim))
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    return np.column_stack([g.ravel() for g in mesh])


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

    # Test cases - comprehensive coverage of 5D parameter space
    # Format: [S, K, T, sigma, r], "Name"
    test_cases = [
        # Vary S (moneyness)
        ([100.0, 100.0, 1.0, 0.25, 0.05], "ATM"),
        ([110.0, 100.0, 1.0, 0.25, 0.05], "ITM"),
        ([90.0, 100.0, 1.0, 0.25, 0.05], "OTM"),
        # Vary T (maturity)
        ([100.0, 100.0, 0.5, 0.25, 0.05], "Short T"),
        ([100.0, 100.0, 0.25, 0.25, 0.05], "Very Short T"),
        # Vary sigma (volatility)
        ([100.0, 100.0, 1.0, 0.15, 0.05], "Low vol"),
        ([100.0, 100.0, 1.0, 0.35, 0.05], "High vol"),
        # Vary r (interest rate)
        ([100.0, 100.0, 1.0, 0.25, 0.01], "Low r"),
        ([100.0, 100.0, 1.0, 0.25, 0.08], "High r"),
        # Corner cases (multiple params vary)
        ([85.0, 105.0, 0.5, 0.20, 0.03], "Corner1"),  # OTM, short, low vol, low r
        ([115.0, 95.0, 0.75, 0.30, 0.07], "Corner2"),  # ITM, med T, high vol, high r
    ]

    # Define Greek derivative indices
    greek_specs = [
        ('Price', [0, 0, 0, 0, 0]),
        ('Delta', [1, 0, 0, 0, 0]),
        ('Gamma', [2, 0, 0, 0, 0]),
        ('Vega',  [0, 0, 0, 1, 0]),
        ('Rho',   [0, 0, 0, 0, 1]),
        ('Vanna', [1, 0, 0, 1, 0]),
        ('Charm', [1, 0, 1, 0, 0]),
        ('Vomma', [0, 0, 0, 2, 0]),
        ('Veta',  [0, 0, 1, 1, 0]),
    ]

    # =========================================================================
    # UNIFIED TABLE: Price + Greeks for ALL Scenarios
    # =========================================================================
    all_errors = []

    for test_point, case_name in test_cases:
        S, K, T, sigma, r = test_point
        opt = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

        print(f"\n{'='*70}")
        print(f"SCENARIO: {case_name} (S={S:.0f}, K={K:.0f}, T={T:.2f}, σ={sigma:.2f}, r={r:.2f})")
        print(f"{'='*70}")
        print(f"{'Metric':<8} {'Exact':>14} {'MoCaX':>14} {'Error%':>10}")
        print("-" * 50)

        # Get analytical values
        exact_values = {
            'Price': opt.price(),
            'Delta': opt.delta(),
            'Gamma': opt.gamma(),
            'Vega': opt.vega(),
            'Rho': opt.rho(),
            'Vanna': opt.vanna(),
            'Charm': -opt.charm(),  # Negated: MoCaX uses opposite sign convention
            'Vomma': opt.vomma(),
            'Veta': opt.veta(),
        }

        for metric_name, deriv in greek_specs:
            exact = exact_values[metric_name]
            derivative_id = mocax_5d.get_derivative_id(deriv)
            approx = mocax_5d.eval(test_point, derivative_id)

            # Compute error
            if abs(exact) > 1e-10:
                err = abs(approx - exact) / abs(exact) * 100
            else:
                err = abs(approx - exact) * 100

            all_errors.append(err)
            print(f"{metric_name:<8} {exact:>14.6f} {approx:>14.6f} {err:>10.3f}%")

    max_price_err = max([all_errors[i*9] for i in range(len(test_cases))])  # Price is index 0 of each scenario
    max_greek_err = max(all_errors)

    print(f"\nMax errors: Price {max_price_err:.3f}%, Greeks {max_greek_err:.3f}%")

    # =========================================================================
    # COMPREHENSIVE GRID EVALUATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPREHENSIVE GRID EVALUATION")
    print(f"{'='*70}")

    # Configuration (allow env var override)
    n_points_per_dim = int(os.environ.get('N_GRID_POINTS', 10))
    total_points = n_points_per_dim ** 5

    print(f"Grid configuration:")
    print(f"  Points per dimension: {n_points_per_dim}")
    print(f"  Total evaluation points: {total_points:,}")
    print(f"  (Set N_GRID_POINTS env var to change)")

    # Generate grid samples
    print(f"\nGenerating {n_points_per_dim}^5 = {total_points:,} grid points...")
    samples = generate_5d_grid_samples(domain_values, n_points_per_dim)

    # Compute ground truth using analytical Black-Scholes
    print(f"Computing ground truth (analytical Black-Scholes)...")
    ground_truth_prices = np.zeros(total_points)
    for i in tqdm(range(total_points), desc="Ground truth", ncols=80):
        S, K, T, sigma, r = samples[i]
        opt = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        ground_truth_prices[i] = opt.price()

    # Evaluate MoCaX on all grid points (sequential - MoCaX is not thread-safe)
    print(f"Evaluating MoCaX approximation on {total_points:,} points...")
    deriv_price = mocax_5d.get_derivative_id([0, 0, 0, 0, 0])
    mocax_prices = np.zeros(total_points)

    start_eval = time.time()
    for i in tqdm(range(total_points), desc="MoCaX eval", ncols=80):
        point = samples[i].tolist()
        mocax_prices[i] = mocax_5d.eval(point, deriv_price)
    eval_time = time.time() - start_eval

    # Compute errors (percentage relative to ground truth)
    errors = np.abs(mocax_prices - ground_truth_prices) / ground_truth_prices * 100

    # Report results
    print(f"\n{'='*70}")
    print("GRID EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Points evaluated:     {total_points:,}")
    print(f"  Total MoCaX time:     {eval_time:.3f} s")
    print(f"  Time per evaluation:  {eval_time/total_points*1000:.4f} ms")
    print(f"  Evals per second:     {total_points/eval_time:,.0f}")
    print(f"")
    print(f"  Mean error:           {np.mean(errors):.6f}%")
    print(f"  Max error:            {np.max(errors):.6f}%")
    print(f"  Std deviation:        {np.std(errors):.6f}%")
    print(f"  Median error:         {np.median(errors):.6f}%")
    print(f"  95th percentile:      {np.percentile(errors, 95):.6f}%")
    print(f"{'='*70}")

    # Print all grid points with error > 1%
    high_error_mask = errors > 1.0
    n_high_error = np.sum(high_error_mask)
    if n_high_error > 0:
        print(f"\nPoints with error > 1%: {n_high_error} ({n_high_error/total_points*100:.2f}%)")
        print(f"{'S':>10} {'K':>10} {'T':>10} {'sigma':>10} {'r':>10} {'Exact':>12} {'MoCaX':>12} {'Error%':>10}")
        print("-" * 96)
        high_error_indices = np.where(high_error_mask)[0]
        for idx in high_error_indices:
            S, K, T, sigma, r = samples[idx]
            print(f"{S:>10.4f} {K:>10.4f} {T:>10.4f} {sigma:>10.4f} {r:>10.4f} "
                  f"{ground_truth_prices[idx]:>12.6f} {mocax_prices[idx]:>12.6f} {errors[idx]:>10.4f}")
    else:
        print(f"\nNo points with error > 1%")

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
