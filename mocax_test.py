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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mocax_lib'))
os.environ['LD_LIBRARY_PATH'] = os.path.join(os.path.dirname(__file__), 'mocax_lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

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

    # Test multiple points
    test_cases = [
        [100.0, 1.0, 0.25],  # ATM
        [120.0, 1.0, 0.25],  # ITM
        [80.0, 1.0, 0.25],   # OTM
        [100.0, 0.5, 0.25],  # ATM, shorter maturity
        [100.0, 1.0, 0.35],  # ATM, higher vol
    ]

    print("\nValidation against analytical Black-Scholes:")
    print(f"{'Case':<8} {'S':>7} {'T':>6} {'σ':>6} {'Analytical':>12} {'MoCaX':>12} {'Error':>10}")
    print("-" * 70)

    max_error = 0.0
    for i, test_point in enumerate(test_cases):
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

        case_name = ["ATM", "ITM", "OTM", "Short T", "High Vol"][i]
        print(f"{case_name:<8} {S:>7.1f} {T_test:>6.2f} {sigma_test:>6.2f} "
              f"{analytical:>12.6f} {mocax_value:>12.6f} {rel_error:>9.3f}%")

    # Test Delta (∂C/∂S) - first derivative w.r.t. S
    print("\nTesting Delta (∂C/∂S) at ATM:")
    test_point = [100.0, 1.0, 0.25]
    option = BlackScholesCall(S=test_point[0], K=K, T=test_point[1], r=r, sigma=test_point[2], q=q)
    analytical_delta = option.delta()

    derivative_id = mocax_bs.get_derivative_id([1, 0, 0])  # d/dS
    mocax_delta = mocax_bs.eval(test_point, derivative_id)

    delta_error = abs(mocax_delta - analytical_delta)
    delta_rel_error = delta_error / analytical_delta * 100

    print(f"  Analytical Delta: {analytical_delta:.6f}")
    print(f"  MoCaX Delta:      {mocax_delta:.6f}")
    print(f"  Relative error:   {delta_rel_error:.3f}%")

    # Test Vega (∂C/∂σ) - first derivative w.r.t. sigma
    print("\nTesting Vega (∂C/∂σ) at ATM:")
    analytical_vega = option.vega()

    derivative_id = mocax_bs.get_derivative_id([0, 0, 1])  # d/dsigma
    mocax_vega = mocax_bs.eval(test_point, derivative_id)

    vega_error = abs(mocax_vega - analytical_vega)
    vega_rel_error = vega_error / analytical_vega * 100

    print(f"  Analytical Vega: {analytical_vega:.6f}")
    print(f"  MoCaX Vega:      {mocax_vega:.6f}")
    print(f"  Relative error:  {vega_rel_error:.3f}%")

    # Benchmark evaluation speed
    print("\nSpeed benchmark (1000 evaluations):")
    n_eval = 1000
    test_point = [100.0, 1.0, 0.25]
    derivative_id = mocax_bs.get_derivative_id([0, 0, 0])

    # MoCaX speed
    start = time.time()
    for _ in range(n_eval):
        _ = mocax_bs.eval(test_point, derivative_id)
    mocax_time = (time.time() - start) / n_eval * 1e6  # microseconds

    # Analytical speed
    start = time.time()
    for _ in range(n_eval):
        option = BlackScholesCall(S=test_point[0], K=K, T=test_point[1], r=r, sigma=test_point[2], q=q)
        _ = option.price()
    analytical_time = (time.time() - start) / n_eval * 1e6  # microseconds

    print(f"  MoCaX evaluation:     {mocax_time:.2f} μs per call")
    print(f"  Analytical formula:   {analytical_time:.2f} μs per call")
    print(f"  Ratio:                {mocax_time/analytical_time:.2f}x")

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

    # Comprehensive test cases covering the parameter space
    test_cases = [
        # [S, K, T, sigma, r] - Description
        ([100.0, 100.0, 1.0, 0.25, 0.05], "Center point (ATM, mid params)"),
        ([110.0, 100.0, 1.0, 0.25, 0.05], "ITM (S > K)"),
        ([90.0, 100.0, 1.0, 0.25, 0.05], "OTM (S < K)"),
        ([100.0, 95.0, 1.0, 0.25, 0.05], "ITM (K < S)"),
        ([100.0, 105.0, 1.0, 0.25, 0.05], "OTM (K > S)"),
        ([100.0, 100.0, 0.5, 0.25, 0.05], "Shorter maturity"),
        ([100.0, 100.0, 0.25, 0.25, 0.05], "Very short maturity"),
        ([100.0, 100.0, 1.0, 0.35, 0.05], "High volatility"),
        ([100.0, 100.0, 1.0, 0.15, 0.05], "Low volatility"),
        ([100.0, 100.0, 1.0, 0.25, 0.08], "High interest rate"),
        ([100.0, 100.0, 1.0, 0.25, 0.01], "Low interest rate"),
        ([115.0, 95.0, 0.75, 0.30, 0.06], "Deep ITM, mixed params"),
        ([85.0, 105.0, 0.5, 0.20, 0.03], "Deep OTM, mixed params"),
        ([105.0, 105.0, 0.25, 0.35, 0.07], "ATM, extreme params"),
    ]

    print("\n" + "="*85)
    print("VALIDATION: Price Accuracy Across 5D Parameter Space")
    print("="*85)
    print(f"{'Case':<10} {'S':>6} {'K':>6} {'T':>5} {'σ':>5} {'r':>5} {'Analytical':>11} {'MoCaX':>11} {'Error':>8}")
    print("-"*85)

    max_error = 0.0
    errors = []

    for test_point, description in test_cases:
        S, K, T, sigma, r = test_point

        # Analytical value
        option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        analytical = option.price()

        # MoCaX evaluation
        derivative_id = mocax_5d.get_derivative_id([0, 0, 0, 0, 0])
        mocax_value = mocax_5d.eval(test_point, derivative_id)

        error = abs(mocax_value - analytical)
        rel_error = error / analytical * 100 if analytical > 0 else 0
        max_error = max(max_error, rel_error)
        errors.append(rel_error)

        # Shorten description for display
        desc_short = description[:10] if len(description) > 10 else description
        print(f"{desc_short:<10} {S:>6.0f} {K:>6.0f} {T:>5.2f} {sigma:>5.2f} {r:>5.2f} "
              f"{analytical:>11.6f} {mocax_value:>11.6f} {rel_error:>7.3f}%")

    print("-"*85)
    print(f"{'Statistics':<10} {'':>6} {'':>6} {'':>5} {'':>5} {'':>5} "
          f"{'Mean error:':>11} {np.mean(errors):>11.3f}% {'Max:':>4} {max_error:>6.3f}%")

    # Test Greeks with respect to multiple parameters
    print("\n" + "="*70)
    print("GREEKS: Analytical Derivatives Across Parameters")
    print("="*70)

    test_point = [100.0, 100.0, 1.0, 0.25, 0.05]
    option = BlackScholesCall(S=test_point[0], K=test_point[1], T=test_point[2],
                              r=test_point[4], sigma=test_point[3], q=q)

    # Delta: ∂V/∂S
    analytical_delta = option.delta()
    derivative_id = mocax_5d.get_derivative_id([1, 0, 0, 0, 0])
    mocax_delta = mocax_5d.eval(test_point, derivative_id)
    delta_error = abs(mocax_delta - analytical_delta) / analytical_delta * 100

    print(f"\nDelta (∂V/∂S) at center point:")
    print(f"  Analytical: {analytical_delta:>10.6f}")
    print(f"  MoCaX:      {mocax_delta:>10.6f}")
    print(f"  Error:      {delta_error:>9.3f}%")

    # Gamma: ∂²V/∂S²
    analytical_gamma = option.gamma()
    derivative_id = mocax_5d.get_derivative_id([2, 0, 0, 0, 0])
    mocax_gamma = mocax_5d.eval(test_point, derivative_id)
    gamma_error = abs(mocax_gamma - analytical_gamma) / analytical_gamma * 100

    print(f"\nGamma (∂²V/∂S²) at center point:")
    print(f"  Analytical: {analytical_gamma:>10.6f}")
    print(f"  MoCaX:      {mocax_gamma:>10.6f}")
    print(f"  Error:      {gamma_error:>9.3f}%")

    # Vega: ∂V/∂σ
    analytical_vega = option.vega()
    derivative_id = mocax_5d.get_derivative_id([0, 0, 0, 1, 0])
    mocax_vega = mocax_5d.eval(test_point, derivative_id)
    vega_error = abs(mocax_vega - analytical_vega) / analytical_vega * 100

    print(f"\nVega (∂V/∂σ) at center point:")
    print(f"  Analytical: {analytical_vega:>10.6f}")
    print(f"  MoCaX:      {mocax_vega:>10.6f}")
    print(f"  Error:      {vega_error:>9.3f}%")

    # Rho: ∂V/∂r
    analytical_rho = option.rho()
    derivative_id = mocax_5d.get_derivative_id([0, 0, 0, 0, 1])
    mocax_rho = mocax_5d.eval(test_point, derivative_id)
    rho_error = abs(mocax_rho - analytical_rho) / analytical_rho * 100

    print(f"\nRho (∂V/∂r) at center point:")
    print(f"  Analytical: {analytical_rho:>10.6f}")
    print(f"  MoCaX:      {mocax_rho:>10.6f}")
    print(f"  Error:      {rho_error:>9.3f}%")

    # Strike sensitivity: ∂V/∂K (not a standard Greek but useful for calibration)
    # Approximate analytical derivative using finite difference
    eps = 0.01
    option_up = BlackScholesCall(S=test_point[0], K=test_point[1]+eps, T=test_point[2],
                                  r=test_point[4], sigma=test_point[3], q=q)
    option_down = BlackScholesCall(S=test_point[0], K=test_point[1]-eps, T=test_point[2],
                                    r=test_point[4], sigma=test_point[3], q=q)
    analytical_dK = (option_up.price() - option_down.price()) / (2*eps)

    derivative_id = mocax_5d.get_derivative_id([0, 1, 0, 0, 0])
    mocax_dK = mocax_5d.eval(test_point, derivative_id)
    dK_error = abs(mocax_dK - analytical_dK) / abs(analytical_dK) * 100

    print(f"\nStrike Sensitivity (∂V/∂K) at center point:")
    print(f"  Numerical:  {analytical_dK:>10.6f}  (finite difference approx.)")
    print(f"  MoCaX:      {mocax_dK:>10.6f}  (analytical from Chebyshev)")
    print(f"  Error:      {dK_error:>9.3f}%")

    # Speed benchmark
    print("\n" + "="*70)
    print("PERFORMANCE: Speed Comparison")
    print("="*70)

    n_eval = 1000
    test_point = [100.0, 100.0, 1.0, 0.25, 0.05]

    # MoCaX speed (function value)
    derivative_id = mocax_5d.get_derivative_id([0, 0, 0, 0, 0])
    start = time.time()
    for _ in range(n_eval):
        _ = mocax_5d.eval(test_point, derivative_id)
    mocax_time = (time.time() - start) / n_eval * 1e6

    # MoCaX speed (all Greeks: Delta, Gamma, Vega, Rho, dK)
    derivative_ids = [
        mocax_5d.get_derivative_id([1, 0, 0, 0, 0]),  # Delta
        mocax_5d.get_derivative_id([2, 0, 0, 0, 0]),  # Gamma
        mocax_5d.get_derivative_id([0, 0, 0, 1, 0]),  # Vega
        mocax_5d.get_derivative_id([0, 0, 0, 0, 1]),  # Rho
        mocax_5d.get_derivative_id([0, 1, 0, 0, 0]),  # dK
    ]
    start = time.time()
    for _ in range(n_eval):
        for deriv_id in derivative_ids:
            _ = mocax_5d.eval(test_point, deriv_id)
    mocax_greeks_time = (time.time() - start) / n_eval * 1e6

    # Analytical speed (function value)
    start = time.time()
    for _ in range(n_eval):
        option = BlackScholesCall(S=test_point[0], K=test_point[1], T=test_point[2],
                                  r=test_point[4], sigma=test_point[3], q=q)
        _ = option.price()
    analytical_time = (time.time() - start) / n_eval * 1e6

    # Analytical speed (all Greeks)
    start = time.time()
    for _ in range(n_eval):
        option = BlackScholesCall(S=test_point[0], K=test_point[1], T=test_point[2],
                                  r=test_point[4], sigma=test_point[3], q=q)
        _ = option.price()
        _ = option.delta()
        _ = option.gamma()
        _ = option.vega()
        _ = option.rho()
        # dK requires finite difference
        option_up = BlackScholesCall(S=test_point[0], K=test_point[1]+0.01, T=test_point[2],
                                      r=test_point[4], sigma=test_point[3], q=q)
        option_down = BlackScholesCall(S=test_point[0], K=test_point[1]-0.01, T=test_point[2],
                                        r=test_point[4], sigma=test_point[3], q=q)
        _ = (option_up.price() - option_down.price()) / 0.02
    analytical_greeks_time = (time.time() - start) / n_eval * 1e6

    print(f"\nSpeed benchmark ({n_eval} evaluations):")
    print(f"  {'Method':<30} {'Price Only':>15} {'Price + 5 Greeks':>20}")
    print(f"  {'-'*30} {'-'*15} {'-'*20}")
    print(f"  {'MoCaX (Chebyshev)':<30} {mocax_time:>12.2f} μs {mocax_greeks_time:>17.2f} μs")
    print(f"  {'Analytical (Black-Scholes)':<30} {analytical_time:>12.2f} μs {analytical_greeks_time:>17.2f} μs")
    print(f"  {'Ratio (MoCaX / Analytical)':<30} {mocax_time/analytical_time:>14.2f}x {mocax_greeks_time/analytical_greeks_time:>18.2f}x")

    print("\nKey observations:")
    print(f"  • Build time: {build_time:.3f}s for {np.prod(n_values):,} evaluations")
    print(f"  • Break-even: ~{int(build_time*1e6 / (analytical_time - mocax_time))} queries")
    print(f"  • Greeks advantage: {analytical_greeks_time/mocax_greeks_time:.1f}× faster with MoCaX for full Greek set")
    print(f"  • Memory: Chebyshev coefficients (compact) vs. grid values")

    # Comparison with linear interpolation approach
    print("\n" + "="*70)
    print("COMPARISON: MoCaX vs Linear Interpolation on Chebyshev Nodes")
    print("="*70)
    print("\nFrom chebyshev_tensor_demo.py results (5D with CP decomposition):")
    print("  Linear interpolation:")
    print("    • Vega error: 3.22% mean, 7.60% max")
    print("    • Rho error:  1.36% mean, 4.28% max")
    print("    • Greeks via finite differences (numerical)")
    print("\n  MoCaX (this test):")
    print(f"    • Vega error: {vega_error:.2f}% (analytical derivative)")
    print(f"    • Rho error:  {rho_error:.2f}% (analytical derivative)")
    print(f"    • Greeks via Chebyshev differentiation (spectral accuracy)")
    print("\n  Advantage: MoCaX provides true Chebyshev polynomial evaluation")
    print("            with analytical derivatives, not piecewise linear interpolation")

    del mocax_5d

    # Success criteria: <1% error on prices, <5% on Greeks
    greek_errors = [delta_error, gamma_error, vega_error, rho_error]
    max_greek_error = max(greek_errors)

    if max_error < 1.0 and max_greek_error < 5.0:
        print(f"\n✓ Test PASSED: Price error {max_error:.3f}%, Greek error {max_greek_error:.3f}%")
        return True
    else:
        print(f"\n✗ Test FAILED: Price error {max_error:.3f}% or Greek error {max_greek_error:.3f}% too high")
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
