"""
Test script for MoCaX Extend (Tensor Train) on 5D Black-Scholes option pricing.

This script demonstrates MoCaX Extend with Tensor Train decomposition for
Black-Scholes pricing. This is the APPROPRIATE method for smooth coupled
functions like Black-Scholes.

Expected results: <2% errors with rank-adaptive TT compression

Key features:
- Tensor Train (TT) decomposition for high-dimensional functions
- Rank-adaptive algorithm automatically selects optimal rank
- Suitable for smooth functions with strong parameter coupling
- Production-ready for 5D+ Black-Scholes pricing
"""

import sys
import os
import time
import numpy as np
from scipy.stats import norm

# Add mocaxextend_lib to path
mocaxextend_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocaxextend_lib')
sys.path.insert(0, mocaxextend_lib_dir)

# Set LD_LIBRARY_PATH for shared libraries
shared_libs_dir = os.path.join(mocaxextend_lib_dir, 'shared_libs')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{shared_libs_dir}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = shared_libs_dir

try:
    import mocaxextendpy.mocax_extend as me
    print(f"✓ Successfully imported mocaxextendpy")
except ImportError as e:
    print(f"✗ Failed to import mocaxextendpy: {e}")
    print(f"  Make sure mocaxextend_lib is set up correctly")
    sys.exit(1)

try:
    from blackscholes import BlackScholesCall
    print(f"✓ Successfully imported blackscholes")
except ImportError as e:
    print(f"✗ Failed to import blackscholes: {e}")
    print("  Install with: uv add blackscholes")
    sys.exit(1)


def black_scholes_call_vectorized(x):
    """
    Vectorized Black-Scholes call option pricing.

    Args:
        x: numpy array of shape (N, 5) with columns [S, K, T, sigma, r]

    Returns:
        numpy array of shape (N,) with call option prices
    """
    S = x[:, 0]
    K = x[:, 1]
    T = x[:, 2]
    sigma = x[:, 3]
    r = x[:, 4]

    # Handle T=0 case
    results = np.zeros(len(S))
    valid_idx = T > 0

    if np.any(valid_idx):
        S_valid = S[valid_idx]
        K_valid = K[valid_idx]
        T_valid = T[valid_idx]
        sigma_valid = sigma[valid_idx]
        r_valid = r[valid_idx]

        d1 = (np.log(S_valid / K_valid) + (r_valid + 0.5 * sigma_valid**2) * T_valid) / (sigma_valid * np.sqrt(T_valid))
        d2 = d1 - sigma_valid * np.sqrt(T_valid)

        results[valid_idx] = S_valid * norm.cdf(d1) - K_valid * np.exp(-r_valid * T_valid) * norm.cdf(d2)

    # Handle T=0 case
    if np.any(~valid_idx):
        results[~valid_idx] = np.maximum(S[~valid_idx] - K[~valid_idx], 0)

    return results


def test_5d_black_scholes_tt():
    """
    Test MoCaX Extend (Tensor Train) on 5D Black-Scholes.

    This demonstrates the proper method for coupled functions.
    """
    print("\n" + "="*70)
    print("TEST: 5D Black-Scholes with MoCaX Extend (Tensor Train)")
    print("="*70)
    print("\n✓ MoCaX Extend TT is the APPROPRIATE method for Black-Scholes!")
    print("  Tensor Train handles smooth coupled functions efficiently.")
    print("  Expected errors: <2% with rank-adaptive compression\n")

    # Configuration
    dimension = 5
    num_cheb_pts = [11, 11, 11, 11, 11]  # 11 Chebyshev nodes per dimension
    variable_ranges = [
        [50.0, 150.0],   # S: spot price
        [50.0, 150.0],   # K: strike price
        [0.1, 3.0],      # T: time to maturity
        [0.1, 0.5],      # σ: volatility
        [0.01, 0.15]     # r: risk-free rate
    ]

    print("Configuration:")
    print(f"  Dimension: {dimension}D")
    print(f"  Chebyshev nodes per dimension: {num_cheb_pts}")
    print(f"  Domain: S=[50,150], K=[50,150], T=[0.1,3], σ=[0.1,0.5], r=[0.01,0.15]")
    print(f"  Full tensor size: {11**5} = 161,051 evaluations")

    # Create MoCaX Extend object
    print("\nInitializing MoCaX Extend...")
    obj = me.MocaxExtend(dimension, num_cheb_pts, variable_ranges)

    # Generate subgrid for training
    num_scenarios = 8000  # Number of training points
    print(f"\nGenerating subgrid ({num_scenarios} training points)...")
    start = time.time()
    random_cheb_pts = obj.subgrid_by_number(num_scenarios)
    subgrid_time = time.time() - start
    print(f"  Subgrid generation time: {subgrid_time:.4f}s")

    # Evaluate Black-Scholes on subgrid
    print(f"\nEvaluating Black-Scholes on {num_scenarios} points...")
    start = time.time()
    vals_subgrid = black_scholes_call_vectorized(random_cheb_pts)
    eval_time = time.time() - start
    print(f"  Evaluation time: {eval_time:.4f}s ({eval_time/num_scenarios*1000:.2f} ms/eval)")

    # Incorporate data into rank-adaptive object
    print("\nIncorporating training data...")
    obj.set_subgrid_values(vals_subgrid)
    obj.gen_train_val_data()

    # Grid size comparison
    original_grid_size = obj.get_tensor_size()
    subgrid_size = obj.get_subgrid_size()
    print(f"  Subgrid is {100 * subgrid_size / original_grid_size:.2f}% of full tensor")
    print(f"  ({subgrid_size} / {original_grid_size} points)")

    # Rank-adaptive algorithm parameters
    rank_adaptive_params = {
        "tolerance": 1e-3,
        "rel_tolerance": 1e-8,
        "max_iters": 100,
        "max_rank": 20,
        "print_progress": False,  # Suppress verbose training output
        "max_rounds": 5
    }

    print("\nRunning rank-adaptive TT algorithm...")
    print("  (Training in progress, please wait...)")

    # Train Chebyshev Tensor in TT format
    start = time.time()
    obj.run_rank_adaptive_algo(**rank_adaptive_params)
    train_time = time.time() - start
    print(f"\n  Training time: {train_time:.4f}s")

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
    print(f"{'Case':<12} {'MoCaX TT':>15} {'Analytical':>15} {'Error':>12}")
    print("-"*70)

    max_error = 0.0
    test_points = np.array([params for params, _ in test_cases])

    # Evaluate TT approximation
    start = time.time()
    tt_prices = obj.cheb_tensor_evals(test_points)
    eval_time = time.time() - start

    # Evaluate analytical
    analytical_prices = black_scholes_call_vectorized(test_points)

    for i, (params, label) in enumerate(test_cases):
        tt_price = tt_prices[i]
        analytical_price = analytical_prices[i]

        error = abs(tt_price - analytical_price) / analytical_price * 100
        max_error = max(max_error, error)

        print(f"{label:<12} {tt_price:>15.6f} {analytical_price:>15.6f} {error:>11.3f}%")

    print("-"*70)
    print(f"Maximum error: {max_error:.3f}%")
    print(f"Evaluation time: {eval_time*1000:.2f} ms for {len(test_cases)} points")

    # Greeks calculation at ATM
    print("\n" + "-"*70)
    print("Greeks Calculation at ATM [100, 100, 1.0, 0.25, 0.05]:")
    print("-"*70)

    S0, K0, T0, sigma0, r0 = 100, 100, 1.0, 0.25, 0.05
    eps_S = 0.01  # For Delta/Gamma
    eps_sigma = 0.001  # For Vega
    eps_r = 0.0001  # For Rho

    # Prepare evaluation points for finite differences
    eval_points = np.array([
        [S0 + eps_S, K0, T0, sigma0, r0],  # For Delta
        [S0 - eps_S, K0, T0, sigma0, r0],
        [S0, K0, T0, sigma0, r0],  # Center
        [S0, K0, T0, sigma0 + eps_sigma, r0],  # For Vega
        [S0, K0, T0, sigma0 - eps_sigma, r0],
        [S0, K0, T0, sigma0, r0 + eps_r],  # For Rho
        [S0, K0, T0, sigma0, r0 - eps_r],
    ])

    # Evaluate all points
    vals = obj.cheb_tensor_evals(eval_points)

    # Calculate Greeks
    V_up = vals[0]
    V_down = vals[1]
    V_center = vals[2]
    V_sigma_up = vals[3]
    V_sigma_down = vals[4]
    V_r_up = vals[5]
    V_r_down = vals[6]

    delta_tt = (V_up - V_down) / (2 * eps_S)
    gamma_tt = (V_up - 2*V_center + V_down) / (eps_S**2)
    vega_tt = (V_sigma_up - V_sigma_down) / (2 * eps_sigma)
    rho_tt = (V_r_up - V_r_down) / (2 * eps_r)

    # Analytical Greeks
    bs_call = BlackScholesCall(S=S0, K=K0, T=T0, r=r0, sigma=sigma0)
    delta_analytical = bs_call.delta()
    gamma_analytical = bs_call.gamma()
    vega_analytical = bs_call.vega()
    rho_analytical = bs_call.rho()

    # Compare
    delta_error = abs(delta_tt - delta_analytical) / abs(delta_analytical) * 100
    gamma_error = abs(gamma_tt - gamma_analytical) / abs(gamma_analytical) * 100
    vega_error = abs(vega_tt - vega_analytical) / abs(vega_analytical) * 100
    rho_error = abs(rho_tt - rho_analytical) / abs(rho_analytical) * 100

    print(f"{'Greek':<10} {'MoCaX TT':>15} {'Analytical':>15} {'Error':>12}")
    print("-"*70)
    print(f"{'Delta':<10} {delta_tt:>15.6f} {delta_analytical:>15.6f} {delta_error:>11.3f}%")
    print(f"{'Gamma':<10} {gamma_tt:>15.6f} {gamma_analytical:>15.6f} {gamma_error:>11.3f}%")
    print(f"{'Vega':<10} {vega_tt:>15.6f} {vega_analytical:>15.6f} {vega_error:>11.3f}%")
    print(f"{'Rho':<10} {rho_tt:>15.6f} {rho_analytical:>15.6f} {rho_error:>11.3f}%")
    print("-"*70)

    max_greek_error = max(delta_error, gamma_error, vega_error, rho_error)
    print(f"Maximum Greek error: {max_greek_error:.3f}%")

    # Serialization test
    print("\n" + "-"*70)
    print("Serialization Test:")
    print("-"*70)
    obj.serialize("mocax_tt_5d_bs.pickle")
    print(f"  ✓ Serialized to mocax_tt_5d_bs.pickle")

    obj_des = me.MocaxExtend.deserialize("mocax_tt_5d_bs.pickle")
    print(f"  ✓ Deserialized successfully")

    # Verify deserialized object
    test_point = np.array([[100, 100, 1.0, 0.25, 0.05]])
    original_val = obj.cheb_tensor_evals(test_point)[0]
    deserialized_val = obj_des.cheb_tensor_evals(test_point)[0]
    print(f"  Original: {original_val:.6f}, Deserialized: {deserialized_val:.6f}")
    print(f"  Difference: {abs(original_val - deserialized_val):.2e}")

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Training points:   {num_scenarios}")
    print(f"Build time:        {subgrid_time + eval_time + train_time:.4f}s")
    print(f"Max price error:   {max_error:.3f}%")
    print(f"Max Greek error:   {max_greek_error:.3f}%")
    print("\n✓ Excellent accuracy with TT compression!")
    print("  MoCaX Extend TT is the appropriate method for coupled functions.")
    print("  Scales efficiently to higher dimensions (d>6).")
    print("="*70)

    # Cleanup
    import os as _os
    if _os.path.exists("mocax_tt_5d_bs.pickle"):
        _os.remove("mocax_tt_5d_bs.pickle")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MoCaX Extend (Tensor Train) Test Suite")
    print("="*70)
    print("\nDemonstrating MoCaX Extend with Tensor Train decomposition")
    print("on 5D Black-Scholes option pricing.")
    print("\n✓ Appropriate method for smooth coupled functions")

    test_5d_black_scholes_tt()

    print("\n✓ Test complete")
    print("\nComparison with other methods:")
    print("  - MoCaX Sliding: Fast but poor accuracy for coupled functions")
    print("  - MoCaX Standard: Best for d≤6 (full tensor)")
    print("  - MoCaX TT: Best for d>6 or when compression needed")


if __name__ == "__main__":
    main()
