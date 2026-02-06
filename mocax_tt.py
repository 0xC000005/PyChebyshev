"""
Test script for MoCaX Extend (Tensor Train) on 5D Black-Scholes option pricing.

Uses the SAME domain, scenarios, and metrics as mocax_baseline.py for direct
comparison between MoCaX Standard (full tensor) and MoCaX TT (compressed).

Key difference from MoCaX Standard:
- TT format does NOT support analytical derivatives
- All Greeks computed via finite differences (central differences)
- TT uses ~5% of full tensor grid for training

Expected results: <1% price error, <5% Greek error
"""

import sys
import os
import time
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# Add mocaxextend_lib to path
mocaxextend_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocaxextend_lib')
sys.path.insert(0, mocaxextend_lib_dir)

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


# Fixed dividend yield (same as mocax_baseline.py)
Q = 0.02


def black_scholes_call_vectorized(x):
    """
    Vectorized Black-Scholes call option pricing with dividend yield.

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

    results = np.zeros(len(S))
    valid_idx = T > 0

    if np.any(valid_idx):
        Sv = S[valid_idx]
        Kv = K[valid_idx]
        Tv = T[valid_idx]
        sv = sigma[valid_idx]
        rv = r[valid_idx]

        d1 = (np.log(Sv / Kv) + (rv - Q + 0.5 * sv**2) * Tv) / (sv * np.sqrt(Tv))
        d2 = d1 - sv * np.sqrt(Tv)
        results[valid_idx] = Sv * np.exp(-Q * Tv) * norm.cdf(d1) - Kv * np.exp(-rv * Tv) * norm.cdf(d2)

    if np.any(~valid_idx):
        results[~valid_idx] = np.maximum(S[~valid_idx] - K[~valid_idx], 0)

    return results


def generate_5d_grid_samples(domain_values, n_points_per_dim):
    """Generate regular grid samples in 5D parameter space."""
    grids_1d = []
    for d in range(5):
        min_val, max_val = domain_values[d]
        grids_1d.append(np.linspace(min_val, max_val, n_points_per_dim))
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    return np.column_stack([g.ravel() for g in mesh])


def compute_fd_greeks(obj, S, K, T, sigma, r, domain):
    """
    Compute Price + 8 Greeks via finite differences on a TT object.

    Price is evaluated at the exact input point.
    Greeks use a center point nudged slightly inward from domain boundaries
    to ensure all FD perturbation points stay within bounds.

    FD step sizes are chosen to balance truncation error vs noise amplification
    from TT approximation error (~0.1%). Larger steps reduce noise amplification
    at the cost of truncation error.

    Returns dict with Greek values and 'fd_center' tuple for reference computation.
    """
    # FD step sizes - larger than textbook values to handle TT approximation noise
    # For 1st-order: noise ~ ε_abs/h, truncation ~ f'''·h²/6
    # For 2nd-order: noise ~ 3·ε_abs/h², truncation ~ f''''·h²/12
    eps_S = 0.5       # 1.25% of S range (40)
    eps_T = 0.05      # 6.7% of T range (0.75)
    eps_sigma = 0.01  # 5% of σ range (0.2)
    eps_r = 0.003     # 4.3% of r range (0.07)

    # Nudge center away from boundaries so ±eps stays inside domain
    fd = [S, K, T, sigma, r]
    fd_eps_by_dim = {0: eps_S, 2: eps_T, 3: eps_sigma, 4: eps_r}
    for d, eps in fd_eps_by_dim.items():
        lo, hi = domain[d]
        needed = eps * 1.5
        if fd[d] - lo < needed:
            fd[d] = lo + needed
        if hi - fd[d] < needed:
            fd[d] = hi - needed
    Sf, Kf, Tf, sf, rf = fd

    # Build all perturbation points in one batch (20 points)
    pts = np.array([
        [S, K, T, sigma, r],                              # 0: original (for Price)
        [Sf + eps_S, Kf, Tf, sf, rf],                     # 1: S+
        [Sf - eps_S, Kf, Tf, sf, rf],                     # 2: S-
        [Sf, Kf, Tf, sf + eps_sigma, rf],                 # 3: σ+
        [Sf, Kf, Tf, sf - eps_sigma, rf],                 # 4: σ-
        [Sf, Kf, Tf, sf, rf + eps_r],                     # 5: r+
        [Sf, Kf, Tf, sf, rf - eps_r],                     # 6: r-
        [Sf, Kf, Tf, sf, rf],                             # 7: FD center (nudged)
        # Vanna: S × σ
        [Sf + eps_S, Kf, Tf, sf + eps_sigma, rf],         # 8:  S+ σ+
        [Sf + eps_S, Kf, Tf, sf - eps_sigma, rf],         # 9:  S+ σ-
        [Sf - eps_S, Kf, Tf, sf + eps_sigma, rf],         # 10: S- σ+
        [Sf - eps_S, Kf, Tf, sf - eps_sigma, rf],         # 11: S- σ-
        # Charm: S × T
        [Sf + eps_S, Kf, Tf + eps_T, sf, rf],             # 12: S+ T+
        [Sf + eps_S, Kf, Tf - eps_T, sf, rf],             # 13: S+ T-
        [Sf - eps_S, Kf, Tf + eps_T, sf, rf],             # 14: S- T+
        [Sf - eps_S, Kf, Tf - eps_T, sf, rf],             # 15: S- T-
        # Veta: T × σ
        [Sf, Kf, Tf + eps_T, sf + eps_sigma, rf],         # 16: T+ σ+
        [Sf, Kf, Tf + eps_T, sf - eps_sigma, rf],         # 17: T+ σ-
        [Sf, Kf, Tf - eps_T, sf + eps_sigma, rf],         # 18: T- σ+
        [Sf, Kf, Tf - eps_T, sf - eps_sigma, rf],         # 19: T- σ-
    ])

    v = obj.cheb_tensor_evals(pts)

    fc = v[7]  # FD center value (at nudged point)
    return {
        'Price': v[0],                                                          # exact center
        'fd_center': (Sf, Kf, Tf, sf, rf),
        'Delta': (v[1] - v[2]) / (2 * eps_S),
        'Gamma': (v[1] - 2 * fc + v[2]) / (eps_S ** 2),
        'Vega': (v[3] - v[4]) / (2 * eps_sigma),
        'Rho': (v[5] - v[6]) / (2 * eps_r),
        'Vanna': (v[8] - v[9] - v[10] + v[11]) / (4 * eps_S * eps_sigma),
        'Charm': (v[12] - v[13] - v[14] + v[15]) / (4 * eps_S * eps_T),
        'Vomma': (v[3] - 2 * fc + v[4]) / (eps_sigma ** 2),
        'Veta': (v[16] - v[17] - v[18] + v[19]) / (4 * eps_T * eps_sigma),
    }


def test_5d_black_scholes_tt():
    """
    Test MoCaX Extend (Tensor Train) on 5D Black-Scholes.
    Same domain, scenarios, and metrics as mocax_baseline.py.
    """
    print("\n" + "=" * 70)
    print("TEST: 5D Black-Scholes with MoCaX Extend (Tensor Train)")
    print("=" * 70)

    # =========================================================================
    # Configuration - SAME domain as mocax_baseline.py
    # =========================================================================
    dimension = 5
    num_cheb_pts = [11, 11, 11, 11, 11]
    variable_ranges = [
        [80.0, 120.0],    # S: spot price
        [90.0, 110.0],    # K: strike price
        [0.25, 1.0],      # T: time to maturity
        [0.15, 0.35],     # σ: volatility
        [0.01, 0.08]      # r: risk-free rate
    ]

    print(f"\nConfiguration:")
    print(f"  Dimensions: {dimension} (S, K, T, σ, r)")
    print(f"  Domain: S ∈ [80, 120], K ∈ [90, 110], T ∈ [0.25, 1.0]")
    print(f"          σ ∈ [0.15, 0.35], r ∈ [0.01, 0.08]")
    print(f"  Chebyshev nodes per dimension: {num_cheb_pts}")
    print(f"  Full tensor size: {np.prod(num_cheb_pts):,} evaluations")
    print(f"  Fixed: q = {Q} (dividend yield)")

    # =========================================================================
    # Build TT approximation
    # =========================================================================
    print("\nInitializing MoCaX Extend...")
    obj = me.MocaxExtend(dimension, num_cheb_pts, variable_ranges)

    num_scenarios = 8000
    print(f"\nGenerating subgrid ({num_scenarios} training points)...")
    start = time.time()
    random_cheb_pts = obj.subgrid_by_number(num_scenarios)
    subgrid_time = time.time() - start
    print(f"  Subgrid generation time: {subgrid_time:.4f}s")

    print(f"\nEvaluating Black-Scholes on {num_scenarios} points...")
    start = time.time()
    vals_subgrid = black_scholes_call_vectorized(random_cheb_pts)
    eval_time = time.time() - start
    print(f"  Evaluation time: {eval_time:.4f}s ({eval_time / num_scenarios * 1000:.2f} ms/eval)")

    print("\nIncorporating training data...")
    obj.set_subgrid_values(vals_subgrid)
    obj.gen_train_val_data()

    original_grid_size = obj.get_tensor_size()
    subgrid_size = obj.get_subgrid_size()
    print(f"  Subgrid is {100 * subgrid_size / original_grid_size:.2f}% of full tensor")
    print(f"  ({subgrid_size} / {original_grid_size} points)")

    rank_adaptive_params = {
        "tolerance": 1e-3,
        "rel_tolerance": 1e-8,
        "max_iters": 100,
        "max_rank": 20,
        "print_progress": False,
        "max_rounds": 5,
    }

    print("\nRunning rank-adaptive TT algorithm...")
    print("  (Training in progress, please wait...)")

    start = time.time()
    obj.run_rank_adaptive_algo(**rank_adaptive_params)
    train_time = time.time() - start
    build_time = subgrid_time + eval_time + train_time
    print(f"\n  Training time: {train_time:.3f} s")
    print(f"  Total build time: {build_time:.3f} s")

    # =========================================================================
    # UNIFIED TABLE: Price + Greeks for ALL Scenarios (same as baseline)
    # =========================================================================
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
        ([85.0, 105.0, 0.5, 0.20, 0.03], "Corner1"),
        ([115.0, 95.0, 0.75, 0.30, 0.07], "Corner2"),
    ]

    metric_names = ['Price', 'Delta', 'Gamma', 'Vega', 'Rho',
                    'Vanna', 'Charm', 'Vomma', 'Veta']
    first_order = {'Price', 'Delta', 'Vega', 'Rho'}
    second_order = {'Gamma', 'Vanna', 'Charm', 'Vomma', 'Veta'}

    price_errors = []
    first_order_errors = []
    second_order_errors = []
    second_order_errors_meaningful = []  # |exact| > 1.0

    for test_point, case_name in test_cases:
        S, K, T, sigma, r = test_point

        # TT values via finite differences
        tt_values = compute_fd_greeks(obj, S, K, T, sigma, r, variable_ranges)

        # Analytical Price at original point
        opt_orig = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=Q)
        exact_price = opt_orig.price()

        # Analytical Greeks at FD center (nudged point) for fair comparison
        Sf, Kf, Tf, sf, rf = tt_values['fd_center']
        opt_fd = BlackScholesCall(S=Sf, K=Kf, T=Tf, r=rf, sigma=sf, q=Q)
        exact_values = {
            'Price': exact_price,
            'Delta': opt_fd.delta(),
            'Gamma': opt_fd.gamma(),
            'Vega': opt_fd.vega(),
            'Rho': opt_fd.rho(),
            'Vanna': opt_fd.vanna(),
            'Charm': -opt_fd.charm(),
            'Vomma': opt_fd.vomma(),
            'Veta': opt_fd.veta(),
        }

        print(f"\n{'=' * 70}")
        print(f"SCENARIO: {case_name} (S={S:.0f}, K={K:.0f}, T={T:.2f}, σ={sigma:.2f}, r={r:.2f})")
        print(f"{'=' * 70}")
        print(f"{'Metric':<8} {'Exact':>14} {'TT (FD)':>14} {'Error%':>10}")
        print("-" * 50)

        for metric_name in metric_names:
            exact = exact_values[metric_name]
            approx = tt_values[metric_name]

            if abs(exact) > 1e-10:
                err = abs(approx - exact) / abs(exact) * 100
            else:
                err = abs(approx - exact) * 100

            if metric_name == 'Price':
                price_errors.append(err)
            elif metric_name in first_order:
                first_order_errors.append(err)
            else:
                second_order_errors.append(err)
                if abs(exact) > 1.0:
                    second_order_errors_meaningful.append(err)

            # Mark near-zero exact values where relative error is unreliable
            if metric_name in second_order and abs(exact) < 1.0:
                tag = " ** (near-zero)"
            elif metric_name in second_order:
                tag = " *"
            else:
                tag = ""
            print(f"{metric_name:<8} {exact:>14.6f} {approx:>14.6f} {err:>10.3f}%{tag}")

    print(f"\n  *  = 2nd-order/cross-term Greek")
    print(f"  ** = exact value near zero (relative error unreliable)")
    print(f"\n  Price max error:           {max(price_errors):.3f}%")
    print(f"  1st-order Greek max error: {max(first_order_errors):.3f}%  (Delta, Vega, Rho)")
    if second_order_errors_meaningful:
        print(f"  2nd-order Greek max error: {max(second_order_errors_meaningful):.3f}%  "
              f"(|exact| > 1.0, excl. near-zero)")
    print(f"  2nd-order Greek max (all): {max(second_order_errors):.3f}%  (incl. near-zero exact values)")

    # =========================================================================
    # COMPREHENSIVE GRID EVALUATION (price only)
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE GRID EVALUATION (price only)")
    print(f"{'=' * 70}")

    n_points_per_dim = int(os.environ.get('N_GRID_POINTS', 10))
    total_points = n_points_per_dim ** 5

    print(f"Grid configuration:")
    print(f"  Points per dimension: {n_points_per_dim}")
    print(f"  Total evaluation points: {total_points:,}")
    print(f"  (Set N_GRID_POINTS env var to change)")

    print(f"\nGenerating {n_points_per_dim}^5 = {total_points:,} grid points...")
    samples = generate_5d_grid_samples(variable_ranges, n_points_per_dim)

    print(f"Computing ground truth (analytical Black-Scholes)...")
    ground_truth_prices = black_scholes_call_vectorized(samples)

    print(f"Evaluating TT approximation on {total_points:,} points...")
    start_eval = time.time()
    tt_prices = obj.cheb_tensor_evals(samples)
    eval_time = time.time() - start_eval

    # Compute errors (filter near-zero prices to avoid inflated relative errors)
    abs_errors = np.abs(tt_prices - ground_truth_prices)
    min_price_threshold = 0.50  # Only compute relative error for prices > $0.50
    meaningful = ground_truth_prices > min_price_threshold
    n_meaningful = np.sum(meaningful)
    rel_errors_meaningful = abs_errors[meaningful] / ground_truth_prices[meaningful] * 100
    rel_errors_all = np.where(ground_truth_prices > 1e-10,
                              abs_errors / ground_truth_prices * 100,
                              abs_errors * 100)

    print(f"\n{'=' * 70}")
    print("GRID EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Points evaluated:     {total_points:,}")
    print(f"  Total TT time:        {eval_time:.3f} s")
    print(f"  Time per evaluation:  {eval_time / total_points * 1000:.4f} ms")
    print(f"  Evals per second:     {total_points / eval_time:,.0f}")
    print(f"")
    print(f"  --- All {total_points:,} points ---")
    print(f"  Mean abs error:       ${np.mean(abs_errors):.6f}")
    print(f"  Max abs error:        ${np.max(abs_errors):.6f}")
    print(f"  Median abs error:     ${np.median(abs_errors):.6f}")
    print(f"")
    print(f"  --- {n_meaningful:,} points with price > ${min_price_threshold:.2f} ---")
    print(f"  Mean error:           {np.mean(rel_errors_meaningful):.4f}%")
    print(f"  Max error:            {np.max(rel_errors_meaningful):.4f}%")
    print(f"  Std deviation:        {np.std(rel_errors_meaningful):.4f}%")
    print(f"  Median error:         {np.median(rel_errors_meaningful):.4f}%")
    print(f"  95th percentile:      {np.percentile(rel_errors_meaningful, 95):.4f}%")
    print(f"{'=' * 70}")

    # Print high-error points (relative, filtered)
    high_error_mask = rel_errors_meaningful > 1.0
    n_high_error = np.sum(high_error_mask)
    meaningful_samples = samples[meaningful]
    meaningful_gt = ground_truth_prices[meaningful]
    meaningful_tt = tt_prices[meaningful]
    if n_high_error > 0:
        print(f"\nPoints with price > ${min_price_threshold:.2f} and error > 1%: "
              f"{n_high_error} ({n_high_error / n_meaningful * 100:.2f}%)")
        print(f"{'S':>10} {'K':>10} {'T':>10} {'sigma':>10} {'r':>10} "
              f"{'Exact':>12} {'TT':>12} {'Error%':>10}")
        print("-" * 96)
        high_error_indices = np.where(high_error_mask)[0]
        for idx in high_error_indices[:20]:
            Si, Ki, Ti, sigi, ri = meaningful_samples[idx]
            print(f"{Si:>10.4f} {Ki:>10.4f} {Ti:>10.4f} {sigi:>10.4f} {ri:>10.4f} "
                  f"{meaningful_gt[idx]:>12.6f} {meaningful_tt[idx]:>12.6f} "
                  f"{rel_errors_meaningful[idx]:>10.4f}")
        if n_high_error > 20:
            print(f"  ... and {n_high_error - 20} more")
    else:
        print(f"\nNo points with price > ${min_price_threshold:.2f} and error > 1%")

    # Cleanup serialization file if exists
    if os.path.exists("mocax_tt_5d_bs.pickle"):
        os.remove("mocax_tt_5d_bs.pickle")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Training points:        {num_scenarios}")
    print(f"  Compression:            {100 * subgrid_size / original_grid_size:.2f}% of full tensor")
    print(f"  Build time:             {build_time:.3f} s")
    print(f"")
    print(f"  Scenario max errors:")
    print(f"    Price:                {max(price_errors):.3f}%")
    print(f"    1st-order Greeks:     {max(first_order_errors):.3f}%  (Delta, Vega, Rho)")
    if second_order_errors_meaningful:
        print(f"    2nd-order Greeks:     {max(second_order_errors_meaningful):.3f}%  "
              f"(|exact| > 1, excl. near-zero)")
    print(f"    2nd-order (all):      {max(second_order_errors):.3f}%  (incl. near-zero)")
    print(f"")
    print(f"  Grid evaluation (price > ${min_price_threshold:.2f}):")
    print(f"    Mean error:           {np.mean(rel_errors_meaningful):.4f}%")
    print(f"    Max error:            {np.max(rel_errors_meaningful):.4f}%")
    print(f"    Median error:         {np.median(rel_errors_meaningful):.4f}%")
    print(f"")
    print(f"  Note: Greeks computed via finite differences (TT has no analytical derivatives)")
    print(f"  Note: 2nd-order Greeks amplify TT approximation noise through FD division")
    print(f"{'=' * 70}")

    return max(price_errors) < 5.0


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MoCaX Extend (Tensor Train) Test Suite")
    print("=" * 70)
    print("\nDemonstrating MoCaX Extend with Tensor Train decomposition")
    print("on 5D Black-Scholes option pricing.")
    print("\nSame domain and scenarios as MoCaX Standard (mocax_baseline.py)")
    print("for direct comparison.")

    passed = test_5d_black_scholes_tt()

    if passed:
        print("\n✓ Test PASSED")
    else:
        print("\n✗ Test FAILED - errors exceeded threshold")

    print("\nComparison with other methods:")
    print("  - MoCaX Standard: Full tensor, analytical derivatives, 0% price error")
    print("  - MoCaX TT:       5% of tensor, FD derivatives, <1% price error")
    print("  - MoCaX Sliding:  Additive decomposition, poor for coupled functions")


if __name__ == "__main__":
    main()
