"""
Comprehensive Comparison of Option Pricing Methods

Compares speed and accuracy of:
1. Chebyshev Barycentric (manual barycentric interpolation)
2. MoCaX Standard (full tensor)
3. (Optional) FDM (Finite Difference Method) [commented out]
4. (Optional) Chebyshev Baseline (NumPy Chebyshev.interpolate) [commented out]

Ground truth: blackscholes library analytical formulas
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Import ground truth
from blackscholes import BlackScholesCall

# Import our implementations
from fdm_baseline import BlackScholesFDM
from chebyshev_baseline import ChebyshevApproximation as ChebyshevBaseline
from chebyshev_barycentric import ChebyshevApproximation as ChebyshevBarycentric


@dataclass
class GreekErrors:
    """Container for Greek errors."""
    price: List[float]
    delta: List[float]
    gamma: List[float]
    vega: List[float]
    theta: List[float]
    rho: List[float]

    def __init__(self):
        self.price = []
        self.delta = []
        self.gamma = []
        self.vega = []
        self.theta = []
        self.rho = []

    def add(self, metric: str, error: float):
        """Add error for a specific metric."""
        getattr(self, metric.lower()).append(error)

    def get_stats(self, metric: str) -> Tuple[float, float]:
        """Get (mean, max) for a metric."""
        errors = getattr(self, metric.lower())
        return np.mean(errors), np.max(errors)


@dataclass
class MethodResult:
    """Results for a single method."""
    name: str
    build_time: Optional[float]  # None for FDM
    eval_time: float
    errors: GreekErrors
    grid_points: Optional[int] = None


def setup_5d_domain():
    """
    Set up 5D parameter space matching existing tests.

    Returns:
        domain: List of (min, max) for each dimension [S, K, T, σ, r]
        q: Fixed dividend yield
    """
    domain = [
        (80.0, 120.0),    # S: spot price
        (90.0, 110.0),    # K: strike price
        (0.25, 1.0),      # T: time to maturity
        (0.15, 0.35),     # σ: volatility
        (0.01, 0.08)      # r: risk-free rate
    ]
    q = 0.02  # dividend yield (fixed)

    return domain, q


def generate_random_samples(domain: List[Tuple[float, float]], n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate random samples in 5D parameter space.

    Args:
        domain: List of (min, max) for each dimension
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        samples: Array of shape (n_samples, 5)
    """
    np.random.seed(seed)
    samples = np.zeros((n_samples, 5))

    for i, (min_val, max_val) in enumerate(domain):
        samples[:, i] = np.random.uniform(min_val, max_val, n_samples)

    return samples


def compute_ground_truth(samples: np.ndarray, q: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute analytical prices and Greeks using blackscholes library.

    Args:
        samples: Array of shape (n_samples, 5) with columns [S, K, T, σ, r]
        q: Dividend yield

    Returns:
        prices: Array of shape (n_samples,)
        greeks: Dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'
                each containing array of shape (n_samples,)
    """
    n_samples = len(samples)
    prices = np.zeros(n_samples)
    greeks = {
        'delta': np.zeros(n_samples),
        'gamma': np.zeros(n_samples),
        'vega': np.zeros(n_samples),
        'theta': np.zeros(n_samples),
        'rho': np.zeros(n_samples)
    }

    for i in range(n_samples):
        S, K, T, sigma, r = samples[i]
        call = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

        prices[i] = call.price()

        # Get core Greeks from library
        core_greeks = call.get_core_greeks()
        for name in greeks.keys():
            greeks[name][i] = core_greeks[name]

    return prices, greeks


def build_chebyshev_baseline(domain: List[Tuple[float, float]], q: float) -> Tuple[ChebyshevBaseline, float]:
    """Build Chebyshev baseline approximation."""
    print("\n" + "="*70)
    print("Building Chebyshev Baseline (NumPy Chebyshev.interpolate)")
    print("="*70)

    def bs_5d(x, _):
        return BlackScholesCall(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=q).price()

    cheb = ChebyshevBaseline(
        bs_5d,
        5,
        domain,
        [11, 11, 11, 11, 11],
        max_derivative_order=2
    )

    start = time.time()
    cheb.build()
    build_time = time.time() - start

    return cheb, build_time


def build_chebyshev_barycentric(domain: List[Tuple[float, float]], q: float) -> Tuple[ChebyshevBarycentric, float]:
    """Build Chebyshev barycentric approximation."""
    print("\n" + "="*70)
    print("Building Chebyshev Barycentric (Manual Barycentric)")
    print("="*70)

    def bs_5d(x, _):
        return BlackScholesCall(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=q).price()

    cheb = ChebyshevBarycentric(
        bs_5d,
        5,
        domain,
        [11, 11, 11, 11, 11],
        max_derivative_order=2
    )

    start = time.time()
    cheb.build()
    build_time = time.time() - start

    return cheb, build_time


def build_mocax(domain: List[Tuple[float, float]], q: float) -> Tuple[Optional[object], Optional[float], Optional[int]]:
    """Build MoCaX approximation (returns None if library not available)."""
    print("\n" + "="*70)
    print("Building MoCaX Standard (Full Tensor)")
    print("="*70)

    try:
        # Add mocax_lib to path
        mocax_lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mocax_lib')
        sys.path.insert(0, mocax_lib_dir)

        # Patch ctypes for libmocaxc.so
        from ctypes import CDLL
        import ctypes

        _original_cdll_init = CDLL.__init__

        def _patched_cdll_init(self, name, mode=ctypes.DEFAULT_MODE, handle=None, use_errno=False, use_last_error=False, winmode=None):
            if isinstance(name, str) and ('libmocaxc.so' in name or name == 'libmocaxc.so'):
                name = os.path.join(mocax_lib_dir, 'libmocaxc.so')
            _original_cdll_init(self, name, mode, handle, use_errno, use_last_error, winmode)

        CDLL.__init__ = _patched_cdll_init

        import mocaxpy
        print(f"✓ MoCaX version: {mocaxpy.get_version_id()}")

        # Define wrapper function
        def bs_call_wrapper(x, additional_data):
            # Use indexing to extract first 5 values (avoid unpacking error)
            S = x[0]
            K = x[1]
            T = x[2]
            sigma = x[3]
            r = x[4]
            return BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q).price()

        # MoCaX configuration
        num_dimensions = 5
        domain_values = [list(d) for d in domain]
        mocax_domain = mocaxpy.MocaxDomain(domain_values)

        n_values = [11, 11, 11, 11, 11]  # 11 nodes per dimension
        ns = mocaxpy.MocaxNs(n_values)
        max_derivative_order = 2

        print(f"  Domain: S=[80,120], K=[90,110], T=[0.25,1], σ=[0.15,0.35], r=[0.01,0.08]")
        print(f"  Chebyshev nodes: {n_values}")

        start = time.time()
        mocax_obj = mocaxpy.Mocax(bs_call_wrapper, num_dimensions, mocax_domain,
                                   None, ns, max_derivative_order=max_derivative_order)
        build_time = time.time() - start

        print(f"✓ Built in {build_time:.3f}s")
        print("="*70)

        return mocax_obj, build_time, int(np.prod(n_values))

    except Exception as e:
        print(f"✗ MoCaX not available: {e}")
        print("  Skipping MoCaX comparison")
        print("="*70)
        return None, None, None


def evaluate_chebyshev_method(
    cheb_obj,
    samples: np.ndarray,
    ground_truth_prices: np.ndarray,
    ground_truth_greeks: Dict[str, np.ndarray],
    method_name: str,
    use_fast_eval: bool = False
) -> Tuple[float, GreekErrors]:
    """
    Evaluate Chebyshev method on random samples.

    Args:
        cheb_obj: Chebyshev approximation object
        samples: Array of shape (n_samples, 5)
        ground_truth_prices: Array of shape (n_samples,)
        ground_truth_greeks: Dict of ground truth Greeks
        method_name: Name for progress bar
        use_fast_eval: If True, use fast_eval() instead of eval() (for barycentric)

    Returns:
        eval_time: Total evaluation time
        errors: GreekErrors object
    """
    n_samples = len(samples)
    errors = GreekErrors()

    # Derivative IDs for Greeks
    deriv_ids = {
        'price': [0, 0, 0, 0, 0],
        'delta': [1, 0, 0, 0, 0],
        'gamma': [2, 0, 0, 0, 0],
        'vega': [0, 0, 0, 1, 0],
        'theta_deriv': [0, 0, 1, 0, 0],  # Will negate for theta
        'rho': [0, 0, 0, 0, 1]
    }

    # Select evaluation method
    eval_method = cheb_obj.fast_eval if use_fast_eval else cheb_obj.eval

    start = time.time()

    for i in tqdm(range(n_samples), desc=f"{method_name:20s}", ncols=80):
        point = samples[i].tolist()

        # Price
        price_approx = eval_method(point, deriv_ids['price'])
        price_error = abs(price_approx - ground_truth_prices[i]) / ground_truth_prices[i] * 100
        errors.add('price', price_error)

        # Delta
        delta_approx = eval_method(point, deriv_ids['delta'])
        delta_error = abs(delta_approx - ground_truth_greeks['delta'][i]) / abs(ground_truth_greeks['delta'][i]) * 100
        errors.add('delta', delta_error)

        # Gamma
        gamma_approx = eval_method(point, deriv_ids['gamma'])
        gamma_error = abs(gamma_approx - ground_truth_greeks['gamma'][i]) / abs(ground_truth_greeks['gamma'][i]) * 100
        errors.add('gamma', gamma_error)

        # Vega
        vega_approx = eval_method(point, deriv_ids['vega'])
        vega_error = abs(vega_approx - ground_truth_greeks['vega'][i]) / abs(ground_truth_greeks['vega'][i]) * 100
        errors.add('vega', vega_error)

        # Theta (negative time derivative)
        theta_deriv = eval_method(point, deriv_ids['theta_deriv'])
        theta_approx = -theta_deriv
        theta_error = abs(theta_approx - ground_truth_greeks['theta'][i]) / abs(ground_truth_greeks['theta'][i]) * 100
        errors.add('theta', theta_error)

        # Rho
        rho_approx = eval_method(point, deriv_ids['rho'])
        rho_error = abs(rho_approx - ground_truth_greeks['rho'][i]) / abs(ground_truth_greeks['rho'][i]) * 100
        errors.add('rho', rho_error)

    eval_time = time.time() - start

    return eval_time, errors


def evaluate_mocax_method(
    mocax_obj,
    samples: np.ndarray,
    ground_truth_prices: np.ndarray,
    ground_truth_greeks: Dict[str, np.ndarray]
) -> Tuple[float, GreekErrors]:
    """Evaluate MoCaX method on random samples."""
    n_samples = len(samples)
    errors = GreekErrors()

    # Derivative IDs
    deriv_price = mocax_obj.get_derivative_id([0, 0, 0, 0, 0])
    deriv_delta = mocax_obj.get_derivative_id([1, 0, 0, 0, 0])
    deriv_gamma = mocax_obj.get_derivative_id([2, 0, 0, 0, 0])
    deriv_vega = mocax_obj.get_derivative_id([0, 0, 0, 1, 0])
    deriv_theta = mocax_obj.get_derivative_id([0, 0, 1, 0, 0])
    deriv_rho = mocax_obj.get_derivative_id([0, 0, 0, 0, 1])

    start = time.time()

    for i in tqdm(range(n_samples), desc=f"{'MoCaX Standard':20s}", ncols=80):
        point = samples[i].tolist()

        # Price
        price_approx = mocax_obj.eval(point, deriv_price)
        price_error = abs(price_approx - ground_truth_prices[i]) / ground_truth_prices[i] * 100
        errors.add('price', price_error)

        # Delta
        delta_approx = mocax_obj.eval(point, deriv_delta)
        delta_error = abs(delta_approx - ground_truth_greeks['delta'][i]) / abs(ground_truth_greeks['delta'][i]) * 100
        errors.add('delta', delta_error)

        # Gamma
        gamma_approx = mocax_obj.eval(point, deriv_gamma)
        gamma_error = abs(gamma_approx - ground_truth_greeks['gamma'][i]) / abs(ground_truth_greeks['gamma'][i]) * 100
        errors.add('gamma', gamma_error)

        # Vega
        vega_approx = mocax_obj.eval(point, deriv_vega)
        vega_error = abs(vega_approx - ground_truth_greeks['vega'][i]) / abs(ground_truth_greeks['vega'][i]) * 100
        errors.add('vega', vega_error)

        # Theta (negative time derivative)
        theta_deriv = mocax_obj.eval(point, deriv_theta)
        theta_approx = -theta_deriv
        theta_error = abs(theta_approx - ground_truth_greeks['theta'][i]) / abs(ground_truth_greeks['theta'][i]) * 100
        errors.add('theta', theta_error)

        # Rho
        rho_approx = mocax_obj.eval(point, deriv_rho)
        rho_error = abs(rho_approx - ground_truth_greeks['rho'][i]) / abs(ground_truth_greeks['rho'][i]) * 100
        errors.add('rho', rho_error)

    eval_time = time.time() - start

    return eval_time, errors


def evaluate_fdm_method(
    samples: np.ndarray,
    ground_truth_prices: np.ndarray,
    ground_truth_greeks: Dict[str, np.ndarray],
    q: float
) -> Tuple[float, GreekErrors]:
    """
    Evaluate FDM method on random samples.

    Note: FDM solves PDE from scratch for each sample, so this will be slow!
    """
    n_samples = len(samples)
    errors = GreekErrors()

    # FDM configuration 
    M, N = 200, 500

    start = time.time()

    for i in tqdm(range(n_samples), desc=f"{'FDM':20s}", ncols=80):
        S, K, T, sigma, r = samples[i]

        # Solve PDE
        S_max = 3 * K
        fdm = BlackScholesFDM(
            S_max=S_max, K=K, T=T, r=r, sigma=sigma, q=q,
            M=M, N=N, option_type='call'
        )
        fdm.solve_crank_nicolson()

        # Price
        price_approx = fdm.get_price(S)
        price_error = abs(price_approx - ground_truth_prices[i]) / ground_truth_prices[i] * 100
        errors.add('price', price_error)

        # Delta
        delta_approx = fdm.get_delta(S)
        delta_error = abs(delta_approx - ground_truth_greeks['delta'][i]) / abs(ground_truth_greeks['delta'][i]) * 100
        errors.add('delta', delta_error)

        # Gamma
        gamma_approx = fdm.get_gamma(S)
        gamma_error = abs(gamma_approx - ground_truth_greeks['gamma'][i]) / abs(ground_truth_greeks['gamma'][i]) * 100
        errors.add('gamma', gamma_error)

        # Theta
        theta_approx = fdm.get_theta(S)
        theta_error = abs(theta_approx - ground_truth_greeks['theta'][i]) / abs(ground_truth_greeks['theta'][i]) * 100
        errors.add('theta', theta_error)

        # Vega (requires re-solving PDE)
        vega_approx = fdm.get_vega_fd(S)
        vega_error = abs(vega_approx - ground_truth_greeks['vega'][i]) / abs(ground_truth_greeks['vega'][i]) * 100
        errors.add('vega', vega_error)

        # Rho (requires re-solving PDE)
        rho_approx = fdm.get_rho_fd(S)
        rho_error = abs(rho_approx - ground_truth_greeks['rho'][i]) / abs(ground_truth_greeks['rho'][i]) * 100
        errors.add('rho', rho_error)

    eval_time = time.time() - start

    return eval_time, errors


def print_build_time_table(results: List[MethodResult]):
    """Print Table 1: Build/Precomputation Time."""
    print("\n" + "="*80)
    print("TABLE 1: Build/Precomputation Time")
    print("="*80)
    print(f"{'Method':<22} | {'Build Time (s)':>14} | {'Grid Points':>12} | {'Notes':<24}")
    print("-"*80)

    for result in results:
        if result.build_time is None:
            build_str = "N/A"
            grid_str = "N/A"
            notes = "No precomputation"
        else:
            build_str = f"{result.build_time:.3f}"
            grid_str = f"{result.grid_points:,}" if result.grid_points else "—"

            if "Baseline" in result.name:
                notes = "Partial precompute"
            elif "Barycentric" in result.name:
                notes = "Fast barycentric eval"
            elif "MoCaX" in result.name:
                notes = "Full tensor"
            else:
                notes = ""

        print(f"{result.name:<22} | {build_str:>14} | {grid_str:>12} | {notes:<24}")

    print("="*80)


def print_accuracy_table(results: List[MethodResult]):
    """Print Table 2: Accuracy & Speed."""
    print("\n" + "="*80)
    print("TABLE 2: Accuracy & Speed (random samples)")
    print("="*80)

    metrics = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']

    # Header
    header = f"{'Method':<20} |"
    for metric in metrics:
        header += f" {metric.capitalize():>15} |"
    header += f" {'Total Time':>11}"
    print(header)

    subheader = f"{'':<20} |"
    for _ in metrics:
        subheader += f" {'Avg% / Max%':>15} |"
    subheader += f" {'(s)':>11}"
    print(subheader)
    print("-"*len(header))

    # Data rows
    for result in results:
        row = f"{result.name:<20} |"

        for metric in metrics:
            avg_err, max_err = result.errors.get_stats(metric)
            row += f" {avg_err:>5.2f}% / {max_err:>5.2f}% |"

        row += f" {result.eval_time:>11.2f}"
        print(row)

    print("="*80)


def main():
    """Run comprehensive comparison experiment."""
    print("="*80)
    print("COMPREHENSIVE COMPARISON: Speed & Accuracy of Option Pricing Methods")
    print("="*80)
    print("\nMethods:")
    print("  1. Chebyshev Barycentric - Manual barycentric interpolation (fast_eval)")
    print("  2. MoCaX Standard - Full tensor (if available)")
    print("  3. (Optional) FDM - Finite Difference PDE [commented out]")
    print("  4. (Optional) Chebyshev Baseline - NumPy Chebyshev.interpolate [commented out]")
    print("\nGround Truth: blackscholes library analytical formulas")
    # Allow overriding sample count via env var N_SAMPLES
    n_samples_env = os.environ.get('N_SAMPLES')
    try:
        n_samples = int(n_samples_env) if n_samples_env else 100
    except ValueError:
        n_samples = 100
    print(f"Test: {n_samples} random samples in 5D parameter space")

    # Setup
    domain, q = setup_5d_domain()

    print("\n5D Parameter Space:")
    print(f"  S (spot):      [{domain[0][0]}, {domain[0][1]}]")
    print(f"  K (strike):    [{domain[1][0]}, {domain[1][1]}]")
    print(f"  T (maturity):  [{domain[2][0]}, {domain[2][1]}]")
    print(f"  σ (volatility):[{domain[3][0]}, {domain[3][1]}]")
    print(f"  r (rate):      [{domain[4][0]}, {domain[4][1]}]")
    print(f"  q (dividend):  {q} (fixed)")

    # Build/precompute methods
    print("\n" + "="*80)
    print("PHASE 1: Build/Precompute Methods")
    print("="*80)

    # cheb_baseline, cheb_baseline_time = build_chebyshev_baseline(domain, q)
    cheb_barycentric, cheb_barycentric_time = build_chebyshev_barycentric(domain, q)
    mocax_obj, mocax_time, mocax_grid = build_mocax(domain, q)

    # Generate random samples
    print("\n" + "="*80)
    print("PHASE 2: Generate Random Samples & Compute Ground Truth")
    print("="*80)

    print(f"\nGenerating {n_samples} random samples...")
    samples = generate_random_samples(domain, n_samples)

    print(f"Computing ground truth (analytical formulas)...")
    ground_truth_prices, ground_truth_greeks = compute_ground_truth(samples, q)
    print(f"✓ Ground truth computed for {n_samples} samples")

    # Evaluate methods
    print("\n" + "="*80)
    print("PHASE 3: Evaluate Methods on Random Samples")
    print("="*80)
    print("\nEvaluating price + 5 Greeks (delta, gamma, vega, theta, rho) for each sample...")

    results = []

    # print("\n1. Chebyshev Baseline:")
    # eval_time, errors = evaluate_chebyshev_method(
    #     cheb_baseline, samples, ground_truth_prices, ground_truth_greeks, "Chebyshev Baseline"
    # )
    # results.append(MethodResult("Chebyshev Baseline", cheb_baseline_time, eval_time, errors))

    # Chebyshev Barycentric (with fast_eval)
    print("\n1. Chebyshev Barycentric:")
    eval_time, errors = evaluate_chebyshev_method(
        cheb_barycentric, samples, ground_truth_prices, ground_truth_greeks, "Cheb Barycentric", use_fast_eval=True
    )
    results.append(MethodResult("Chebyshev Barycentric", cheb_barycentric_time, eval_time, errors, grid_points=int(np.prod([11, 11, 11, 11, 11]))))

    # MoCaX (if available)
    if mocax_obj is not None:
        print("\n2. MoCaX Standard:")
        eval_time, errors = evaluate_mocax_method(
            mocax_obj, samples, ground_truth_prices, ground_truth_greeks
        )
        results.append(MethodResult("MoCaX Standard", mocax_time, eval_time, errors, grid_points=mocax_grid))

    # # FDM
    # print("\n3. FDM (this will take a while - solving PDE for each sample):")
    # eval_time, errors = evaluate_fdm_method(
    #     samples, ground_truth_prices, ground_truth_greeks, q
    # )
    # results.append(MethodResult("FDM", None, eval_time, errors))

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print_build_time_table(results)
    print_accuracy_table(results)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Findings:")
    print("  • Build time: Time to pre-compute approximation (N/A for FDM)")
    print("  • Evaluation time: Total time to evaluate 10000 samples")
    print("  • Errors: Percentage error vs analytical formulas")
    print("\nInterpretation:")
    print("  • Chebyshev methods: Fast evaluation after pre-computation")
    print("  • MoCaX: Similar accuracy to Chebyshev, optimized implementation")
    print("  • FDM: Solves PDE from scratch - most accurate but slowest")
    print("="*80)


if __name__ == "__main__":
    main()
