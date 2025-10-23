"""
Chebyshev Barycentric: Multi-Dimensional Approximation via Barycentric Interpolation

Core idea: For d-dimensional function, collapse dimensions one at a time using
barycentric interpolation formula.

Key optimization: Barycentric weights depend ONLY on node positions, not function values.
→ Pre-compute weights for ALL dimensions once!

Uses numpy.polynomial.chebyshev for:
- chebpts1(): Generate Chebyshev nodes

Implements barycentric interpolation manually (simple formula, ~15 lines).
"""

import numpy as np
from numpy.polynomial.chebyshev import chebpts1
import time
import math
from typing import Callable, List, Tuple
from blackscholes import BlackScholesCall


def compute_barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """
    Compute barycentric weights for given nodes.

    Formula: w_i = 1 / ∏(j≠i) (x_i - x_j)

    These weights depend ONLY on node positions, not function values!
    """
    n = len(nodes)
    weights = np.ones(n)

    for i in range(n):
        for j in range(n):
            if j != i:
                weights[i] /= (nodes[i] - nodes[j])

    return weights


def barycentric_interpolate(x: float, nodes: np.ndarray, values: np.ndarray,
                           weights: np.ndarray) -> float:
    """
    Evaluate barycentric interpolation at point x.

    Formula: p(x) = Σ[w_i * f_i / (x - x_i)] / Σ[w_i / (x - x_i)]

    This is O(N) - just arithmetic, no polynomial fitting!
    Vectorized implementation for performance (eliminates Python loops).
    """
    # Check if x coincides with a node (avoid division by zero)
    diffs = np.abs(nodes - x)
    if np.any(diffs < 1e-14):
        return float(values[np.argmin(diffs)])

    # Vectorized barycentric formula (eliminates loops!)
    w = weights / (x - nodes)
    return float(np.sum(w * values) / np.sum(w))


def barycentric_derivative(x: float, nodes: np.ndarray, values: np.ndarray,
                          weights: np.ndarray, order: int = 1) -> float:
    """
    Compute derivative via finite difference on barycentric interpolant.

    Uses adaptive epsilon based on domain scale for better numerical stability.
    """
    # Adaptive epsilon: scale with domain size
    domain_scale = np.ptp(nodes)  # max - min
    eps = max(1e-7 * domain_scale, 1e-8)  # Scale with domain, but not too small

    if order == 1:
        # First derivative: central difference
        f_plus = barycentric_interpolate(x + eps, nodes, values, weights)
        f_minus = barycentric_interpolate(x - eps, nodes, values, weights)
        return (f_plus - f_minus) / (2 * eps)
    elif order == 2:
        # Second derivative: use better finite difference formula
        # 5-point stencil for better accuracy
        h = eps
        f_pp = barycentric_interpolate(x + 2*h, nodes, values, weights)
        f_p = barycentric_interpolate(x + h, nodes, values, weights)
        f_c = barycentric_interpolate(x, nodes, values, weights)
        f_m = barycentric_interpolate(x - h, nodes, values, weights)
        f_mm = barycentric_interpolate(x - 2*h, nodes, values, weights)

        # 5-point formula: (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h²)
        return (-f_pp + 16*f_p - 30*f_c + 16*f_m - f_mm) / (12 * h**2)
    else:
        raise ValueError(f"Derivative order {order} not supported")


class ChebyshevApproximation:
    """
    Multi-dimensional Chebyshev approximation using barycentric interpolation.

    Key advantage: Pre-compute barycentric weights for ALL dimensions!

    Strategy:
    1. Build:
       - Evaluate function at all Chebyshev node combinations
       - Pre-compute barycentric weights for each dimension (just depends on nodes!)
    2. Query:
       - Use barycentric formula for ALL dimensions (uniform O(N) approach)
       - No polynomial fitting needed during evaluation!

    Example (3D):
    - Build:
        * f(x,y,z) at all node combinations → tensor[i,j,k]
        * weights_x = compute_weights(nodes_x)
        * weights_y = compute_weights(nodes_y)
        * weights_z = compute_weights(nodes_z)
    - Query f(x*, y*, z*):
        * For each (i,j): temp[i,j] = barycentric(z*, nodes_z, tensor[i,j,:], weights_z)
        * For each i: temp[i] = barycentric(y*, nodes_y, temp[i,:], weights_y)
        * result = barycentric(x*, nodes_x, temp[:], weights_x)
    """

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        max_derivative_order: int = 2
    ):
        """
        Args:
            function: f(x: List[float], additional_data) -> float
            num_dimensions: Number of dimensions
            domain: [(min, max), ...] for each dimension
            n_nodes: [n1, n2, ...] nodes per dimension
            max_derivative_order: Max derivative order (1 or 2)
        """
        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.n_nodes = n_nodes
        self.max_derivative_order = max_derivative_order

        # Generate Chebyshev nodes for each dimension
        self.nodes = []
        for d in range(num_dimensions):
            nodes_std = chebpts1(n_nodes[d])
            a, b = domain[d]
            nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
            self.nodes.append(np.sort(nodes))

        # Storage
        self.tensor_values = None
        self.weights = None  # Barycentric weights for ALL dimensions
        self.build_time = 0
        self.n_evaluations = 0

    def build(self):
        """Evaluate function and pre-compute barycentric weights."""
        print(f"\n{'='*70}")
        print(f"Building {self.num_dimensions}D Chebyshev Approximation (Barycentric)")
        print(f"{'='*70}")
        total = np.prod(self.n_nodes)
        print(f"Nodes per dimension: {self.n_nodes}")
        print(f"Total grid points: {total:,}")

        start = time.time()

        # Step 1: Evaluate at all node combinations
        self.tensor_values = np.zeros(self.n_nodes)
        for idx in np.ndindex(*self.n_nodes):
            point = [self.nodes[d][idx[d]] for d in range(self.num_dimensions)]
            self.tensor_values[idx] = self.function(point, None)

        self.n_evaluations = np.prod(self.n_nodes)

        # Step 2: Pre-compute barycentric weights for ALL dimensions
        # This is the key optimization: weights depend only on nodes, not values!
        print(f"Pre-computing barycentric weights for all {self.num_dimensions} dimensions...")
        self.weights = []
        for d in range(self.num_dimensions):
            w = compute_barycentric_weights(self.nodes[d])
            self.weights.append(w)

        self.build_time = time.time() - start

        total_weights = sum(len(w) for w in self.weights)
        print(f"✓ Built in {self.build_time:.3f}s")
        print(f"  Function evaluations: {self.n_evaluations:,}")
        print(f"  Pre-computed weights: {total_weights} floats ({total_weights * 8} bytes)")
        print(f"  Uniform O(N) evaluation for all dimensions!")
        print(f"{'='*70}")

    def eval(self, point: List[float], derivative_order: List[int]) -> float:
        """
        Evaluate using dimensional decomposition with barycentric interpolation.

        Key: Uses pre-computed weights for ALL dimensions - uniform O(N) approach!
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        current = self.tensor_values  # Use reference, not copy (never modified in place)

        # Collapse from last dimension to first
        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]
            nodes = self.nodes[d]
            weights = self.weights[d]

            if d == 0:
                # Final dimension: collapse to scalar
                if deriv == 0:
                    return barycentric_interpolate(x, nodes, current, weights)
                else:
                    return barycentric_derivative(x, nodes, current, weights, deriv)
            else:
                # Intermediate dimension: collapse to lower-dimensional array
                shape = current.shape[:d]
                new = np.zeros(shape)

                for idx in np.ndindex(*shape):
                    # Extract 1D slice
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]

                    # Barycentric interpolation (uses pre-computed weights!)
                    if deriv == 0:
                        new[idx] = barycentric_interpolate(x, nodes, values_1d, weights)
                    else:
                        new[idx] = barycentric_derivative(x, nodes, values_1d, weights, deriv)

                current = new

    def get_derivative_id(self, derivative_order: List[int]) -> List[int]:
        """Get derivative ID (for API compatibility)."""
        return derivative_order


# ============================================================================
# Tests
# ============================================================================

def test_simple_3d():
    """Test 3D: sin(x) + sin(y) + sin(z)"""
    print("\n" + "="*70)
    print("TEST 1: sin(x) + sin(y) + sin(z)")
    print("="*70)

    def f(x, _):
        return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

    cheb = ChebyshevApproximation(
        f, 3, [[-1, 1], [-1, 1], [1, 3]], [10, 8, 4], max_derivative_order=2
    )
    cheb.build()

    # Test point
    p = [0.1, 0.3, 1.7]
    original = f(p, None)
    approx = cheb.eval(p, [0, 0, 0])
    error = abs(approx - original) / abs(original) * 100

    print(f"\nAt {p}:")
    print(f"  Original:  {original:.10f}")
    print(f"  Barycentric: {approx:.10f}")
    print(f"  Error:     {error:.4f}%")

    # Test derivative df/dy
    deriv_approx = cheb.eval(p, [0, 1, 0])
    deriv_exact = math.cos(p[1])
    deriv_error = abs(deriv_approx - deriv_exact)

    print(f"\ndf/dy:")
    print(f"  Exact:     {deriv_exact:.10f}")
    print(f"  Barycentric: {deriv_approx:.10f}")
    print(f"  Error:     {deriv_error:.2e}")

    return error < 1.0


def test_black_scholes_3d():
    """Test 3D Black-Scholes: C(S, T, σ)"""
    print("\n" + "="*70)
    print("TEST 2: Black-Scholes C(S, T, σ)")
    print("="*70)

    K, r, q = 100.0, 0.05, 0.02

    def bs(x, _):
        return BlackScholesCall(S=x[0], K=K, T=x[1], r=r, sigma=x[2], q=q).price()

    cheb = ChebyshevApproximation(
        bs, 3, [[50, 150], [0.1, 2.0], [0.1, 0.5]], [15, 12, 10], max_derivative_order=2
    )
    cheb.build()

    # Test cases
    cases = [
        ([100, 1.0, 0.25], "ATM"),
        ([120, 1.0, 0.25], "ITM"),
        ([80, 1.0, 0.25], "OTM"),
    ]

    print(f"\n{'Case':<6} {'Price (Exact)':>13} {'Price (Bary)':>13} {'Error':>8}")
    print("-" * 50)

    max_err = 0
    for p, name in cases:
        exact = BlackScholesCall(S=p[0], K=K, T=p[1], r=r, sigma=p[2], q=q).price()
        approx = cheb.eval(p, [0, 0, 0])
        err = abs(approx - exact) / exact * 100
        max_err = max(max_err, err)
        print(f"{name:<6} {exact:>13.6f} {approx:>13.6f} {err:>7.3f}%")

    # Delta at ATM
    p = [100, 1.0, 0.25]
    opt = BlackScholesCall(S=p[0], K=K, T=p[1], r=r, sigma=p[2], q=q)
    delta_exact = opt.delta()
    delta_approx = cheb.eval(p, [1, 0, 0])
    delta_err = abs(delta_approx - delta_exact) / delta_exact * 100

    print(f"\nDelta at ATM:")
    print(f"  Exact:     {delta_exact:.6f}")
    print(f"  Barycentric: {delta_approx:.6f}")
    print(f"  Error:     {delta_err:.3f}%")

    return max_err < 0.5


def test_5d_black_scholes():
    """Test 5D: V(S, K, T, σ, r)"""
    print("\n" + "="*70)
    print("TEST 3: 5D Black-Scholes V(S, K, T, σ, r)")
    print("="*70)

    q = 0.02

    def bs_5d(x, _):
        return BlackScholesCall(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=q).price()

    cheb = ChebyshevApproximation(
        bs_5d,
        5,
        [[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
        [11, 11, 11, 11, 11],
        max_derivative_order=2
    )
    cheb.build()

    # Test cases
    cases = [
        ([100, 100, 1.0, 0.25, 0.05], "ATM"),
        ([110, 100, 1.0, 0.25, 0.05], "ITM"),
        ([90, 100, 1.0, 0.25, 0.05], "OTM"),
        ([100, 100, 0.5, 0.25, 0.05], "Short T"),
        ([100, 100, 1.0, 0.35, 0.05], "High vol"),
    ]

    print(f"\n{'Case':<10} {'Price (Exact)':>13} {'Price (Bary)':>13} {'Error':>8}")
    print("-" * 50)

    errors = []
    for p, name in cases:
        exact = BlackScholesCall(S=p[0], K=p[1], T=p[2], r=p[4], sigma=p[3], q=q).price()
        approx = cheb.eval(p, [0, 0, 0, 0, 0])
        err = abs(approx - exact) / exact * 100
        errors.append(err)
        print(f"{name:<10} {exact:>13.6f} {approx:>13.6f} {err:>7.3f}%")

    # Greeks at ATM
    p = [100, 100, 1.0, 0.25, 0.05]
    opt = BlackScholesCall(S=p[0], K=p[1], T=p[2], r=p[4], sigma=p[3], q=q)

    greeks = {
        'Delta': ([1, 0, 0, 0, 0], opt.delta()),
        'Gamma': ([2, 0, 0, 0, 0], opt.gamma()),
        'Vega': ([0, 0, 0, 1, 0], opt.vega()),
        'Rho': ([0, 0, 0, 0, 1], opt.rho()),
    }

    print(f"\nGreeks at ATM:")
    print(f"{'Greek':<8} {'Exact':>12} {'Barycentric':>12} {'Error':>8}")
    print("-" * 50)

    greek_errors = []
    for name, (deriv, exact) in greeks.items():
        approx = cheb.eval(p, deriv)
        err = abs(approx - exact) / exact * 100
        greek_errors.append(err)
        print(f"{name:<8} {exact:>12.6f} {approx:>12.6f} {err:>7.3f}%")

    max_price_err = max(errors)
    max_greek_err = max(greek_errors)

    print(f"\nMax errors: Price {max_price_err:.3f}%, Greeks {max_greek_err:.3f}%")

    # Show optimization benefit
    total_weights = sum(len(w) for w in cheb.weights)
    print(f"\nKey advantages:")
    print(f"  • Pre-computed weights: {total_weights} floats (vs {np.prod(cheb.n_nodes[:-1]):,} polynomials)")
    print(f"  • Uniform O(N) evaluation for ALL dimensions")
    print(f"  • No polynomial fitting during queries!")

    return max_price_err < 1.0 and max_greek_err < 10.0


def main():
    """Run all tests."""
    print("="*70)
    print("Chebyshev Barycentric: Pure NumPy + Manual Barycentric")
    print("="*70)
    print("Strategy: Dimensional decomposition with barycentric interpolation")
    print("Uses: chebpts1() for nodes + manual barycentric formula")
    print("Optimization: Pre-compute weights for ALL dimensions (not just innermost!)")

    results = [
        ("Simple 3D", test_simple_3d),
        ("Black-Scholes 3D", test_black_scholes_3d),
        ("5D Parametric BS", test_5d_black_scholes),
    ]

    passed = []
    for name, test_fn in results:
        try:
            result = test_fn()
            passed.append((name, result))
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"\n{status}")
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            passed.append((name, False))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in passed:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    all_passed = all(r for _, r in passed)
    print("="*70)
    print("✓ All tests PASSED!" if all_passed else "✗ Some tests FAILED")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
