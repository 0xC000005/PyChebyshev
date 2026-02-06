"""
Chebyshev Baseline: Multi-Dimensional Approximation via 1D Chebyshev Interpolation

Core idea: For d-dimensional function, collapse dimensions one at a time using
1D Chebyshev polynomial interpolation.

Uses numpy.polynomial.chebyshev throughout:
- chebpts1(): Generate Chebyshev nodes
- Chebyshev.interpolate(): Build interpolating polynomial (computes coefficients)
- poly(x): Evaluate pre-computed polynomial
- .deriv(): Analytical derivatives
"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebpts1
import time
import math
from typing import Callable, List, Tuple
from blackscholes import BlackScholesCall


class ChebyshevApproximation:
    """
    Multi-dimensional Chebyshev approximation using dimensional decomposition.

    Strategy:
    1. Build:
       - Evaluate function at all Chebyshev node combinations
       - Pre-compute interpolating polynomials for innermost dimension
    2. Query:
       - Use pre-computed polynomials for innermost dimension
       - Dynamically interpolate outer dimensions (values depend on query point)

    Example (3D):
    - Build: f(x,y,z) at all node combinations → tensor[i,j,k]
           Pre-compute polynomials for each (i,j): poly[i,j](z)
    - Query f(x*, y*, z*):
      * Evaluate pre-computed poly[i,j](z*) → temp[i,j]  (fast!)
      * For each i: interpolate temp[i,:] → poly[i](y*) → temp[i]
      * Interpolate temp[:] → poly(x*) → result
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
            # Get standard Chebyshev nodes on [-1, 1]
            nodes_std = chebpts1(n_nodes[d])
            # Map to [a, b]
            a, b = domain[d]
            nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
            self.nodes.append(np.sort(nodes))

        # Storage
        self.tensor_values = None
        self.innermost_polys = None  # Pre-computed polynomials
        self.build_time = 0
        self.n_evaluations = 0

    def build(self):
        """Evaluate function at all Chebyshev node combinations and pre-compute polynomials."""
        print(f"\n{'='*70}")
        print(f"Building {self.num_dimensions}D Chebyshev Approximation")
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

        # Step 2: Pre-compute interpolating polynomials for innermost dimension
        # This is the optimization: we can pre-compute these because they don't
        # depend on the query point, only on the tensor values
        if self.num_dimensions > 1:
            print(f"Pre-computing interpolating polynomials for innermost dimension...")
            outer_shape = self.n_nodes[:-1]
            self.innermost_polys = np.empty(outer_shape, dtype=object)

            for idx in np.ndindex(*outer_shape):
                # Get 1D slice along innermost dimension
                values_1d = self.tensor_values[idx + (slice(None),)]

                # Build interpolating polynomial (computes Chebyshev coefficients)
                # This returns a Chebyshev object with stored coefficients
                poly = Chebyshev.interpolate(
                    lambda x, v=values_1d: np.interp(x, self.nodes[-1], v),
                    deg=self.n_nodes[-1] - 1,
                    domain=self.domain[-1]
                )
                self.innermost_polys[idx] = poly

        self.build_time = time.time() - start

        print(f"✓ Built in {self.build_time:.3f}s")
        print(f"  Function evaluations: {self.n_evaluations:,}")
        print(f"  Pre-computed polynomials: {np.prod(self.n_nodes[:-1]) if self.num_dimensions > 1 else 0:,}")
        print(f"{'='*70}")

    def eval(self, point: List[float], derivative_order: List[int]) -> float:
        """
        Evaluate using dimensional decomposition with pre-computed polynomials.

        Key optimization: Innermost dimension uses pre-computed polynomials (no re-interpolation!).
        Outer dimensions must be interpolated dynamically (depend on query point).
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        # Special case: 1D
        if self.num_dimensions == 1:
            poly = Chebyshev.interpolate(
                lambda x: np.interp(x, self.nodes[0], self.tensor_values),
                deg=self.n_nodes[0] - 1,
                domain=self.domain[0]
            )
            if derivative_order[0] > 0:
                poly = poly.deriv(derivative_order[0])
            return float(poly(point[0]))

        # Multi-dimensional: Use pre-computed for innermost, interpolate others
        current = None

        # Start from innermost dimension
        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]

            if d == self.num_dimensions - 1:
                # Innermost dimension: Use pre-computed polynomials!
                outer_shape = self.n_nodes[:-1]
                current = np.zeros(outer_shape)

                for idx in np.ndindex(*outer_shape):
                    poly = self.innermost_polys[idx]
                    if deriv > 0:
                        poly = poly.deriv(deriv)
                    current[idx] = poly(x)

            elif d == 0:
                # Outermost dimension: final collapse to scalar
                poly = Chebyshev.interpolate(
                    lambda x_val, v=current, nodes=self.nodes[d]: np.interp(x_val, nodes, v),
                    deg=self.n_nodes[d] - 1,
                    domain=self.domain[d]
                )
                if deriv > 0:
                    poly = poly.deriv(deriv)
                return float(poly(x))

            else:
                # Middle dimensions: interpolate dynamically
                shape = current.shape[:d]
                new = np.zeros(shape)

                for idx in np.ndindex(*shape):
                    # Extract 1D slice
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]

                    # Build interpolating polynomial
                    poly = Chebyshev.interpolate(
                        lambda x_val, v=values_1d, nodes=self.nodes[d]: np.interp(x_val, nodes, v),
                        deg=self.n_nodes[d] - 1,
                        domain=self.domain[d]
                    )
                    if deriv > 0:
                        poly = poly.deriv(deriv)
                    new[idx] = poly(x)

                current = new

        return float(current)

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
    print(f"  Chebyshev: {approx:.10f}")
    print(f"  Error:     {error:.4f}%")

    # Test derivative df/dy
    deriv_approx = cheb.eval(p, [0, 1, 0])
    deriv_exact = math.cos(p[1])
    deriv_error = abs(deriv_approx - deriv_exact)

    print(f"\ndf/dy:")
    print(f"  Exact:     {deriv_exact:.10f}")
    print(f"  Chebyshev: {deriv_approx:.10f}")
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

    print(f"\n{'Case':<6} {'Price (Exact)':>13} {'Price (Cheb)':>13} {'Error':>8}")
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
    print(f"  Chebyshev: {delta_approx:.6f}")
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

    print(f"\n{'Case':<10} {'Price (Exact)':>13} {'Price (Cheb)':>13} {'Error':>8}")
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
    print(f"{'Greek':<8} {'Exact':>12} {'Chebyshev':>12} {'Error':>8}")
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
    print(f"\nOptimization: Pre-computed {np.prod(cheb.n_nodes[:-1]):,} polynomials")
    print(f"  (Innermost dimension uses pre-computed polys - no re-interpolation!)")

    return max_price_err < 1.0 and max_greek_err < 10.0


def main():
    """Run all tests."""
    print("="*70)
    print("Chebyshev Baseline: Pure NumPy Implementation")
    print("="*70)
    print("Strategy: Dimensional decomposition with pre-computed polynomials")
    print("Uses: Chebyshev.interpolate() + pre-computing")

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
