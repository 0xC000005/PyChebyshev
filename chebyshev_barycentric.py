"""
Chebyshev Barycentric: Multi-Dimensional Approximation via Barycentric Interpolation

Core idea: For d-dimensional function, collapse dimensions one at a time using
barycentric interpolation formula.

Key optimization: Barycentric weights depend ONLY on node positions, not function values.
→ Pre-compute weights for ALL dimensions once!

Performance optimizations:
- Numba JIT compilation for hot-path functions
- Pre-allocated evaluation cache
- Optional fast_eval() method that skips validation

Uses numpy.polynomial.chebyshev for:
- chebpts1(): Generate Chebyshev nodes

Implements barycentric interpolation manually (simple formula, ~15 lines).
"""

import numpy as np
from numpy.polynomial.chebyshev import chebpts1
from numba import njit
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


def compute_differentiation_matrix(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute differentiation matrix for barycentric interpolation.

    Based on Berrut & Trefethen (2004), Section 9.3.
    Implementation follows scipy's BarycentricInterpolator.

    The differentiation matrix D satisfies: (D @ f) gives derivative values at nodes.

    For i ≠ j:
        D[i,j] = (w_j / w_i) / (x_i - x_j)

    For i = j:
        D[i,i] = -Σ(k≠i) D[i,k]

    Args:
        nodes: Interpolation nodes
        weights: Barycentric weights (pre-computed)

    Returns:
        D: Differentiation matrix (n × n)
    """
    n = len(nodes)

    # Compute node differences: c[i,j] = x_i - x_j
    c = nodes[:, np.newaxis] - nodes

    # Avoid division by zero on diagonal (temporarily set to 1)
    np.fill_diagonal(c, 1.0)

    # Apply barycentric weights: c[i,j] = (w_j / w_i) / (x_i - x_j)
    c = weights / (c * weights[:, np.newaxis])

    # Zero out diagonal temporarily
    np.fill_diagonal(c, 0.0)

    # Diagonal entries are negative sum of row: D[i,i] = -Σ(j≠i) D[i,j]
    d = -c.sum(axis=1)
    np.fill_diagonal(c, d)

    return c


@njit(cache=True, fastmath=True)
def barycentric_interpolate_jit(x: float, nodes: np.ndarray, values: np.ndarray,
                                weights: np.ndarray) -> float:
    """
    JIT-compiled barycentric interpolation (no node coincidence check).

    Formula: p(x) = Σ[w_i * f_i / (x - x_i)] / Σ[w_i / (x - x_i)]

    This is O(N) - just arithmetic, no polynomial fitting!
    Numba compiles this to machine code for 10-20× speedup.

    IMPORTANT: Assumes x does NOT coincide with any node (no division by zero check).
    Use barycentric_interpolate() wrapper for validation if needed.
    """
    sum_numerator = 0.0
    sum_denominator = 0.0

    for i in range(len(nodes)):
        w_i = weights[i] / (x - nodes[i])
        sum_numerator += w_i * values[i]
        sum_denominator += w_i

    return sum_numerator / sum_denominator


def barycentric_interpolate(x: float, nodes: np.ndarray, values: np.ndarray,
                           weights: np.ndarray, skip_check: bool = False) -> float:
    """
    Evaluate barycentric interpolation at point x.

    Formula: p(x) = Σ[w_i * f_i / (x - x_i)] / Σ[w_i / (x - x_i)]

    This is O(N) - just arithmetic, no polynomial fitting!

    Args:
        x: Evaluation point
        nodes: Interpolation nodes
        values: Function values at nodes
        weights: Pre-computed barycentric weights
        skip_check: If True, skip node coincidence check (faster but less safe)
    """
    if not skip_check:
        # Check if x coincides with a node (avoid division by zero)
        diffs = np.abs(nodes - x)
        if np.any(diffs < 1e-14):
            return float(values[np.argmin(diffs)])

    # Use JIT-compiled version
    return barycentric_interpolate_jit(x, nodes, values, weights)


def barycentric_derivative_analytical(x: float, nodes: np.ndarray, values: np.ndarray,
                                     weights: np.ndarray, diff_matrix: np.ndarray,
                                     order: int = 1) -> float:
    """
    Compute analytical derivative using differentiation matrix.

    Based on Berrut & Trefethen (2004), Section 9.3.

    Process:
    1. Apply differentiation matrix to values → derivative values at nodes
    2. Interpolate these derivative values to point x using barycentric formula
    3. For higher orders: apply differentiation matrix recursively

    Args:
        x: Evaluation point
        nodes: Interpolation nodes
        values: Function values at nodes
        weights: Barycentric weights (pre-computed)
        diff_matrix: Differentiation matrix (pre-computed)
        order: Derivative order (1 or 2)

    Returns:
        Derivative value at x
    """
    if order == 1:
        # First derivative: D @ values gives derivative values at nodes
        deriv_values = diff_matrix @ values
        # Interpolate derivative values to point x
        return barycentric_interpolate(x, nodes, deriv_values, weights)
    elif order == 2:
        # Second derivative: Apply differentiation matrix twice
        # D @ (D @ values) gives second derivative values at nodes
        deriv_values = diff_matrix @ (diff_matrix @ values)
        # Interpolate second derivative values to point x
        return barycentric_interpolate(x, nodes, deriv_values, weights)
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
        self.diff_matrices = None  # Differentiation matrices for analytical derivatives
        self.build_time = 0
        self.n_evaluations = 0

        # Pre-allocated evaluation cache (reused across evaluations for speed)
        # Maps dimension index to pre-allocated array for dimensional collapse
        self._eval_cache = {}

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

        # Step 3: Pre-compute differentiation matrices for analytical derivatives
        # Differentiation matrices depend only on nodes and weights, not function values!
        print(f"Pre-computing differentiation matrices for all {self.num_dimensions} dimensions...")
        self.diff_matrices = []
        for d in range(self.num_dimensions):
            D = compute_differentiation_matrix(self.nodes[d], self.weights[d])
            self.diff_matrices.append(D)

        self.build_time = time.time() - start

        # Pre-allocate evaluation cache for fast_eval()
        # Cache arrays for dimensional collapse (reused across evaluations)
        for d in range(self.num_dimensions - 1, 0, -1):
            shape = tuple(self.n_nodes[i] for i in range(d))
            self._eval_cache[d] = np.zeros(shape)

        total_weights = sum(len(w) for w in self.weights)
        total_diff_matrix_elements = sum(D.size for D in self.diff_matrices)
        cache_size = sum(arr.size for arr in self._eval_cache.values())
        print(f"✓ Built in {self.build_time:.3f}s")
        print(f"  Function evaluations: {self.n_evaluations:,}")
        print(f"  Pre-computed weights: {total_weights} floats ({total_weights * 8} bytes)")
        print(f"  Pre-computed diff matrices: {total_diff_matrix_elements} floats ({total_diff_matrix_elements * 8} bytes)")
        print(f"  Pre-allocated cache: {cache_size} floats ({cache_size * 8} bytes)")
        print(f"  Uniform O(N) evaluation for all dimensions!")
        print(f"  Analytical derivatives via differentiation matrices!")
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
            diff_matrix = self.diff_matrices[d]

            if d == 0:
                # Final dimension: collapse to scalar
                if deriv == 0:
                    return barycentric_interpolate(x, nodes, current, weights)
                else:
                    return barycentric_derivative_analytical(x, nodes, current, weights, diff_matrix, deriv)
            else:
                # Intermediate dimension: collapse to lower-dimensional array
                shape = current.shape[:d]
                new = np.zeros(shape)

                for idx in np.ndindex(*shape):
                    # Extract 1D slice
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]

                    # Barycentric interpolation (uses pre-computed weights and diff matrices!)
                    if deriv == 0:
                        new[idx] = barycentric_interpolate(x, nodes, values_1d, weights)
                    else:
                        new[idx] = barycentric_derivative_analytical(x, nodes, values_1d, weights, diff_matrix, deriv)

                current = new

    def fast_eval(self, point: List[float], derivative_order: List[int]) -> float:
        """
        Fast evaluation without validation (use after testing).

        Performance optimizations:
        - Skips node coincidence checks (uses JIT version directly)
        - Reuses pre-allocated cache arrays (no memory allocation)
        - Minimal overhead for production use

        WARNING: Assumes build() has been called and point is valid.
        Use eval() during testing/validation, fast_eval() in production.

        Expected speedup: 20-50× faster than original eval() on 5D problems.
        """
        current = self.tensor_values

        # Collapse from last dimension to first
        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]
            nodes = self.nodes[d]
            weights = self.weights[d]
            diff_matrix = self.diff_matrices[d]

            # Check node coincidence once per dimension (avoid div-by-zero in JIT)
            coincident_idx = None
            diffs = np.abs(x - nodes)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 1e-14:
                coincident_idx = int(min_idx)

            if d == 0:
                # Final dimension: collapse to scalar
                if deriv == 0:
                    if coincident_idx is not None:
                        return float(current[coincident_idx])
                    return barycentric_interpolate_jit(x, nodes, current, weights)
                else:
                    return barycentric_derivative_analytical(x, nodes, current, weights, diff_matrix, deriv)
            else:
                # Intermediate dimension: collapse to lower-dimensional array
                # Reuse pre-allocated cache (no new allocation!)
                shape = current.shape[:d]
                cache = self._eval_cache[d]

                for idx in np.ndindex(*shape):
                    # Extract 1D slice
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]

                    # Barycentric interpolation (uses JIT version and analytical derivatives!)
                    if deriv == 0:
                        if coincident_idx is not None:
                            cache[idx] = values_1d[coincident_idx]
                        else:
                            cache[idx] = barycentric_interpolate_jit(x, nodes, values_1d, weights)
                    else:
                        cache[idx] = barycentric_derivative_analytical(x, nodes, values_1d, weights, diff_matrix, deriv)

                current = cache

    def vectorized_eval(self, point: List[float], derivative_order: List[int]) -> float:
        """
        Fully vectorized evaluation using NumPy matrix operations.

        Replaces the Python np.ndindex loop with NumPy @ (matmul) operations:
        - current @ w_over_diff contracts the last axis (barycentric interpolation)
        - current @ D.T applies the differentiation matrix along the last axis

        For 5D with 11 nodes: 5 BLAS calls instead of 16,105 Python iterations.
        Expected ~30-50× faster than fast_eval().
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        current = self.tensor_values

        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]

            # Apply diff matrix along last axis if derivative needed
            if deriv > 0:
                D_T = self.diff_matrices[d].T
                for _ in range(deriv):
                    current = current @ D_T

            # Barycentric interpolation: contract last axis
            diff = x - self.nodes[d]
            exact = np.where(np.abs(diff) < 1e-14)[0]
            if len(exact) > 0:
                current = current[..., exact[0]]
            else:
                w_over_diff = self.weights[d] / diff
                current = current @ w_over_diff / np.sum(w_over_diff)

        return float(current)

    def vectorized_eval_batch(self, points: np.ndarray, derivative_order: List[int]) -> np.ndarray:
        """
        Evaluate at N points simultaneously. points shape: (N, num_dims).

        Returns array of shape (N,) with interpolated values.
        """
        N = points.shape[0]
        results = np.empty(N)
        for i in range(N):
            results[i] = self.vectorized_eval(points[i], derivative_order)
        return results

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

    # Verify vectorized_eval produces same results
    print(f"\n--- Vectorized evaluation verification ---")
    p_test = [100, 100, 1.0, 0.25, 0.05]
    price_orig = cheb.eval(p_test, [0, 0, 0, 0, 0])
    price_vec = cheb.vectorized_eval(p_test, [0, 0, 0, 0, 0])
    print(f"  eval():            {price_orig:.10f}")
    print(f"  vectorized_eval(): {price_vec:.10f}")
    print(f"  Difference:        {abs(price_orig - price_vec):.2e}")

    for name, (deriv, exact) in greeks.items():
        v_orig = cheb.eval(p_test, deriv)
        v_vec = cheb.vectorized_eval(p_test, deriv)
        diff = abs(v_orig - v_vec)
        print(f"  {name}: diff = {diff:.2e}")

    # Performance comparison
    print(f"\n--- Performance comparison (100 evals, price only) ---")
    n_bench = 100
    # Use random points in domain interior to avoid Chebyshev node coincidence
    rng = np.random.default_rng(42)
    test_points = []
    for _ in range(n_bench):
        test_points.append([
            rng.uniform(85, 115),   # S
            rng.uniform(92, 108),   # K
            rng.uniform(0.3, 0.9),  # T
            rng.uniform(0.17, 0.33),# sigma
            rng.uniform(0.02, 0.07) # r
        ])
    zero_deriv = [0, 0, 0, 0, 0]

    # Warmup JIT
    cheb.fast_eval(test_points[0], zero_deriv)

    import time as _time

    t0 = _time.perf_counter()
    for pt in test_points:
        cheb.eval(pt, zero_deriv)
    t_eval = (_time.perf_counter() - t0) / n_bench * 1000

    t0 = _time.perf_counter()
    for pt in test_points:
        cheb.fast_eval(pt, zero_deriv)
    t_fast = (_time.perf_counter() - t0) / n_bench * 1000

    t0 = _time.perf_counter()
    for pt in test_points:
        cheb.vectorized_eval(pt, zero_deriv)
    t_vec = (_time.perf_counter() - t0) / n_bench * 1000

    print(f"  eval():            {t_eval:.3f} ms/query")
    print(f"  fast_eval():       {t_fast:.3f} ms/query")
    print(f"  vectorized_eval(): {t_vec:.3f} ms/query")
    print(f"  Speedup (vec vs fast_eval): {t_fast / t_vec:.1f}×")

    # Price + Greeks timing
    print(f"\n--- Performance comparison (100 evals, price + 4 Greeks) ---")
    all_derivs = [zero_deriv, [1,0,0,0,0], [2,0,0,0,0], [0,0,0,1,0], [0,0,0,0,1]]

    t0 = _time.perf_counter()
    for pt in test_points:
        for d in all_derivs:
            cheb.fast_eval(pt, d)
    t_fast_all = (_time.perf_counter() - t0) / n_bench * 1000

    t0 = _time.perf_counter()
    for pt in test_points:
        for d in all_derivs:
            cheb.vectorized_eval(pt, d)
    t_vec_all = (_time.perf_counter() - t0) / n_bench * 1000

    print(f"  fast_eval():       {t_fast_all:.3f} ms/sample (5 metrics)")
    print(f"  vectorized_eval(): {t_vec_all:.3f} ms/sample (5 metrics)")
    print(f"  Speedup: {t_fast_all / t_vec_all:.1f}×")

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
