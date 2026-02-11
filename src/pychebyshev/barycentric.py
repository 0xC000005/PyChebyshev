"""Multi-dimensional Chebyshev approximation via barycentric interpolation.

This module implements the core algorithm: dimensional decomposition of an
N-dimensional tensor using the barycentric interpolation formula. Barycentric
weights depend only on node positions, enabling full pre-computation across
all dimensions.

References
----------
- Berrut & Trefethen (2004), "Barycentric Lagrange Interpolation",
  SIAM Review 46(3):501-517
- Gaß et al. (2018), "Chebyshev Interpolation for Parametric Option Pricing",
  Finance and Stochastics 22(3):701-731
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Callable, List, Tuple

import numpy as np
from numpy.polynomial.chebyshev import chebpts1

from pychebyshev._jit import barycentric_interpolate_jit


def compute_barycentric_weights(nodes: np.ndarray) -> np.ndarray:
    """Compute barycentric weights for given nodes.

    Parameters
    ----------
    nodes : ndarray
        Interpolation nodes of shape (n,).

    Returns
    -------
    ndarray
        Barycentric weights w_i = 1 / prod_{j!=i} (x_i - x_j).
    """
    n = len(nodes)
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                weights[i] /= (nodes[i] - nodes[j])
    return weights


def compute_differentiation_matrix(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute spectral differentiation matrix for barycentric interpolation.

    Based on Berrut & Trefethen (2004), Section 9.3.

    Parameters
    ----------
    nodes : ndarray
        Interpolation nodes of shape (n,).
    weights : ndarray
        Barycentric weights of shape (n,).

    Returns
    -------
    ndarray
        Differentiation matrix D of shape (n, n) such that D @ f gives
        derivative values at nodes.
    """
    n = len(nodes)
    c = nodes[:, np.newaxis] - nodes
    np.fill_diagonal(c, 1.0)
    c = weights / (c * weights[:, np.newaxis])
    np.fill_diagonal(c, 0.0)
    d = -c.sum(axis=1)
    np.fill_diagonal(c, d)
    return c


def barycentric_interpolate(x: float, nodes: np.ndarray, values: np.ndarray,
                            weights: np.ndarray, skip_check: bool = False) -> float:
    """Evaluate barycentric interpolation at a single point.

    Parameters
    ----------
    x : float
        Evaluation point.
    nodes : ndarray
        Interpolation nodes.
    values : ndarray
        Function values at nodes.
    weights : ndarray
        Barycentric weights.
    skip_check : bool, optional
        If True, skip node coincidence check (faster but may divide by zero).

    Returns
    -------
    float
        Interpolated value p(x).
    """
    if not skip_check:
        diffs = np.abs(nodes - x)
        if np.any(diffs < 1e-14):
            return float(values[np.argmin(diffs)])
    return barycentric_interpolate_jit(x, nodes, values, weights)


def barycentric_derivative_analytical(x: float, nodes: np.ndarray, values: np.ndarray,
                                      weights: np.ndarray, diff_matrix: np.ndarray,
                                      order: int = 1) -> float:
    """Compute analytical derivative using the differentiation matrix.

    Parameters
    ----------
    x : float
        Evaluation point.
    nodes : ndarray
        Interpolation nodes.
    values : ndarray
        Function values at nodes.
    weights : ndarray
        Barycentric weights.
    diff_matrix : ndarray
        Spectral differentiation matrix.
    order : int, optional
        Derivative order (1 or 2). Default is 1.

    Returns
    -------
    float
        Derivative value at x.

    Raises
    ------
    ValueError
        If order is not 1 or 2.
    """
    if order == 1:
        deriv_values = diff_matrix @ values
        return barycentric_interpolate(x, nodes, deriv_values, weights)
    elif order == 2:
        deriv_values = diff_matrix @ (diff_matrix @ values)
        return barycentric_interpolate(x, nodes, deriv_values, weights)
    else:
        raise ValueError(f"Derivative order {order} not supported (use 1 or 2)")


class ChebyshevApproximation:
    """Multi-dimensional Chebyshev approximation using barycentric interpolation.

    Pre-computes barycentric weights for all dimensions at build time,
    enabling uniform O(N) evaluation complexity for every dimension.
    Supports analytical derivatives via spectral differentiation matrices.

    Parameters
    ----------
    function : callable
        Function to approximate. Signature: ``f(point, data) -> float``
        where ``point`` is a list of floats and ``data`` is arbitrary
        additional data (can be None).
    num_dimensions : int
        Number of input dimensions.
    domain : list of (float, float)
        Bounds [(lo, hi), ...] for each dimension.
    n_nodes : list of int
        Number of Chebyshev nodes per dimension.
    max_derivative_order : int, optional
        Maximum derivative order to support. Default is 2.

    Examples
    --------
    >>> import math
    >>> def f(x, _):
    ...     return math.sin(x[0]) + math.sin(x[1])
    >>> cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
    >>> cheb.build()  # doctest: +SKIP
    >>> cheb.vectorized_eval([0.5, 0.3], [0, 0])  # doctest: +SKIP
    0.7764...
    """

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        max_derivative_order: int = 2,
    ):
        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.n_nodes = n_nodes
        self.max_derivative_order = max_derivative_order

        # Generate Chebyshev nodes for each dimension
        self.nodes: List[np.ndarray] = []
        for d in range(num_dimensions):
            nodes_std = chebpts1(n_nodes[d])
            a, b = domain[d]
            nodes = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
            self.nodes.append(np.sort(nodes))

        self.tensor_values: np.ndarray | None = None
        self.weights: List[np.ndarray] | None = None
        self.diff_matrices: List[np.ndarray] | None = None
        self.build_time: float = 0.0
        self.n_evaluations: int = 0
        self._eval_cache: dict = {}
        self._cached_error_estimate: float | None = None

    def build(self, verbose: bool = True) -> None:
        """Evaluate the function at all node combinations and pre-compute weights.

        Parameters
        ----------
        verbose : bool, optional
            If True, print build progress. Default is True.
        """
        total = int(np.prod(self.n_nodes))
        if verbose:
            print(f"Building {self.num_dimensions}D Chebyshev approximation "
                  f"({total:,} evaluations)...")

        start = time.time()
        self._cached_error_estimate = None

        # Step 1: Evaluate at all node combinations
        self.tensor_values = np.zeros(self.n_nodes)
        for idx in np.ndindex(*self.n_nodes):
            point = [self.nodes[d][idx[d]] for d in range(self.num_dimensions)]
            self.tensor_values[idx] = self.function(point, None)
        self.n_evaluations = total

        # Step 2: Pre-compute barycentric weights
        self.weights = []
        for d in range(self.num_dimensions):
            self.weights.append(compute_barycentric_weights(self.nodes[d]))

        # Step 3: Pre-compute differentiation matrices
        self.diff_matrices = []
        for d in range(self.num_dimensions):
            self.diff_matrices.append(
                compute_differentiation_matrix(self.nodes[d], self.weights[d])
            )

        self.build_time = time.time() - start

        # Pre-allocate evaluation cache for fast_eval() (deprecated)
        for d in range(self.num_dimensions - 1, 0, -1):
            shape = tuple(self.n_nodes[i] for i in range(d))
            self._eval_cache[d] = np.zeros(shape)

        if verbose:
            total_weights = sum(len(w) for w in self.weights)
            print(f"  Built in {self.build_time:.3f}s "
                  f"({total_weights} weights, {total_weights * 8} bytes)")

    def eval(self, point: List[float], derivative_order: List[int]) -> float:
        """Evaluate using dimensional decomposition with barycentric interpolation.

        Parameters
        ----------
        point : list of float
            Query point, one coordinate per dimension.
        derivative_order : list of int
            Derivative order per dimension (0 = function value, 1 = first
            derivative, 2 = second derivative).

        Returns
        -------
        float
            Interpolated value or derivative at the query point.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        current = self.tensor_values

        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]
            nodes = self.nodes[d]
            weights = self.weights[d]
            diff_matrix = self.diff_matrices[d]

            if d == 0:
                if deriv == 0:
                    return barycentric_interpolate(x, nodes, current, weights)
                else:
                    return barycentric_derivative_analytical(
                        x, nodes, current, weights, diff_matrix, deriv
                    )
            else:
                shape = current.shape[:d]
                new = np.zeros(shape)
                for idx in np.ndindex(*shape):
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]
                    if deriv == 0:
                        new[idx] = barycentric_interpolate(x, nodes, values_1d, weights)
                    else:
                        new[idx] = barycentric_derivative_analytical(
                            x, nodes, values_1d, weights, diff_matrix, deriv
                        )
                current = new

    def fast_eval(self, point: List[float], derivative_order: List[int]) -> float:
        """Fast evaluation using pre-allocated cache (skips validation).

        .. deprecated:: 0.3.0
            Use :meth:`vectorized_eval` instead, which is ~150x faster via
            BLAS GEMV and requires no optional dependencies.

        Parameters
        ----------
        point : list of float
            Query point.
        derivative_order : list of int
            Derivative order per dimension.

        Returns
        -------
        float
            Interpolated value or derivative.
        """
        warnings.warn(
            "fast_eval() is deprecated and will be removed in a future version. "
            "Use vectorized_eval() instead — it is ~150x faster via BLAS GEMV "
            "and requires no optional dependencies.",
            DeprecationWarning,
            stacklevel=2,
        )
        current = self.tensor_values

        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]
            nodes = self.nodes[d]
            weights = self.weights[d]
            diff_matrix = self.diff_matrices[d]

            coincident_idx = None
            diffs = np.abs(x - nodes)
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 1e-14:
                coincident_idx = int(min_idx)

            if d == 0:
                if deriv == 0:
                    if coincident_idx is not None:
                        return float(current[coincident_idx])
                    return barycentric_interpolate_jit(x, nodes, current, weights)
                else:
                    return barycentric_derivative_analytical(
                        x, nodes, current, weights, diff_matrix, deriv
                    )
            else:
                shape = current.shape[:d]
                cache = self._eval_cache[d]
                for idx in np.ndindex(*shape):
                    slice_idx = idx + (slice(None),) + (0,) * (len(current.shape) - d - 1)
                    values_1d = current[slice_idx]
                    if deriv == 0:
                        if coincident_idx is not None:
                            cache[idx] = values_1d[coincident_idx]
                        else:
                            cache[idx] = barycentric_interpolate_jit(
                                x, nodes, values_1d, weights
                            )
                    else:
                        cache[idx] = barycentric_derivative_analytical(
                            x, nodes, values_1d, weights, diff_matrix, deriv
                        )
                current = cache

    @staticmethod
    def _matmul_last_axis(current: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Contract last axis of N-D array with a vector or matrix.

        For 3-D+ arrays, reshapes to 2-D first to expose BLAS GEMV/GEMM.
        """
        if current.ndim > 2:
            lead_shape = current.shape[:-1]
            flat = current.reshape(-1, current.shape[-1]) @ rhs
            if rhs.ndim == 1:
                return flat.reshape(lead_shape)
            return flat.reshape(lead_shape + (rhs.shape[-1],))
        return current @ rhs

    def vectorized_eval(self, point: List[float], derivative_order: List[int]) -> float:
        """Fully vectorized evaluation using NumPy matrix operations.

        Replaces the Python loop with BLAS matrix-vector products.
        For 5-D with 11 nodes: 5 BLAS calls instead of 16,105 Python iterations.

        Parameters
        ----------
        point : list of float
            Query point, one coordinate per dimension.
        derivative_order : list of int
            Derivative order per dimension.

        Returns
        -------
        float
            Interpolated value or derivative.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        current = self.tensor_values
        _matmul = self._matmul_last_axis

        for d in range(self.num_dimensions - 1, -1, -1):
            x = point[d]
            deriv = derivative_order[d]

            if deriv > 0:
                D_T = self.diff_matrices[d].T
                for _ in range(deriv):
                    current = _matmul(current, D_T)

            diff = x - self.nodes[d]
            exact = np.where(np.abs(diff) < 1e-14)[0]
            if len(exact) > 0:
                current = current[..., exact[0]]
            else:
                w_over_diff = self.weights[d] / diff
                current = _matmul(current, w_over_diff) / np.sum(w_over_diff)

        return float(current)

    def vectorized_eval_batch(self, points: np.ndarray, derivative_order: List[int]) -> np.ndarray:
        """Evaluate at multiple points.

        Parameters
        ----------
        points : ndarray
            Points of shape (N, num_dimensions).
        derivative_order : list of int
            Derivative order per dimension.

        Returns
        -------
        ndarray
            Results of shape (N,).
        """
        N = points.shape[0]
        results = np.empty(N)
        for i in range(N):
            results[i] = self.vectorized_eval(points[i], derivative_order)
        return results

    def vectorized_eval_multi(
        self, point: List[float], derivative_orders: List[List[int]]
    ) -> List[float]:
        """Evaluate multiple derivative orders at the same point, sharing weights.

        Pre-computes normalized barycentric weights once per dimension and
        reuses them across all derivative orders. Computing price + 5 Greeks
        costs ~0.29 ms instead of 6 x 0.065 ms = 0.39 ms.

        Parameters
        ----------
        point : list of float
            Query point.
        derivative_orders : list of list of int
            Each inner list specifies derivative order per dimension.

        Returns
        -------
        list of float
            One result per derivative order.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        dim_info = []
        for d in range(self.num_dimensions):
            x = point[d]
            diff = x - self.nodes[d]
            abs_diff = np.abs(diff)
            min_idx = np.argmin(abs_diff)
            if abs_diff[min_idx] < 1e-14:
                dim_info.append((True, int(min_idx), None))
            else:
                w_over_diff = self.weights[d] / diff
                w_norm = w_over_diff / np.sum(w_over_diff)
                dim_info.append((False, None, w_norm))

        _matmul = self._matmul_last_axis
        results = []
        for deriv_order in derivative_orders:
            current = self.tensor_values
            for d in range(self.num_dimensions - 1, -1, -1):
                deriv = deriv_order[d]
                if deriv > 0:
                    D_T = self.diff_matrices[d].T
                    for _ in range(deriv):
                        current = _matmul(current, D_T)
                is_exact, exact_idx, w_norm = dim_info[d]
                if is_exact:
                    current = current[..., exact_idx]
                else:
                    current = _matmul(current, w_norm)
            results.append(float(current))
        return results

    def get_derivative_id(self, derivative_order: List[int]) -> List[int]:
        """Return derivative order as-is (for API compatibility)."""
        return derivative_order

    # ------------------------------------------------------------------
    # Error estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _chebyshev_coefficients_1d(values: np.ndarray) -> np.ndarray:
        """Compute Chebyshev expansion coefficients from values at Type I nodes.

        Uses DCT-II (`scipy.fft.dct`) matching the Type I (``chebpts1``)
        node distribution used by this library.

        Parameters
        ----------
        values : ndarray of shape (n,)
            Function values at Chebyshev Type I nodes in ascending order.

        Returns
        -------
        ndarray of shape (n,)
            Chebyshev coefficients c_0, c_1, ..., c_{n-1}.

        References
        ----------
        Ruiz & Zeron (2021), Section 3.4 — Ex Ante Error Estimation.
        """
        from scipy.fft import dct

        n = len(values)
        # Reverse to decreasing-node order for DCT-II convention
        coeffs = dct(values[::-1], type=2) / n
        coeffs[0] /= 2
        return coeffs

    def error_estimate(self) -> float:
        """Estimate the supremum-norm interpolation error.

        Computes Chebyshev expansion coefficients via DCT-II for each
        1-D slice of the tensor, and returns the sum of per-dimension
        maximum last-coefficient magnitudes:

        .. math::

            \\hat{E} = \\sum_{d=1}^{D}
                \\max_{\\text{slices along } d} |c_{n_d - 1}|

        This follows the ex ante error estimation from Ruiz & Zeron
        (2021), Section 3.4, adapted for Type I Chebyshev nodes.

        Returns
        -------
        float
            Estimated maximum interpolation error (sup-norm).

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        if self._cached_error_estimate is not None:
            return self._cached_error_estimate

        total_error = 0.0
        for d in range(self.num_dimensions):
            max_err_this_dim = 0.0
            # Build shape of indices for all dims except d
            other_shape = tuple(
                self.n_nodes[i]
                for i in range(self.num_dimensions)
                if i != d
            )
            for idx in np.ndindex(*other_shape):
                # Insert slice(None) at position d to extract 1-D slice
                full_idx = list(idx)
                full_idx.insert(d, slice(None))
                values_1d = self.tensor_values[tuple(full_idx)]
                coeffs = self._chebyshev_coefficients_1d(values_1d)
                max_err_this_dim = max(max_err_this_dim, abs(coeffs[-1]))
            total_error += max_err_this_dim

        self._cached_error_estimate = total_error
        return total_error

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return picklable state, excluding the original function and eval cache."""
        from pychebyshev._version import __version__

        state = self.__dict__.copy()
        state["function"] = None
        state.pop("_eval_cache", None)
        state["_pychebyshev_version"] = __version__
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and reconstruct the eval cache."""
        from pychebyshev._version import __version__

        saved_version = state.pop("_pychebyshev_version", None)
        if saved_version is not None and saved_version != __version__:
            warnings.warn(
                f"This object was saved with pychebyshev {saved_version}, "
                f"but you are loading it with {__version__}. "
                f"Evaluation results may differ if internal data layout changed.",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(state)
        self.function = None

        # Ensure fields added in later versions exist (backward compat)
        if not hasattr(self, "_cached_error_estimate"):
            self._cached_error_estimate = None

        # Reconstruct pre-allocated eval cache for fast_eval() (deprecated)
        self._eval_cache = {}
        if self.tensor_values is not None:
            for d in range(self.num_dimensions - 1, 0, -1):
                shape = tuple(self.n_nodes[i] for i in range(d))
                self._eval_cache[d] = np.zeros(shape)

    def save(self, path: str | os.PathLike) -> None:
        """Save the built interpolant to a file.

        The original function is **not** saved — only the numerical data
        needed for evaluation. The saved file can be loaded with
        :meth:`load` without access to the original function.

        Parameters
        ----------
        path : str or path-like
            Destination file path.

        Raises
        ------
        RuntimeError
            If the interpolant has not been built yet.
        """
        if self.tensor_values is None:
            raise RuntimeError(
                "Cannot save an unbuilt interpolant. Call build() first."
            )
        with open(os.fspath(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChebyshevApproximation":
        """Load a previously saved interpolant from a file.

        The loaded object can evaluate immediately; no rebuild is needed.
        The ``function`` attribute will be ``None``. Assign a new function
        before calling ``build()`` again if a rebuild is desired.

        Parameters
        ----------
        path : str or path-like
            Path to the saved file.

        Returns
        -------
        ChebyshevApproximation
            The restored interpolant.

        Warns
        -----
        UserWarning
            If the file was saved with a different PyChebyshev version.

        .. warning::

            This method uses :mod:`pickle` internally. Pickle can execute
            arbitrary code during deserialization. **Only load files you
            trust.**
        """
        with open(os.fspath(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        built = self.tensor_values is not None
        return (
            f"ChebyshevApproximation("
            f"dims={self.num_dimensions}, "
            f"nodes={self.n_nodes}, "
            f"built={built})"
        )

    def __str__(self) -> str:
        built = self.tensor_values is not None
        total_nodes = int(np.prod(self.n_nodes))
        status = "built" if built else "not built"

        # Truncate display for high-dimensional objects (>6 dims)
        max_display = 6
        if self.num_dimensions > max_display:
            nodes_str = (
                "[" + ", ".join(str(n) for n in self.n_nodes[:max_display])
                + ", ...]"
            )
            domain_str = (
                " x ".join(
                    f"[{lo}, {hi}]"
                    for lo, hi in self.domain[:max_display]
                )
                + " x ..."
            )
        else:
            nodes_str = str(self.n_nodes)
            domain_str = " x ".join(
                f"[{lo}, {hi}]" for lo, hi in self.domain
            )

        lines = [
            f"ChebyshevApproximation ({self.num_dimensions}D, {status})",
            f"  Nodes:       {nodes_str} ({total_nodes:,} total)",
            f"  Domain:      {domain_str}",
        ]

        if built:
            lines.append(
                f"  Build:       {self.build_time:.3f}s, "
                f"{self.n_evaluations:,} evaluations"
            )
            lines.append(f"  Error est:   {self.error_estimate():.2e}")

        lines.append(
            f"  Derivatives: up to order {self.max_derivative_order}"
        )

        return "\n".join(lines)
