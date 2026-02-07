"""Optional Numba JIT-compiled kernels for barycentric interpolation.

If Numba is not installed, falls back to pure Python implementations.
Install with: pip install pychebyshev[jit]
"""

import numpy as np

try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def barycentric_interpolate_jit(x: float, nodes: np.ndarray, values: np.ndarray,
                                    weights: np.ndarray) -> float:
        """JIT-compiled barycentric interpolation (no node coincidence check).

        Parameters
        ----------
        x : float
            Evaluation point.
        nodes : ndarray
            Chebyshev nodes.
        values : ndarray
            Function values at nodes.
        weights : ndarray
            Barycentric weights.

        Returns
        -------
        float
            Interpolated value.
        """
        sum_numerator = 0.0
        sum_denominator = 0.0

        for i in range(len(nodes)):
            w_i = weights[i] / (x - nodes[i])
            sum_numerator += w_i * values[i]
            sum_denominator += w_i

        return sum_numerator / sum_denominator

    HAS_NUMBA = True

except ImportError:

    def barycentric_interpolate_jit(x: float, nodes: np.ndarray, values: np.ndarray,
                                    weights: np.ndarray) -> float:
        """Pure Python barycentric interpolation fallback (no Numba)."""
        diff = x - nodes
        w_over_diff = weights / diff
        return float(np.dot(w_over_diff, values) / np.sum(w_over_diff))

    HAS_NUMBA = False
