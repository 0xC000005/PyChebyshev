"""Piecewise Chebyshev interpolation (Chebyshev Splines).

Implements the Chebyshev Spline technique from Section 3.8 of Ruiz & Zeron
(2021), "Machine Learning for Risk Calculations". Partitions the domain at
user-specified knots and builds an independent Chebyshev interpolant on each
sub-interval.  This eliminates the Gibbs phenomenon near discontinuities or
kinks, restoring spectral (exponential) convergence on every piece.

References
----------
- Ruiz & Zeron (2021), "Machine Learning for Risk Calculations",
  Wiley Finance, Section 3.8: Chebyshev Splines
"""

from __future__ import annotations

import itertools
import os
import pickle
import time
import warnings
from typing import Callable, List, Tuple

import numpy as np

from pychebyshev.barycentric import ChebyshevApproximation


class ChebyshevSpline:
    """Piecewise Chebyshev interpolation with user-specified knots.

    Partitions the domain into sub-intervals at interior knots and builds
    an independent :class:`ChebyshevApproximation` on each piece.  Query
    points are routed to the appropriate piece for evaluation.

    This is the correct approach when the target function has known
    singularities (kinks, discontinuities) at specific locations: place
    knots at those locations so that each piece is smooth, restoring
    spectral convergence.

    Parameters
    ----------
    function : callable
        Function to approximate.  Signature: ``f(point, data) -> float``
        where ``point`` is a list of floats and ``data`` is arbitrary
        additional data (can be None).
    num_dimensions : int
        Number of input dimensions.
    domain : list of (float, float)
        Bounds [lo, hi] for each dimension.
    n_nodes : list of int
        Number of Chebyshev nodes per dimension *per piece*.
    knots : list of list of float
        Interior knots for each dimension.  Each sub-list must be sorted
        and every knot must lie strictly inside the corresponding domain
        interval.  Use an empty list ``[]`` for dimensions with no knots.
    max_derivative_order : int, optional
        Maximum derivative order to pre-compute (default 2).

    Examples
    --------
    >>> import math
    >>> def f(x, _):
    ...     return abs(x[0])
    >>> sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
    >>> sp.build(verbose=False)
    >>> round(sp.eval([0.5], [0]), 10)
    0.5
    >>> round(sp.eval([-0.3], [0]), 10)
    0.3
    """

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        knots: List[List[float]],
        max_derivative_order: int = 2,
    ):
        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.n_nodes = n_nodes
        self.knots = knots
        self.max_derivative_order = max_derivative_order

        # Validate knots: each must be strictly inside domain and sorted
        for d in range(num_dimensions):
            lo, hi = domain[d]
            for k in knots[d]:
                if not (lo < k < hi):
                    raise ValueError(
                        f"Knot {k} for dimension {d} is not strictly "
                        f"inside domain [{lo}, {hi}]"
                    )
            if list(knots[d]) != sorted(knots[d]):
                raise ValueError(
                    f"Knots for dimension {d} must be sorted"
                )

        # Compute per-dimension intervals
        # _intervals[d] = [(lo, k1), (k1, k2), ..., (kn, hi)]
        self._intervals: List[List[Tuple[float, float]]] = []
        for d in range(num_dimensions):
            lo, hi = domain[d]
            edges = [lo] + list(knots[d]) + [hi]
            self._intervals.append(
                [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            )

        # Shape: per-dimension piece counts
        self._shape = tuple(len(intervals) for intervals in self._intervals)

        # Flat storage for pieces
        self._pieces: List[ChebyshevApproximation | None] = (
            [None] * int(np.prod(self._shape))
        )
        self._built = False
        self._build_time = 0.0
        self._cached_error_estimate: float | None = None

    def build(self, verbose: bool = True) -> None:
        """Build all pieces by evaluating the function on each sub-domain.

        Each piece is an independent :class:`ChebyshevApproximation` built
        on the Cartesian product of per-dimension sub-intervals.

        Parameters
        ----------
        verbose : bool, optional
            If True, print build progress.  Default is True.
        """
        start = time.time()
        self._cached_error_estimate = None
        total_pieces = int(np.prod(self._shape))
        per_piece_evals = int(np.prod(self.n_nodes))

        if verbose:
            print(
                f"Building {self.num_dimensions}D Chebyshev Spline "
                f"({total_pieces} pieces, "
                f"{total_pieces * per_piece_evals:,} total evaluations)..."
            )

        for flat_idx, multi_idx in enumerate(
            itertools.product(*[range(s) for s in self._shape])
        ):
            # Compute sub-domain for this piece
            sub_domain = [
                list(self._intervals[d][multi_idx[d]])
                for d in range(self.num_dimensions)
            ]

            piece = ChebyshevApproximation(
                self.function,
                self.num_dimensions,
                sub_domain,
                self.n_nodes,
                max_derivative_order=self.max_derivative_order,
            )
            piece.build(verbose=False)
            self._pieces[flat_idx] = piece

            if verbose:
                print(
                    f"  Piece {flat_idx + 1}/{total_pieces}: "
                    f"domain {sub_domain}"
                )

        self._build_time = time.time() - start
        self._built = True

        if verbose:
            print(f"Build complete in {self._build_time:.3f}s")

    def _find_piece(
        self, point: List[float]
    ) -> Tuple[int, ChebyshevApproximation]:
        """Find the piece containing the given point.

        Parameters
        ----------
        point : list of float
            Query point in the full domain.

        Returns
        -------
        flat_idx : int
            Flat index into ``self._pieces``.
        piece : ChebyshevApproximation
            The piece whose sub-domain contains ``point``.
        """
        multi_idx = []
        for d in range(self.num_dimensions):
            if len(self.knots[d]) == 0:
                multi_idx.append(0)
            else:
                # side='right' means a point exactly at a knot goes to the
                # right piece (the piece whose left edge is the knot).
                idx = int(
                    np.searchsorted(self.knots[d], point[d], side="right")
                )
                # Clamp to valid range (handles points at domain boundary)
                idx = min(idx, self._shape[d] - 1)
                multi_idx.append(idx)
        flat = int(np.ravel_multi_index(multi_idx, self._shape))
        return flat, self._pieces[flat]

    def _check_knot_boundary(
        self, point: List[float], derivative_order: List[int]
    ) -> None:
        """Raise ``ValueError`` if point is at a knot and derivatives requested.

        Derivatives are not defined at knot boundaries because the left and
        right polynomial pieces may have different derivative values there.

        Parameters
        ----------
        point : list of float
            Query point.
        derivative_order : list of int
            Derivative order per dimension.

        Raises
        ------
        ValueError
            If the point coincides with a knot in a dimension where a
            non-zero derivative is requested.
        """
        if all(d == 0 for d in derivative_order):
            return  # Pure function value is fine at knots
        for d in range(self.num_dimensions):
            if derivative_order[d] > 0:
                for k in self.knots[d]:
                    if abs(point[d] - k) < 1e-14:
                        raise ValueError(
                            f"Derivative w.r.t. dimension {d} is not defined "
                            f"at knot x[{d}]={k}. The left and right "
                            f"derivatives may differ at this point."
                        )

    def eval(self, point: List[float], derivative_order: List[int]) -> float:
        """Evaluate the spline approximation at a point.

        Routes the query to the piece whose sub-domain contains ``point``
        and delegates to its
        :meth:`~ChebyshevApproximation.vectorized_eval`.

        Parameters
        ----------
        point : list of float
            Evaluation point in the full domain.
        derivative_order : list of int
            Derivative order for each dimension (0 = function value).

        Returns
        -------
        float
            Approximated function value or derivative.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If the point is at a knot and a non-zero derivative is requested.
        """
        if not self._built:
            raise RuntimeError("Call build() before eval().")
        self._check_knot_boundary(point, derivative_order)
        _, piece = self._find_piece(point)
        return piece.vectorized_eval(point, derivative_order)

    def eval_multi(
        self, point: List[float], derivative_orders: List[List[int]]
    ) -> List[float]:
        """Evaluate multiple derivative orders at one point, sharing weights.

        Routes to a single piece and delegates to its
        :meth:`~ChebyshevApproximation.vectorized_eval_multi` so that
        barycentric weight computation is shared across all requested
        derivative orders.

        Parameters
        ----------
        point : list of float
            Evaluation point in the full domain.
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
        ValueError
            If the point is at a knot and a non-zero derivative is requested.
        """
        if not self._built:
            raise RuntimeError("Call build() before eval_multi().")
        for do in derivative_orders:
            self._check_knot_boundary(point, do)
        _, piece = self._find_piece(point)
        return piece.vectorized_eval_multi(point, derivative_orders)

    def eval_batch(
        self, points: np.ndarray, derivative_order: List[int]
    ) -> np.ndarray:
        """Evaluate at multiple points, grouping by piece for efficiency.

        Vectorises the piece-routing step using ``np.searchsorted`` and
        evaluates each piece's batch via
        :meth:`~ChebyshevApproximation.vectorized_eval_batch`.

        Parameters
        ----------
        points : ndarray of shape (N, num_dimensions)
            Evaluation points.
        derivative_order : list of int
            Derivative order for each dimension.

        Returns
        -------
        ndarray of shape (N,)
            Approximated values or derivatives at each point.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if not self._built:
            raise RuntimeError("Call build() before eval_batch().")
        points = np.asarray(points, dtype=float)
        N = points.shape[0]
        results = np.empty(N)

        # Compute piece index for each point (vectorised per dimension)
        multi_indices = np.zeros((N, self.num_dimensions), dtype=int)
        for d in range(self.num_dimensions):
            if len(self.knots[d]) > 0:
                multi_indices[:, d] = np.searchsorted(
                    self.knots[d], points[:, d], side="right"
                )
                np.clip(
                    multi_indices[:, d],
                    0,
                    self._shape[d] - 1,
                    out=multi_indices[:, d],
                )

        flat_indices = np.ravel_multi_index(multi_indices.T, self._shape)

        # Group by piece and batch-eval
        for piece_idx in np.unique(flat_indices):
            mask = flat_indices == piece_idx
            piece = self._pieces[piece_idx]
            results[mask] = piece.vectorized_eval_batch(
                points[mask], derivative_order
            )

        return results

    # ------------------------------------------------------------------
    # Error estimation
    # ------------------------------------------------------------------

    def error_estimate(self) -> float:
        """Estimate the supremum-norm interpolation error.

        Returns the **maximum** error estimate across all pieces.  Since
        pieces cover disjoint sub-domains, the interpolation error at any
        point is bounded by the error of the piece containing that point.
        The worst-case error is therefore the maximum over all pieces
        (not the sum, unlike :class:`ChebyshevSlider` where all slides
        contribute to every point).

        Returns
        -------
        float
            Estimated maximum interpolation error (sup-norm).

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if not self._built:
            raise RuntimeError("Call build() before error_estimate().")
        if self._cached_error_estimate is not None:
            return self._cached_error_estimate
        self._cached_error_estimate = max(
            piece.error_estimate() for piece in self._pieces
        )
        return self._cached_error_estimate

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_pieces(self) -> int:
        """Total number of pieces (Cartesian product of per-dimension intervals)."""
        return int(np.prod(self._shape))

    @property
    def total_build_evals(self) -> int:
        """Total number of function evaluations used during build."""
        return self.num_pieces * int(np.prod(self.n_nodes))

    @property
    def build_time(self) -> float:
        """Wall-clock time (seconds) for the most recent ``build()`` call."""
        return self._build_time

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return picklable state, excluding the original function."""
        from pychebyshev._version import __version__

        state = self.__dict__.copy()
        state["function"] = None
        state["_pychebyshev_version"] = __version__
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state from a pickled dict."""
        from pychebyshev._version import __version__

        saved_version = state.pop("_pychebyshev_version", None)
        if saved_version is not None and saved_version != __version__:
            warnings.warn(
                f"This object was saved with pychebyshev {saved_version}, "
                f"but you are loading it with {__version__}. "
                f"Evaluation results may differ if internal data layout "
                f"changed.",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(state)
        self.function = None

        # Ensure fields added in later versions exist (backward compat)
        if not hasattr(self, "_cached_error_estimate"):
            self._cached_error_estimate = None

    def save(self, path: str | os.PathLike) -> None:
        """Save the built spline to a file.

        The original function is **not** saved -- only the numerical data
        needed for evaluation.  The saved file can be loaded with
        :meth:`load` without access to the original function.

        Parameters
        ----------
        path : str or path-like
            Destination file path.

        Raises
        ------
        RuntimeError
            If the spline has not been built yet.
        """
        if not self._built:
            raise RuntimeError(
                "Cannot save an unbuilt spline. Call build() first."
            )
        with open(os.fspath(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChebyshevSpline":
        """Load a previously saved spline from a file.

        The loaded object can evaluate immediately; no rebuild is needed.
        The ``function`` attribute will be ``None``.  Assign a new function
        before calling ``build()`` again if a rebuild is desired.

        Parameters
        ----------
        path : str or path-like
            Path to the saved file.

        Returns
        -------
        ChebyshevSpline
            The restored spline.

        Warns
        -----
        UserWarning
            If the file was saved with a different PyChebyshev version.

        .. warning::

            This method uses :mod:`pickle` internally.  Pickle can execute
            arbitrary code during deserialization.  **Only load files you
            trust.**
        """
        with open(os.fspath(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, "
                f"got {type(obj).__name__}"
            )
        return obj

    # ------------------------------------------------------------------
    # Internal factory for arithmetic operators
    # ------------------------------------------------------------------

    @classmethod
    def _from_pieces(cls, source, pieces):
        """Create a new spline sharing grid metadata from *source* with new *pieces*."""
        obj = object.__new__(cls)
        obj.function = None
        obj.num_dimensions = source.num_dimensions
        obj.domain = [list(bounds) for bounds in source.domain]
        obj.n_nodes = list(source.n_nodes)
        obj.max_derivative_order = source.max_derivative_order
        obj.knots = [list(k) for k in source.knots]
        obj._intervals = source._intervals
        obj._shape = source._shape
        obj._pieces = pieces
        obj._built = True
        obj._build_time = 0.0
        obj._cached_error_estimate = None
        return obj

    def _check_spline_compatible(self, other):
        """Validate that two splines can be combined arithmetically."""
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        if self.knots != other.knots:
            raise ValueError(f"Knot mismatch: {self.knots} vs {other.knots}")

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        self._check_spline_compatible(other)
        pieces = [
            ChebyshevApproximation._from_grid(p_self, p_self.tensor_values + p_other.tensor_values)
            for p_self, p_other in zip(self._pieces, other._pieces)
        ]
        return ChebyshevSpline._from_pieces(self, pieces)

    def __sub__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        self._check_spline_compatible(other)
        pieces = [
            ChebyshevApproximation._from_grid(p_self, p_self.tensor_values - p_other.tensor_values)
            for p_self, p_other in zip(self._pieces, other._pieces)
        ]
        return ChebyshevSpline._from_pieces(self, pieces)

    def __mul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        s = float(scalar)
        pieces = [
            ChebyshevApproximation._from_grid(p, p.tensor_values * s)
            for p in self._pieces
        ]
        return ChebyshevSpline._from_pieces(self, pieces)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        return self.__mul__(1.0 / float(scalar))

    def __neg__(self):
        return self.__mul__(-1.0)

    def __iadd__(self, other):
        self._check_spline_compatible(other)
        for p_self, p_other in zip(self._pieces, other._pieces):
            p_self.tensor_values = p_self.tensor_values + p_other.tensor_values
            p_self._cached_error_estimate = None
        self._cached_error_estimate = None
        return self

    def __isub__(self, other):
        self._check_spline_compatible(other)
        for p_self, p_other in zip(self._pieces, other._pieces):
            p_self.tensor_values = p_self.tensor_values - p_other.tensor_values
            p_self._cached_error_estimate = None
        self._cached_error_estimate = None
        return self

    def __imul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        s = float(scalar)
        for p in self._pieces:
            p.tensor_values = p.tensor_values * s
            p._cached_error_estimate = None
        self._cached_error_estimate = None
        return self

    def __itruediv__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        return self.__imul__(1.0 / float(scalar))

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        built = self._built
        return (
            f"ChebyshevSpline("
            f"dims={self.num_dimensions}, "
            f"pieces={self.num_pieces}, "
            f"shape={self._shape}, "
            f"built={built})"
        )

    def __str__(self) -> str:
        built = self._built
        status = "built" if built else "not built"
        total_evals = self.total_build_evals

        max_display = 6

        # Nodes line
        if self.num_dimensions > max_display:
            nodes_str = (
                "["
                + ", ".join(str(n) for n in self.n_nodes[:max_display])
                + ", ...]"
            )
        else:
            nodes_str = str(self.n_nodes)

        # Knots line
        if self.num_dimensions > max_display:
            knots_str = (
                "["
                + ", ".join(str(k) for k in self.knots[:max_display])
                + ", ...]"
            )
        else:
            knots_str = str(self.knots)

        # Pieces line
        shape_str = " x ".join(str(s) for s in self._shape)

        # Domain line
        if self.num_dimensions > max_display:
            domain_str = (
                " x ".join(
                    f"[{lo}, {hi}]"
                    for lo, hi in self.domain[:max_display]
                )
                + " x ..."
            )
        else:
            domain_str = " x ".join(
                f"[{lo}, {hi}]" for lo, hi in self.domain
            )

        lines = [
            f"ChebyshevSpline ({self.num_dimensions}D, {status})",
            f"  Nodes:       {nodes_str} per piece",
            f"  Knots:       {knots_str}",
            f"  Pieces:      {self.num_pieces} ({shape_str})",
        ]

        if built:
            lines.append(
                f"  Build:       {self._build_time:.3f}s "
                f"({total_evals:,} function evals)"
            )

        lines.append(f"  Domain:      {domain_str}")

        if built:
            lines.append(f"  Error est:   {self.error_estimate():.2e}")

        return "\n".join(lines)
