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
        if self.function is None:
            raise RuntimeError(
                "Cannot build: no function assigned. "
                "This object was created via from_values() or load()."
            )
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
    # Pre-computed values: nodes first, values later
    # ------------------------------------------------------------------

    @staticmethod
    def nodes(
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        knots: List[List[float]],
    ) -> dict:
        """Generate Chebyshev nodes for every piece without evaluating any function.

        Use this to obtain the per-piece grid points, evaluate your function
        externally, then pass the results to :meth:`from_values`.

        Parameters
        ----------
        num_dimensions : int
            Number of dimensions.
        domain : list of (float, float)
            Lower and upper bounds for each dimension.
        n_nodes : list of int
            Number of Chebyshev nodes per dimension *per piece*.
        knots : list of list of float
            Knot positions for each dimension (may be empty).

        Returns
        -------
        dict
            ``'pieces'`` : list of dicts, one per piece in C-order
            (``np.ndindex(*piece_shape)``).  Each dict contains:

            - ``'piece_index'`` : tuple — multi-index of this piece
            - ``'sub_domain'`` : list of (float, float) — bounds for this piece
            - ``'nodes_per_dim'`` : list of 1-D arrays
            - ``'full_grid'`` : 2-D array, shape ``(prod(n_nodes), num_dimensions)``
            - ``'shape'`` : tuple of int

            ``'num_pieces'`` : int — total number of pieces.

            ``'piece_shape'`` : tuple of int — per-dimension piece counts.

        Examples
        --------
        >>> info = ChebyshevSpline.nodes(1, [[-1, 1]], [10], [[0.0]])
        >>> info['num_pieces']
        2
        >>> info['pieces'][0]['sub_domain']
        [(-1, 0.0)]
        """
        # Validate domain and knots
        for d in range(num_dimensions):
            lo, hi = domain[d]
            if lo >= hi:
                raise ValueError(
                    f"domain[{d}]: lo={lo} must be strictly less than hi={hi}"
                )
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
            if len(knots[d]) != len(set(knots[d])):
                raise ValueError(
                    f"Knots for dimension {d} contain duplicates"
                )

        # Compute per-dimension intervals
        intervals = []
        for d in range(num_dimensions):
            lo, hi = domain[d]
            edges = [lo] + list(knots[d]) + [hi]
            intervals.append(
                [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            )

        piece_shape = tuple(len(ivs) for ivs in intervals)
        pieces_info = []

        for multi_idx in np.ndindex(*piece_shape):
            sub_domain = [
                intervals[d][multi_idx[d]]
                for d in range(num_dimensions)
            ]
            piece_nodes = ChebyshevApproximation.nodes(
                num_dimensions,
                [list(sd) for sd in sub_domain],
                n_nodes,
            )
            pieces_info.append({
                "piece_index": multi_idx,
                "sub_domain": sub_domain,
                "nodes_per_dim": piece_nodes["nodes_per_dim"],
                "full_grid": piece_nodes["full_grid"],
                "shape": piece_nodes["shape"],
            })

        return {
            "pieces": pieces_info,
            "num_pieces": int(np.prod(piece_shape)),
            "piece_shape": piece_shape,
        }

    @classmethod
    def from_values(
        cls,
        piece_values: List[np.ndarray],
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        knots: List[List[float]],
        max_derivative_order: int = 2,
    ) -> "ChebyshevSpline":
        """Create a spline from pre-computed function values on each piece.

        The resulting object is fully functional: evaluation, derivatives,
        integration, rootfinding, algebra, extrusion/slicing, and
        serialization all work exactly as if ``build()`` had been called.

        Parameters
        ----------
        piece_values : list of numpy.ndarray
            Function values for each piece.  Length must equal the total
            number of pieces (``prod`` of per-dimension piece counts).
            Order follows ``np.ndindex(*piece_shape)`` (C-order), matching
            the ``'pieces'`` list returned by :meth:`nodes`.  Each array
            must have shape ``tuple(n_nodes)``.
        num_dimensions : int
            Number of dimensions.
        domain : list of (float, float)
            Lower and upper bounds for each dimension.
        n_nodes : list of int
            Number of Chebyshev nodes per dimension *per piece*.
        knots : list of list of float
            Knot positions for each dimension (may be empty).
        max_derivative_order : int, optional
            Maximum derivative order (default 2).

        Returns
        -------
        ChebyshevSpline
            A fully built spline with ``function=None``.

        Raises
        ------
        ValueError
            If the number of pieces does not match, or any piece has
            the wrong shape or contains NaN/Inf.
        """
        # Validate domain and knots (same as nodes())
        for d in range(num_dimensions):
            lo, hi = domain[d]
            if lo >= hi:
                raise ValueError(
                    f"domain[{d}]: lo={lo} must be strictly less than hi={hi}"
                )
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
            if len(knots[d]) != len(set(knots[d])):
                raise ValueError(
                    f"Knots for dimension {d} contain duplicates"
                )

        # Compute intervals and shape
        intervals = []
        for d in range(num_dimensions):
            lo, hi = domain[d]
            edges = [lo] + list(knots[d]) + [hi]
            intervals.append(
                [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            )

        piece_shape = tuple(len(ivs) for ivs in intervals)
        total_pieces = int(np.prod(piece_shape))

        if len(piece_values) != total_pieces:
            raise ValueError(
                f"Expected {total_pieces} piece_values, got {len(piece_values)}"
            )

        # Validate per-piece shapes before building
        expected_shape = tuple(n_nodes)
        for flat_idx, pv in enumerate(piece_values):
            pv_arr = np.asarray(pv)
            if pv_arr.shape != expected_shape:
                raise ValueError(
                    f"piece_values[{flat_idx}] has shape {pv_arr.shape}, "
                    f"expected {expected_shape}"
                )

        # Build each piece via ChebyshevApproximation.from_values()
        pieces = []
        for flat_idx, multi_idx in enumerate(np.ndindex(*piece_shape)):
            sub_domain = [
                list(intervals[d][multi_idx[d]])
                for d in range(num_dimensions)
            ]
            piece = ChebyshevApproximation.from_values(
                piece_values[flat_idx],
                num_dimensions,
                sub_domain,
                n_nodes,
                max_derivative_order=max_derivative_order,
            )
            pieces.append(piece)

        # Assemble via object.__new__()
        obj = object.__new__(cls)
        obj.function = None
        obj.num_dimensions = num_dimensions
        obj.domain = [list(bounds) for bounds in domain]
        obj.n_nodes = list(n_nodes)
        obj.max_derivative_order = max_derivative_order
        obj.knots = [list(k) for k in knots]
        obj._intervals = intervals
        obj._shape = piece_shape
        obj._pieces = pieces
        obj._built = True
        obj._build_time = 0.0
        obj._cached_error_estimate = None

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

    # ------------------------------------------------------------------
    # Extrusion and slicing
    # ------------------------------------------------------------------

    def extrude(self, params):
        """Add new dimensions where the function is constant.

        Each piece is extruded independently via
        :meth:`ChebyshevApproximation.extrude`.  The extruded spline
        evaluates identically to the original regardless of the new
        coordinate(s), because Chebyshev basis functions form a partition
        of unity.  The new dimension gets ``knots=[]`` and a single
        interval ``(lo, hi)``.

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, (lo, hi), n_nodes)`` or a list of such
            tuples.  ``dim_index`` is the position in the **output** space
            (0-indexed).  ``n_nodes`` must be >= 2 and ``lo < hi``.

        Returns
        -------
        ChebyshevSpline
            A new, higher-dimensional spline (already built).
            The result has ``function=None`` and ``build_time=0.0``.

        Raises
        ------
        RuntimeError
            If the spline has not been built yet.
        TypeError
            If ``dim_index`` is not an integer.
        ValueError
            If ``dim_index`` is out of range, duplicated, ``lo >= hi``,
            or ``n_nodes < 2``.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._extrude_slice import _normalize_extrusion_params

        sorted_params = _normalize_extrusion_params(params, self.num_dimensions)

        knots = [list(k) for k in self.knots]
        intervals = [list(iv) for iv in self._intervals]
        shape = list(self._shape)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)

        for dim_idx, (lo, hi), n in sorted_params:
            knots.insert(dim_idx, [])
            intervals.insert(dim_idx, [(lo, hi)])
            shape.insert(dim_idx, 1)
            domain.insert(dim_idx, [lo, hi])
            n_nodes.insert(dim_idx, n)

        # Extrude each piece
        pieces = []
        for piece in self._pieces:
            p = piece
            for dim_idx, bounds, n in sorted_params:
                p = p.extrude((dim_idx, bounds, n))
            pieces.append(p)

        new_ndim = self.num_dimensions + len(sorted_params)
        obj = object.__new__(ChebyshevSpline)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.knots = knots
        obj._intervals = intervals
        obj._shape = tuple(shape)
        obj._pieces = pieces
        obj._built = True
        obj._build_time = 0.0
        obj._cached_error_estimate = None
        return obj

    def slice(self, params):
        """Fix one or more dimensions at given values, reducing dimensionality.

        For each sliced dimension, only the pieces whose interval contains
        the slice value survive.  Each surviving piece is then sliced via
        :meth:`ChebyshevApproximation.slice`, which contracts the tensor
        along that axis using the barycentric interpolation formula.  When
        the slice value coincides with a Chebyshev node (within 1e-14),
        the contraction reduces to an exact ``np.take`` (fast path).

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, value)`` or a list of such tuples.
            ``value`` must lie within the domain for that dimension.

        Returns
        -------
        ChebyshevSpline
            A new, lower-dimensional spline (already built).
            The result has ``function=None`` and ``build_time=0.0``.

        Raises
        ------
        RuntimeError
            If the spline has not been built yet.
        TypeError
            If ``dim_index`` is not an integer.
        ValueError
            If a slice value is outside the domain, if slicing all
            dimensions, or if ``dim_index`` is out of range or duplicated.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._extrude_slice import _normalize_slicing_params

        sorted_params = _normalize_slicing_params(params, self.num_dimensions)

        # Validate values within domain
        for dim_idx, value in sorted_params:
            lo, hi = self.domain[dim_idx]
            if value < lo or value > hi:
                raise ValueError(
                    f"Slice value {value} for dim {dim_idx} is outside "
                    f"domain [{lo}, {hi}]"
                )

        knots = [list(k) for k in self.knots]
        intervals = [list(iv) for iv in self._intervals]
        shape = list(self._shape)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)
        # Work with pieces as a multi-dimensional array for easy indexing
        pieces_arr = np.array(self._pieces, dtype=object).reshape(self._shape)

        for dim_idx, value in sorted_params:  # descending order
            # Find which interval contains the value along this dim
            knots_d = knots[dim_idx]
            if len(knots_d) == 0:
                interval_idx = 0
            else:
                interval_idx = int(np.searchsorted(knots_d, value, side="right"))
                interval_idx = min(interval_idx, shape[dim_idx] - 1)

            # Select only pieces at this interval index along dim_idx
            pieces_arr = np.take(pieces_arr, interval_idx, axis=dim_idx)

            # Slice each surviving piece
            flat_pieces = pieces_arr.ravel()
            for i in range(len(flat_pieces)):
                flat_pieces[i] = flat_pieces[i].slice((dim_idx, value))
            pieces_arr = flat_pieces.reshape(pieces_arr.shape)

            del knots[dim_idx]
            del intervals[dim_idx]
            del shape[dim_idx]
            del domain[dim_idx]
            del n_nodes[dim_idx]

        new_ndim = self.num_dimensions - len(sorted_params)
        obj = object.__new__(ChebyshevSpline)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.knots = knots
        obj._intervals = intervals
        obj._shape = tuple(shape)
        obj._pieces = list(pieces_arr.ravel())
        obj._built = True
        obj._build_time = 0.0
        obj._cached_error_estimate = None
        return obj

    # ------------------------------------------------------------------
    # Calculus: integration, roots, optimization
    # ------------------------------------------------------------------

    def integrate(self, dims=None, bounds=None):
        """Integrate the spline over one or more dimensions.

        For full integration, sums the integrals of each piece (pieces
        cover disjoint sub-domains).  For partial integration, pieces
        along the integrated dimension are summed and the result is a
        lower-dimensional spline.

        Parameters
        ----------
        dims : int, list of int, or None
            Dimensions to integrate out.  If ``None``, integrates over
            **all** dimensions and returns a scalar.
        bounds : tuple, list of tuple/None, or None
            Sub-interval bounds for integration.  ``None`` (default)
            integrates over the full domain of each dimension.  A single
            tuple ``(lo, hi)`` applies to a single *dims* entry.  A list
            of tuples/``None`` provides per-dimension bounds with
            positional correspondence to *dims*.

        Returns
        -------
        float or ChebyshevSpline
            Scalar for full integration; lower-dimensional spline for
            partial integration.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If any dimension index is out of range or duplicated, or if
            bounds are outside the domain.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _normalize_bounds

        # Normalize dims
        if dims is None:
            dims = list(range(self.num_dimensions))
        elif isinstance(dims, int):
            dims = [dims]
        dims = sorted(set(dims))

        for d in dims:
            if d < 0 or d >= self.num_dimensions:
                raise ValueError(
                    f"dim {d} out of range [0, {self.num_dimensions - 1}]"
                )

        per_dim_bounds = _normalize_bounds(dims, bounds, self.domain)
        dim_to_idx = {d: i for i, d in enumerate(dims)}

        # Full integration: sum piece integrals (with bounds)
        if len(dims) == self.num_dimensions:
            total = 0.0
            pieces_arr = np.array(self._pieces, dtype=object).reshape(self._shape)
            for idx in np.ndindex(*self._shape):
                piece = pieces_arr[idx]
                piece_bounds = []
                skip = False
                for d in range(self.num_dimensions):
                    bd = per_dim_bounds[dim_to_idx[d]]
                    if bd is None:
                        piece_bounds.append(None)
                    else:
                        piece_lo, piece_hi = self._intervals[d][idx[d]]
                        overlap_lo = max(bd[0], piece_lo)
                        overlap_hi = min(bd[1], piece_hi)
                        if overlap_lo >= overlap_hi:
                            skip = True
                            break
                        if abs(overlap_lo - piece_lo) < 1e-14 and abs(overlap_hi - piece_hi) < 1e-14:
                            piece_bounds.append(None)
                        else:
                            piece_bounds.append((overlap_lo, overlap_hi))
                if skip:
                    continue
                if all(b is None for b in piece_bounds):
                    total += piece.integrate()
                else:
                    total += piece.integrate(bounds=piece_bounds)
            return total

        # Partial integration
        pieces_arr = np.array(self._pieces, dtype=object).reshape(self._shape)
        knots = [list(k) for k in self.knots]
        intervals = [list(iv) for iv in self._intervals]
        shape = list(self._shape)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)

        for d in sorted(dims, reverse=True):
            bd = per_dim_bounds[dim_to_idx[d]]
            # Integrate each piece along dim d, then sum along that axis
            new_shape = [s for i, s in enumerate(pieces_arr.shape) if i != d]
            new_pieces = np.empty(new_shape, dtype=object) if new_shape else np.empty((), dtype=object)

            if new_shape:
                for idx in np.ndindex(*new_shape):
                    full_idx = list(idx)
                    full_idx.insert(d, slice(None))
                    dim_pieces = pieces_arr[tuple(full_idx)]
                    integrated = []
                    for piece_idx, p in enumerate(dim_pieces.ravel()):
                        if bd is None:
                            integrated.append(p.integrate(dims=[d]))
                        else:
                            piece_lo, piece_hi = intervals[d][piece_idx]
                            overlap_lo = max(bd[0], piece_lo)
                            overlap_hi = min(bd[1], piece_hi)
                            if overlap_lo >= overlap_hi:
                                continue
                            if abs(overlap_lo - piece_lo) < 1e-14 and abs(overlap_hi - piece_hi) < 1e-14:
                                integrated.append(p.integrate(dims=[d]))
                            else:
                                integrated.append(p.integrate(dims=[d], bounds=[(overlap_lo, overlap_hi)]))
                    if not integrated:
                        # Zero contribution: build a zero piece
                        integrated.append(dim_pieces.ravel()[0].integrate(dims=[d]) * 0.0)
                    result = integrated[0]
                    for other in integrated[1:]:
                        result = result + other
                    new_pieces[idx] = result
            else:
                integrated = []
                for piece_idx, p in enumerate(pieces_arr.ravel()):
                    if bd is None:
                        integrated.append(p.integrate(dims=[d]))
                    else:
                        piece_lo, piece_hi = intervals[d][piece_idx]
                        overlap_lo = max(bd[0], piece_lo)
                        overlap_hi = min(bd[1], piece_hi)
                        if overlap_lo >= overlap_hi:
                            continue
                        if abs(overlap_lo - piece_lo) < 1e-14 and abs(overlap_hi - piece_hi) < 1e-14:
                            integrated.append(p.integrate(dims=[d]))
                        else:
                            integrated.append(p.integrate(dims=[d], bounds=[(overlap_lo, overlap_hi)]))
                if not integrated:
                    integrated.append(pieces_arr.ravel()[0].integrate(dims=[d]) * 0.0)
                result = integrated[0]
                for other in integrated[1:]:
                    result = result + other
                new_pieces[()] = result

            pieces_arr = new_pieces
            del knots[d]
            del intervals[d]
            del shape[d]
            del domain[d]
            del n_nodes[d]

        # If 0D result, return float
        if len(shape) == 0:
            return float(pieces_arr.item().integrate())

        new_ndim = self.num_dimensions - len(dims)
        obj = object.__new__(ChebyshevSpline)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.knots = knots
        obj._intervals = intervals
        obj._shape = tuple(shape)
        obj._pieces = list(pieces_arr.ravel())
        obj._built = True
        obj._build_time = 0.0
        obj._cached_error_estimate = None
        return obj

    def roots(self, dim=None, fixed=None):
        """Find all roots of the spline along a specified dimension.

        Slices the spline to 1-D, then finds roots in each piece and
        merges the results.

        Parameters
        ----------
        dim : int or None
            Dimension along which to find roots.
        fixed : dict or None
            For multi-D, dict ``{dim_index: value}`` for all other dims.

        Returns
        -------
        ndarray
            Sorted array of root locations in the physical domain.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _roots_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        # Slice to 1D spline
        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        # Find roots in each piece
        all_roots = []
        for piece in sliced._pieces:
            piece_roots = _roots_1d(piece.tensor_values, piece.domain[0])
            all_roots.append(piece_roots)

        if not all_roots:
            return np.array([], dtype=float)

        combined = np.concatenate(all_roots)
        combined = np.sort(combined)

        # Deduplicate near knot boundaries
        if len(combined) > 1:
            domain_scale = abs(self.domain[dim][1] - self.domain[dim][0]) + 1
            mask = np.concatenate([[True], np.diff(combined) > 1e-10 * domain_scale])
            combined = combined[mask]

        return combined

    def minimize(self, dim=None, fixed=None):
        """Find the minimum value of the spline along a dimension.

        Parameters
        ----------
        dim : int or None
            Dimension along which to minimize.
        fixed : dict or None
            For multi-D, dict ``{dim_index: value}`` for all other dims.

        Returns
        -------
        (value, location) : (float, float)

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _optimize_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        best_val, best_loc = float("inf"), 0.0
        for piece in sliced._pieces:
            val, loc = _optimize_1d(
                piece.tensor_values, piece.nodes[0], piece.weights[0],
                piece.diff_matrices[0], piece.domain[0], mode="min",
            )
            if val < best_val:
                best_val, best_loc = val, loc
        return best_val, best_loc

    def maximize(self, dim=None, fixed=None):
        """Find the maximum value of the spline along a dimension.

        Parameters
        ----------
        dim : int or None
            Dimension along which to maximize.
        fixed : dict or None
            For multi-D, dict ``{dim_index: value}`` for all other dims.

        Returns
        -------
        (value, location) : (float, float)

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _optimize_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        best_val, best_loc = float("-inf"), 0.0
        for piece in sliced._pieces:
            val, loc = _optimize_1d(
                piece.tensor_values, piece.nodes[0], piece.weights[0],
                piece.diff_matrices[0], piece.domain[0], mode="max",
            )
            if val > best_val:
                best_val, best_loc = val, loc
        return best_val, best_loc

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
