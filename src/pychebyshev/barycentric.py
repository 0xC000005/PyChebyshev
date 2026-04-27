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


def _validate_special_points_shape(
    special_points: List[List[float]],
    n_nodes: List[int | None] | List[List[int | None]] | None,
    num_dimensions: int,
    domain: List[Tuple[float, float]],
) -> None:
    """Validate special_points and nested n_nodes shape (v0.12).

    Called from ``ChebyshevApproximation.__new__`` before dispatching to
    ``ChebyshevSpline``.  Raises ValueError on any shape or content
    violation; returns None on success.

    The outer length check (``len(special_points) == num_dimensions``)
    and per-dim list-ness are handled by ``__new__`` so that typos like
    ``special_points=[0.0]`` or an all-empty-but-wrong-length input fail
    loudly before reaching this helper.
    """
    for d in range(num_dimensions):
        lo, hi = domain[d]
        pts = list(special_points[d])
        for k in pts:
            if not (lo < k < hi):
                raise ValueError(
                    f"Special point {k} for dimension {d} is not strictly "
                    f"inside domain [{lo}, {hi}]"
                )
        if pts != sorted(pts):
            raise ValueError(
                f"special_points for dimension {d} must be sorted"
            )
        if len(set(pts)) != len(pts):
            raise ValueError(
                f"Coinciding special points in dimension {d}"
            )

    if n_nodes is None:
        return  # auto-N per piece is fine; spline will handle

    any_nested = any(isinstance(x, (list, tuple)) for x in n_nodes)
    all_nested = all(isinstance(x, (list, tuple)) for x in n_nodes)
    if any_nested and not all_nested:
        raise ValueError(
            f"n_nodes must be fully nested (all dims as lists) when any "
            f"dim is nested; got mixed form {n_nodes!r}"
        )
    if not all_nested:
        raise ValueError(
            f"n_nodes must be nested as List[List[int]] when special_points "
            f"is present; got {n_nodes!r}"
        )
    for d in range(num_dimensions):
        expected = len(special_points[d]) + 1
        got = len(n_nodes[d])
        if got != expected:
            raise ValueError(
                f"n_nodes[{d}] must have {expected} entries "
                f"(one per sub-interval); got {got}"
            )


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
    n_nodes : list of (int or None), optional
        Number of Chebyshev nodes per dimension. Entries may be None
        when ``error_threshold`` is set, signalling auto-N mode for
        that dimension. If omitted entirely, ``error_threshold`` must
        be provided. Default is None.
    max_derivative_order : int, optional
        Maximum derivative order to support. Default is 2.
    error_threshold : float, optional
        When set, enables error-driven auto-N construction: any
        dimension with an unresolved (None) entry in ``n_nodes`` is
        refined via the build-time doubling loop until the sup-norm
        error estimate falls below this threshold (or ``max_n`` is
        reached). Default is None.
    max_n : int, optional
        Upper cap on per-dimension node count when the doubling loop
        refines auto dims. Default is 64.
    special_points : list of list of float, optional
        Per-dimension kinks or discontinuities.  When any dimension has at
        least one point, construction transparently returns a
        :class:`ChebyshevSpline` via ``__new__`` dispatch (precedent:
        :class:`pathlib.Path`).  Each inner list must be strictly inside
        its domain interval, sorted, and free of duplicates.  When set,
        ``n_nodes`` must be nested as ``List[List[int | None]]`` with
        ``len(n_nodes[d]) == len(special_points[d]) + 1``.  Default is
        ``None``.

    Examples
    --------
    >>> import math
    >>> def f(x, _):
    ...     return math.sin(x[0]) + math.sin(x[1])
    >>> cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
    >>> cheb.build()  # doctest: +SKIP
    >>> cheb.vectorized_eval([0.5, 0.3], [0, 0])  # doctest: +SKIP
    0.7764...

    Notes
    -----
    When ``special_points`` is provided with any kink, the return value is
    a :class:`ChebyshevSpline` instance — not a :class:`ChebyshevApproximation`.
    Use ``type(obj)`` or ``isinstance(obj, ChebyshevSpline)`` to distinguish
    if needed.  All downstream features (eval, integrate, algebra,
    extrude/slice, save/load) work identically on either type.
    """

    def __new__(
        cls,
        function: Callable | None = None,
        num_dimensions: int | None = None,
        domain: List[Tuple[float, float]] | None = None,
        n_nodes: List[int | None] | List[List[int | None]] | None = None,
        max_derivative_order: int = 2,
        error_threshold: float | None = None,
        max_n: int = 64,
        special_points: List[List[float]] | None = None,
        additional_data: object = None,
        *,
        defer_build: bool = False,
        n_workers: int | None = None,
    ):
        """Dispatch to ChebyshevSpline when special_points declares any kink.

        Python skips ``__init__`` on this class when ``__new__`` returns an
        instance that is not an instance of ``cls`` (or subclass).
        ChebyshevSpline is not a subclass of ChebyshevApproximation, so the
        spline's own ``__init__`` (already run inside the return expression
        below) is not overwritten.

        All parameters default to ``None`` so pickle/copy protocols can call
        ``__new__(cls)`` without positional arguments; real construction
        still goes through ``__init__``.
        """
        # Unwrap typed helpers (v0.16). Lazy import avoids circular dependency.
        from pychebyshev import Domain, Ns, SpecialPoints as _SP
        if isinstance(domain, Domain):
            domain = list(domain.bounds)
        if isinstance(n_nodes, Ns):
            n_nodes = list(n_nodes.counts)
        if isinstance(special_points, _SP):
            special_points = [list(k) for k in special_points.knots_per_dim]
        if special_points is not None:
            # Outer validation runs whether or not any dim is non-empty, so
            # typos like special_points=[[]] on a 2D problem or
            # special_points=[0.0] (missing inner list) surface immediately.
            if num_dimensions is not None and len(special_points) != num_dimensions:
                raise ValueError(
                    f"special_points must have {num_dimensions} entries, "
                    f"got {len(special_points)}"
                )
            for d, sp in enumerate(special_points):
                if not isinstance(sp, (list, tuple)):
                    raise ValueError(
                        f"special_points[{d}] must be a list/tuple of floats, "
                        f"got {type(sp).__name__}: {sp!r}"
                    )
            if any(len(sp) > 0 for sp in special_points):
                from pychebyshev.spline import ChebyshevSpline
                _validate_special_points_shape(
                    special_points, n_nodes, num_dimensions, domain
                )
                return ChebyshevSpline(
                    function,
                    num_dimensions,
                    domain,
                    n_nodes=n_nodes,
                    knots=special_points,
                    max_derivative_order=max_derivative_order,
                    error_threshold=error_threshold,
                    max_n=max_n,
                    additional_data=additional_data,
                    defer_build=defer_build,
                    n_workers=n_workers,
                )
        return super().__new__(cls)

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int | None] | List[List[int | None]] | None = None,
        max_derivative_order: int = 2,
        error_threshold: float | None = None,
        max_n: int = 64,
        special_points: List[List[float]] | None = None,
        additional_data: object = None,
        *,
        defer_build: bool = False,
        n_workers: int | None = None,
    ):
        # Unwrap typed helpers (v0.16). Lazy import avoids circular dependency.
        from pychebyshev import Domain, Ns, SpecialPoints as _SP
        if isinstance(domain, Domain):
            domain = list(domain.bounds)
        if isinstance(n_nodes, Ns):
            n_nodes = list(n_nodes.counts)
        if isinstance(special_points, _SP):
            special_points = [list(k) for k in special_points.knots_per_dim]
        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.error_threshold = error_threshold
        if max_n < 3:
            raise ValueError(
                f"max_n must be at least 3 (the initial N of the doubling "
                f"loop), got max_n={max_n}. For a grid smaller than 3 per "
                f"dimension, pass n_nodes explicitly instead of using "
                f"error-threshold auto-calibration."
            )
        self.max_n = max_n
        self.max_derivative_order = max_derivative_order
        self.special_points = special_points
        self.descriptor: str = ""
        self.additional_data = additional_data
        from pychebyshev._parallel import _normalize_n_workers
        self.n_workers = _normalize_n_workers(n_workers)
        self._derivative_id_registry: dict[tuple[int, ...], int] = {}
        self._derivative_id_to_orders: list[tuple[int, ...]] = []

        # Normalize n_nodes — None means "auto this dim"
        if n_nodes is None:
            if error_threshold is None and not defer_build:
                raise ValueError(
                    "Must provide either n_nodes (explicit) or error_threshold "
                    "(auto-N). Got neither."
                )
            n_nodes = [None] * num_dimensions
        else:
            n_nodes = list(n_nodes)
            if any(n is None for n in n_nodes) and error_threshold is None:
                raise ValueError(
                    "None entries in n_nodes require error_threshold to be set "
                    "(auto-N mode)."
                )

        self.n_nodes = n_nodes
        # Preserve the user's original intent (None sentinels intact) so
        # a second build() call after the doubling loop has resolved all
        # Ns can still dispatch to the auto-N path when the user
        # tightens error_threshold and rebuilds.
        self._original_n_nodes: List[int | None] = list(self.n_nodes)

        self.tensor_values: np.ndarray | None = None
        self.weights: List[np.ndarray] | None = None
        self.diff_matrices: List[np.ndarray] | None = None
        self.build_time: float = 0.0
        self.n_evaluations: int = 0
        self._eval_cache: dict = {}
        self._cached_error_estimate: float | None = None

        # Deferred-build path: grid metadata only, no function evaluation.
        if defer_build:
            if function is not None:
                raise ValueError(
                    "defer_build=True requires function=None (the deferred-construction "
                    "workflow expects values to be supplied via "
                    "set_original_function_values() later)"
                )
            if self.n_nodes is None or any(
                not isinstance(n, int) or n <= 0 for n in self.n_nodes
            ):
                raise ValueError(
                    "defer_build=True requires explicit positive int n_nodes; "
                    "auto-N (error_threshold) is not supported in deferred mode"
                )
            self._initialize_grid_only()
            return

        # Generate nodes only if n_nodes is fully resolved; otherwise
        # _build_with_threshold() (Task 3) will (re)generate on each iteration.
        self.nodes: List[np.ndarray] = []
        if all(n is not None for n in self.n_nodes):
            self._generate_nodes()

    def _generate_nodes(self) -> None:
        """Populate self.nodes (Chebyshev grid, sorted) from self.n_nodes.

        Weights and differentiation matrices are generated by build(),
        not here — preserves the existing separation of concerns so the
        doubling loop in Task 3 can reuse _build_fixed_grid unchanged.
        """
        self.nodes = []
        for d in range(self.num_dimensions):
            nodes_std = chebpts1(self.n_nodes[d])
            a, b = self.domain[d]
            nodes_scaled = 0.5 * (a + b) + 0.5 * (b - a) * nodes_std
            self.nodes.append(np.sort(nodes_scaled))

    def _initialize_grid_only(self) -> None:
        """Populate nodes, weights, diff_matrices, and eval_cache without evaluating
        the function.  Called by ``__init__`` when ``defer_build=True``.

        Uses the same code path as :meth:`from_values` to guarantee bit-identical
        grid metadata.
        """
        from pychebyshev._extrude_slice import _make_nodes_for_dim

        self.nodes = []
        for d in range(self.num_dimensions):
            lo, hi = self.domain[d]
            self.nodes.append(_make_nodes_for_dim(lo, hi, self.n_nodes[d]))

        self.weights = []
        for d in range(self.num_dimensions):
            self.weights.append(compute_barycentric_weights(self.nodes[d]))

        self.diff_matrices = []
        for d in range(self.num_dimensions):
            self.diff_matrices.append(
                compute_differentiation_matrix(self.nodes[d], self.weights[d])
            )

        # Pre-allocate eval cache for deprecated fast_eval()
        self._eval_cache = {}
        for d in range(self.num_dimensions - 1, 0, -1):
            shape = tuple(self.n_nodes[i] for i in range(d))
            self._eval_cache[d] = np.zeros(shape)

    def set_original_function_values(self, values) -> None:
        """In-place mutator: populate this interpolant's tensor with explicit
        function values.  After this call the interpolant is fully evaluable.

        Pairs with ``defer_build=True`` ctor: construct an empty grid-metadata
        object, then fill its tensor here — an in-place alternative to the
        :meth:`from_values` classmethod factory.

        Parameters
        ----------
        values : array_like
            Tensor of shape ``tuple(self.n_nodes)`` containing the function
            values at the points returned by :meth:`get_evaluation_points`,
            laid out in C-order (matching :meth:`nodes` output).

        Raises
        ------
        RuntimeError
            If this interpolant is already constructed (``build()`` was called,
            or this method was already called).
        ValueError
            On shape mismatch or if *values* contains NaN or Inf.
        """
        if self.tensor_values is not None:
            raise RuntimeError(
                "interpolant is already constructed; "
                "set_original_function_values() is for defer_build=True objects"
            )
        arr = np.asarray(values, dtype=np.float64)
        expected_shape = tuple(self.n_nodes)
        if arr.shape != expected_shape:
            raise ValueError(
                f"values shape {arr.shape} does not match expected {expected_shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("values contains NaN or Inf (must be finite)")
        self.tensor_values = arr.copy()
        self.function = None  # function-less interpolant

    def build(self, verbose: bool | int = True) -> None:
        """Build the Chebyshev approximation by evaluating the function on the grid.

        If ``error_threshold`` was provided to ``__init__``, runs the
        doubling loop until the target precision is reached (or
        ``max_n`` is hit on all auto dims). Otherwise builds on the
        fixed grid specified by ``n_nodes``.

        Parameters
        ----------
        verbose : bool or int, optional
            If True or 1, print build progress. If 2, also show a tqdm
            progress bar (requires ``pychebyshev[viz]``). Default is True.

        Notes
        -----
        In auto-N mode (``error_threshold`` set), ``n_evaluations``
        and ``build_time`` are **accumulated across doubling
        iterations** and reflect the total work performed, not just
        the final iteration. In fixed-N mode they reflect a single
        build.

        Dispatch keys off the user's *original* ``n_nodes``
        (preserved as ``_original_n_nodes``), so a second
        ``build()`` after mutating ``error_threshold`` correctly
        re-runs the doubling loop instead of silently bypassing it.

        Raises
        ------
        RuntimeError
            If ``function`` is None (e.g., object was produced via
            ``from_values()``, ``load()``, algebra, ``slice``, or
            ``extrude`` — use those factories directly).
        """
        if self.function is None:
            raise RuntimeError(
                "Cannot build: no function assigned. "
                "This object was created via from_values() or load()."
            )
        if any(n is None for n in self._original_n_nodes):
            self._build_with_threshold(verbose=verbose)
        else:
            self._build_fixed_grid(verbose=verbose)

    def _build_with_threshold(self, verbose: bool | int = True) -> None:
        """Iteratively double auto-dim Ns until error_estimate <= threshold.

        The dim with the largest per-dimension last-coefficient
        magnitude is doubled each iteration (capped at ``max_n``).
        Accumulates ``n_evaluations`` and ``build_time`` across
        iterations so the post-build counters reflect the *total*
        work done, not just the final iteration.
        """
        assert self.error_threshold is not None
        assert any(n is None for n in self._original_n_nodes), (
            "_build_with_threshold called with all Ns resolved"
        )

        # Resolve: ints stay, None starts at 3
        current: List[int] = [
            n if n is not None else 3 for n in self._original_n_nodes
        ]
        auto_dims = [
            i for i, n in enumerate(self._original_n_nodes) if n is None
        ]

        total_evals = 0
        total_build_time = 0.0

        while True:
            self.n_nodes = list(current)
            self._cached_error_estimate = None
            self._generate_nodes()
            self._build_fixed_grid(verbose=verbose)
            total_evals += self.n_evaluations
            total_build_time += self.build_time

            # Compute per-dim errors once per iteration; derive the
            # total from their sum and seed the cache so any external
            # error_estimate() call on the final object hits the cache.
            per_dim = self._error_estimate_per_dim()
            err = float(sum(per_dim))
            self._cached_error_estimate = err

            if verbose:
                print(f"[auto-N] n_nodes={current}, error={err:.3e}")

            if err <= self.error_threshold:
                break

            # Pick worst auto dim not yet at max_n
            candidates = [
                (per_dim[i], i)
                for i in auto_dims
                if current[i] < self.max_n
            ]
            if not candidates:
                warnings.warn(
                    f"max_n={self.max_n} reached on all auto dims "
                    f"before error_threshold={self.error_threshold:.2e} "
                    f"satisfied (last error={err:.3e}). "
                    f"Increase max_n or relax error_threshold.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                break

            # Largest per-dim error first; ties → lowest index
            candidates.sort(key=lambda t: (-t[0], t[1]))
            worst_dim = candidates[0][1]
            # Schedule: plain doubling of the single worst-contributing
            # dim, capped at max_n. Alternatives considered in the
            # v0.11 design spec §8 (n+2, ceil(1.5*n), adaptive per-dim)
            # were deferred pending empirics; doubling is the safe
            # default for exponentially-convergent analytic functions.
            # A ``schedule=`` kwarg can be threaded through __init__ if
            # mixed-frequency functions ever prove the default
            # inefficient in practice.
            current[worst_dim] = min(2 * current[worst_dim], self.max_n)

        # Commit accumulated counters (overwrite per-iteration values)
        self.n_evaluations = total_evals
        self.build_time = total_build_time

    def _build_fixed_grid(self, verbose: bool | int = True) -> None:
        """Build tensor values on the already-resolved (all-int) grid.

        Original body of build() — now called by the public build()
        wrapper for the fixed-N path, and by _build_with_threshold for
        each iteration of the doubling loop. The public wrapper
        already enforces ``function is not None``, so no guard here.
        """
        total = int(np.prod(self.n_nodes))
        if verbose:
            print(f"Building {self.num_dimensions}D Chebyshev approximation "
                  f"({total:,} evaluations)...")

        start = time.time()
        self._cached_error_estimate = None

        # Step 1: Evaluate at all node combinations
        if self.n_workers is None or self.n_workers == 1:
            # Fast path: in-place fill (original behavior, no intermediate list)
            self.tensor_values = np.zeros(self.n_nodes)
            for idx in np.ndindex(*self.n_nodes):
                point = [self.nodes[d][idx[d]] for d in range(self.num_dimensions)]
                self.tensor_values[idx] = float(
                    self.function(point, self.additional_data)
                )
        else:
            # Parallel path: build points list, dispatch to pool, reshape
            from pychebyshev._parallel import _evaluate_in_parallel
            points = [
                [self.nodes[d][idx[d]] for d in range(self.num_dimensions)]
                for idx in np.ndindex(*self.n_nodes)
            ]
            flat = _evaluate_in_parallel(
                self.function, points, self.additional_data, self.n_workers
            )
            self.tensor_values = flat.reshape(self.n_nodes)
        self.n_evaluations = total

        # Guard: reject NaN / Inf before proceeding with weight computation.
        if not np.isfinite(self.tensor_values).all():
            n_bad = int(np.sum(~np.isfinite(self.tensor_values)))
            raise ValueError(
                f"function returned non-finite values at {n_bad} grid point(s); "
                "build cannot proceed with NaN/Inf in tensor_values"
            )

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

    def eval(
        self,
        point: List[float],
        derivative_order: List[int] | None = None,
        *,
        derivative_id: int | None = None,
    ) -> float:
        """Evaluate using dimensional decomposition with barycentric interpolation.

        Parameters
        ----------
        point : list of float
            Query point, one coordinate per dimension.
        derivative_order : list of int, optional
            Derivative order per dimension (0 = function value, 1 = first
            derivative, 2 = second derivative). Provide exactly one of
            ``derivative_order`` or ``derivative_id``.
        derivative_id : int, optional
            Stable session-local id returned by :meth:`get_derivative_id`.
            Provide exactly one of ``derivative_order`` or ``derivative_id``.

        Returns
        -------
        float
            Interpolated value or derivative at the query point.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If both or neither of ``derivative_order`` / ``derivative_id``
            are provided.
        KeyError
            If ``derivative_id`` is unknown (not yet registered).
        """
        derivative_order = self._resolve_derivative_args(
            derivative_order, derivative_id
        )
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

    def fast_eval(
        self,
        point: List[float],
        derivative_order: List[int] | None = None,
        *,
        derivative_id: int | None = None,
    ) -> float:
        """Fast evaluation using pre-allocated cache (skips validation).

        .. deprecated:: 0.3.0
            Use :meth:`vectorized_eval` instead, which is ~150x faster via
            BLAS GEMV and requires no optional dependencies.

        Parameters
        ----------
        point : list of float
            Query point.
        derivative_order : list of int, optional
            Derivative order per dimension. Provide exactly one of
            ``derivative_order`` or ``derivative_id``.
        derivative_id : int, optional
            Stable session-local id returned by :meth:`get_derivative_id`.
            Provide exactly one of ``derivative_order`` or ``derivative_id``.

        Returns
        -------
        float
            Interpolated value or derivative.
        """
        derivative_order = self._resolve_derivative_args(
            derivative_order, derivative_id
        )
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

    def vectorized_eval(
        self,
        point: List[float],
        derivative_order: List[int] | None = None,
        *,
        derivative_id: int | None = None,
    ) -> float:
        """Fully vectorized evaluation using NumPy matrix operations.

        Replaces the Python loop with BLAS matrix-vector products.
        For 5-D with 11 nodes: 5 BLAS calls instead of 16,105 Python iterations.

        Parameters
        ----------
        point : list of float
            Query point, one coordinate per dimension.
        derivative_order : list of int, optional
            Derivative order per dimension. Provide exactly one of
            ``derivative_order`` or ``derivative_id``.
        derivative_id : int, optional
            Stable session-local id returned by :meth:`get_derivative_id`.
            Provide exactly one of ``derivative_order`` or ``derivative_id``.

        Returns
        -------
        float
            Interpolated value or derivative.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If both or neither of ``derivative_order`` / ``derivative_id``
            are provided.
        KeyError
            If ``derivative_id`` is unknown (not yet registered).
        """
        derivative_order = self._resolve_derivative_args(
            derivative_order, derivative_id
        )
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

    def vectorized_eval_batch(
        self,
        points: np.ndarray,
        derivative_order: List[int] | None = None,
        *,
        derivative_id: int | None = None,
    ) -> np.ndarray:
        """Evaluate at multiple points.

        Parameters
        ----------
        points : ndarray
            Points of shape (N, num_dimensions).
        derivative_order : list of int, optional
            Derivative order per dimension. Provide exactly one of
            ``derivative_order`` or ``derivative_id``.
        derivative_id : int, optional
            Stable session-local id returned by :meth:`get_derivative_id`.
            Provide exactly one of ``derivative_order`` or ``derivative_id``.

        Returns
        -------
        ndarray
            Results of shape (N,).
        """
        derivative_order = self._resolve_derivative_args(
            derivative_order, derivative_id
        )
        N = points.shape[0]
        results = np.empty(N)
        for i in range(N):
            results[i] = self.vectorized_eval(points[i], derivative_order=derivative_order)
        return results

    def vectorized_eval_multi(
        self, point: List[float], derivative_orders: List[List[int]]
    ) -> List[float]:
        """Evaluate multiple derivative orders at the same point, sharing weights.

        Pre-computes normalized barycentric weights once per dimension and
        reuses them across all derivative orders. Computing price + 5 Greeks
        costs ~0.29 ms instead of 6 x 0.065 ms = 0.39 ms.

        .. note::
            ``derivative_id`` is not supported here because this method takes
            a list of orders. For derivative_id-based access, look up orders
            via ``self._derivative_id_to_orders[id]``.

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

    def is_construction_finished(self) -> bool:
        """Return True iff this interpolant is built and usable."""
        return self.tensor_values is not None

    def get_constructor_type(self) -> str:
        """Return the class name (matches MoCaX getConstructorType convention)."""
        return type(self).__name__

    def get_used_ns(self) -> list:
        """Return the resolved per-dim node count after build (or as constructed)."""
        return list(self.n_nodes)

    def set_descriptor(self, descriptor: str) -> None:
        """Set a free-form text label on this interpolant.

        Parameters
        ----------
        descriptor : str
            Label to attach to this interpolant.

        Raises
        ------
        TypeError
            If ``descriptor`` is not a string.
        """
        if not isinstance(descriptor, str):
            raise TypeError(
                f"descriptor must be str, got {type(descriptor).__name__}"
            )
        self.descriptor = descriptor

    def get_descriptor(self) -> str:
        """Return the descriptor label (default ``""``)."""
        return self.descriptor

    def get_max_derivative_order(self) -> int:
        """Return the maximum derivative order this interpolant was constructed
        with. Derivative orders up to and including this value are queryable
        via ``eval(point, derivative_order=...)``."""
        return self.max_derivative_order

    @staticmethod
    def is_dimensionality_allowed(num_dimensions: int) -> bool:
        """Return whether this interpolant class supports the given number of
        dimensions. Returns True for any ``num_dimensions >= 1``. Provided as
        a hook for future per-class capability caps."""
        return isinstance(num_dimensions, int) and num_dimensions >= 1

    def get_special_points(self) -> list[list[float]] | None:
        """Return the special points (kinks/knots) declared at construction.

        Returns ``None`` if no ``special_points`` was passed. If
        ``special_points`` declared any non-empty list, ``__new__``
        dispatched to :class:`ChebyshevSpline`; this getter on a
        :class:`ChebyshevApproximation` therefore returns either ``None``
        or a list-of-empty-lists.
        """
        return self.special_points

    def get_derivative_id(self, derivative_order: List[int]) -> int:
        """Register a derivative-orders tuple and return a stable session-local int.

        Calling with the same ``derivative_order`` returns the same int. IDs
        are sequential, starting at 0, and are persisted across pickle save/load
        but reset on binary `.pcb` save/load.

        Parameters
        ----------
        derivative_order : list of int
            Per-dim derivative orders. Must have length ``num_dimensions``.

        Returns
        -------
        int
            Stable session-local id for this orders tuple.

        Raises
        ------
        ValueError
            If ``derivative_order`` has the wrong length, or any entry is
            negative, or any entry exceeds ``max_derivative_order``.
        """
        if len(derivative_order) != self.num_dimensions:
            raise ValueError(
                f"derivative_order length {len(derivative_order)} does not "
                f"match num_dimensions {self.num_dimensions}"
            )
        for d, o in enumerate(derivative_order):
            if not isinstance(o, (int, np.integer)):
                raise ValueError(
                    f"derivative_order[{d}] must be int, got {type(o).__name__}"
                )
            if o < 0 or o > self.max_derivative_order:
                raise ValueError(
                    f"derivative_order[{d}]={o} out of range "
                    f"[0, {self.max_derivative_order}]"
                )
        key = tuple(int(o) for o in derivative_order)
        if key in self._derivative_id_registry:
            return self._derivative_id_registry[key]
        new_id = len(self._derivative_id_to_orders)
        self._derivative_id_registry[key] = new_id
        self._derivative_id_to_orders.append(key)
        return new_id

    def _resolve_derivative_args(
        self,
        derivative_order: List[int] | None,
        derivative_id: int | None,
    ) -> List[int]:
        """Resolve the derivative spec from kwargs (orders xor id).

        Returns the resolved ``derivative_order`` list. Raises ``ValueError``
        if both or neither kwarg is provided. Raises ``KeyError`` if
        ``derivative_id`` is unknown.
        """
        if derivative_order is not None and derivative_id is not None:
            raise ValueError(
                "provide exactly one of derivative_order or derivative_id, not both"
            )
        if derivative_order is None and derivative_id is None:
            raise ValueError("must provide derivative_order or derivative_id")
        if derivative_id is not None:
            if derivative_id < 0 or derivative_id >= len(self._derivative_id_to_orders):
                raise KeyError(
                    f"unknown derivative_id {derivative_id}; "
                    f"register via get_derivative_id() first"
                )
            return list(self._derivative_id_to_orders[derivative_id])
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

    def _error_estimate_per_dim(self) -> List[float]:
        """Per-dimension max last-coefficient magnitudes.

        Returns one float per dimension; ``error_estimate()`` returns the sum.
        Split out so ``_build_with_threshold`` can pick the worst-contributing
        dim to refine.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        per_dim: List[float] = []
        for d in range(self.num_dimensions):
            max_err_this_dim = 0.0
            other_shape = tuple(
                self.n_nodes[i]
                for i in range(self.num_dimensions)
                if i != d
            )
            for idx in np.ndindex(*other_shape):
                full_idx = list(idx)
                full_idx.insert(d, slice(None))
                values_1d = self.tensor_values[tuple(full_idx)]
                coeffs = self._chebyshev_coefficients_1d(values_1d)
                max_err_this_dim = max(max_err_this_dim, abs(coeffs[-1]))
            per_dim.append(max_err_this_dim)
        return per_dim

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
        if self._cached_error_estimate is not None:
            return self._cached_error_estimate
        total = float(sum(self._error_estimate_per_dim()))
        self._cached_error_estimate = total
        return total

    def sobol_indices(self) -> dict:
        """Compute first-order and total-order Sobol sensitivity indices.

        Uses the Chebyshev spectral expansion of the interpolant to compute
        variance-based sensitivity indices analytically (no Monte Carlo).

        Returns
        -------
        dict
            ``{"first_order": {dim: index}, "total_order": {dim: index},
            "variance": float}``

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        from pychebyshev._sensitivity import (
            _compute_chebyshev_coefficients,
            _compute_sobol_from_coeffs,
        )

        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        coeffs = _compute_chebyshev_coefficients(
            self.tensor_values, self.num_dimensions
        )
        return _compute_sobol_from_coeffs(coeffs, self.num_dimensions)

    def get_error_threshold(self) -> float | None:
        """Return the error_threshold passed to ``__init__``, or None if unset.

        This is the *target* precision specified at construction time, not
        the achieved error after build. For the post-build achieved error
        estimate, use :meth:`error_estimate`.

        Returns
        -------
        float or None
            The ``error_threshold`` kwarg from construction, or ``None``
            if the object was built with an explicit fixed grid.
        """
        return self.error_threshold

    def get_num_evaluation_points(self) -> int:
        """Return the number of points where ``f`` was (or will be) evaluated.

        For a fixed-grid construction this is ``prod(n_nodes)``.

        Returns
        -------
        int
            Total number of grid points at which ``f`` is evaluated.
        """
        return int(np.prod(self.n_nodes))

    def get_evaluation_points(self) -> np.ndarray:
        """Return the grid of points where ``f`` was (or will be) evaluated.

        Returns
        -------
        np.ndarray
            Shape ``(N, num_dimensions)`` where ``N = prod(n_nodes)``. Points
            are listed in C-order: dim-0 outermost, dim-(d-1) innermost.
        """
        grids = np.meshgrid(*self.nodes, indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=-1).astype(np.float64)

    def clone(self) -> "ChebyshevApproximation":
        """Return an independent deep copy of this interpolant.

        All mutable state (tensors, descriptor, additional_data,
        derivative-id registry) is duplicated. Mutating the clone does not
        affect the original.

        Note
        ----
        Like :meth:`save` / :meth:`load`, the source ``function`` callable is
        not duplicated -- the clone has ``function = None``. All evaluation,
        algebra, serialization, and v0.16 surface methods continue to work;
        only :meth:`build` (which requires a function) does not.

        Returns
        -------
        ChebyshevApproximation
            A new instance with deep-copied state.
        """
        import copy
        return copy.deepcopy(self)

    def plot_convergence(self, target_error=None, max_n=64, ax=None):
        """Plot convergence: builds at increasing N, plots error decay.

        Requires the optional ``pychebyshev[viz]`` dependency group.

        Parameters
        ----------
        target_error : float | None
            If provided, draws a horizontal dashed line at this level.
        max_n : int
            Largest N to try in the doubling sweep.
        ax : matplotlib.axes.Axes | None
            Pre-existing axes to plot into. Creates a new figure if None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "plot_convergence requires matplotlib; install with "
                "`pip install pychebyshev[viz]`"
            )

        if self.function is None:
            raise RuntimeError(
                "plot_convergence requires a function-bound interpolant "
                "(this object has function=None — built via from_values, algebra, or load)"
            )

        ns = list(range(4, max_n + 1, 2))
        errors = []
        for n in ns:
            cheb = ChebyshevApproximation(
                self.function, self.num_dimensions, self.domain,
                n_nodes=[n] * self.num_dimensions,
                additional_data=self.additional_data,
            )
            cheb.build(verbose=False)
            errors.append(cheb.error_estimate())

        if ax is None:
            _, ax = plt.subplots()
        ax.semilogy(ns, errors, marker="o")
        ax.set_xlabel("Number of nodes per dimension (N)")
        ax.set_ylabel("Error estimate (log scale)")
        ax.set_title(f"Convergence — {self.num_dimensions}-D Chebyshev")
        if target_error is not None:
            ax.axhline(target_error, linestyle="--", color="red", label=f"target={target_error}")
            ax.legend()
        return ax

    def plot_1d(self, ax=None, n_points=200, fixed=None):
        """Plot the 1-D slice of this interpolant.

        Requires the optional ``pychebyshev[viz]`` dependency group.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Pre-existing axes (creates a new figure if None).
        n_points : int
            Number of sample points along the free dim.
        fixed : dict[int, float] | None
            Map of dim → value to constrain other dims, leaving exactly one free.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from pychebyshev._viz import _plot_1d_impl
        return _plot_1d_impl(self, ax=ax, n_points=n_points, fixed=fixed)

    def plot_2d_surface(self, ax=None, n_points=50, fixed=None):
        """Plot a 3-D surface for the 2-D slice. Requires matplotlib."""
        from pychebyshev._viz import _plot_2d_surface_impl
        return _plot_2d_surface_impl(self, ax=ax, n_points=n_points, fixed=fixed)

    def plot_2d_contour(self, ax=None, n_points=50, n_levels=20, fixed=None):
        """Plot a filled-contour 2-D slice. Requires matplotlib."""
        from pychebyshev._viz import _plot_2d_contour_impl
        return _plot_2d_contour_impl(
            self, ax=ax, n_points=n_points, n_levels=n_levels, fixed=fixed
        )

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
        if not hasattr(self, "_original_n_nodes"):
            # v0.10 and earlier: n_nodes was always fully resolved
            self._original_n_nodes = list(self.n_nodes)
        if not hasattr(self, "descriptor"):
            self.descriptor = ""
        if not hasattr(self, "additional_data"):
            self.additional_data = None
        if not hasattr(self, "_derivative_id_registry"):
            self._derivative_id_registry = {}
        if not hasattr(self, "_derivative_id_to_orders"):
            self._derivative_id_to_orders = []
        if not hasattr(self, "special_points"):
            self.special_points = None
        if not hasattr(self, "n_workers"):
            self.n_workers = None

        # Reconstruct pre-allocated eval cache for fast_eval() (deprecated)
        self._eval_cache = {}
        if self.tensor_values is not None:
            for d in range(self.num_dimensions - 1, 0, -1):
                shape = tuple(self.n_nodes[i] for i in range(d))
                self._eval_cache[d] = np.zeros(shape)

    def save(
        self,
        path: str | os.PathLike,
        format: str = "pickle",
    ) -> None:
        """Save the built interpolant to a file.

        The original function is **not** saved — only the numerical data
        needed for evaluation. The saved file can be loaded with
        :meth:`load` without access to the original function.

        Parameters
        ----------
        path : str or path-like
            Destination file path.
        format : {'pickle', 'binary'}, optional
            ``'pickle'`` (default) writes the standard Python pickle
            stream — bit-identical to previous versions. ``'binary'``
            writes the portable ``.pcb`` format documented at
            ``docs/user-guide/binary-format.md``, which can be read by
            consumers in C, Rust, Julia, and other languages.

        Raises
        ------
        RuntimeError
            If the interpolant has not been built yet.
        ValueError
            If ``format`` is not ``'pickle'`` or ``'binary'``.

        Notes
        -----
        Pickle stays the default and is unchanged. Binary support was
        added in v0.14 and is opt-in.
        """
        if self.tensor_values is None:
            raise RuntimeError(
                "Cannot save an unbuilt ChebyshevApproximation. Call build() first."
            )

        if format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "binary":
            from pychebyshev import _binary
            with open(path, "wb") as f:
                _binary.write_approx(f, self)
        else:
            raise ValueError(
                f"format must be 'pickle' or 'binary', got {format!r}"
            )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChebyshevApproximation":
        """Load a previously-saved interpolant from a file.

        The file format is auto-detected by inspecting the first 4 bytes:
        files beginning with the magic ``b"PCB\\x00"`` are read as the
        portable v0.14 binary format; everything else is read as a
        Python pickle stream.

        Parameters
        ----------
        path : str or path-like
            Source file path.

        Returns
        -------
        ChebyshevApproximation
            The loaded interpolant. ``function`` is always ``None`` —
            re-bind it manually if you need to call ``build()`` again.

        Notes
        -----
        Pickle deserialization runs arbitrary code from the file. Only
        load files you trust. The binary format is safe to load from
        untrusted sources but does not preserve the function object.
        """
        from pychebyshev import _binary
        fmt = _binary.detect_format(path)
        if fmt == "binary":
            with open(path, "rb") as f:
                return _binary.read_approx(f)
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected a {cls.__name__} instance, got {type(obj).__name__}"
            )
        return obj

    @staticmethod
    def peek_format_version(filename: str) -> int:
        """Read a .pcb file's header and return its major format version
        integer. Does not deserialize the body.

        See ``docs/user-guide/binary-format.md`` for the format spec.

        Parameters
        ----------
        filename : str
            Path to a .pcb file.

        Returns
        -------
        int
            The major format version (currently 1).

        Raises
        ------
        ValueError
            If the file's first 4 bytes do not match the .pcb magic, or if
            the file is shorter than the 12-byte header.
        FileNotFoundError
            If the file does not exist.
        IOError
            If the file cannot be opened.
        """
        from pychebyshev._binary import peek_format_version
        return peek_format_version(filename)

    # ------------------------------------------------------------------
    # Pre-computed values: nodes first, values later
    # ------------------------------------------------------------------

    @staticmethod
    def nodes(
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
    ) -> dict:
        """Generate Chebyshev nodes without evaluating any function.

        Use this to obtain the grid points, evaluate your function externally
        (e.g. on an HPC cluster), then pass the results to :meth:`from_values`.

        Parameters
        ----------
        num_dimensions : int
            Number of dimensions.
        domain : list of (float, float)
            Lower and upper bounds for each dimension.
        n_nodes : list of int
            Number of Chebyshev nodes per dimension.

        Returns
        -------
        dict
            ``'nodes_per_dim'`` : list of 1-D arrays — Chebyshev nodes for
            each dimension, sorted ascending.

            ``'full_grid'`` : 2-D array, shape ``(prod(n_nodes), num_dimensions)``
            — Cartesian product of all nodes.  Row order matches
            ``np.ndindex(*n_nodes)`` (C-order), so
            ``values.reshape(info['shape'])`` produces the correct tensor.

            ``'shape'`` : tuple of int — expected shape of the tensor
            (``== tuple(n_nodes)``).

        Examples
        --------
        >>> info = ChebyshevApproximation.nodes(1, [[-1, 1]], [5])
        >>> info['shape']
        (5,)
        >>> info['full_grid'].shape
        (5, 1)
        """
        if len(domain) != num_dimensions or len(n_nodes) != num_dimensions:
            raise ValueError(
                f"len(domain)={len(domain)} and len(n_nodes)={len(n_nodes)} "
                f"must both equal num_dimensions={num_dimensions}"
            )
        from pychebyshev._extrude_slice import _make_nodes_for_dim

        nodes_per_dim = []
        for d in range(num_dimensions):
            lo, hi = domain[d]
            nodes_per_dim.append(_make_nodes_for_dim(lo, hi, n_nodes[d]))

        grids = np.meshgrid(*nodes_per_dim, indexing="ij")
        full_grid = np.column_stack([g.ravel() for g in grids])

        return {
            "nodes_per_dim": nodes_per_dim,
            "full_grid": full_grid,
            "shape": tuple(n_nodes),
        }

    @classmethod
    def get_optimal_n1(
        cls,
        function: Callable,
        domain_1d: List[float] | Tuple[float, float],
        error_threshold: float,
        max_n: int = 64,
    ) -> int:
        """Smallest N such that a 1-D Chebyshev build hits ``error_threshold``.

        Useful as a capacity estimate before committing to a full
        multi-dimensional build. Internally runs the same doubling
        loop as ``__init__(error_threshold=...)`` on a 1-D interpolant.

        Parameters
        ----------
        function : callable
            Signature ``f(point, data) -> float`` where ``point`` is a
            list of length 1.
        domain_1d : tuple or list
            ``(lo, hi)`` bounds.
        error_threshold : float
            Target supremum-norm error.
        max_n : int, optional
            Cap on the returned N. Default is 64. If the doubling loop
            cannot achieve ``error_threshold`` within this cap, a
            ``RuntimeWarning`` is emitted and ``max_n`` is returned.

        Returns
        -------
        int
            Resolved N on dimension 0.
        """
        lo, hi = domain_1d
        cheb = cls(
            function, 1, [[lo, hi]],
            error_threshold=error_threshold, max_n=max_n,
        )
        # Bypass the public build() wrapper and invoke the doubling
        # loop directly.  Rationale: the RuntimeWarning in
        # _build_with_threshold uses stacklevel=3, which points at
        #   user_code --(stacklevel=3)--> cheb.build() --> _build_with_threshold
        # when called through build().  When called through
        # get_optimal_n1, the extra build() frame pushes the pointer
        # off the user's call site.  Calling _build_with_threshold
        # directly re-aligns the stack so stacklevel=3 correctly
        # points at the user's get_optimal_n1(...) line.
        cheb._build_with_threshold(verbose=False)
        return int(cheb.n_nodes[0])

    @classmethod
    def from_values(
        cls,
        tensor_values: np.ndarray,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        max_derivative_order: int = 2,
    ) -> "ChebyshevApproximation":
        """Create an interpolant from pre-computed function values.

        The resulting object is fully functional: evaluation, derivatives,
        integration, rootfinding, algebra, extrusion/slicing, and
        serialization all work exactly as if ``build()`` had been called.

        Parameters
        ----------
        tensor_values : numpy.ndarray
            Function values on the Chebyshev grid.  Shape must equal
            ``tuple(n_nodes)``.  Entry ``tensor_values[i0, i1, ...]``
            corresponds to the function evaluated at
            ``(nodes[0][i0], nodes[1][i1], ...)``, where ``nodes`` are the
            arrays returned by :meth:`nodes`.
        num_dimensions : int
            Number of dimensions.
        domain : list of (float, float)
            Lower and upper bounds for each dimension.
        n_nodes : list of int
            Number of Chebyshev nodes per dimension.
        max_derivative_order : int, optional
            Maximum derivative order (default 2).

        Returns
        -------
        ChebyshevApproximation
            A fully built interpolant with ``function=None``.

        Raises
        ------
        ValueError
            If *tensor_values* shape does not match *n_nodes*, contains
            NaN or Inf, or if dimension parameters are inconsistent.

        Examples
        --------
        >>> import math
        >>> info = ChebyshevApproximation.nodes(1, [[0, 3.15]], [20])
        >>> vals = np.sin(info['full_grid'][:, 0]).reshape(info['shape'])
        >>> cheb = ChebyshevApproximation.from_values(vals, 1, [[0, 3.15]], [20])
        >>> abs(cheb.vectorized_eval([1.0], [0]) - math.sin(1.0)) < 1e-10
        True
        """
        tensor_values = np.asarray(tensor_values, dtype=float)

        # --- validation ---
        if len(domain) != num_dimensions or len(n_nodes) != num_dimensions:
            raise ValueError(
                f"len(domain)={len(domain)} and len(n_nodes)={len(n_nodes)} "
                f"must both equal num_dimensions={num_dimensions}"
            )
        expected_shape = tuple(n_nodes)
        if tensor_values.shape != expected_shape:
            raise ValueError(
                f"tensor_values.shape={tensor_values.shape} does not match "
                f"n_nodes={expected_shape}"
            )
        if not np.isfinite(tensor_values).all():
            raise ValueError("tensor_values contains NaN or Inf")
        for d in range(num_dimensions):
            lo, hi = domain[d]
            if lo >= hi:
                raise ValueError(
                    f"domain[{d}]: lo={lo} must be strictly less than hi={hi}"
                )

        # --- build the object without calling __init__ ---
        from pychebyshev._extrude_slice import _make_nodes_for_dim

        obj = object.__new__(cls)
        obj.function = None
        obj.num_dimensions = num_dimensions
        obj.domain = [list(bounds) for bounds in domain]
        obj.n_nodes = list(n_nodes)
        obj.max_derivative_order = max_derivative_order

        # Chebyshev nodes
        obj.nodes = []
        for d in range(num_dimensions):
            lo, hi = domain[d]
            obj.nodes.append(_make_nodes_for_dim(lo, hi, n_nodes[d]))

        obj.tensor_values = tensor_values.copy()

        # Barycentric weights
        obj.weights = []
        for d in range(num_dimensions):
            obj.weights.append(compute_barycentric_weights(obj.nodes[d]))

        # Differentiation matrices
        obj.diff_matrices = []
        for d in range(num_dimensions):
            obj.diff_matrices.append(
                compute_differentiation_matrix(obj.nodes[d], obj.weights[d])
            )

        obj.build_time = 0.0
        obj.n_evaluations = 0
        obj._cached_error_estimate = None
        obj.special_points = None
        obj.descriptor = ""
        obj.additional_data = None
        obj.n_workers = None
        obj._derivative_id_registry = {}
        obj._derivative_id_to_orders = []

        # Pre-allocate eval cache for deprecated fast_eval()
        obj._eval_cache = {}
        for d in range(obj.num_dimensions - 1, 0, -1):
            shape = tuple(obj.n_nodes[i] for i in range(d))
            obj._eval_cache[d] = np.zeros(shape)

        return obj

    # ------------------------------------------------------------------
    # Internal factory for arithmetic operators
    # ------------------------------------------------------------------

    @classmethod
    def _from_grid(cls, source, tensor_values):
        """Create a new instance sharing grid data from *source* with new *tensor_values*.

        Internal factory for arithmetic operators. The result is already 'built'
        (tensor_values is set) but has function=None and build_time=0.0.
        """
        obj = object.__new__(cls)
        obj.function = None
        obj.num_dimensions = source.num_dimensions
        obj.domain = [list(bounds) for bounds in source.domain]
        obj.n_nodes = list(source.n_nodes)
        obj.max_derivative_order = source.max_derivative_order
        obj.nodes = source.nodes          # list of 1-D arrays -- shared, not copied
        obj.weights = source.weights      # list of 1-D arrays -- shared
        obj.diff_matrices = source.diff_matrices  # shared
        obj.tensor_values = tensor_values
        obj.build_time = 0.0
        obj.n_evaluations = 0
        obj._cached_error_estimate = None
        obj.special_points = None
        obj.descriptor = ""
        obj.additional_data = None
        obj.n_workers = None
        obj._derivative_id_registry = {}
        obj._derivative_id_to_orders = []
        # Pre-allocate eval cache for deprecated fast_eval()
        obj._eval_cache = {}
        for d in range(obj.num_dimensions - 1, 0, -1):
            shape = tuple(obj.n_nodes[i] for i in range(d))
            obj._eval_cache[d] = np.zeros(shape)
        return obj

    # ------------------------------------------------------------------
    # Extrusion and slicing
    # ------------------------------------------------------------------

    def extrude(self, params):
        """Add new dimensions where the function is constant.

        The extruded interpolant evaluates identically to the original
        regardless of the new coordinate(s), because Chebyshev basis
        functions form a partition of unity: the barycentric weights
        sum to 1, so replicating tensor values along a new axis
        produces the same result for any coordinate in the new domain.

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, (lo, hi), n_nodes)`` or a list of such
            tuples.  ``dim_index`` is the position in the **output** space
            (0-indexed).  ``n_nodes`` must be >= 2 and ``lo < hi``.

        Returns
        -------
        ChebyshevApproximation
            A new, higher-dimensional interpolant (already built).
            The result has ``function=None`` and ``build_time=0.0``.

        Raises
        ------
        RuntimeError
            If the interpolant has not been built yet.
        TypeError
            If ``dim_index`` is not an integer.
        ValueError
            If ``dim_index`` is out of range, duplicated, ``lo >= hi``,
            or ``n_nodes < 2``.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._extrude_slice import (
            _extrude_tensor,
            _make_nodes_for_dim,
            _normalize_extrusion_params,
        )

        sorted_params = _normalize_extrusion_params(params, self.num_dimensions)

        tensor = self.tensor_values.copy()
        nodes = list(self.nodes)
        weights = list(self.weights)
        diff_matrices = list(self.diff_matrices)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)

        for dim_idx, (lo, hi), n in sorted_params:
            tensor = _extrude_tensor(tensor, dim_idx, n)
            new_nodes = _make_nodes_for_dim(lo, hi, n)
            new_weights = compute_barycentric_weights(new_nodes)
            new_diff_mat = compute_differentiation_matrix(new_nodes, new_weights)
            nodes.insert(dim_idx, new_nodes)
            weights.insert(dim_idx, new_weights)
            diff_matrices.insert(dim_idx, new_diff_mat)
            domain.insert(dim_idx, [lo, hi])
            n_nodes.insert(dim_idx, n)

        new_ndim = self.num_dimensions + len(sorted_params)
        obj = object.__new__(ChebyshevApproximation)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.nodes = nodes
        obj.weights = weights
        obj.diff_matrices = diff_matrices
        obj.tensor_values = tensor
        obj.build_time = 0.0
        obj.n_evaluations = 0
        obj.special_points = None
        obj.descriptor = ""
        obj.additional_data = None
        obj.n_workers = None
        obj._cached_error_estimate = None
        obj._derivative_id_registry = {}
        obj._derivative_id_to_orders = []
        obj._eval_cache = {}
        for d in range(new_ndim - 1, 0, -1):
            shape = tuple(n_nodes[i] for i in range(d))
            obj._eval_cache[d] = np.zeros(shape)
        return obj

    def slice(self, params):
        """Fix one or more dimensions at given values, reducing dimensionality.

        Contracts the tensor along each sliced dimension using the
        barycentric interpolation formula: for each sliced axis the
        normalized weight vector ``w_i / (x - x_i) / sum(w_j / (x - x_j))``
        is contracted with the tensor via ``np.tensordot``.  When the
        slice value coincides with a Chebyshev node (within 1e-14), the
        contraction reduces to an exact ``np.take`` (fast path).

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, value)`` or a list of such tuples.
            ``value`` must lie within the domain for that dimension.

        Returns
        -------
        ChebyshevApproximation
            A new, lower-dimensional interpolant (already built).
            The result has ``function=None`` and ``build_time=0.0``.

        Raises
        ------
        RuntimeError
            If the interpolant has not been built yet.
        TypeError
            If ``dim_index`` is not an integer.
        ValueError
            If a slice value is outside the domain, if slicing all
            dimensions, or if ``dim_index`` is out of range or duplicated.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._extrude_slice import (
            _normalize_slicing_params,
            _slice_tensor,
        )

        sorted_params = _normalize_slicing_params(params, self.num_dimensions)

        # Validate values within domain
        for dim_idx, value in sorted_params:
            lo, hi = self.domain[dim_idx]
            if value < lo or value > hi:
                raise ValueError(
                    f"Slice value {value} for dim {dim_idx} is outside "
                    f"domain [{lo}, {hi}]"
                )

        tensor = self.tensor_values.copy()
        nodes = list(self.nodes)
        weights = list(self.weights)
        diff_matrices = list(self.diff_matrices)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)

        for dim_idx, value in sorted_params:  # descending order
            tensor = _slice_tensor(tensor, dim_idx, nodes[dim_idx], weights[dim_idx], value)
            del nodes[dim_idx]
            del weights[dim_idx]
            del diff_matrices[dim_idx]
            del domain[dim_idx]
            del n_nodes[dim_idx]

        new_ndim = self.num_dimensions - len(sorted_params)
        obj = object.__new__(ChebyshevApproximation)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.nodes = nodes
        obj.weights = weights
        obj.diff_matrices = diff_matrices
        obj.tensor_values = tensor
        obj.build_time = 0.0
        obj.n_evaluations = 0
        obj.special_points = None
        obj.descriptor = ""
        obj.additional_data = None
        obj.n_workers = None
        obj._cached_error_estimate = None
        obj._derivative_id_registry = {}
        obj._derivative_id_to_orders = []
        obj._eval_cache = {}
        for d in range(new_ndim - 1, 0, -1):
            shape = tuple(n_nodes[i] for i in range(d))
            obj._eval_cache[d] = np.zeros(shape)
        return obj

    # ------------------------------------------------------------------
    # Calculus: integration, roots, optimization
    # ------------------------------------------------------------------

    def integrate(self, dims=None, bounds=None):
        """Integrate the interpolant over one or more dimensions.

        Uses Fejér-1 quadrature weights (Waldvogel 2006) at Chebyshev
        Type I nodes, computed in O(n log n) via DCT-III.  For multi-D
        tensors, each dimension is contracted via ``np.tensordot``.

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
        float or ChebyshevApproximation
            If all dimensions are integrated, returns the scalar integral.
            Otherwise returns a lower-dimensional interpolant.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If any dimension index is out of range or duplicated, or if
            bounds are outside the domain.

        References
        ----------
        Waldvogel (2006), "Fast Construction of the Fejér and
        Clenshaw–Curtis Quadrature Rules", BIT Numer. Math. 46(2):195–202.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import (
            _compute_fejer1_weights,
            _compute_sub_interval_weights,
            _normalize_bounds,
        )

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

        tensor = self.tensor_values.copy()
        nodes = list(self.nodes)
        wts = list(self.weights)
        diff_matrices = list(self.diff_matrices)
        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)

        # Process dimensions in descending order to avoid index shift
        for d in sorted(dims, reverse=True):
            a, b = domain[d]
            scale = (b - a) / 2.0
            bd = per_dim_bounds[dim_to_idx[d]]
            if bd is None:
                quad_w = _compute_fejer1_weights(n_nodes[d])
            else:
                t_lo = 2.0 * (bd[0] - a) / (b - a) - 1.0
                t_hi = 2.0 * (bd[1] - a) / (b - a) - 1.0
                quad_w = _compute_sub_interval_weights(n_nodes[d], t_lo, t_hi)
            tensor = np.tensordot(tensor, quad_w * scale, axes=([d], [0]))
            del nodes[d]
            del wts[d]
            del diff_matrices[d]
            del domain[d]
            del n_nodes[d]

        new_ndim = self.num_dimensions - len(dims)
        if new_ndim == 0:
            return float(tensor)

        obj = object.__new__(ChebyshevApproximation)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.nodes = nodes
        obj.weights = wts
        obj.diff_matrices = diff_matrices
        obj.tensor_values = tensor
        obj.build_time = 0.0
        obj.n_evaluations = 0
        obj.special_points = None
        obj.descriptor = ""
        obj.additional_data = None
        obj.n_workers = None
        obj._cached_error_estimate = None
        obj._derivative_id_registry = {}
        obj._derivative_id_to_orders = []
        obj._eval_cache = {}
        for d in range(new_ndim - 1, 0, -1):
            shape = tuple(n_nodes[i] for i in range(d))
            obj._eval_cache[d] = np.zeros(shape)
        return obj

    def roots(self, dim=None, fixed=None):
        """Find all roots of the interpolant along a specified dimension.

        Uses the colleague matrix eigenvalue method (Good 1961) via
        ``numpy.polynomial.chebyshev.chebroots``.  For multi-D interpolants,
        all dimensions except the target must be fixed at specific values
        (the interpolant is sliced to 1-D first).

        Parameters
        ----------
        dim : int or None
            Dimension along which to find roots.  For 1-D interpolants,
            defaults to 0.
        fixed : dict or None
            For multi-D interpolants, a dict ``{dim_index: value}`` for
            **all** dimensions except *dim*.

        Returns
        -------
        ndarray
            Sorted array of root locations in the physical domain.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails or values are out of domain.

        References
        ----------
        Good (1961), "The colleague matrix", Quarterly J. Math. 12(1):61–68.
        Trefethen (2013), "Approximation Theory and Approximation Practice",
        SIAM, Chapter 18.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _roots_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        # Slice to 1D
        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        return _roots_1d(sliced.tensor_values, sliced.domain[0])

    def minimize(self, dim=None, fixed=None):
        """Find the minimum value of the interpolant along a dimension.

        Computes derivative roots to locate critical points, then
        evaluates the interpolant at all critical points and domain
        endpoints.

        Parameters
        ----------
        dim : int or None
            Dimension along which to minimize.  Defaults to 0 for 1-D.
        fixed : dict or None
            For multi-D, dict ``{dim_index: value}`` for all other dims.

        Returns
        -------
        (value, location) : (float, float)
            The minimum value and its coordinate in the target dimension.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails.

        References
        ----------
        Trefethen (2013), "Approximation Theory and Approximation Practice",
        SIAM, Chapter 18.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _optimize_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        return _optimize_1d(
            sliced.tensor_values, sliced.nodes[0], sliced.weights[0],
            sliced.diff_matrices[0], sliced.domain[0], mode="min",
        )

    def maximize(self, dim=None, fixed=None):
        """Find the maximum value of the interpolant along a dimension.

        Computes derivative roots to locate critical points, then
        evaluates the interpolant at all critical points and domain
        endpoints.

        Parameters
        ----------
        dim : int or None
            Dimension along which to maximize.  Defaults to 0 for 1-D.
        fixed : dict or None
            For multi-D, dict ``{dim_index: value}`` for all other dims.

        Returns
        -------
        (value, location) : (float, float)
            The maximum value and its coordinate in the target dimension.

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        ValueError
            If *dim* / *fixed* validation fails.

        References
        ----------
        Trefethen (2013), "Approximation Theory and Approximation Practice",
        SIAM, Chapter 18.
        """
        if self.tensor_values is None:
            raise RuntimeError("Call build() first")

        from pychebyshev._calculus import _optimize_1d, _validate_calculus_args

        dim, slice_params = _validate_calculus_args(
            self.num_dimensions, dim, fixed, self.domain
        )

        if slice_params:
            sliced = self.slice(slice_params)
        else:
            sliced = self

        return _optimize_1d(
            sliced.tensor_values, sliced.nodes[0], sliced.weights[0],
            sliced.diff_matrices[0], sliced.domain[0], mode="max",
        )

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        return ChebyshevApproximation._from_grid(
            self, self.tensor_values + other.tensor_values
        )

    def __sub__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        return ChebyshevApproximation._from_grid(
            self, self.tensor_values - other.tensor_values
        )

    def __mul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        return ChebyshevApproximation._from_grid(
            self, self.tensor_values * float(scalar)
        )

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
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        self.tensor_values = self.tensor_values + other.tensor_values
        self._cached_error_estimate = None
        return self

    def __isub__(self, other):
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        self.tensor_values = self.tensor_values - other.tensor_values
        self._cached_error_estimate = None
        return self

    def __imul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        self.tensor_values = self.tensor_values * float(scalar)
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
        built = self.tensor_values is not None
        return (
            f"ChebyshevApproximation("
            f"dims={self.num_dimensions}, "
            f"nodes={self.n_nodes}, "
            f"built={built})"
        )

    def __str__(self) -> str:
        built = self.tensor_values is not None
        has_none = any(n is None for n in self.n_nodes)
        if has_none:
            # Auto-N mode, not yet resolved: avoid np.prod([None]) TypeError
            total_nodes_str = "auto"
        else:
            total_nodes_str = f"{int(np.prod(self.n_nodes)):,}"
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
            f"  Nodes:       {nodes_str} ({total_nodes_str} total)",
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
