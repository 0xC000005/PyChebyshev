"""Chebyshev Sliding approximation for high-dimensional functions.

Implements the Sliding Technique from Chapter 7 of Ruiz & Zeron (2021),
"Machine Learning for Risk Calculations". Decomposes a high-dimensional
function into a sum of low-dimensional Chebyshev interpolants (slides)
around a pivot point.

References
----------
- Ruiz & Zeron (2021), "Machine Learning for Risk Calculations",
  Wiley Finance, Chapter 7: Sliding Technique
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Callable, List, Tuple

import numpy as np

from pychebyshev.barycentric import ChebyshevApproximation


class ChebyshevSlider:
    """Chebyshev Sliding approximation for high-dimensional functions.

    Decomposes f(x_1, ..., x_n) into a sum of low-dimensional Chebyshev
    interpolants (slides) around a pivot point z:

        f(x) ≈ f(z) + Σ_i [s_i(x_group_i) - f(z)]

    where each slide s_i is a ChebyshevApproximation built on a subset
    of dimensions with the remaining dimensions fixed at z.

    This trades accuracy for dramatically reduced build cost: instead of
    evaluating f at n_1 × n_2 × ... × n_d grid points (exponential),
    the slider evaluates at n_1 × n_2 + n_3 × n_4 + ... (sum of products
    within each group).

    Parameters
    ----------
    function : callable
        Function to approximate. Signature: ``f(point, data) -> float``
        where ``point`` is a list of floats and ``data`` is arbitrary
        additional data (can be None).
    num_dimensions : int
        Total number of input dimensions.
    domain : list of (float, float)
        Bounds [lo, hi] for each dimension.
    n_nodes : list of int
        Number of Chebyshev nodes per dimension.
    partition : list of list of int
        Grouping of dimension indices into slides. Each dimension must
        appear in exactly one group. E.g. ``[[0,1,2], [3,4]]`` creates
        a 3D slide for dims 0,1,2 and a 2D slide for dims 3,4.
    pivot_point : list of float
        Reference point z around which slides are built.
    max_derivative_order : int, optional
        Maximum derivative order to pre-compute (default 2).

    Examples
    --------
    >>> import math
    >>> def f(x, _):
    ...     return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])
    >>> slider = ChebyshevSlider(
    ...     f, 3, [[-1,1], [-1,1], [-1,1]], [11,11,11],
    ...     partition=[[0], [1], [2]],
    ...     pivot_point=[0.0, 0.0, 0.0],
    ... )
    >>> slider.build(verbose=False)
    >>> round(slider.eval([0.5, 0.3, 0.1], [0,0,0]), 4)
    0.8764
    """

    def __init__(
        self,
        function: Callable,
        num_dimensions: int,
        domain: List[Tuple[float, float]],
        n_nodes: List[int],
        partition: List[List[int]],
        pivot_point: List[float],
        max_derivative_order: int = 2,
    ):
        self.function = function
        self.num_dimensions = num_dimensions
        self.domain = domain
        self.n_nodes = n_nodes
        self.partition = partition
        self.pivot_point = list(pivot_point)
        self.max_derivative_order = max_derivative_order

        # Validate partition
        all_dims = sorted(d for group in partition for d in group)
        if all_dims != list(range(num_dimensions)):
            raise ValueError(
                f"Partition must cover all dimensions 0..{num_dimensions-1} "
                f"exactly once. Got dimensions: {all_dims}"
            )

        # Map each dimension to its slide index
        self._dim_to_slide = {}
        for slide_idx, group in enumerate(partition):
            for d in group:
                self._dim_to_slide[d] = slide_idx

        self.slides: List[ChebyshevApproximation] = []
        self.pivot_value: float = 0.0
        self._built = False
        self._cached_error_estimate: float | None = None

    def build(self, verbose: bool = True) -> None:
        """Build all slides by evaluating the function at slide-specific grids.

        For each slide, dimensions outside the slide group are fixed at
        their pivot values.

        Parameters
        ----------
        verbose : bool, optional
            If True, print build progress. Default is True.
        """
        start = time.time()
        self._cached_error_estimate = None

        # Evaluate pivot value
        self.pivot_value = self.function(self.pivot_point, None)

        total_evals = sum(
            int(np.prod([self.n_nodes[d] for d in group]))
            for group in self.partition
        )

        if verbose:
            print(
                f"Building {self.num_dimensions}D Chebyshev Slider "
                f"({len(self.partition)} slides, {total_evals:,} evaluations "
                f"vs {int(np.prod(self.n_nodes)):,} for full tensor)..."
            )

        self.slides = []
        for slide_idx, group in enumerate(self.partition):
            slide_dim = len(group)
            slide_domain = [self.domain[d] for d in group]
            slide_n_nodes = [self.n_nodes[d] for d in group]

            # Create wrapper function that fixes non-group dims at pivot
            pivot = self.pivot_point

            def make_slide_func(grp, pvt):
                def slide_func(sub_point, data):
                    full_point = list(pvt)
                    for local_i, global_d in enumerate(grp):
                        full_point[global_d] = sub_point[local_i]
                    return self.function(full_point, data)
                return slide_func

            slide_func = make_slide_func(group, pivot)

            slide = ChebyshevApproximation(
                slide_func,
                slide_dim,
                slide_domain,
                slide_n_nodes,
                max_derivative_order=self.max_derivative_order,
            )
            slide.build(verbose=False)
            self.slides.append(slide)

            if verbose:
                slide_evals = int(np.prod(slide_n_nodes))
                print(f"  Slide {slide_idx + 1}/{len(self.partition)}: "
                      f"dims {group}, {slide_evals:,} evals")

        elapsed = time.time() - start
        if verbose:
            print(f"Build complete in {elapsed:.3f}s")

        self._built = True

    def eval(self, point: List[float], derivative_order: List[int]) -> float:
        """Evaluate the slider approximation at a point.

        Uses Equation 7.5 from Ruiz & Zeron (2021):
            f(x) ≈ f(z) + Σ_i [s_i(x_i) - f(z)]

        For derivatives, only the slide containing that dimension contributes.

        Parameters
        ----------
        point : list of float
            Evaluation point in the full n-dimensional space.
        derivative_order : list of int
            Derivative order for each dimension (0 = function value).

        Returns
        -------
        float
            Approximated function value or derivative.
        """
        if not self._built:
            raise RuntimeError("Call build() before eval().")

        is_derivative = any(d > 0 for d in derivative_order)

        if is_derivative:
            # For derivatives, only the slide containing the differentiated
            # dimensions contributes. The pivot_value is constant → its
            # derivative is 0. Cross-group mixed partials (e.g. d²f/dx₀dx₃
            # when x₀ and x₃ are in different slides) are exactly 0 because
            # slides are independent functions of disjoint variable groups.
            active_slides = {
                self._dim_to_slide[d]
                for d, order in enumerate(derivative_order)
                if order > 0
            }
            if len(active_slides) > 1:
                return 0.0

            # Single slide contributes
            slide_idx = active_slides.pop()
            group = self.partition[slide_idx]
            sub_point = [point[d] for d in group]
            sub_deriv = [derivative_order[d] for d in group]
            return self.slides[slide_idx].vectorized_eval(
                sub_point, sub_deriv
            )
        else:
            # Eq 7.5: f(x) ≈ v + Σ [s_i(x_i) - v]
            result = self.pivot_value
            for slide_idx, group in enumerate(self.partition):
                sub_point = [point[d] for d in group]
                sub_deriv = [derivative_order[d] for d in group]
                slide_val = self.slides[slide_idx].vectorized_eval(
                    sub_point, sub_deriv
                )
                result += slide_val - self.pivot_value
            return result

    def eval_multi(
        self, point: List[float], derivative_orders: List[List[int]]
    ) -> List[float]:
        """Evaluate slider at multiple derivative orders for the same point.

        Parameters
        ----------
        point : list of float
            Evaluation point in the full n-dimensional space.
        derivative_orders : list of list of int
            Each inner list specifies derivative order per dimension.

        Returns
        -------
        list of float
            Results for each derivative order.
        """
        return [self.eval(point, do) for do in derivative_orders]

    # ------------------------------------------------------------------
    # Error estimation
    # ------------------------------------------------------------------

    def error_estimate(self) -> float:
        """Estimate the sliding approximation error.

        Returns the sum of per-slide Chebyshev error estimates.
        Each slide's error is estimated independently using the
        Chebyshev coefficient method from Ruiz & Zeron (2021),
        Section 3.4.

        Note: This captures per-slide interpolation error only.
        Cross-group interaction error (inherent to the sliding
        decomposition) is **not** included.

        Returns
        -------
        float
            Estimated interpolation error (per-slide sum).

        Raises
        ------
        RuntimeError
            If ``build()`` has not been called.
        """
        if not self._built:
            raise RuntimeError("Call build() before error_estimate().")
        if self._cached_error_estimate is not None:
            return self._cached_error_estimate
        self._cached_error_estimate = sum(
            slide.error_estimate() for slide in self.slides
        )
        return self._cached_error_estimate

    @property
    def total_build_evals(self) -> int:
        """Total number of function evaluations used during build."""
        return sum(
            int(np.prod([self.n_nodes[d] for d in group]))
            for group in self.partition
        )

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
                f"Evaluation results may differ if internal data layout changed.",
                UserWarning,
                stacklevel=2,
            )

        self.__dict__.update(state)
        self.function = None

        # Ensure fields added in later versions exist (backward compat)
        if not hasattr(self, "_cached_error_estimate"):
            self._cached_error_estimate = None

    def save(self, path: str | os.PathLike) -> None:
        """Save the built slider to a file.

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
            If the slider has not been built yet.
        """
        if not self._built:
            raise RuntimeError(
                "Cannot save an unbuilt slider. Call build() first."
            )
        with open(os.fspath(path), "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ChebyshevSlider":
        """Load a previously saved slider from a file.

        The loaded object can evaluate immediately; no rebuild is needed.
        The ``function`` attribute will be ``None``. Assign a new function
        before calling ``build()`` again if a rebuild is desired.

        Parameters
        ----------
        path : str or path-like
            Path to the saved file.

        Returns
        -------
        ChebyshevSlider
            The restored slider.

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
    # Internal factory for arithmetic operators
    # ------------------------------------------------------------------

    @classmethod
    def _from_slides(cls, source, slides, pivot_value):
        """Create a new slider sharing grid metadata from *source* with new *slides* and *pivot_value*."""
        obj = object.__new__(cls)
        obj.function = None
        obj.num_dimensions = source.num_dimensions
        obj.domain = [list(bounds) for bounds in source.domain]
        obj.n_nodes = list(source.n_nodes)
        obj.max_derivative_order = source.max_derivative_order
        obj.partition = [list(group) for group in source.partition]
        obj.pivot_point = list(source.pivot_point)
        obj.slides = slides
        obj.pivot_value = pivot_value
        obj._dim_to_slide = source._dim_to_slide
        obj._built = True
        obj._cached_error_estimate = None
        return obj

    # ------------------------------------------------------------------
    # Extrusion and slicing
    # ------------------------------------------------------------------

    def extrude(self, params):
        """Add new dimensions where the function is constant.

        Each new dimension becomes its own single-dim slide group with
        ``tensor_values = np.full(n, pivot_value)``, so that
        ``s_new(x) - pivot_value = 0`` for all x (no contribution to
        the sliding sum).  This is the partition-of-unity property:
        the barycentric weights sum to 1, so a constant tensor
        produces the same value for any coordinate.

        Existing slide groups have their dimension indices remapped to
        account for the inserted dimensions.

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, (lo, hi), n_nodes)`` or a list of such
            tuples.  ``dim_index`` is the position in the **output** space
            (0-indexed).  ``n_nodes`` must be >= 2 and ``lo < hi``.

        Returns
        -------
        ChebyshevSlider
            A new, higher-dimensional slider (already built).
            The result has ``function=None``.

        Raises
        ------
        RuntimeError
            If the slider has not been built yet.
        TypeError
            If ``dim_index`` is not an integer.
        ValueError
            If ``dim_index`` is out of range, duplicated, ``lo >= hi``,
            or ``n_nodes < 2``.
        """
        if not self._built:
            raise RuntimeError("Call build() first")

        from pychebyshev._extrude_slice import (
            _make_nodes_for_dim,
            _normalize_extrusion_params,
        )
        from pychebyshev.barycentric import (
            compute_barycentric_weights,
            compute_differentiation_matrix,
        )

        sorted_params = _normalize_extrusion_params(params, self.num_dimensions)

        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)
        pivot_point = list(self.pivot_point)
        partition = [list(g) for g in self.partition]
        slides = list(self.slides)

        for dim_idx, (lo, hi), n in sorted_params:
            # Remap partition indices: increment all indices >= dim_idx
            for group in partition:
                for i in range(len(group)):
                    if group[i] >= dim_idx:
                        group[i] += 1

            # Create new 1-dim slide: constant at pivot_value
            new_nodes = _make_nodes_for_dim(lo, hi, n)
            new_weights = compute_barycentric_weights(new_nodes)
            new_diff_mat = compute_differentiation_matrix(new_nodes, new_weights)
            new_tensor = np.full(n, self.pivot_value)

            new_slide = object.__new__(ChebyshevApproximation)
            new_slide.function = None
            new_slide.num_dimensions = 1
            new_slide.domain = [[lo, hi]]
            new_slide.n_nodes = [n]
            new_slide.max_derivative_order = self.max_derivative_order
            new_slide.nodes = [new_nodes]
            new_slide.weights = [new_weights]
            new_slide.diff_matrices = [new_diff_mat]
            new_slide.tensor_values = new_tensor
            new_slide.build_time = 0.0
            new_slide.n_evaluations = 0
            new_slide._cached_error_estimate = None
            new_slide._eval_cache = {}

            # Add new group and slide
            partition.append([dim_idx])
            slides.append(new_slide)

            # Insert into domain/n_nodes/pivot_point
            domain.insert(dim_idx, [lo, hi])
            n_nodes.insert(dim_idx, n)
            pivot_point.insert(dim_idx, 0.5 * (lo + hi))

        new_ndim = self.num_dimensions + len(sorted_params)

        # Rebuild _dim_to_slide
        dim_to_slide = {}
        for slide_idx, group in enumerate(partition):
            for d in group:
                dim_to_slide[d] = slide_idx

        obj = object.__new__(ChebyshevSlider)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.partition = partition
        obj.pivot_point = pivot_point
        obj.slides = slides
        obj.pivot_value = self.pivot_value
        obj._dim_to_slide = dim_to_slide
        obj._built = True
        obj._cached_error_estimate = None
        return obj

    def slice(self, params):
        """Fix one or more dimensions at given values, reducing dimensionality.

        Two cases per sliced dimension:

        - **Multi-dim group**: The slide's ``ChebyshevApproximation`` is
          sliced at the local dimension index via barycentric contraction.
          When the slice value coincides with a Chebyshev node (within
          1e-14), the contraction reduces to an exact ``np.take``
          (fast path).  The dimension is removed from the group.
        - **Single-dim group**: The slide is evaluated at the value,
          giving a constant ``s_val``.  The shift
          ``delta = s_val - pivot_value`` is absorbed into
          ``pivot_value`` and each remaining slide's
          ``tensor_values``, and the group is removed entirely.

        Remaining dimension indices in all groups are remapped downward
        to stay contiguous.

        Parameters
        ----------
        params : tuple or list of tuples
            Single ``(dim_index, value)`` or a list of such tuples.
            ``value`` must lie within the domain for that dimension.

        Returns
        -------
        ChebyshevSlider
            A new, lower-dimensional slider (already built).
            The result has ``function=None``.

        Raises
        ------
        RuntimeError
            If the slider has not been built yet.
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

        domain = [list(b) for b in self.domain]
        n_nodes = list(self.n_nodes)
        pivot_point = list(self.pivot_point)
        partition = [list(g) for g in self.partition]
        slides = list(self.slides)
        pivot_value = self.pivot_value

        for dim_idx, value in sorted_params:  # descending order
            # Find which slide group contains dim_idx
            slide_idx = None
            local_dim_idx = None
            for si, group in enumerate(partition):
                if dim_idx in group:
                    slide_idx = si
                    local_dim_idx = group.index(dim_idx)
                    break

            if len(partition[slide_idx]) > 1:
                # Case 1: Multi-dim group — slice the slide's ChebyshevApproximation
                slides[slide_idx] = slides[slide_idx].slice(
                    (local_dim_idx, value)
                )
                partition[slide_idx].remove(dim_idx)
            else:
                # Case 2: Single-dim group — evaluate and absorb
                s_val = slides[slide_idx].vectorized_eval([value], [0])
                delta = s_val - pivot_value

                # Add delta to each remaining slide's tensor_values
                for i in range(len(slides)):
                    if i != slide_idx:
                        slides[i] = ChebyshevApproximation._from_grid(
                            slides[i],
                            slides[i].tensor_values + delta,
                        )

                pivot_value = s_val

                # Remove group and slide
                del partition[slide_idx]
                del slides[slide_idx]

            # Remap all partition indices > dim_idx down by 1
            for group in partition:
                for i in range(len(group)):
                    if group[i] > dim_idx:
                        group[i] -= 1

            del domain[dim_idx]
            del n_nodes[dim_idx]
            del pivot_point[dim_idx]

        new_ndim = self.num_dimensions - len(sorted_params)

        # Rebuild _dim_to_slide
        dim_to_slide = {}
        for si, group in enumerate(partition):
            for d in group:
                dim_to_slide[d] = si

        obj = object.__new__(ChebyshevSlider)
        obj.function = None
        obj.num_dimensions = new_ndim
        obj.domain = domain
        obj.n_nodes = n_nodes
        obj.max_derivative_order = self.max_derivative_order
        obj.partition = partition
        obj.pivot_point = pivot_point
        obj.slides = slides
        obj.pivot_value = pivot_value
        obj._dim_to_slide = dim_to_slide
        obj._built = True
        obj._cached_error_estimate = None
        return obj

    def _check_slider_compatible(self, other):
        """Validate that two sliders can be combined arithmetically."""
        from pychebyshev._algebra import _check_compatible
        _check_compatible(self, other)
        if self.partition != other.partition:
            raise ValueError(f"Partition mismatch: {self.partition} vs {other.partition}")
        if self.pivot_point != other.pivot_point:
            raise ValueError(f"Pivot point mismatch: {self.pivot_point} vs {other.pivot_point}")

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        self._check_slider_compatible(other)
        slides = [
            ChebyshevApproximation._from_grid(s_self, s_self.tensor_values + s_other.tensor_values)
            for s_self, s_other in zip(self.slides, other.slides)
        ]
        return ChebyshevSlider._from_slides(self, slides, self.pivot_value + other.pivot_value)

    def __sub__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        self._check_slider_compatible(other)
        slides = [
            ChebyshevApproximation._from_grid(s_self, s_self.tensor_values - s_other.tensor_values)
            for s_self, s_other in zip(self.slides, other.slides)
        ]
        return ChebyshevSlider._from_slides(self, slides, self.pivot_value - other.pivot_value)

    def __mul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        s = float(scalar)
        slides = [
            ChebyshevApproximation._from_grid(sl, sl.tensor_values * s)
            for sl in self.slides
        ]
        return ChebyshevSlider._from_slides(self, slides, self.pivot_value * s)

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
        self._check_slider_compatible(other)
        for s_self, s_other in zip(self.slides, other.slides):
            s_self.tensor_values = s_self.tensor_values + s_other.tensor_values
            s_self._cached_error_estimate = None
        self.pivot_value += other.pivot_value
        self._cached_error_estimate = None
        return self

    def __isub__(self, other):
        self._check_slider_compatible(other)
        for s_self, s_other in zip(self.slides, other.slides):
            s_self.tensor_values = s_self.tensor_values - s_other.tensor_values
            s_self._cached_error_estimate = None
        self.pivot_value -= other.pivot_value
        self._cached_error_estimate = None
        return self

    def __imul__(self, scalar):
        from pychebyshev._algebra import _is_scalar
        if not _is_scalar(scalar):
            return NotImplemented
        s = float(scalar)
        for sl in self.slides:
            sl.tensor_values = sl.tensor_values * s
            sl._cached_error_estimate = None
        self.pivot_value *= s
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
            f"ChebyshevSlider("
            f"dims={self.num_dimensions}, "
            f"slides={len(self.partition)}, "
            f"partition={self.partition}, "
            f"built={built})"
        )

    def __str__(self) -> str:
        built = self._built
        status = "built" if built else "not built"
        total_slide_evals = self.total_build_evals
        full_tensor_evals = int(np.prod(self.n_nodes))

        max_display = 6

        # Nodes line
        if self.num_dimensions > max_display:
            nodes_str = (
                "[" + ", ".join(str(n) for n in self.n_nodes[:max_display])
                + ", ...]"
            )
        else:
            nodes_str = str(self.n_nodes)

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

        # Pivot line
        if self.num_dimensions > max_display:
            pivot_str = (
                "[" + ", ".join(str(v) for v in self.pivot_point[:max_display])
                + ", ...]"
            )
        else:
            pivot_str = str(self.pivot_point)

        # Partition line
        if len(self.partition) > max_display:
            partition_str = (
                "["
                + ", ".join(str(g) for g in self.partition[:max_display])
                + ", ...]"
            )
        else:
            partition_str = str(self.partition)

        lines = [
            f"ChebyshevSlider ({self.num_dimensions}D, "
            f"{len(self.partition)} slides, {status})",
            f"  Partition: {partition_str}",
            f"  Pivot:     {pivot_str}",
            f"  Nodes:     {nodes_str} "
            f"({total_slide_evals:,} vs {full_tensor_evals:,} full tensor)",
            f"  Domain:    {domain_str}",
        ]

        if built and self.slides:
            lines.append(f"  Error est: {self.error_estimate():.2e}")
            lines.append("  Slides:")
            for i, (group, slide) in enumerate(
                zip(self.partition, self.slides)
            ):
                slide_evals = int(
                    np.prod([self.n_nodes[d] for d in group])
                )
                lines.append(
                    f"    [{i}] dims {group}: "
                    f"{slide_evals:,} evals, "
                    f"built in {slide.build_time:.3f}s"
                )

        return "\n".join(lines)
