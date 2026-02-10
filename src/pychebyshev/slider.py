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
