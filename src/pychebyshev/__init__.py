"""PyChebyshev: Fast multi-dimensional Chebyshev tensor interpolation.

Provides the :class:`ChebyshevApproximation` class for building and
evaluating multi-dimensional Chebyshev interpolants with analytical
derivatives via spectral differentiation matrices, the
:class:`ChebyshevSlider` class for high-dimensional approximation
via the Sliding Technique, the :class:`ChebyshevSpline` class for
piecewise Chebyshev interpolation at user-specified knots, and the
:class:`ChebyshevTT` class for Tensor Train Chebyshev interpolation
of 5+ dimensional functions.

Example
-------
>>> import math
>>> from pychebyshev import ChebyshevApproximation
>>> def f(x, _):
...     return math.sin(x[0]) + math.sin(x[1])
>>> cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
>>> cheb.build(verbose=False)
>>> round(cheb.vectorized_eval([0.5, 0.3], [0, 0]), 4)
0.7764
"""

from __future__ import annotations

from dataclasses import dataclass

from pychebyshev._version import __version__
from pychebyshev.barycentric import ChebyshevApproximation
from pychebyshev.slider import ChebyshevSlider
from pychebyshev.spline import ChebyshevSpline
from pychebyshev.tensor_train import ChebyshevTT


@dataclass(frozen=True)
class Domain:
    """Typed container for an interpolant's per-dimension bounds.

    Equivalent to a raw ``list[tuple[float, float]]``. Constructors of
    all four PyChebyshev classes accept either form.
    """

    bounds: list[tuple[float, float]]


@dataclass(frozen=True)
class Ns:
    """Typed container for an interpolant's per-dimension node counts.

    Equivalent to a raw ``list[int]``. Accepted by
    :class:`ChebyshevApproximation`, :class:`ChebyshevSpline`,
    :class:`ChebyshevSlider`, and :class:`ChebyshevTT`.
    """

    counts: list[int]


@dataclass(frozen=True)
class SpecialPoints:
    """Typed container for per-dimension kink/knot locations.

    Equivalent to a raw ``list[list[float]]``. Accepted by
    :class:`ChebyshevApproximation` and :class:`ChebyshevSpline`.
    """

    knots_per_dim: list[list[float]]


__all__ = [
    "ChebyshevApproximation",
    "ChebyshevSlider",
    "ChebyshevSpline",
    "ChebyshevTT",
    "Domain",
    "Ns",
    "SpecialPoints",
    "__version__",
]
