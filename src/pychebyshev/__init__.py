"""PyChebyshev: Fast multi-dimensional Chebyshev tensor interpolation.

Provides the :class:`ChebyshevApproximation` class for building and
evaluating multi-dimensional Chebyshev interpolants with analytical
derivatives via spectral differentiation matrices.

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

from pychebyshev._version import __version__
from pychebyshev.barycentric import ChebyshevApproximation

__all__ = ["ChebyshevApproximation", "__version__"]
