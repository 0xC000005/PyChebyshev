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

from pychebyshev._version import __version__
from pychebyshev.barycentric import ChebyshevApproximation
from pychebyshev.slider import ChebyshevSlider
from pychebyshev.spline import ChebyshevSpline
from pychebyshev.tensor_train import ChebyshevTT

__all__ = ["ChebyshevApproximation", "ChebyshevSlider", "ChebyshevSpline", "ChebyshevTT", "__version__"]
