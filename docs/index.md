# PyChebyshev

**Fast multi-dimensional Chebyshev tensor interpolation with analytical derivatives.**

PyChebyshev builds a Chebyshev interpolant of any smooth function in up to N dimensions, then evaluates it and its derivatives in microseconds using vectorized NumPy operations. Four classes cover different use cases:

- **[`ChebyshevApproximation`](user-guide/usage.md)** — full tensor interpolation with analytical derivatives (up to ~5 dimensions)
- **[`ChebyshevSpline`](user-guide/spline.md)** — piecewise Chebyshev interpolation with knots at known singularities (kinks, discontinuities)
- **[`ChebyshevTT`](user-guide/tensor-train.md)** — Tensor Train format via TT-Cross for 5+ dimensions
- **[`ChebyshevSlider`](user-guide/sliding.md)** — additive decomposition for separable high-dimensional functions

## Key Features

- **Spectral accuracy** — exponential error decay as node count increases
- **Chebyshev Splines** — piecewise interpolation at kinks restores spectral convergence for non-smooth functions
- **Analytical derivatives** — via spectral differentiation matrices (no finite differences)
- **Tensor Train** — TT-Cross builds from O(d·n·r²) evaluations instead of O(n^d)
- **Fast evaluation** — ~0.065 ms per query (price), ~0.29 ms for price + 5 Greeks
- **Save & load** — persist built interpolants to disk; rebuild-free deployment
- **Pure Python** — NumPy + SciPy only, no compiled extensions needed

## Quick Example

```python
import math
from pychebyshev import ChebyshevApproximation

# Define any smooth function
def my_func(x, _):
    return math.sin(x[0]) * math.exp(-x[1])

# Build interpolant
cheb = ChebyshevApproximation(
    my_func,
    num_dimensions=2,
    domain=[[-1, 1], [0, 2]],
    n_nodes=[15, 15],
)
cheb.build()

# Evaluate
value = cheb.vectorized_eval([0.5, 1.0], [0, 0])

# First derivative with respect to x[0]
dfdx = cheb.vectorized_eval([0.5, 1.0], [1, 0])
```

## Installation

```bash
pip install pychebyshev
```

<!-- No optional dependencies needed — all evaluation uses BLAS via NumPy -->

## Performance

| Method | Price Error | Greek Error | Build Time | Query Time |
|--------|-------------|-------------|------------|------------|
| **Chebyshev Barycentric** | 0.000% | 1.980% | ~0.35s | ~0.065ms |
| **Chebyshev TT** | 0.014% | 0.029% | ~0.35s | ~0.004ms |
| **MoCaX Standard (C++)** | 0.000% | 1.980% | ~1.04s | ~0.47ms |
| **FDM** | 0.803% | 2.234% | N/A | ~500ms |

Based on 5D Black-Scholes tests with 11 nodes per dimension.
TT uses ~7,400 function evaluations (vs 161,051 for full tensor methods).
See [Benchmarks](benchmarks.md) for detailed comparisons including MoCaX TT.
