# PyChebyshev

**Fast multi-dimensional Chebyshev tensor interpolation with analytical derivatives.**

PyChebyshev builds a Chebyshev interpolant of any smooth function in up to N dimensions, then evaluates it and its derivatives in microseconds using vectorized NumPy operations.

## Key Features

- **Spectral accuracy** — exponential error decay as node count increases
- **Analytical derivatives** — via spectral differentiation matrices (no finite differences)
- **Fast evaluation** — ~0.065 ms per query (price), ~0.29 ms for price + 5 Greeks
- **Minimal storage** — 55 floats (440 bytes) for a 5D interpolant with 11 nodes per dimension
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
| **MoCaX Standard (C++)** | 0.000% | 1.980% | ~1.04s | ~0.47ms |
| **FDM** | 0.803% | 2.234% | N/A | ~500ms |

Based on 5D Black-Scholes tests with 11 nodes per dimension.
