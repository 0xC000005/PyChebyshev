# Getting Started

## Installation

### From PyPI

```bash
pip install pychebyshev
```

### From source (development)

```bash
git clone https://github.com/0xC000005/PyChebyshev.git
cd PyChebyshev
pip install -e ".[dev]"
```

## Quick Start

### 1. Define your function

PyChebyshev can approximate any smooth function. The function signature is
`f(point, data) -> float`, where `point` is a list of coordinates and `data`
is optional additional data (pass `None` if unused).

```python
import math

def my_func(x, _):
    return math.sin(x[0]) + math.cos(x[1])
```

### 2. Build the interpolant

```python
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(
    function=my_func,
    num_dimensions=2,
    domain=[[-3, 3], [-3, 3]],  # bounds per dimension
    n_nodes=[15, 15],            # Chebyshev nodes per dimension
)
cheb.build()
```

### 3. Evaluate

```python
# Function value
value = cheb.vectorized_eval([1.0, 2.0], [0, 0])

# First derivative w.r.t. x[0]
dfdx0 = cheb.vectorized_eval([1.0, 2.0], [1, 0])

# Second derivative w.r.t. x[1]
d2fdx1 = cheb.vectorized_eval([1.0, 2.0], [0, 2])
```

### 4. Evaluate price + all Greeks at once

For maximum efficiency when computing multiple derivatives at the same point:

```python
results = cheb.vectorized_eval_multi(
    [1.0, 2.0],
    [
        [0, 0],  # function value
        [1, 0],  # df/dx0
        [0, 1],  # df/dx1
        [2, 0],  # d2f/dx0^2
    ],
)
# results = [value, dfdx0, dfdx1, d2fdx0]
```

This shares barycentric weights across all derivative orders, saving ~25% compared
to separate calls.

### 5. Save for later

Save the built interpolant to skip rebuilding next time:

```python
cheb.save("my_interpolant.pkl")
```

Load it back — no rebuild needed:

```python
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation.load("my_interpolant.pkl")
value = cheb.vectorized_eval([1.0, 2.0], [0, 0])
```

See [Saving & Loading](user-guide/serialization.md) for details.

## Choosing Node Counts

- **10-15 nodes** per dimension is typical for smooth analytic functions
- More nodes = higher accuracy but more build-time evaluations ($n_1 \times n_2 \times \cdots$)
- For 5D with 11 nodes: $11^5 = 161{,}051$ function evaluations at build time
- Convergence is **exponential** for analytic functions — a few extra nodes can eliminate errors entirely

## Choosing the Right Class

| Class | Dimensions | Build Cost | Derivatives | Best For |
|-------|-----------|-----------|-------------|----------|
| [`ChebyshevApproximation`](user-guide/usage.md) | 1--5 | $n^d$ evals | Analytical | Full accuracy with spectral derivatives |
| [`ChebyshevSpline`](user-guide/spline.md) | 1--5 | $\text{pieces} \times n^d$ evals | Analytical (per piece) | Functions with kinks at known locations |
| [`ChebyshevTT`](user-guide/tensor-train.md) | 5+ | $O(d \cdot n \cdot r^2)$ evals | Finite differences | High-dimensional problems where full grids are infeasible |
| [`ChebyshevSlider`](user-guide/sliding.md) | 5+ | Sum of slide grids | Analytical (per slide) | Functions with additive/separable structure |

## Next Steps

- [Computing Greeks](user-guide/greeks.md) -- analytical derivatives for pricing
- [Error Estimation](user-guide/error-estimation.md) -- validate accuracy without test points
- [Saving & Loading](user-guide/serialization.md) -- persist built interpolants
- [Benchmarks](benchmarks.md) -- performance comparison with MoCaX C++
