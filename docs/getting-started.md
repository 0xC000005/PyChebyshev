# Getting Started

## Installation

### From PyPI

```bash
pip install pychebyshev
```

### With Numba JIT acceleration

```bash
pip install pychebyshev[jit]
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

## Choosing Node Counts

- **10-15 nodes** per dimension is typical for smooth analytic functions
- More nodes = higher accuracy but more build-time evaluations ($n_1 \times n_2 \times \cdots$)
- For 5D with 11 nodes: $11^5 = 161{,}051$ function evaluations at build time
- Convergence is **exponential** for analytic functions — a few extra nodes can eliminate errors entirely
