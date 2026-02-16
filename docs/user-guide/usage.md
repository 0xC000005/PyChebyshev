# Usage Patterns

## Basic Workflow

Every PyChebyshev workflow follows three steps:

1. **Define** a callable function
2. **Build** the interpolant (evaluates at Chebyshev nodes, pre-computes weights)
3. **Query** at arbitrary points

```python
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(func, num_dimensions, domain, n_nodes)
cheb.build()
result = cheb.vectorized_eval(point, derivative_order)
```

## Evaluation Methods

PyChebyshev provides several evaluation methods with different speed/safety tradeoffs:

| Method | Speed | Safety | Use When |
|--------|-------|--------|----------|
| `eval()` | Slowest | Full validation | Testing and debugging |
| `vectorized_eval()` | Fastest | Full validation | **Default choice** |
| `vectorized_eval_multi()` | Fastest (multi) | Full validation | Price + Greeks at same point |

!!! note "Why no JIT?"
    Earlier versions offered a Numba JIT `fast_eval()` path, but `vectorized_eval()`
    is **~150x faster** because it routes N-D tensor contractions through BLAS GEMV
    — a single optimized matrix-vector multiply per dimension. JIT compilation cannot
    beat BLAS for this workload. `fast_eval()` is deprecated and will be removed in
    a future version.

### `vectorized_eval()` — Recommended

Uses BLAS matrix-vector products. For 5D with 11 nodes, replaces 16,105 Python loop
iterations with 5 BLAS calls:

```python
price = cheb.vectorized_eval([100, 100, 1.0, 0.25, 0.05], [0, 0, 0, 0, 0])
```

### `vectorized_eval_multi()` — Best for multiple derivatives

Pre-computes normalized barycentric weights once, reuses across all derivative orders:

```python
results = cheb.vectorized_eval_multi(
    [100, 100, 1.0, 0.25, 0.05],
    [
        [0, 0, 0, 0, 0],  # price
        [1, 0, 0, 0, 0],  # delta (dV/dS)
        [2, 0, 0, 0, 0],  # gamma (d²V/dS²)
        [0, 0, 0, 1, 0],  # vega  (dV/dσ)
        [0, 0, 0, 0, 1],  # rho   (dV/dr)
    ],
)
price, delta, gamma, vega, rho = results
```

## Batch Evaluation

For evaluating at many points:

```python
import numpy as np

points = np.array([
    [100, 100, 1.0, 0.25, 0.05],
    [110, 100, 1.0, 0.25, 0.05],
    [90, 100, 1.0, 0.25, 0.05],
])
prices = cheb.vectorized_eval_batch(points, [0, 0, 0, 0, 0])
```

## Function Signature

The function passed to `ChebyshevApproximation` must accept:

- `point` — a list of floats (one per dimension)
- `data` — arbitrary additional data (use `None` if not needed)

```python
def my_func(point, data):
    x, y, z = point
    return x**2 + y * z
```

## Next Steps

- [Computing Greeks](greeks.md) -- analytical derivatives via spectral differentiation
- [Chebyshev Algebra](algebra.md) -- combine interpolants via `+`, `-`, `*`, `/`
- [Chebyshev Calculus](calculus.md) -- integration, rootfinding & optimization on interpolants
- [Error Estimation](error-estimation.md) -- check accuracy without test points
- [Saving & Loading](serialization.md) -- persist built interpolants for production
- For 5+ dimensions, see [Tensor Train](tensor-train.md) or [Sliding Technique](sliding.md)
