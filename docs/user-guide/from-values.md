# Pre-computed Values

## Motivation -- Nodes First, Values Later

In production environments -- HPC clusters, distributed pricing engines, cloud
compute -- the function to be interpolated often cannot be called from within
Python.  The evaluation may run on a separate process, a remote node, or inside
a proprietary pricing library.

PyChebyshev's standard workflow (`__init__` + `build()`) requires a Python
callable.  The **pre-computed values** workflow decouples node generation from
function evaluation:

1. **Generate nodes** -- call `nodes()` to get the Chebyshev grid points.
2. **Evaluate externally** -- feed the points to your own pricing engine.
3. **Load values** -- call `from_values()` to construct the interpolant.

The resulting object is *fully functional*: evaluation, derivatives,
integration, rootfinding, optimization, algebra, extrusion/slicing, and
serialization all work identically to a `build()`-based interpolant.

## Workflow

### ChebyshevApproximation

```python
import numpy as np
from pychebyshev import ChebyshevApproximation

# Step 1: Get nodes (no function needed)
info = ChebyshevApproximation.nodes(
    num_dimensions=2,
    domain=[[-1, 1], [0, 2]],
    n_nodes=[15, 11],
)
# info['nodes_per_dim']  -> [array(15,), array(11,)]
# info['full_grid']      -> array(165, 2)
# info['shape']          -> (15, 11)

# Step 2: Evaluate externally
# Each row of full_grid is one (x0, x1) point.
# Feed these to your pricing engine, collect results.
values = my_pricing_engine(info['full_grid'])

# Step 3: Reshape and build
tensor = values.reshape(info['shape'])
cheb = ChebyshevApproximation.from_values(
    tensor, num_dimensions=2,
    domain=[[-1, 1], [0, 2]], n_nodes=[15, 11],
)

# Now use it like any other interpolant
price     = cheb.vectorized_eval([0.5, 1.0], [0, 0])
delta     = cheb.vectorized_eval([0.5, 1.0], [1, 0])
integral  = cheb.integrate()
```

### ChebyshevSpline

For piecewise interpolation with knots, the same pattern applies per piece:

```python
from pychebyshev import ChebyshevSpline

# Step 1: Get per-piece nodes
info = ChebyshevSpline.nodes(
    num_dimensions=1,
    domain=[[-1, 1]],
    n_nodes=[15],
    knots=[[0.0]],
)
# info['pieces']       -> list of 2 dicts (one per piece)
# info['num_pieces']   -> 2
# info['piece_shape']  -> (2,)

# Step 2: Evaluate each piece externally
piece_values = []
for piece in info['pieces']:
    vals = my_engine(piece['full_grid']).reshape(piece['shape'])
    piece_values.append(vals)

# Step 3: Build
spline = ChebyshevSpline.from_values(
    piece_values, num_dimensions=1,
    domain=[[-1, 1]], n_nodes=[15], knots=[[0.0]],
)
```

## Indexing Convention

The tensor entries must satisfy:

$$
\texttt{tensor\_values}[i_0, i_1, \ldots] = f\!\bigl(\texttt{nodes\_per\_dim}[0][i_0],\; \texttt{nodes\_per\_dim}[1][i_1],\; \ldots\bigr)
$$

The rows of `full_grid` follow `np.ndindex(*n_nodes)` order (C-order,
row-major), so `values.reshape(info['shape'])` produces the correct tensor
automatically.

!!! warning "Use C-order reshape"
    Always reshape with the default C-order: `values.reshape(info['shape'])`.
    Do **not** use `order='F'` (Fortran-order), which would silently produce
    incorrect tensor entries and lead to wrong interpolation results.

For **spline pieces**, the list order follows `np.ndindex(*piece_shape)`
(C-order).  In 2D with knots `[[0.0], [1.0]]` on domain `[[-1,1], [0,2]]`,
the four pieces are:

| Index | piece_index | sub_domain                    |
|-------|-------------|-------------------------------|
| 0     | (0, 0)      | [(-1, 0), (0, 1)]            |
| 1     | (0, 1)      | [(-1, 0), (1, 2)]            |
| 2     | (1, 0)      | [(0, 1), (0, 1)]             |
| 3     | (1, 1)      | [(0, 1), (1, 2)]             |

## Mathematical Justification

Everything the interpolant needs -- *except* the function values themselves --
depends only on the node positions:

| Pre-computed data          | Depends on          |
|----------------------------|---------------------|
| Chebyshev nodes            | domain, n_nodes     |
| Barycentric weights        | nodes               |
| Differentiation matrices   | nodes, weights      |
| Fejér quadrature weights   | nodes               |

The function values appear only in `tensor_values`.  The Chebyshev nodes are
the zeros of $T_n(x)$, mapped to each dimension's domain (Trefethen 2013,
Ch. 3).  Since `from_values()` computes all of the above from the same node
formula as `build()`, the resulting interpolant is **bit-identical** to one
built the traditional way.

## Examples

### 1-D: sin(x) on [0, pi]

```python
import numpy as np
from pychebyshev import ChebyshevApproximation

info = ChebyshevApproximation.nodes(1, [[0, 3.14159]], [20])
values = np.sin(info['full_grid'][:, 0]).reshape(info['shape'])
cheb = ChebyshevApproximation.from_values(values, 1, [[0, 3.14159]], [20])

# Integral of sin(x) on [0, pi] ~ 2.0
print(cheb.integrate())  # 1.9999999...
```

### Combine with Algebra

```python
# Two proxies built externally
cheb_a = ChebyshevApproximation.from_values(vals_a, 2, domain, n_nodes)
cheb_b = ChebyshevApproximation.from_values(vals_b, 2, domain, n_nodes)

# Portfolio-level proxy
portfolio = 0.6 * cheb_a + 0.4 * cheb_b
```

### Save & Load

```python
cheb.save("my_proxy.pkl")
loaded = ChebyshevApproximation.load("my_proxy.pkl")
```

## Error Handling

| Error                        | Raised when                                     |
|------------------------------|--------------------------------------------------|
| `ValueError` (shape)        | `tensor_values.shape != tuple(n_nodes)`          |
| `ValueError` (NaN/Inf)      | `tensor_values` contains non-finite values       |
| `ValueError` (dimensions)   | `len(domain) != num_dimensions` or `len(n_nodes) != num_dimensions` |
| `ValueError` (domain)       | `lo >= hi` for any dimension                     |
| `RuntimeError` (build)      | Calling `build()` on a `from_values` object (no function) |

!!! note "Calling `build()` on a `from_values` result"
    Objects created via `from_values()` have `function=None`.  To re-build
    with a different set of values, create a new object via `from_values()`.
    To re-build from a callable, assign a function first:
    `cheb.function = my_func`, then `cheb.build()`.

## Combining with Other Features

Objects created via `from_values()` support **all** PyChebyshev operations:

- **Evaluation & derivatives** -- `vectorized_eval()`, `vectorized_eval_multi()`
- **Calculus** -- `integrate()`, `roots()`, `minimize()`, `maximize()`
- **Algebra** -- `+`, `-`, `*`, `/` (including with `build()`-based objects)
- **Extrusion & slicing** -- `extrude()`, `slice()`
- **Serialization** -- `save()`, `load()`
- **Error estimation** -- `error_estimate()`

## API Reference

### `ChebyshevApproximation.nodes()`

| Parameter         | Type                     | Description                    |
|-------------------|--------------------------|--------------------------------|
| `num_dimensions`  | int                      | Number of dimensions           |
| `domain`          | list of (float, float)   | Bounds per dimension           |
| `n_nodes`         | list of int              | Nodes per dimension            |

**Returns** `dict` with keys `'nodes_per_dim'`, `'full_grid'`, `'shape'`.

### `ChebyshevApproximation.from_values()`

| Parameter               | Type              | Description                        |
|-------------------------|-------------------|------------------------------------|
| `tensor_values`         | numpy.ndarray     | Function values, shape `tuple(n_nodes)` |
| `num_dimensions`        | int               | Number of dimensions               |
| `domain`                | list of (float, float) | Bounds per dimension          |
| `n_nodes`               | list of int       | Nodes per dimension                |
| `max_derivative_order`  | int (default 2)   | Maximum derivative order           |

**Returns** `ChebyshevApproximation` with `function=None`.

### `ChebyshevSpline.nodes()`

| Parameter         | Type                     | Description                    |
|-------------------|--------------------------|--------------------------------|
| `num_dimensions`  | int                      | Number of dimensions           |
| `domain`          | list of (float, float)   | Bounds per dimension           |
| `n_nodes`         | list of int              | Nodes per dimension per piece  |
| `knots`           | list of list of float    | Knot positions per dimension   |

**Returns** `dict` with keys `'pieces'`, `'num_pieces'`, `'piece_shape'`.

### `ChebyshevSpline.from_values()`

| Parameter               | Type                  | Description                        |
|-------------------------|-----------------------|------------------------------------|
| `piece_values`          | list of numpy.ndarray | Per-piece values in C-order        |
| `num_dimensions`        | int                   | Number of dimensions               |
| `domain`                | list of (float, float)| Bounds per dimension               |
| `n_nodes`               | list of int           | Nodes per dimension per piece      |
| `knots`                 | list of list of float | Knot positions per dimension       |
| `max_derivative_order`  | int (default 2)       | Maximum derivative order           |

**Returns** `ChebyshevSpline` with `function=None`.

## Comparison with MoCaX Extend

PyChebyshev's `nodes()` + `from_values()` is analogous to the MoCaX Extend
workflow, which also supports decoupled node generation and value loading:

| Step                | PyChebyshev                                | MoCaX Extend                                   |
|---------------------|--------------------------------------------|-------------------------------------------------|
| Get nodes           | `ChebyshevApproximation.nodes()`           | `MocaxExtend.cheb_pts_per_dimension`            |
| Load values         | `ChebyshevApproximation.from_values()`     | `MocaxExtend.set_subgrid_values()`              |
| Finalize            | (immediate)                                | `MocaxExtend.run_rank_adaptive_algo()`          |
| Grid type           | Full tensor grid                           | Random subgrid (TT compression)                 |
| Compression         | None (exact on grid)                       | TT decomposition (lossy)                        |

## See Also

- [Chebyshev Algebra](algebra.md) -- combine from_values interpolants
- [Chebyshev Calculus](calculus.md) -- integrate, find roots, optimize
- [Extrusion & Slicing](extrude-slice.md) -- add/fix dimensions
- [Saving & Loading](serialization.md) -- persist from_values objects
