# Chebyshev Algebra

## Motivation -- Portfolio Combination

In counterparty credit risk (CCR), thousands of trades sharing common risk factors
must be priced at millions of Monte Carlo scenarios.  Building a Chebyshev proxy per
trade reduces pricing cost, but evaluating 1,000 separate proxies at each scenario is
still $O(\text{num\_trades})$.

Algebraic combination lets you **pre-combine** trade proxies into a single
netting-set-level proxy:

```python
portfolio = w1 * trade_1 + w2 * trade_2 + ... + wn * trade_n
```

One evaluation of `portfolio` gives the netting set price -- $O(1)$ regardless of
the number of trades.

!!! tip "When to use algebra"
    Use algebraic combination when multiple Chebyshev interpolants share the **same
    grid** (same domain, node counts, derivative order) and you want to combine them
    into a single interpolant for faster evaluation.

## Mathematical Basis

The barycentric interpolation formula evaluates a Chebyshev Tensor (CT) at any point
$\mathbf{x}$:

$$
p_n(\mathbf{x}) = \sum_{i_1, \ldots, i_d} v_{i_1, \ldots, i_d} \prod_{k=1}^{d} \ell^{(k)}_{i_k}(x_k)
$$

where $\ell^{(k)}_{i_k}$ are the barycentric basis functions.  This is **linear in
the values** $v_{i_1, \ldots, i_d}$.

**Theorem (Linearity of CT operations).**  Let $T_f$ and $T_g$ be two CTs on the
same grid.  Then:

1. **Addition**: $T_f + T_g$ (element-wise on grid values) is the CT for $f + g$
2. **Scalar multiplication**: $c \cdot T_f$ is the CT for $c \cdot f$
3. **Subtraction**: $T_f - T_g$ is the CT for $f - g$

*Proof.*  Direct from linearity of the barycentric formula.

**Corollary (Derivatives).**  Since the spectral differentiation matrix
$\mathcal{D}_k$ depends only on grid points:

$$
\mathcal{D}_k (v_f + v_g) = \mathcal{D}_k v_f + \mathcal{D}_k v_g
$$

Derivatives of a combined CT equal the combined derivatives.

**Error bound.**  By the triangle inequality:

$$
\|(f + g) - (p_f + p_g)\|_\infty \leq \epsilon_f + \epsilon_g
$$

For scalar multiplication: $\|cf - cp_f\|_\infty = |c| \cdot \epsilon_f$.

!!! note "Book reference"
    The linearity of Chebyshev Tensor operations is described in Section 3.9 of Ruiz
    & Zeron (2021), *Machine Learning for Risk Calculations*, Wiley Finance.

## Quick Start

```python
import math
from pychebyshev import ChebyshevApproximation

# Two functions on the same grid
def f(x, _):
    return math.sin(x[0]) + math.sin(x[1])

def g(x, _):
    return math.cos(x[0]) * math.cos(x[1])

a = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
b = ChebyshevApproximation(g, 2, [[-1, 1], [-1, 1]], [11, 11])
a.build(); b.build()

# Combine into a portfolio proxy
portfolio = 0.6 * a + 0.4 * b

# Evaluate price and Greeks at any point
point = [0.5, 0.3]
price = portfolio.vectorized_eval(point, [0, 0])
delta = portfolio.vectorized_eval(point, [1, 0])
gamma = portfolio.vectorized_eval(point, [2, 0])

print(portfolio)
```

Output:

```
ChebyshevApproximation (2D, built)
  Nodes:       [11, 11] (121 total)
  Domain:      [-1, 1] x [-1, 1]
  Build:       0.000s, 0 evaluations
  Error est:   4.22e-10
  Derivatives: up to order 2
```

The combined `portfolio` is a regular `ChebyshevApproximation` -- all existing
evaluation methods (`eval`, `vectorized_eval`, `vectorized_eval_multi`,
`vectorized_eval_batch`) work unchanged.

## Supported Operations

| Operator | Example | Result |
|----------|---------|--------|
| `+` | `cheb_a + cheb_b` | Element-wise add tensor values |
| `-` | `cheb_a - cheb_b` | Element-wise subtract |
| `*` scalar | `3.0 * cheb` or `cheb * 3.0` | Scale all tensor values |
| `/` scalar | `cheb / 2.0` | Divide all tensor values |
| unary `-` | `-cheb` | Negate all tensor values |
| `+=` | `cheb_a += cheb_b` | In-place add |
| `-=` | `cheb_a -= cheb_b` | In-place subtract |
| `*=` | `cheb *= 3.0` | In-place scale |
| `/=` | `cheb /= 2.0` | In-place divide |

## Compatibility Requirements

Both operands must share:

- **Same type** -- both `ChebyshevApproximation`, both `ChebyshevSpline`, etc.
- **Same `num_dimensions`** -- number of interpolation dimensions
- **Same `domain`** -- identical domain bounds in every dimension
- **Same `n_nodes`** -- same node counts in every dimension
- **Same `max_derivative_order`** -- same spectral differentiation depth
- **Both must be built** -- `build()` must have been called on each operand

Additional requirements for specific classes:

- **`ChebyshevSpline`**: same `knots` in every dimension
- **`ChebyshevSlider`**: same `partition` and same `pivot_point`

A `ValueError` is raised if any of these conditions are not met.

!!! example "Checking compatibility"
    ```python
    # These will raise ValueError:
    a = ChebyshevApproximation(f, 2, [[0, 1], [0, 1]], [10, 10])
    b = ChebyshevApproximation(g, 2, [[0, 1], [0, 2]], [10, 10])  # different domain
    a.build(); b.build()
    c = a + b  # ValueError: domain mismatch
    ```

## Derivatives

Derivatives propagate automatically through algebraic operations.  The combined
interpolant inherits the spectral differentiation matrices from its operands, so
no re-computation is needed:

```python
# Build two interpolants
call.build()
put.build()

# Combine
portfolio = 0.6 * call + 0.4 * put

# Delta of the portfolio = 0.6 * delta_call + 0.4 * delta_put
delta = portfolio.vectorized_eval(point, [1, 0, 0])

# Gamma works too
gamma = portfolio.vectorized_eval(point, [2, 0, 0])
```

This follows directly from the linearity of the spectral differentiation matrices
$\mathcal{D}_k$.

## Error Estimation

`error_estimate()` recomputes from the combined Chebyshev coefficients (DCT of the
combined tensor values).  In practice this may give a **tighter bound** than the
triangle inequality $\epsilon_f + \epsilon_g$, because cancellation between the
high-order coefficients of $f$ and $g$ can reduce the estimated tail.

```python
portfolio = 0.6 * call + 0.4 * put
err = portfolio.error_estimate()
print(f"Portfolio error estimate: {err:.2e}")
```

## Serialization

Combined interpolants support `save()` and `load()` just like any other built
interpolant.  The underlying function reference is lost (`function=None`), but all
tensor values, grid data, and differentiation matrices are preserved:

```python
portfolio = 0.6 * call + 0.4 * put
portfolio.save("portfolio.pkl")

loaded = ChebyshevApproximation.load("portfolio.pkl")
loaded.vectorized_eval(point, [0, 0])  # works identically
```

## Spline and Slider Examples

### ChebyshevSpline addition

Two splines with the **same knots** can be combined:

```python
from pychebyshev import ChebyshevSpline

spline_a = ChebyshevSpline(
    f, 2, [[80, 120], [0.25, 1.0]], [15, 15],
    knots=[[100.0], []],
)
spline_b = ChebyshevSpline(
    g, 2, [[80, 120], [0.25, 1.0]], [15, 15],
    knots=[[100.0], []],
)
spline_a.build(); spline_b.build()

combined = spline_a + spline_b
price = combined.eval([110.0, 0.5], [0, 0])
```

Each piece is combined independently -- the combined spline has the same knot
structure as its operands.

### ChebyshevSlider addition

Two sliders with the **same partition and pivot point** can be combined:

```python
from pychebyshev import ChebyshevSlider

slider_a = ChebyshevSlider(
    f, 5, domain, [11] * 5,
    partition=[[0, 1], [2, 3, 4]],
    pivot_point=[0.0] * 5,
)
slider_b = ChebyshevSlider(
    g, 5, domain, [11] * 5,
    partition=[[0, 1], [2, 3, 4]],
    pivot_point=[0.0] * 5,
)
slider_a.build(); slider_b.build()

combined = slider_a + slider_b
val = combined.eval([0.5] * 5, [0] * 5)
```

Each slide is combined independently, preserving the additive decomposition structure.

## Why Pointwise Products are NOT Supported

The product $f \cdot g$ is **not** $v_f \odot v_g$ (element-wise product of grid
values).  The product of two polynomials of degree $n$ has degree $2n$, which
cannot be represented on the same $n$-point grid.

Only **linear combinations** (addition, subtraction, scalar multiplication) are exact
on the same grid.  Pointwise multiplication of two Chebyshev interpolants requires a
grid refinement step and is not supported.

!!! note "Workaround for products"
    If you need to approximate $f \cdot g$, build a single Chebyshev interpolant
    for the product function directly: define `h(x, aux) = f(x) * g(x)` and call
    `ChebyshevApproximation(h, ...).build()`.

## Limitations

- **No `ChebyshevTT` operators** -- TT addition requires rank control (rank of
  $T_f + T_g$ is $r_f + r_g$) and is not currently implemented.
- **No cross-type operations** -- you cannot add a `ChebyshevApproximation` to a
  `ChebyshevSpline` or a `ChebyshevSlider`.
- **Operands must share exact grid parameters** -- domain, node counts, derivative
  order, and (where applicable) knots or partition must be identical.
- **Result has `function=None`** -- the combined interpolant cannot call `build()`
  again, since it has no underlying function reference.
