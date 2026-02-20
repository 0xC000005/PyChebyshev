# Chebyshev Splines

## The Gibbs Phenomenon

Chebyshev interpolation converges **exponentially** for smooth (analytic) functions
-- this is the spectral advantage that makes the method so powerful.  But when the
target function has a **discontinuity** or a **kink**, this advantage disappears.

**Jump discontinuities.**  For a function \(f\) with a jump discontinuity at
\(c \in (a, b)\), the Chebyshev interpolant converges only as \(O(1/n)\) pointwise
away from \(c\) (Trefethen 2013, Ch. 9).  Near \(c\), oscillations persist
regardless of how many nodes you add -- this is the classical **Gibbs phenomenon**.

**Kinks.**  For a function that is continuous but whose derivative is discontinuous
at \(c\) -- for example \(|x|\) at \(x = 0\) or a call payoff
\(\max(S - K, 0)\) at \(S = K\) -- the situation is better but still algebraic:
convergence is \(O(1/n^2)\) instead of exponential.  This means that increasing
the node count from 15 to 30 only halves the error, rather than reducing it by
orders of magnitude as it would for a smooth function.

!!! example "Interpolating \(|x|\) on \([-1, 1]\)"
    With global Chebyshev interpolation, the error near \(x = 0\) plateaus at
    approximately \(0.01\) regardless of whether you use 10, 20, or 40 nodes.
    This is exactly the algebraic \(O(1/n^2)\) convergence rate -- the spectral
    advantage of Chebyshev is lost.

In quantitative finance this problem is ubiquitous: option payoffs have kinks at
strike prices, barrier levels, and exercise boundaries.  Applying global Chebyshev
interpolation to such functions wastes nodes fighting the Gibbs oscillations instead
of refining the smooth parts of the function.

## Why Piecewise Chebyshev Restores Spectral Convergence

The key to understanding why piecewise interpolation helps lies in the
**Bernstein ellipse theorem** (Trefethen 2013, Ch. 8).

### The Bernstein ellipse

For a function \(f\) analytic in the interior of the **Bernstein ellipse**
\(\mathcal{E}_\rho\) -- the ellipse in the complex plane with foci at \(\pm 1\)
and semi-axis sum \(\rho > 1\) -- the Chebyshev interpolation error on \([-1, 1]\)
with \(n\) nodes satisfies:

\[
\| f - p_n \|_\infty \leq \frac{2 M}{\rho^n (\rho - 1)}
\]

where \(M = \max_{z \in \mathcal{E}_\rho} |f(z)|\).  The rate \(\rho^{-n}\) is
**exponential** in \(n\) -- this is spectral convergence.

### How kinks destroy analyticity

A function with a kink at \(c\) is **not analytic** at \(c\).  The Bernstein
ellipse cannot extend past the singularity: it collapses to the real interval
near \(c\), forcing \(\rho \to 1\) and reducing convergence to algebraic.

### How splitting restores it

By placing a **knot** at \(c\) and interpolating each sub-interval separately,
each piece sees a **smooth** function:

- \(|x|\) restricted to \([-1, 0]\) is just \(-x\), which is entire (analytic
  everywhere in \(\mathbb{C}\)).
- \(|x|\) restricted to \([0, 1]\) is just \(x\), also entire.

Each piece has a large Bernstein ellipse parameter \(\rho_k \gg 1\), and the error
on piece \(k\) with \(n\) Chebyshev nodes is:

$$
E_k \leq \frac{2 M_k}{\rho_k^n (\rho_k - 1)}
$$

where \(M_k = \max_{z \in \mathcal{E}_{\rho_k}} |f(z)|\) on that piece's Bernstein
ellipse.  Because pieces cover disjoint sub-domains, the overall interpolation error
is:

$$
\| f - \mathcal{S}_n f \|_\infty = \max_k \, E_k
$$

This is exponential in \(n\) -- **spectral convergence is restored**.

!!! note "Book reference"
    The Chebyshev Spline technique is described in Section 3.8 of Ruiz & Zeron
    (2021), *Machine Learning for Risk Calculations*, Wiley Finance.  The book
    demonstrates that pricing a European call near the strike kink requires 95
    nodes with global Chebyshev but only 25 nodes (split into two pieces at
    \(K\)) with a Chebyshev spline -- same accuracy, 4x fewer evaluations.

## Quick Start

```python
import math
from pychebyshev import ChebyshevSpline

# European call payoff max(S - K, 0) * exp(-rT) with kink at K = 100
def payoff(x, _):
    return max(x[0] - 100.0, 0.0) * math.exp(-0.05 * x[1])

# Place a knot at S = 100 (the strike), no knots in the T dimension
spline = ChebyshevSpline(
    payoff,
    num_dimensions=2,
    domain=[[80, 120], [0.25, 1.0]],
    n_nodes=[15, 15],
    knots=[[100.0], []],   # knot at S=K, none in T
)
spline.build()

# Evaluate in-the-money
price_itm = spline.eval([110.0, 0.5], [0, 0])

# Evaluate out-of-the-money
price_otm = spline.eval([90.0, 0.5], [0, 0])

# Delta (dV/dS) on the in-the-money side
delta = spline.eval([110.0, 0.5], [1, 0])

print(spline)
```

Output:

```
ChebyshevSpline (2D, built)
  Nodes:       [15, 15] per piece
  Knots:       [[100.0], []]
  Pieces:      2 (2 x 1)
  Build:       0.012s (450 function evals)
  Domain:      [80, 120] x [0.25, 1.0]
  Error est:   1.23e-10
```

With global `ChebyshevApproximation` on the same domain, you would need
approximately 95 nodes to achieve similar accuracy.  The spline uses 2 pieces
of 15 nodes each (450 total evaluations vs. 9,025).

## Choosing Knots

Place knots at the locations where the function is **non-smooth**:

| Singularity type | Example | Knot location |
|-----------------|---------|---------------|
| Payoff kink | European call \(\max(S - K, 0)\) | \(S = K\) |
| Barrier level | Knock-out option | \(S = B\) |
| Exercise boundary | American option (if known) | \(S = S^*(T)\) |
| Absolute value | \(\|x\|\) | \(x = 0\) |

**Guidelines:**

- **Only add knots where the function is non-smooth.**  For smooth functions,
  knots provide no benefit -- you pay extra build cost for no accuracy gain.
- **More knots = more pieces = more build evaluations.**  Each dimension with
  \(k_d\) interior knots creates \(k_d + 1\) sub-intervals.  The total number
  of pieces is the Cartesian product \(\prod_d (k_d + 1)\).
- **Knots must be known a priori.**  `ChebyshevSpline` does not detect
  singularities automatically; you must specify where they are.

## Multiple Knots and Multi-Dimensional Problems

### Multiple knots in one dimension

```python
# Two knots in dimension 0: at x = -1 and x = 1
# This creates 3 pieces in dimension 0
spline = ChebyshevSpline(
    my_func, 1,
    domain=[[-3, 3]],
    n_nodes=[15],
    knots=[[-1.0, 1.0]],
)
```

### Multi-dimensional knots

Each dimension has its own independent list of knots.  The total number of pieces
is the Cartesian product of per-dimension intervals:

```python
# 2D: 2 knots in dim 0, 1 knot in dim 1
# Pieces: (2+1) x (1+1) = 3 x 2 = 6
spline = ChebyshevSpline(
    my_func, 2,
    domain=[[0, 10], [0, 5]],
    n_nodes=[15, 15],
    knots=[[3.0, 7.0], [2.5]],
)
```

### No knots in a dimension

Use an empty list `[]` for dimensions where the function is smooth:

```python
# Knot at S = 100 in price dimension, none in time or vol
spline = ChebyshevSpline(
    bs_func, 3,
    domain=[[80, 120], [0.25, 1.0], [0.15, 0.35]],
    n_nodes=[15, 15, 15],
    knots=[[100.0], [], []],
)
```

### Degenerate case: no knots at all

If every dimension has an empty knot list, the spline has exactly one piece and
behaves identically to a plain `ChebyshevApproximation`.

## Derivatives

Within each piece, derivatives are computed **analytically** via spectral
differentiation matrices -- the same mechanism used by `ChebyshevApproximation`.
No finite differences are needed.

```python
# First derivative w.r.t. dimension 0
dfdx0 = spline.eval([110.0, 0.5], [1, 0])

# Second derivative w.r.t. dimension 0
d2fdx0 = spline.eval([110.0, 0.5], [2, 0])

# Multiple derivatives at once (shared barycentric weights)
results = spline.eval_multi(
    [110.0, 0.5],
    [
        [0, 0],  # function value
        [1, 0],  # dV/dS
        [2, 0],  # d2V/dS2
        [0, 1],  # dV/dT
    ],
)
```

### Derivatives at knot boundaries

Derivatives are **not defined** at knot boundaries.  At a kink, the left and right
polynomial pieces have different derivative values.  Requesting a derivative at a
knot raises `ValueError`:

```python
# This raises ValueError:
# "Derivative w.r.t. dimension 0 is not defined at knot x[0]=100.0"
spline.eval([100.0, 0.5], [1, 0])

# Function values are fine at knots:
spline.eval([100.0, 0.5], [0, 0])  # OK
```

!!! tip "Evaluate near the knot instead"
    If you need a derivative near a knot, evaluate slightly to one side:
    `spline.eval([100.001, 0.5], [1, 0])` gives the right-side derivative.

## Error Estimation

`error_estimate()` returns the **maximum** error estimate across all pieces:

```python
spline.build()
print(f"Error estimate: {spline.error_estimate():.2e}")
```

Since pieces cover disjoint sub-domains, the interpolation error at any point is
bounded by the error of the piece containing that point.  The worst-case error is
therefore the **maximum** over all pieces:

$$
\hat{E} = \max_k \hat{E}_k
$$

This differs from `ChebyshevSlider`, where all slides contribute to every point
and the error estimate is the **sum** over slides.

## When to Use ChebyshevSpline

| Scenario | Recommended class | Why |
|----------|-------------------|-----|
| Smooth function, \(\leq\) 5D | `ChebyshevApproximation` | Full tensor is feasible; spectral convergence without knots |
| Function with kinks at known locations | **`ChebyshevSpline`** | Restores spectral convergence by splitting at singularities |
| High-dimensional (\(5+\)D), general | `ChebyshevTT` | TT-Cross builds from \(O(d \cdot n \cdot r^2)\) evaluations |
| High-dimensional, additively separable | `ChebyshevSlider` | Additive decomposition; cheapest build |

Use `ChebyshevSpline` when:

- The function has **known non-smooth points** (kinks, discontinuities).
- The dimension count is low enough for full tensor grids (typically \(\leq 5\)D).
- You need **analytical derivatives** (not finite differences).
- You want **spectral accuracy** on a function that would otherwise converge slowly.

## Batch Evaluation

For evaluating many points at once, `eval_batch()` vectorises the piece-routing
step and groups points by piece:

```python
import numpy as np

points = np.column_stack([
    np.random.uniform(80, 120, 1000),
    np.random.uniform(0.25, 1.0, 1000),
])
values = spline.eval_batch(points, [0, 0])
```

Points that fall in the same piece are batched together for efficient evaluation.

## Serialization

Save and load splines using the same pattern as other PyChebyshev classes:

```python
# Save
spline.save("payoff_spline.pkl")

# Load (no rebuild needed)
from pychebyshev import ChebyshevSpline
loaded = ChebyshevSpline.load("payoff_spline.pkl")
val = loaded.eval([110.0, 0.5], [0, 0])
```

The original function is **not** saved -- only the numerical data needed for
evaluation.  Assign a new function before calling `build()` again if a rebuild
is desired.

## References

- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.*
  Wiley Finance. Section 3.8.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM. Chapters 8--9.

## Limitations

- **Knots must be known a priori.**  `ChebyshevSpline` does not automatically
  detect singularities.  You must know where the function is non-smooth and place
  knots accordingly.

- **Build cost scales with the number of pieces.**  Total evaluations are
  \(\text{num\_pieces} \times \prod_d n_d\).  Many knots in many dimensions
  creates many pieces: 3 knots in each of 4 dimensions means
  \(4^4 = 256\) pieces.

- **Low-dimensional only.**  Like `ChebyshevApproximation`, each piece requires
  a full tensor grid.  For high-dimensional functions with kinks, a future
  extension could compose `ChebyshevSpline` with `ChebyshevTT` or
  `ChebyshevSlider`.

## API Reference

::: pychebyshev.spline.ChebyshevSpline
    options:
      show_source: false
      docstring_style: numpy
      members_order: source
