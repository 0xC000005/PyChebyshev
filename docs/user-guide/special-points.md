# Special Points in the Core API

*Available since v0.12.0.*

Declare domain kinks at construction time on `ChebyshevApproximation`.
PyChebyshev routes the build to a piecewise `ChebyshevSpline` under the
hood -- the call site stays simple.

## Motivation: kinks break spectral convergence

Chebyshev polynomial approximation of an analytic function converges
**exponentially** in the number of nodes (Trefethen 2013, Ch. 8):

$$
\|f - p_N\|_\infty \le C\,\rho^{-N}
$$

where $\rho > 1$ depends on the analyticity of $f$. When $f$ has a
**kink** -- a point where $f$ is continuous but its derivative is not
-- this rate collapses to algebraic:

$$
\|f - p_N\|_\infty \sim \mathcal{O}(N^{-k})
$$

for a $C^{k-1}$ function (Trefethen 2013, Ch. 6). The classic example
is $f(x) = |x|$ on $[-1, 1]$: even at $N = 100$ nodes, the supremum
error stays near $10^{-2}$.

The fix is piecewise construction: split the domain at each kink so
every piece is analytic, then interpolate each piece with Chebyshev.
This restores exponential convergence per piece. MoCaX calls these
"special points"; PyChebyshev now does too.

!!! tip "Cross-reference"
    This page is the ergonomic front door to
    [Chebyshev Splines](spline.md). That page explains *why* piecewise
    construction restores spectral convergence (Bernstein ellipses,
    Gibbs phenomenon); this page covers *how* to declare kinks on the
    main `ChebyshevApproximation` API without reaching for the spline
    class directly.

## API

```python
from pychebyshev import ChebyshevApproximation

def f(x, _):
    return abs(x[0])

cheb = ChebyshevApproximation(
    f, 1, [[-1, 1]],
    n_nodes=[[11, 11]],           # per-piece Ns: 11 nodes each side
    special_points=[[0.0]],       # kink at origin
)
cheb.build()
```

- `special_points: list[list[float]] | None = None` -- one inner list
  per dimension. Points must be strictly interior to the domain,
  sorted, and distinct. Empty lists mean "no kinks on this dim".
- `n_nodes` must be **nested** (`list[list[int | None]]`) when any dim
  has special points. For each dim $d$,
  `len(n_nodes[d]) == len(special_points[d]) + 1`.
- When `special_points` declares any kink, `ChebyshevApproximation(...)`
  returns a `ChebyshevSpline` instance (transparent dispatch; precedent:
  `pathlib.Path()`). All spline features -- `eval`, `integrate`,
  `roots`, algebra (`+ - * /`), `extrude`/`slice`, `save`/`load` --
  work identically.

## Worked example 1: `abs(x)`

Without special points, the error plateaus:

```python
cheb = ChebyshevApproximation(lambda x, _: abs(x[0]), 1, [[-1, 1]], [31])
cheb.build()
# max |cheb.eval(x) - |x||  ~  0.03 across [-1, 1]
```

With special points, each linear piece is interpolated exactly:

```python
cheb = ChebyshevApproximation(
    lambda x, _: abs(x[0]), 1, [[-1, 1]],
    n_nodes=[[11, 11]],
    special_points=[[0.0]],
)
cheb.build()
# max error  ~  1e-14
```

## Worked example 2: Barrier option payoff

A down-and-out call has a payoff with kinks at the barrier $B$ and
strike $K$:

$$
\text{payoff}(S) = \begin{cases}
0 & S \le B \\
\max(S - K,\, 0) & S > B
\end{cases}
$$

Two kinks ($B$ and $K$ assuming $B < K$), one smooth piece on each
side:

```python
def barrier_call(x, params):
    S, K, B = x[0], params['K'], params['B']
    if S <= B:
        return 0.0
    return max(S - K, 0.0)

cheb = ChebyshevApproximation(
    lambda x, _: barrier_call(x, {'K': 100.0, 'B': 95.0}),
    1, [[90.0, 150.0]],
    n_nodes=[[7, 7, 15]],
    special_points=[[95.0, 100.0]],  # kinks at barrier and strike
)
cheb.build()
```

Piece 1 is identically zero, piece 2 is zero, piece 3 is linear in $S$
-- all three recover exactly.

## Per-piece auto-N with `error_threshold`

Combine kink declaration with v0.11 error-driven construction:

```python
cheb = ChebyshevApproximation(
    lambda x, _: abs(x[0]) ** 1.5, 1, [[-1, 1]],
    special_points=[[0.0]],
    error_threshold=1e-8,
)
cheb.build()
```

Each piece runs its own doubling loop until its DCT last-coefficient
drops below `error_threshold`. Pieces with sharper curvature land at
larger $N$; smooth pieces terminate early. `error_estimate()` returns
the $\max$ across pieces -- the global bound.

## Relationship to `ChebyshevSpline`

`ChebyshevSpline` remains the direct API for power users and is what
`ChebyshevApproximation(..., special_points=...)` dispatches to:

```python
from pychebyshev import ChebyshevSpline

sp = ChebyshevSpline(f, 1, [[-1, 1]], n_nodes=[[11, 11]], knots=[[0.0]])
```

`special_points` on the main constructor is an alias for `knots` on the
spline -- choose whichever reads better at your call site.

## Current limitation: `nodes()` / `from_values()` + nested Ns

`ChebyshevSpline.nodes()` and `ChebyshevSpline.from_values()` (the
"precompute values, then build" workflow introduced in v0.10) accept
`knots` but **not** per-sub-interval nested `n_nodes` -- they require a
single flat `n_nodes` shared across all pieces in a dim. If you need
per-sub-interval Ns via that workflow, build via the `__init__` path
instead:

```python
# Works: per-sub-interval Ns via __init__
cheb = ChebyshevApproximation(
    f, 1, [[-1, 1]], n_nodes=[[11, 15]], special_points=[[0.0]]
)

# Not yet supported: per-sub-interval Ns via nodes() + from_values()
# Nested n_nodes here raises NotImplementedError with a link back to
# this page.
# grids = ChebyshevSpline.nodes(1, [[-1, 1]], n_nodes=[[11, 15]], knots=[[0.0]])
# ChebyshevSpline.from_values(..., n_nodes=[[11, 15]], ...)
```

Extending the precompute workflow to nested Ns is tracked for a future
release.

## See Also

- [Chebyshev Splines](spline.md) -- the underlying piecewise machinery,
  the Bernstein-ellipse argument for why splitting at kinks restores
  spectral convergence, and the Gibbs phenomenon for true
  discontinuities.
- [Error-Driven Construction](error-driven-construction.md) -- the
  doubling loop reused per-piece when `error_threshold` is set.
- [Mathematical Concepts](concepts.md) -- Bernstein ellipses and the
  exponential convergence rate for analytic functions.

## References

- Trefethen, L. N. (2013). *Approximation Theory and Approximation
  Practice.* SIAM. Chapters 6 and 8.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk
  Calculations.* Wiley Finance. Section 3.8.
- MoCaX Intelligence 4.3.1 manual, `MocaxSpecialPoints` class reference.
