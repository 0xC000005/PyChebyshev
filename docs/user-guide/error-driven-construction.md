# Error-Driven Construction

*Available since v0.11.0.*

## Motivation

For smooth (analytic) functions, Chebyshev interpolation converges
*exponentially* in the number of nodes. For a function analytic in a
Bernstein ellipse with parameter $\rho > 1$ (see
[Mathematical Concepts](concepts.md)), the approximation error decays
as $\mathcal{O}(\rho^{-N})$, where $\rho$ is determined by the region
of analyticity. In practice this means the "right" number of nodes is
a property of the function, not something the user should have to
guess.

PyChebyshev v0.11 lets you specify a **target error** instead of a
node count -- the library picks $N$ for you by iteratively refining
the grid until the built-in error estimate meets the threshold.

!!! tip "Cross-reference"
    This page builds directly on
    [Error Estimation](error-estimation.md). The doubling loop reuses
    the same DCT-II last-coefficient bound documented there; it is
    worth reading that page first if you want to understand *why* the
    loop converges.

## Construction Modes

`ChebyshevApproximation` (and equivalently `ChebyshevSpline`, applied
per piece) supports three construction modes:

```python
from pychebyshev import ChebyshevApproximation

# Mode 1 -- explicit Ns (unchanged from previous versions)
cheb = ChebyshevApproximation(f, 3, domain, n_nodes=[11, 11, 11])

# Mode 2 -- auto-pick N for every dim
cheb = ChebyshevApproximation(f, 3, domain, error_threshold=1e-6)

# Mode 3 -- semi-variable: fix some dims, auto-refine others
cheb = ChebyshevApproximation(
    f, 3, domain,
    n_nodes=[None, 15, 15],       # None means "auto this dim"
    error_threshold=1e-6,
)

# Optional cap on auto-N growth (default 64)
cheb = ChebyshevApproximation(
    f, 3, domain,
    error_threshold=1e-6, max_n=128,
)

cheb.build()

# Query post-build
cheb.get_error_threshold()   # -> float | None (the target)
cheb.error_estimate()        # -> float        (the achieved estimate)
cheb.n_nodes                 # -> list[int]    (resolved)
```

| Mode | `n_nodes` | `error_threshold` | Behaviour                         |
|------|-----------|-------------------|-----------------------------------|
| 1    | all ints  | omitted           | Fixed grid (classical build)      |
| 2    | omitted   | float             | Doubling loop over all dims       |
| 3    | mix of int and `None` | float | Doubling loop over `None` dims only |

If `n_nodes` is omitted *and* `error_threshold` is unset, construction
raises `ValueError`. Similarly, `None` entries in `n_nodes` without a
threshold raise `ValueError`.

## Algorithm

The doubling loop is simple:

1. Start with $n_d = 3$ for every *auto* dim (dims passed as `None`).
   Fixed dims keep their user-supplied value.
2. Build the interpolant on the current grid.
3. Compute `error_estimate()`. If it meets the threshold, **stop**.
4. Otherwise, pick the auto dim with the largest per-dim
   last-coefficient magnitude -- the worst contributor to the sum --
   and double its $n_d$, capped at `max_n`.
5. If every auto dim is already at `max_n`, emit a
   `RuntimeWarning` and stop with whatever grid is built.

The error estimate reused here is the same DCT-II last-coefficient
bound documented in [Error Estimation](error-estimation.md) (Ruiz &
Zeron 2021, Section 3.4):

$$
\hat{E} = \sum_{d=1}^{D}
    \max_{\text{slices along } d} |c_{n_d - 1}|.
$$

Decomposing this sum per dim lets the loop refine the dim whose
Chebyshev series is truncating most aggressively, rather than
doubling every dim indiscriminately (which would waste memory on
dimensions that have already converged).

!!! note "Why start at N = 3?"

    Three nodes is the minimum grid that defines a non-trivial
    Chebyshev expansion (and therefore a well-defined last
    coefficient). Starting small keeps the first few iterations
    cheap; the exponential convergence then does the heavy lifting
    once the function's smoothness kicks in.

## One-Dimensional Capacity Estimate

For planning -- e.g., deciding whether a 5D function will fit in
memory before committing to the full build -- there is a 1-D helper
classmethod:

```python
n = ChebyshevApproximation.get_optimal_n1(
    lambda x, _: f_1d(x[0]),
    domain_1d=[0.0, 1.0],
    error_threshold=1e-8,
    max_n=64,
)  # -> int
```

This runs the doubling loop on a 1-D slice of a function and returns
the resolved $N$. Use it as an order-of-magnitude estimator for
per-dimension resolution before you commit to the full tensor build.

## Usage Example -- Black-Scholes in 5D

```python
import math
from scipy.stats import norm
from pychebyshev import ChebyshevApproximation

def bs_call(point, _):
    """European call price: V(S, K, T, sigma, r)."""
    S, K, T, sigma, r = point
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

cheb = ChebyshevApproximation(
    bs_call, 5,
    [[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
    error_threshold=1e-8,
)
cheb.build()

print(cheb.n_nodes)           # e.g. [11, 9, 15, 11, 7]
print(cheb.error_estimate())  # <= 1e-8
```

Each dimension converges at its own rate: maturity $T$ typically needs
more nodes than the interest rate $r$, which is nearly linear in the
price. The doubling loop discovers this automatically.

## Rebuilding with a Tighter Threshold

The original `n_nodes` argument (including `None` sentinels) is
preserved, so you can call `build()` again after tightening
`error_threshold` and the doubling loop will re-run on the auto dims:

```python
cheb.error_threshold = 1e-12
cheb.build()                 # re-runs auto-N on previously-None dims
```

In auto-N mode, `n_evaluations` and `build_time` are **accumulated
across doubling iterations** and reflect total work, not just the
final iteration.

## Edge Cases and Limits

- **`max_n` exhausted**: the doubling loop emits `RuntimeWarning`
  and returns with whatever grid was built. The object is still
  usable -- check `error_estimate()` against your target to decide
  whether to rebuild with a larger cap.
- **Non-smooth functions**: doubling converges slowly (or stalls at
  `max_n`) for functions with kinks or discontinuities. Use
  [`ChebyshevSpline`](spline.md) with knots placed at the kinks to
  restore spectral convergence on each piece. `ChebyshevSpline`
  also supports `error_threshold` -- applied independently per
  piece -- so the piece containing a kink refines until each side
  is smooth.
- **Pathological test functions**: functions whose Chebyshev series
  exhibits vanishing coefficients at small $N$ (e.g., an odd
  function on a symmetric domain whose last even coefficient
  happens to alias to zero at $N=3$) can fool the loop into
  stopping prematurely. This does not arise in real-world pricing
  problems; when in doubt, pass an explicit floor via the
  semi-variable mode.

## API Reference

- [`ChebyshevApproximation`](../api/reference.md#pychebyshev.ChebyshevApproximation)
  -- the `error_threshold` and `max_n` kwargs on the constructor.
- [`ChebyshevApproximation.get_error_threshold()`](../api/reference.md#pychebyshev.ChebyshevApproximation.get_error_threshold)
  -- query the stored target.
- [`ChebyshevApproximation.get_optimal_n1()`](../api/reference.md#pychebyshev.ChebyshevApproximation.get_optimal_n1)
  -- 1-D capacity estimator classmethod.
- [`ChebyshevSpline`](../api/reference.md#pychebyshev.ChebyshevSpline)
  -- same kwargs, applied per piece.

## See Also

- [Error Estimation](error-estimation.md) -- the DCT-II coefficient
  bound that powers the doubling loop.
- [Mathematical Concepts](concepts.md) -- Bernstein ellipses and
  why Chebyshev interpolation converges exponentially for analytic
  functions.
- [Chebyshev Splines](spline.md) -- recover spectral convergence
  on non-smooth functions by placing knots at singularities.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange
  Interpolation." *SIAM Review* 46(3):501--517.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk
  Calculations.* Wiley Finance. Section 3.4.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation
  Practice.* SIAM. Chapters 7--8.
