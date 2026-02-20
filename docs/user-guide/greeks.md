# Computing Greeks

PyChebyshev computes option Greeks (and any partial derivatives) analytically
using spectral differentiation matrices — no finite differences needed.

## Derivative Specification

Derivatives are specified as a list of integers, one per dimension. Each integer
is the derivative order with respect to that dimension.

For a 5D function $V(S, K, T, \sigma, r)$:

| Greek | `derivative_order` | Mathematical |
|-------|-------------------|--------------|
| Price | `[0, 0, 0, 0, 0]` | $V$ |
| Delta | `[1, 0, 0, 0, 0]` | $\partial V / \partial S$ |
| Gamma | `[2, 0, 0, 0, 0]` | $\partial^2 V / \partial S^2$ |
| Vega | `[0, 0, 0, 1, 0]` | $\partial V / \partial \sigma$ |
| Rho | `[0, 0, 0, 0, 1]` | $\partial V / \partial r$ |

## Example: Black-Scholes Greeks

```python
from pychebyshev import ChebyshevApproximation

def black_scholes_call(x, _):
    S, K, T, sigma, r = x
    # ... your pricing function here
    return price

cheb = ChebyshevApproximation(
    black_scholes_call, 5,
    domain=[[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
    n_nodes=[11, 11, 11, 11, 11],
)
cheb.build()

point = [100, 100, 1.0, 0.25, 0.05]

# All Greeks at once (most efficient)
price, delta, gamma, vega, rho = cheb.vectorized_eval_multi(point, [
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])
```

## How It Works

1. **At build time**: Pre-compute differentiation matrix $D$ from node positions
2. **At query time**: Apply $D$ to the function value tensor before barycentric interpolation
3. For second derivatives: apply $D$ twice ($D^2 \mathbf{f}$)
4. Interpolate the resulting derivative values to the query point

This provides **exact derivatives of the interpolating polynomial**. Because the
differentiation matrix computes derivatives of the degree-$n$ polynomial $p(x)$
exactly (within machine precision), and $p(x)$ converges spectrally to $f(x)$,
the derivative $p'(x)$ also converges spectrally to $f'(x)$
(Trefethen 2013, Ch. 11).

See [Berrut & Trefethen (2004)](https://people.maths.ox.ac.uk/trefethen/barycentric.pdf)
for the derivation of the differentiation matrix formulas. For the full theory of
spectral differentiation, see Trefethen (2013), *Approximation Theory and Approximation
Practice*, SIAM, Chapter 11.

!!! note "Tensor Train derivatives"
    `ChebyshevTT` uses **finite differences** instead of analytical derivatives,
    because the spectral differentiation matrix requires the full tensor (which TT
    avoids storing). FD derivatives are still accurate to within a few hundredths of
    a percent for first derivatives. See
    [Tensor Train: Derivatives](tensor-train.md#derivatives-via-finite-differences).

## Accuracy

With 11 nodes per dimension on a 5D Black-Scholes test:

| Greek | Max Error |
|-------|-----------|
| Delta | < 0.01% |
| Gamma | < 0.01% |
| Vega | ~1.98% |
| Rho | < 0.01% |

Vega has slightly higher error because volatility sensitivity involves a product of
multiple terms, but remains well within practical tolerance.
