# Tensor Train ALS and Algebra

PyChebyshev v0.13 adds Alternating Least Squares (ALS) as a third TT build
method and exposes TT-level algebra primitives: `run_completion`,
`inner_product`, and in-place orthogonalization sweeps (`orth_left`,
`orth_right`).

## Motivation

`ChebyshevTT` already supports two build methods:

- **`method='cross'`** (default): TT-Cross with maxvol pivoting. Fast for
  smooth functions. Rank grows adaptively via SVD.
- **`method='svd'`**: TT-SVD on the explicit tensor grid. Requires the full
  tensor in memory (feasible only in low dimension).

For smooth high-dimensional functions where Cross's pivot quality degrades,
or when you want to refine an existing TT without rebuilding, v0.13 adds:

- **`method='als'`**: rank-adaptive Alternating Least Squares.
- **`run_completion()`**: fixed-rank ALS sweeps on any built TT.
- **`inner_product()`**: exact TT inner product.
- **`orth_left` / `orth_right`**: canonicalization sweeps.

## Mathematical background

ALS alternately optimizes one TT core at a time while holding the others
fixed. At each position $k$, the remaining cores are canonicalized so the
per-core subproblem reduces to a linear least-squares system:

$$
\min_{\mathbf{c}_k} \left\lVert A_k \mathbf{c}_k - \mathbf{b} \right\rVert_2
$$

where $\mathbf{c}_k$ is the flattened core at position $k$, $A_k$ is the
design matrix built from the canonicalized neighbors, and $\mathbf{b}$ is
the vector of target values on the Chebyshev grid. The rank-adaptive
outer loop starts at rank 1 and increments by 1 per iteration until the
sampled error estimate falls below `tolerance` or `max_rank` is reached.

Reference: Holtz, Rohwedder, and Schneider (2012), *The Alternating
Linear Scheme for Tensor Optimization in the Tensor Train Format*.

## Cross vs SVD vs ALS

| You have | Use |
|---|---|
| A cheap function, smooth target | `method='cross'` (default) |
| The full tensor already (e.g. via `from_values`) | `method='svd'` |
| A smooth high-d function, want tight tolerance | `method='als'` |
| An existing build you want to sharpen | `run_completion()` |

## Worked example 1: ALS build on Black-Scholes 5D

```python
from pychebyshev import ChebyshevTT
import numpy as np
from scipy.stats import norm

def black_scholes(x):
    S, K, r, sigma, T = x
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

domain = [(80.0, 120.0), (90.0, 110.0), (0.02, 0.06), (0.15, 0.35), (0.5, 2.0)]
tt = ChebyshevTT(black_scholes, 5, domain, [8, 8, 6, 6, 6],
                 method='als', tolerance=1e-5, max_rank=10)
tt.build(verbose=True)
print(tt.eval([100.0, 100.0, 0.04, 0.25, 1.0]))
```

## Worked example 2: Refining a Cross build with completion

```python
tt = ChebyshevTT(f, 3, domain, n_nodes, method='cross', tolerance=1e-3)
tt.build()
print(tt.error_estimate())       # e.g. 3e-4

tt.run_completion(tolerance=1e-10, max_iter=20)
print(tt.error_estimate())       # e.g. 2e-8 — same cores, sharpened
```

Completion does not re-sample the function — it refines coefficients on
the existing cached grid. If the TT was loaded from disk and the original
function is unavailable, `run_completion()` raises `RuntimeError`.

## Inner product

```python
tt_a.inner_product(tt_b)    # -> float
```

Exact core-by-core contraction. Both TTs must share the same `domain`
and `n_nodes`; mismatches raise `ValueError`. Complexity:
O(d · n · r_a² · r_b²) ops, O(r_a · r_b) extra memory.

## Orthogonalization

```python
tt.orth_left(position=k)     # cores [0..k-1] become left-orthogonal
tt.orth_right(position=k)    # cores [k+1..d-1] become right-orthogonal
```

In-place mutation: the represented tensor is unchanged (QR absorbs R/L
into the neighbor core), only core canonical form shifts. This is the
convention in TT-Toolbox, t3f, and MoCaX — it differs from the
PyChebyshev algebra ops (`+`, `-`, `*`, `/`) which return new objects.

Use orthogonalization when you need numerically stable downstream
operations (e.g., before repeated `inner_product` calls, or when
composing with other TT operations manually).

## Caveats

- `error_estimate()` is a **sampled** estimator (20 random points by
  default), not an exact error. For very tight tolerances you may see
  occasional false convergence; increase `max_rank` and rebuild, or
  follow with `run_completion` at higher `max_iter`.
- `run_completion()` reuses grid samples from the original build; it
  does **not** call `function` again after initial re-sampling into its
  local cache. If the cache coverage is poor (e.g., coming from a Cross
  build that visited few grid points), completion may not converge as
  tightly as a fresh `method='als'` build.

## References

- Holtz, S., Rohwedder, T., & Schneider, R. (2012). The Alternating
  Linear Scheme for Tensor Optimization in the Tensor Train Format.
  *SIAM J. Sci. Comput.* 34(2), A683-A713.
- Oseledets, I. V. (2011). Tensor-Train Decomposition.
  *SIAM J. Sci. Comput.* 33(5), 2295-2317.
- Schollwöck, U. (2011). The density-matrix renormalization group in
  the age of matrix product states. *Ann. Phys.* 326(1), 96-192.
