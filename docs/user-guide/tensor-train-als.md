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

def black_scholes(x, _=None):
    S, K, r, sigma, T = x
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

domain = [(80.0, 120.0), (90.0, 110.0), (0.02, 0.06), (0.15, 0.35), (0.5, 2.0)]
tt = ChebyshevTT(black_scholes, 5, domain, [8, 8, 6, 6, 6],
                 tolerance=1e-5, max_rank=10)
tt.build(verbose=True, method='als')
print(tt.eval([100.0, 100.0, 0.04, 0.25, 1.0]))
```

## Worked example 2: Refining a Cross build with completion

```python
tt = ChebyshevTT(f, 3, domain, n_nodes, tolerance=1e-3)
tt.build(method='cross')  # 'cross' is the default, but shown for clarity
print(tt.error_estimate())       # e.g. 3e-4

tt.run_completion(tolerance=1e-10, max_iter=20)
print(tt.error_estimate())       # e.g. 2e-8 — same cores, sharpened
```

Completion evaluates `function` on the full Chebyshev product grid
(`prod(n_nodes)` points) into a fresh local cache and then runs
fixed-rank ALS sweeps. Rank does not grow. If the TT was loaded from
disk without a callable function, `run_completion()` raises
`RuntimeError` — reassign `tt.function = f` before calling.

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

- `error_estimate()` is a heuristic based on the magnitude of the last
  Chebyshev coefficient slice of each TT core (the TT-format analog of
  the standard Chebyshev truncation-error proxy, Trefethen 2013 Ch. 18).
  It is deterministic and cheap but conservative on highly-aliased
  functions; for tight tolerances, cross-check against a held-out
  evaluation set.
- `run_completion()` materializes the full Chebyshev product grid
  (`prod(n_nodes)` function evaluations in a fresh local cache). For
  TTs originally built with `method='cross'` (which samples O(d·n·r²)
  points), completion's full-grid cost may dwarf the original build
  cost — consider this before calling on very large grids.

## References

- Holtz, S., Rohwedder, T., & Schneider, R. (2012). The Alternating
  Linear Scheme for Tensor Optimization in the Tensor Train Format.
  *SIAM J. Sci. Comput.* 34(2), A683-A713.
- Oseledets, I. V. (2011). Tensor-Train Decomposition.
  *SIAM J. Sci. Comput.* 33(5), 2295-2317.
- Schollwöck, U. (2011). The density-matrix renormalization group in
  the age of matrix product states. *Ann. Phys.* 326(1), 96-192.
