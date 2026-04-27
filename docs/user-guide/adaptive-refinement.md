# Adaptive Refinement (v0.20)

Three new features for handling functions you don't fully understand a priori.

## Auto-knot detection

Discover kinks in your function automatically and build a `ChebyshevSpline`
with knots placed at them:

```python
import math
from pychebyshev import ChebyshevSpline

def f(x, _):
    return abs(x[0] - 0.3) + math.sin(x[0])

spl = ChebyshevSpline.auto_knots(
    f, 1, [[-1, 1]],
    max_knots_per_dim=5,
    n_scan_points=200,
    threshold_factor=5.0,
)
spl.eval([0.3], [0])  # works without you specifying knots=[[0.3]]
```

The algorithm scans `f` per-dimension, computes second differences as a curvature
proxy, and clusters spikes where `|d²f|` exceeds a robust threshold. It's
heuristic — tune `threshold_factor` upward for noisy inputs, downward to catch
subtler kinks.

- `max_knots_per_dim` — max interior knots per dimension (prevents over-segmentation)
- `n_scan_points` — density of grid points to scan each dimension
- `threshold_factor` — multiplier on the mean-based threshold (`mean(|d²f|)`); default 5.0

Returns a `ChebyshevSpline` with auto-discovered knots and equal nodes per piece.

## Sobol indices

Compute first-order and total-order Sobol sensitivity indices directly
from spectral coefficients (no Monte Carlo, no extra function evals):

```python
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(f, 3, domain, n_nodes=20).build()

result = cheb.sobol_indices()
# {
#   "first_order": {0: 0.32, 1: 0.51, 2: 0.17},
#   "total_order": {0: 0.35, 1: 0.55, 2: 0.20},
#   "variance": 1.234,
# }
```

- `first_order[i]` — fraction of total variance driven by dimension `i` alone
- `total_order[i]` — fraction including all interactions involving dimension `i`
- `variance` — global variance of the function over the domain

First-order indices sum to 1 only for purely additive models; the deficit
`1 − Σ S_i` captures interactions. Total-order indices satisfy `Σ S_T_i ≥ 1`,
with the excess attributable to interactions. Available on `ChebyshevApproximation`
and `ChebyshevSpline`. See Saltelli et al. (2010) for the mathematical foundation.

!!! note
    `ChebyshevTT` does not yet support Sobol; this is deferred to v0.21.

## Auto dim-reordering for Tensor Train

`ChebyshevTT` rank depends on dimension order. `with_auto_order` tries multiple
permutations and returns the lowest-rank result:

```python
from pychebyshev import ChebyshevTT

tt = ChebyshevTT.with_auto_order(
    f, 5, domain, n_nodes=10,
    max_rank=10,
    n_trials=5,
    method="greedy_swap",   # or "random"
)
tt.eval([s, k, t, r, sigma])  # original dim order — permutation is transparent
print(tt.dim_order)            # e.g. [4, 0, 2, 1, 3]
```

- `max_rank` — cap on TT rank during build
- `n_trials` — number of permutations to try (cost scales linearly)
- `method` — permutation strategy:
  - `"greedy_swap"` — iteratively swap dimensions to reduce local rank (O(d²))
  - `"random"` — sample random permutations (parallelizable, but may miss optimum)

Returns a `ChebyshevTT` with `dim_order` transparently applied, so user code
stays in original dimension order. Internally, the TT is reordered; all TT
methods respect the permutation transparently (see "Full TT surface" below).

Cost: `n_trials` × normal TT build time. Use only when TT rank is a bottleneck
(typically d ≥ 5) or you're unsure of the best ordering.

### Realigning two auto-ordered TTs with `reorder()`

When two TTs are independently built with `with_auto_order`, they
typically end up with different best dim_orders. Adding them directly
raises `ValueError` (PyChebyshev intentionally avoids hidden rank
blow-up). Use `reorder()` to opt into the alignment cost explicitly:

```python
import numpy as np
from pychebyshev import ChebyshevTT

def f(p, _):
    return np.sin(p[0]) + p[1] ** 2 + np.cos(p[2])

a = ChebyshevTT.with_auto_order(
    f, 3, [[-1, 1]] * 3, [11] * 3, tolerance=1e-8, method="greedy_swap"
)
b = ChebyshevTT.with_auto_order(
    f, 3, [[-1, 1]] * 3, [11] * 3, tolerance=1e-8, method="random", n_trials=5
)

# a + b  →  ValueError if a.dim_order != b.dim_order
b_aligned = b.reorder(a.dim_order)        # explicit alignment via TT-swap
s = a + b_aligned                          # works; result inherits a.dim_order

# Optional: tighten max_rank during the swap
b_tight = b.reorder(a.dim_order, max_rank=8, tolerance=1e-6)
```

`reorder()` performs a sequence of adjacent-axis SVDs (TT-swap) to
realign storage. Each swap may grow the local bond rank; `max_rank=`
and `tolerance=` (defaulting to `self.max_rank` / `self.tolerance`)
control truncation.

### Full TT surface under non-identity `dim_order`

After v0.20.1, every `ChebyshevTT` method accepts non-identity
`dim_order` transparently. Users always work in the original-dim
numbering; PyChebyshev translates at the API boundary.

```python
tt = ChebyshevTT.with_auto_order(
    f, 3, [[-1, 1]] * 3, [11] * 3, tolerance=1e-8
)
# tt.dim_order may be e.g. [2, 0, 1] — storage is permuted internally.

tt.eval([0.3, -0.7, 0.5])              # original-dim order; works
tt.eval_multi([0.3, -0.7, 0.5], [[1, 0, 0]])  # 1st-derivative wrt orig dim 0
tt.slice((1, 0.0))                     # slice original dim 1 at 0
tt.extrude((0, (-2, 2), 5))            # insert new dim at result-position 0
tt.integrate(dims=[2])                 # partial integrate original dim 2
tt.to_dense()                          # axes returned in original-dim order
(-tt).eval([0.1, 0.2, 0.3])            # unary algebra preserves dim_order
2.0 * tt + tt                          # binary algebra succeeds (matching orders)
```

## Sensitivity analysis workflow

Combine auto-knots + Sobol to understand and simplify complex models:

```python
# 1. Auto-discover kinks
spl = ChebyshevSpline.auto_knots(f, 5, domain, max_knots_per_dim=3)

# 2. Compute Sobol indices
sobol = spl.sobol_indices()

# 3. Drop low-sensitivity dims (or use them for conditioning)
important_dims = [i for i, s in sobol["first_order"].items() if s > 0.01]
print(f"Effective dimensionality: {len(important_dims)} / 5")

# 4. Build reduced model if needed, or keep full model with low-rank Slider
```

## Beyond MoCaX

MoCaX has none of these — auto-knot discovery, Sobol indices, and dim
reordering are PyChebyshev-unique adaptive features as of v0.20.

## See Also

- [Chebyshev Splines](spline.md) — manual knot placement and the piecewise interpolation strategy
- [Tensor Train](tensor-train.md) — TT construction and rank reduction
- [Error-Driven Construction](error-driven-construction.md) — auto-N doubling for per-piece refinement

## References

- Saltelli, A., Annoni, P., Azzini, I., et al. (2010). "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index." *Computer Physics Communications*, 181(2), 259–270.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.* SIAM. Chapter 6.
