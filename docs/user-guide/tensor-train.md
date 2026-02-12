# Tensor Train Interpolation

## Introduction

The **Tensor Train (TT)** format enables Chebyshev interpolation of functions with
5 or more dimensions by decomposing the full coefficient tensor into a chain of
small 3D cores. Instead of storing and building the full $n^d$ tensor (infeasible
for $d \geq 6$), TT stores $O(d \cdot n \cdot r^2)$ elements, where $r$ is the
**TT rank** -- a measure of the function's internal complexity.

### When to use which class

| Scenario | Class | Why |
|----------|-------|-----|
| $d \leq 5$ | `ChebyshevApproximation` | Full tensor is feasible; analytical derivatives |
| $d \geq 5$, general function | `ChebyshevTT` | TT-Cross builds from $O(d \cdot n \cdot r^2)$ evaluations |
| $d \gg 5$, separable function | `ChebyshevSlider` | Additive decomposition; cheapest build, but loses cross-group coupling |

`ChebyshevTT` fills the gap between the full tensor approach (limited to low
dimensions) and the sliding technique (limited to separable functions). It handles
general functions -- including those with strong multiplicative coupling like
Black-Scholes -- at a fraction of the full tensor cost.

## Quick Start

```python
import math
from scipy.stats import norm
from pychebyshev import ChebyshevTT

def black_scholes_5d(x, _):
    """European call price: V(S, K, T, sigma, r)."""
    S, K, T, sigma, r = x
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

tt = ChebyshevTT(
    black_scholes_5d, 5,
    domain=[[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
    n_nodes=[11, 11, 11, 11, 11],
    max_rank=15,
)
tt.build()
price = tt.eval([100, 100, 1.0, 0.25, 0.05])
print(f"TT price: {price:.6f}")
```

After building, `print(tt)` shows a summary including TT ranks and compression ratio:

```
ChebyshevTT (5D, built)
  Nodes:       [11, 11, 11, 11, 11]
  TT ranks:    [1, 11, 11, 11, 7, 1]
  Compression: 161,051 -> 3,707 elements (43.4x)
  Build:       0.351s (7,419 function evals)
  Domain:      [80, 120] x [90, 110] x [0.25, 1.0] x [0.15, 0.35] x [0.01, 0.08]
  Error est:   1.23e-06
```

## How It Works

### TT format

A $d$-dimensional tensor $\mathcal{A}$ with indices $j_1, \ldots, j_d$ is
represented as a chain of 3D cores:

$$\mathcal{A}(j_1, \ldots, j_d) = G_1(j_1) \cdot G_2(j_2) \cdots G_d(j_d)$$

Each core $G_k$ is a 3D array of shape $(r_{k-1}, n_k, r_k)$, and for a fixed
index $j_k$ the slice $G_k(j_k)$ is an $r_{k-1} \times r_k$ matrix. The product
of these matrices yields the tensor element. The boundary ranks are $r_0 = r_d = 1$,
so the result is a scalar.

**Storage**: The full tensor has $\prod_k n_k$ elements. The TT format stores
$\sum_k r_{k-1} \cdot n_k \cdot r_k$ elements -- exponentially smaller when ranks
are moderate.

### TT-Cross build algorithm

The key advantage of `ChebyshevTT` over full tensor interpolation is how it is
**built**. Instead of evaluating the function at every node combination ($n^d$
points), the TT-Cross algorithm (Oseledets & Tyrtyshnikov 2010) evaluates at
strategically selected grid points. For 5D with $n = 11$ and `max_rank=15`:

- Full tensor: **161,051** evaluations
- TT-Cross: **~7,400** unique evaluations (**21.7x fewer**)

The algorithm performs alternating sweeps:

**1. Initialize.** Randomly select multi-index sets $J_{\text{right}}[k]$ for each
dimension, each containing $r_k$ multi-indices into the "right" dimensions $k+1, \ldots, d-1$.

**2. Left-to-right sweep.** For each dimension $k = 0, \ldots, d-2$:

- **Build a cross matrix** $C$ of shape $(r_{k-1} \cdot n_k, r_k)$.
  Each row corresponds to a (left multi-index, node index) pair, and each
  column to a right multi-index. Each entry is a function evaluation at
  the combined grid point. This is the most expensive step -- it evaluates
  the function at $r_{k-1} \cdot n_k \cdot r_k$ grid points (with caching,
  many of these are free on subsequent sweeps).

- **SVD for rank selection.** Compute $C = U \Sigma V^T$ and determine the
  effective rank by counting singular values above a tight threshold
  ($10^{-12} \cdot \sigma_{\max}$). This is further capped by the per-mode
  rank bound (see [Optimizations](#optimizations) below).

- **Maxvol pivot selection.** Apply the maxvol algorithm to the left
  singular vectors $U$ to select the $r$ rows whose submatrix has
  approximately maximal determinant. These pivots identify the most
  "informative" (left, node) index pairs.

- **Form the TT core** via cross interpolation:
  $\hat{C} = U \cdot U[\text{pivots}]^{-1}$, then reshape to
  $(r_{k-1}, n_k, r_k)$. The identity $\hat{C}[\text{pivots}] = I$
  ensures exact interpolation at the selected cross points.

- **Update left index set** for dimension $k+1$ by expanding each
  pivot into its constituent (left multi-index, node index) pair.

**3. Convergence check.** After completing the L$\to$R sweep, evaluate the
TT at 20 random grid points and compare against exact function values.
If the relative error is below `tolerance`, stop immediately (skipping the
R$\to$L sweep).

**4. Right-to-left sweep.** Analogous to L$\to$R but processes dimensions
from $d-1$ down to 1. The transposed cross matrix $C^T$ is used, and
the maxvol pivots update the right index sets.

**5. Convergence and plateau check.** After the R$\to$L sweep, evaluate error
again. The algorithm tracks the best cores seen across all checks (see
[Best-cores tracking](#best-cores-tracking)) and stops if no improvement
has been observed for 3 consecutive checks.

**6. Repeat** sweeps until converged, stale, or `max_sweeps` is reached.

### Maxvol algorithm

The **maximum-volume** (maxvol) algorithm is a key subroutine within TT-Cross.
Given a tall matrix $A$ of shape $(m, r)$ with $m \geq r$, it finds $r$ row
indices such that the submatrix $A[\text{idx}]$ has approximately maximal
$|\det|$.

**Why it matters:** In TT-Cross, maxvol selects which cross points to use for
the next dimension. Points with maximal volume correspond to the most linearly
independent function samples, giving the best cross interpolation.

The implementation uses two phases:

1. **Initialization via column-pivoted QR.** Compute
   `Q, R, piv = qr(A.T, pivoting=True)`. The first $r$ pivot indices
   identify the $r$ most linearly independent rows of $A$ -- a good
   starting point.

2. **Iterative refinement.** Compute the coefficient matrix
   $B = A \cdot A[\text{idx}]^{-1}$ (shape $m \times r$). While
   $\max|B_{ij}| > 1.05$:
    - Find the entry $(i, j)$ with largest $|B_{ij}|$.
    - Swap: replace row $j$ in the index set with row $i$.
    - Rank-1 update of $B$ (avoids re-inverting the full matrix).

Each swap strictly increases $|\det(A[\text{idx}])|$, and the algorithm
converges in $O(r^2)$ iterations in practice.

### Coefficient conversion

After TT-Cross produces cores containing function values at Chebyshev nodes, a
**DCT-II** is applied along the middle axis of each core to convert from function
values to Chebyshev expansion coefficients:

```python
from scipy.fft import dct

# core shape: (r_{k-1}, n_k, r_k)
coeff_core = dct(core[:, ::-1, :], type=2, axis=1) / n_k
coeff_core[:, 0, :] /= 2
```

The reversal (`::-1`) accounts for the ordering of Chebyshev Type I nodes.
This is the same transform used by `ChebyshevApproximation` (see
[Error Estimation](error-estimation.md) for details on the DCT), extended to
3D cores.

### Evaluation via TT inner product

Given the pre-computed coefficient cores $\mathcal{C}$ and a query point $p$,
evaluation computes the inner product:

$$I(p) = \langle \mathcal{C},\, \mathcal{T}_p \rangle$$

where $\mathcal{T}_p$ is a rank-1 tensor whose entries are Chebyshev polynomial
values $[T_0(\tilde{x}_k), T_1(\tilde{x}_k), \ldots, T_{n_k-1}(\tilde{x}_k)]$
at the scaled query coordinate $\tilde{x}_k$ in each dimension $k$.

In practice, this inner product is computed as a chain of matrix contractions:

```python
result = np.ones((1, 1))
for k in range(num_dims):
    q = chebyshev_polynomials(scaled_x[k], n_nodes[k])   # (n_k,)
    v = np.einsum('j,ijk->ik', q, coeff_cores[k])        # (r_{k-1}, r_k)
    result = result @ v
value = result[0, 0]
```

Each step contracts one dimension, reducing the chain until a scalar remains.
The cost is $O(d \cdot n \cdot r^2)$ per point.

## Optimizations

PyChebyshev's TT-Cross implementation includes several optimizations that
reduce function evaluations by **10--20x** compared to a naive implementation.

### Eval caching

Function evaluations are cached in a dictionary keyed by the grid index tuple.
When the same grid point is needed again -- whether during the R$\to$L sweep
of the same iteration, or in subsequent sweeps -- the cached value is returned
instantly. This is the single largest optimization: for 5D Black-Scholes with
`max_rank=15`, caching reduces evaluations from ~85,000 to ~7,400.

### Per-mode rank caps

At bond $k$, the theoretical maximum TT rank is
$\min(\prod_{i<k} n_i,\; \prod_{i \geq k} n_i)$. For example, with 5 dimensions
of 11 nodes each, the first bond can have rank at most $\min(11, 14641) = 11$.
PyChebyshev automatically caps the rank at each bond to this theoretical limit,
preventing the algorithm from attempting ranks that are mathematically impossible.

### SVD-based adaptive rank

Instead of always using `max_rank` columns from a QR decomposition, the cross
matrix is decomposed via SVD. Singular values below $10^{-12} \cdot \sigma_{\max}$
are dropped, so dimensions where the function has low effective rank naturally
get smaller cores. For the 5D Black-Scholes example, this produces adaptive ranks
`[1, 11, 11, 11, 7, 1]` instead of a uniform `[1, 11, 11, 11, 11, 1]` -- the
last bond only needs rank 7 because the interest rate dimension has simpler
structure.

### Half-sweep convergence

Error is checked after the L$\to$R half-sweep. If the TT already reproduces
the function to within `tolerance` at random test points, the R$\to$L sweep
is skipped entirely. For separable functions like $\sin(x) + \sin(y) + \sin(z)$,
this means convergence in a single L$\to$R pass with only 159 evaluations.

### Best-cores tracking

TT-Cross error can oscillate between L$\to$R and R$\to$L sweeps -- a good
L$\to$R result may be partially degraded by R$\to$L rebalancing. PyChebyshev
keeps a copy of the best cores (lowest error) seen across all convergence
checks, and returns those when the algorithm stops. This prevents the final
result from being worse than an intermediate result.

The algorithm also counts "stale checks" -- consecutive convergence checks
that fail to improve the best error by at least 10%. After 3 stale checks
(and best error below $10^{-3}$), the algorithm stops early, returning the
best cores found.

### TT-SVD (validation build)

For moderate dimensions ($d \leq 6$), the `method='svd'` option builds the
full tensor and decomposes it via sequential truncated SVD. This produces
**optimal** TT ranks (up to the SVD truncation tolerance) and is useful for:

- Validating TT-Cross accuracy against the best possible TT decomposition.
- Confirming that the function's intrinsic TT rank structure matches
  expectations.
- Problems where the function is cheap to evaluate and a full tensor build
  is acceptable.

```python
# TT-SVD — for validation or moderate dimensions
tt.build(method="svd")
```

In TT-SVD, singular values below `tolerance * sigma_max` are discarded at each
unfolding. For a separable function like $\sin(x) + \sin(y) + \sin(z)$, TT-SVD
finds exact rank `[1, 2, 2, 1]`.

## Controlling Accuracy

### `max_rank`

The TT rank controls how much "coupling" between dimensions the approximation can
capture. Higher rank means more accurate, but more expensive to build and store.

| `max_rank` | Typical use |
|------------|-------------|
| 5--10 | Smooth, nearly separable functions |
| 10--15 | General smooth functions (e.g., Black-Scholes) |
| 20--30 | Functions with strong nonlinear coupling |

!!! tip "Start low, increase if needed"
    Begin with `max_rank=10` and check `error_estimate()`. If the error is too
    large, increase to 15 or 20. Smooth functions like Black-Scholes options
    typically converge well with ranks of 5--15.

The table below shows how `max_rank` affects the 5D Black-Scholes approximation
(11 nodes per dimension, `seed=42`):

| `max_rank` | Unique evals | Max price error | TT ranks |
|------------|-------------|-----------------|----------|
| 8 | ~4,500 | 0.58% | [1, 8, 8, 8, 6, 1] |
| 10 | ~7,500 | 0.09% | [1, 10, 10, 10, 7, 1] |
| 15 | ~7,400 | 0.19% | [1, 11, 11, 11, 7, 1] |

Note that `max_rank=15` and `max_rank=10` use a similar number of evaluations
because per-mode rank caps limit interior ranks to at most $n = 11$.

### `tolerance`

The convergence tolerance for TT-Cross. The algorithm checks relative error
at random grid points after each half-sweep. When the error drops below this
threshold, iteration stops. The default `1e-6` is appropriate for most problems.

!!! note "Tolerance does not control SVD truncation"
    The `tolerance` parameter only controls when the sweep loop stops. Rank
    selection within each mode uses a fixed threshold of $10^{-12}$ to drop
    only numerically zero singular values. Rank is primarily controlled by
    `max_rank` and the per-mode caps.

### `max_sweeps`

The maximum number of alternating left-right sweeps. The default of 10 is
sufficient for most well-behaved functions. In practice, the best-cores tracking
and stale-check stopping mean the algorithm often finishes in 2--3 sweeps.

### Build method

The `build()` method accepts a `method` parameter:

- `method='cross'` (default): TT-Cross algorithm -- evaluates only
  $O(d \cdot n \cdot r^2)$ points. Use for high-dimensional problems.
- `method='svd'`: Builds the full tensor and decomposes via truncated SVD.
  Only feasible for moderate dimensions ($d \leq 6$), but produces optimal
  ranks and is useful for validation.

```python
# TT-Cross (default) — efficient for high dimensions
tt.build(method="cross", seed=42)

# TT-SVD — for validation or moderate dimensions
tt.build(method="svd")
```

### Error estimation

Like `ChebyshevApproximation`, the TT class supports ex ante error estimation from
its coefficient cores:

```python
tt.build()
print(f"Estimated error: {tt.error_estimate():.2e}")
```

!!! warning "Approximate for TT"
    The TT error estimate is based on the trailing Chebyshev coefficients within
    each core. It captures **per-core truncation error** but does not account for
    rank truncation error. The true error may be somewhat larger than the estimate,
    especially at low ranks.

## Derivatives via Finite Differences

The TT format does not support analytical derivatives (the spectral differentiation
matrix approach used by `ChebyshevApproximation` requires the full tensor). Instead,
`ChebyshevTT` computes derivatives via **central finite differences**:

$$\frac{\partial f}{\partial x_k} \approx \frac{f(x + h\, e_k) - f(x - h\, e_k)}{2h}$$

The `eval_multi()` method computes price and Greeks in a single call:

```python
point = [100, 100, 1.0, 0.25, 0.05]

results = tt.eval_multi(point, [
    [0, 0, 0, 0, 0],  # price
    [1, 0, 0, 0, 0],  # Delta (dV/dS)
    [2, 0, 0, 0, 0],  # Gamma (d²V/dS²)
    [0, 0, 0, 1, 0],  # Vega  (dV/dsigma)
    [0, 0, 0, 0, 1],  # Rho   (dV/dr)
])
price, delta, gamma, vega, rho = results
```

The step size $h$ is chosen automatically as $10^{-4}$ times the domain width
in each dimension. Points near domain boundaries are nudged inward to ensure
the FD stencil stays inside the domain.

!!! note "FD accuracy"
    For first-order derivatives, accuracy is typically within a few hundredths
    of a percent of the analytical value. Second-order derivatives (e.g., Gamma)
    are less precise due to the inherent amplification of noise in central
    second differences.

## Batch Evaluation

For evaluating many points at once -- common in portfolio pricing -- use
`eval_batch()`:

```python
import numpy as np

# 1000 random points in the domain
points = np.column_stack([
    np.random.uniform(80, 120, 1000),    # S
    np.random.uniform(90, 110, 1000),    # K
    np.random.uniform(0.25, 1.0, 1000),  # T
    np.random.uniform(0.15, 0.35, 1000), # sigma
    np.random.uniform(0.01, 0.08, 1000), # r
])

prices = tt.eval_batch(points)  # (1000,) array
```

`eval_batch()` vectorizes the TT inner product over all points simultaneously
using `np.einsum` for batched matrix contractions, which is significantly
faster than calling `eval()` in a loop. Typical speedup is 15--20x.

## Performance Comparison

### PyChebyshev vs MoCaX Extend (Tensor Train)

Both PyChebyshev `ChebyshevTT` and MoCaX `MocaxExtend` build Chebyshev
interpolants in Tensor Train format. They differ in **how** the TT is
constructed:

| | PyChebyshev `ChebyshevTT` | MoCaX `MocaxExtend` |
|---|---|---|
| **Build algorithm** | TT-Cross (maxvol pivoting) | Rank-adaptive ALS on random subgrid |
| **Point selection** | Adaptive via cross interpolation | Random subset of full Chebyshev grid |
| **Eval implementation** | Vectorized NumPy (einsum, BLAS) | Python loops + deep copies |
| **Coefficient cores** | Pre-computed via DCT-II | Recomputed on every eval call |

The following benchmarks are from 5D Black-Scholes $V(S, K, T, \sigma, r)$ with
11 Chebyshev nodes per dimension, dividend yield $q = 0.02$.

#### Build comparison

| Metric | PyChebyshev | MoCaX |
|--------|-------------|-------|
| Build time | 0.35s | 5.73s |
| Function evaluations | 7,419 | 8,000 |

PyChebyshev uses slightly fewer evaluations (TT-Cross adaptively selects the
most informative points) and builds **16x faster** (no Python-level ALS
optimization loop).

#### Accuracy (50 random test points)

| Metric | PyChebyshev | MoCaX |
|--------|-------------|-------|
| Mean price error | 0.002% | 0.093% |
| Max price error | 0.014% | 0.712% |
| Median price error | 0.001% | 0.045% |

PyChebyshev is **40--50x more accurate** at comparable evaluation budgets.

#### Evaluation speed (1000 random points)

| Method | PyChebyshev | MoCaX |
|--------|-------------|-------|
| Single eval | 0.065 ms/query | -- |
| Batch eval | 0.004 ms/query | 0.246 ms/query |

PyChebyshev batch evaluation is **58x faster** than MoCaX.

#### Greeks accuracy (10 scenarios vs analytical)

| Greek | PyChebyshev avg error | MoCaX avg error |
|-------|-----------------------|-----------------|
| Delta | 0.029% | 0.379% |
| Gamma | 0.019% | 1.604% |

Both use central finite differences. PyChebyshev's advantage comes from its
more accurate underlying interpolant.

### Reproducing the comparison

The comparison script `compare_tensor_train.py` in the repository root runs
all the benchmarks above. It requires the MoCaX C++ library (not publicly
available); PyChebyshev results are shown regardless.

```bash
# Run the comparison (MoCaX portion is skipped if unavailable)
uv run --with tqdm --with blackscholes python compare_tensor_train.py
```

If you have MoCaX installed, ensure the `mocaxextend_lib/` directory with
`shared_libs/` (containing `libtensorvals.so` and `libhommat.so`) is in the
repository root.

!!! note "MoCaX results are nondeterministic"
    MoCaX uses a random subgrid for its ALS optimization, so its accuracy
    varies between runs. The numbers above are representative of typical runs.

## Comparison with Other PyChebyshev Methods

| | `ChebyshevApproximation` | `ChebyshevTT` | `ChebyshevSlider` |
|---|---|---|---|
| **Dimensions** | $\leq 5$ (practical) | $5+$ | Any (with caveats) |
| **Build cost** | $O(n^d)$ evaluations | $O(d \cdot n \cdot r^2)$ evaluations | $\sum_i O(n_{g_i}^{|G_i|})$ evaluations |
| **Eval cost** | $O(d \cdot n)$ via BLAS GEMV | $O(d \cdot n \cdot r^2)$ via einsum | $O(k \cdot d_g \cdot n)$ per slide |
| **Derivatives** | Analytical (spectral) | Finite differences | Analytical (per-slide) |
| **Accuracy** | Spectral convergence | Rank-dependent | Depends on separability |
| **Best for** | Low-$d$, high-accuracy Greeks | Moderate-$d$, general functions | High-$d$, separable functions |

## Serialization

`ChebyshevTT` supports saving and loading via pickle, following the same pattern
as the other PyChebyshev classes:

```python
# Save after building
tt.save("bs_5d_tt.pkl")

# Load later (no rebuild needed)
from pychebyshev import ChebyshevTT
tt_loaded = ChebyshevTT.load("bs_5d_tt.pkl")
price = tt_loaded.eval([100, 100, 1.0, 0.25, 0.05])
```

!!! note
    The original function is **not** saved -- only the pre-computed coefficient
    cores and metadata. After loading, `eval()`, `eval_batch()`, and `eval_multi()`
    work normally, but `build()` cannot be called again without re-supplying the
    function.

## Limitations

- **No analytical derivatives.** The TT format does not support spectral
  differentiation matrices. Derivatives are computed via finite differences, which
  are less accurate (especially for second-order derivatives like Gamma).
- **Error estimates are approximate.** The `error_estimate()` method captures
  per-core coefficient truncation but not rank truncation error. Always validate
  against known solutions when possible.
- **Convergence depends on function structure.** TT-Cross works best for functions
  with low-rank structure (smooth, with moderate coupling between variables). Not
  all functions have good low-rank TT approximations -- highly oscillatory or
  discontinuous functions may require very high ranks.
- **Build cost grows with rank.** While $O(d \cdot n \cdot r^2)$ is much better
  than $O(n^d)$, a large `max_rank` (say, 50+) can still be expensive for costly
  functions.

## Mathematical Reference

The TT interpolation approach implemented here follows:

- **Ruiz & Zeron (2021)**, *Machine Learning for Risk Calculations*, Wiley Finance,
  Chapter 6: Tensor Train decomposition for Chebyshev interpolation.
- **Oseledets & Tyrtyshnikov (2010)**, "TT-Cross approximation for multidimensional
  arrays" -- the cross approximation algorithm used to build TT cores from function
  evaluations.
- **Goreinov, Tyrtyshnikov & Zamarashkin (1997)** -- maxvol algorithm for pivot
  selection within TT-Cross.

## API Reference

::: pychebyshev.tensor_train.ChebyshevTT
    options:
      show_source: false
      docstring_style: numpy
      members_order: source
