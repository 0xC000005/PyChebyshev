# Error Estimation

## Introduction

After building a Chebyshev interpolant, you often want to know how accurate it is
without comparing against the true function at thousands of test points.
`error_estimate()` provides an **ex ante** estimate of the interpolation error using
only the Chebyshev coefficients already computed during the build step.

This is useful for:

- **Validating node counts** -- check whether your grid is fine enough before deploying.
- **Adaptive refinement** -- increase nodes in dimensions where the error is large.
- **Confidence reporting** -- attach an approximate error magnitude to interpolated values.

## Quick Start

```python
from pychebyshev import ChebyshevApproximation
import math

def f(x, _):
    return math.sin(x[0]) * math.cos(x[1])

cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [15, 15])
cheb.build(verbose=False)
print(f"Error estimate: {cheb.error_estimate():.2e}")
```

No extra function evaluations are needed -- the estimate is computed from the tensor
of function values that `build()` already stored.

## Mathematical Background

### Chebyshev series expansion

Any sufficiently smooth function on $[-1, 1]$ can be expanded in Chebyshev
polynomials:

$$f(x) = \sum_{k=0}^{\infty} c_k\, T_k(x)$$

where $T_k$ is the Chebyshev polynomial of the first kind of degree $k$, and $c_k$
are the expansion coefficients. When we interpolate with $n$ nodes, we compute a
degree-$(n-1)$ polynomial that implicitly truncates this series:

$$p_n(x) = \sum_{k=0}^{n-1} \hat{c}_k\, T_k(x)$$

The interpolation error $f(x) - p_n(x)$ comes from two sources: (1) the omitted
high-degree terms $c_n, c_{n+1}, \ldots$ and (2) aliasing, where these omitted terms
fold back onto the computed coefficients. For well-resolved functions, both sources
are small when the trailing coefficients are small.

### Why the last coefficient estimates the error

For a function analytic in a Bernstein ellipse with parameter $\rho > 1$ (see
[Mathematical Concepts](concepts.md)), the Chebyshev coefficients satisfy
$|c_k| = O(\rho^{-k})$. This means each successive coefficient is roughly $\rho$
times smaller than the previous one. The last included coefficient $|c_{n-1}|$ is
therefore:

1. **An upper bound on the omitted tail.** Since $|c_k| \leq M \rho^{-k}$ and the
   tail sum $\sum_{k=n}^{\infty} |c_k|$ is a geometric series with ratio
   $\rho^{-1} < 1$, we have
   $\sum_{k=n}^{\infty} |c_k| \lesssim |c_{n-1}| / (\rho - 1)$. When
   $\rho$ is even moderately large (say, $\rho > 2$), the omitted tail is
   comparable in magnitude to $|c_{n-1}|$ itself.

2. **A proxy for aliasing error.** The aliased contributions (omitted terms folding
   onto lower coefficients) are bounded by the same geometric decay, so they are also
   $O(|c_{n-1}|)$ for well-resolved functions.

The practical rule: **if $|c_{n-1}|$ is small, both the truncation and aliasing
errors are small, and the interpolant is well-converged.**

!!! warning "Heuristic, not a formal bound"

    This estimate is an empirically reliable proxy, not a rigorous upper bound.
    Ruiz & Zeron (2021, Section 3.4) report that they have never encountered a
    real-world case where small trailing coefficients failed to indicate
    convergence. However, pathological functions (e.g., those with singularities
    just outside the Bernstein ellipse) could have slowly decaying coefficients
    that make the estimate optimistic. Always validate against known solutions
    when possible.

### Computing coefficients via DCT-II

PyChebyshev uses **Type I Chebyshev nodes** (roots of $T_n$, also called
Gauss--Chebyshev nodes; see Trefethen 2013, Ch. 3):

$$x_i = \cos\!\left(\frac{(2i - 1)\,\pi}{2n}\right), \quad i = 1, \ldots, n$$

For values sampled at these $n$ nodes, the Chebyshev expansion coefficients $c_k$ can
be recovered exactly via the **Discrete Cosine Transform (DCT-II)** (Good 1961;
Trefethen 2013, Ch. 3).

**Why DCT-II works.** The connection comes from the orthogonality of Chebyshev
polynomials. Evaluating $T_k(x_i)$ at the Type I nodes and exploiting the identity
$T_k(\cos\theta) = \cos(k\theta)$ turns the coefficient formula into a discrete
cosine sum -- precisely the DCT-II. The computation runs in $O(n \log n)$ via FFT.

In practice, the implementation:

1. **Reverses** the node-order values (`::-1`) because `scipy.fft.dct` expects a
   specific ordering convention, while PyChebyshev stores nodes in ascending order.
2. **Divides** by $n$ (the normalization factor for the DCT-II).
3. **Halves** the zeroth coefficient ($c_0 \mathrel{/}= 2$) because the Chebyshev
   series convention uses $\frac{c_0}{2} T_0(x) + c_1 T_1(x) + \cdots$, while the
   raw DCT includes the full $c_0$.

### Reference

This error estimation approach follows the *ex ante* method described in
Ruiz & Zeron, *Machine Learning for Risk Calculations*, Wiley Finance, 2021,
Section 3.4. For the underlying Chebyshev coefficient theory, see Trefethen
(2013), Chapters 3--4.

## Multi-Dimensional Error

For a $D$-dimensional interpolant on a tensor grid with $n_d$ nodes in dimension
$d$, the error estimate generalizes as follows:

1. **Extract 1-D slices.** For each dimension $d$, fix all other indices and extract
   every 1-D slice of the tensor along dimension $d$.
2. **Compute per-slice error.** For each slice, compute the Chebyshev coefficients
   via DCT-II and take the magnitude of the last coefficient $|c_{n_d - 1}|$.
3. **Maximize over slices.** Take the maximum $|c_{n_d - 1}|$ across all slices for
   dimension $d$. This worst-case slice represents the hardest-to-approximate 1-D
   cross-section of the function along that axis.
4. **Sum across dimensions.** The total error estimate is the sum of per-dimension
   maxima:

$$\hat{E} = \sum_{d=1}^{D} \max_{\text{slices along } d} |c_{n_d - 1}|$$

**Why sum across dimensions?** PyChebyshev evaluates a multi-dimensional interpolant
via *dimensional decomposition* -- contracting one axis at a time (Berrut &
Trefethen 2004; see also [Multi-Dimensional Extension](concepts.md#multi-dimensional-extension)).
At each contraction step, the 1-D interpolation error for that dimension is injected
into the remaining computation. In the worst case, these per-dimension errors add up.
The summation therefore represents a **conservative heuristic**: the total error is
at most the sum of the worst per-dimension errors, assuming the errors do not cancel.

!!! note "Not a tight bound"

    In practice, $\hat{E}$ is often pessimistic because: (a) the worst-case slice
    rarely coincides across all other index combinations, and (b) errors from
    different dimensions tend to partially cancel rather than align. The estimate
    is best used as an order-of-magnitude indicator, not as a precise bound.

## Slider Error Estimation

`ChebyshevSlider` also supports `error_estimate()`. The slider error is the sum of
the error estimates from each individual slide:

```python
import math
from pychebyshev import ChebyshevSlider

def f(x, _):
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

slider = ChebyshevSlider(
    function=f,
    num_dimensions=3,
    domain=[[-1, 1], [-1, 1], [-1, 1]],
    n_nodes=[11, 11, 11],
    partition=[[0], [1], [2]],
    pivot_point=[0.0, 0.0, 0.0],
)
slider.build()
print(f"Slider error estimate: {slider.error_estimate():.2e}")
```

!!! note "Cross-group interaction error"

    The slider error estimate captures **per-slide interpolation error only** --
    the error from approximating each slide's low-dimensional function with
    Chebyshev polynomials. Error from the additive sliding *decomposition itself*
    (i.e., the cross-group coupling that the sliding formula ignores) is **not**
    included. For example, if $f(x_1, x_2) = x_1 \cdot x_2$ and the partition
    is `[[0], [1]]`, the sliding approximation is structurally unable to
    represent the product term, regardless of node count. The error estimate
    will report near-zero (each 1-D slide is well-resolved), but the true error
    can be large. For strongly coupled functions, always validate with test
    points.

## Tensor Train Error Estimation

`ChebyshevTT` also supports `error_estimate()`. The algorithm is analogous to the
full-tensor case: for each core, convert to Chebyshev coefficients via DCT-II, take
the maximum absolute value of the last coefficient slice (across all rank indices),
and sum across dimensions.

```python
from pychebyshev import ChebyshevTT

tt = ChebyshevTT(my_func, 5, domain, [11]*5, max_rank=10)
tt.build()
print(f"TT error estimate: {tt.error_estimate():.2e}")
```

!!! warning "Two sources of TT error"

    The TT error estimate captures **per-core coefficient truncation** -- the
    error from using finitely many Chebyshev nodes within each core.
    It does **not** capture **rank truncation error** -- the error from
    representing the function with a low-rank TT decomposition. Think of it
    this way: even if every core perfectly resolves its 1-D slices
    (coefficient error $\approx 0$), the TT may still be inaccurate because
    the rank is too low to capture all the coupling between dimensions.

    In practice, the rank truncation error dominates at low ranks
    (`max_rank` $\leq 5$), while coefficient truncation dominates at
    high ranks with few nodes. The estimate is most reliable when
    `max_rank` is large enough that increasing it no longer improves accuracy.

## Convergence Example

As the number of nodes increases, the error estimate should decrease -- rapidly for
smooth functions. This example demonstrates spectral convergence for a 1-D function:

```python
from pychebyshev import ChebyshevApproximation
import math

def f(x, _):
    return math.sin(x[0])

for n in [5, 10, 15, 20, 25, 30]:
    cheb = ChebyshevApproximation(f, 1, [[-1, 1]], [n])
    cheb.build(verbose=False)
    print(f"n={n:2d}: error_estimate = {cheb.error_estimate():.2e}")
```

For $\sin(x)$, which is entire (analytic everywhere in the complex plane), the
coefficients decay exponentially, so you should see the estimate drop by several
orders of magnitude as $n$ grows.

## API Reference

For full method signatures and parameter details, see:

- [`ChebyshevApproximation.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevApproximation.error_estimate) -- error estimate for standard tensor interpolation.
- [`ChebyshevSlider.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevSlider.error_estimate) -- error estimate for sliding approximation (sum of per-slide errors).
- [`ChebyshevTT.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevTT.error_estimate) -- error estimate for Tensor Train approximation (from coefficient cores).

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation."
  *SIAM Review* 46(3):501--517.
- Good, I.J. (1961). "The colleague matrix, a Chebyshev analogue of the companion
  matrix." *The Quarterly Journal of Mathematics* 12(1):61--68.
- Ruiz, G. & Zeron, M. (2021). *Machine Learning for Risk Calculations.*
  Wiley Finance. Section 3.4.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM. Chapters 3--4.
