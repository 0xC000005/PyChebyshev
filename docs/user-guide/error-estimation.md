# Error Estimation

## Introduction

After building a Chebyshev interpolant, you often want to know how accurate it is
without comparing against the true function at thousands of test points.
`error_estimate()` provides an **ex ante** estimate of the maximum interpolation
error using only the Chebyshev coefficients already computed during the build step.

This is useful for:

- **Validating node counts** -- check whether your grid is fine enough before deploying.
- **Adaptive refinement** -- increase nodes in dimensions where the error is large.
- **Confidence reporting** -- attach an error bound to interpolated values.

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

$$f(x) \approx \sum_{k=0}^{n-1} c_k\, T_k(x)$$

where $T_k$ is the Chebyshev polynomial of the first kind of degree $k$, and $c_k$
are the expansion coefficients.

### Coefficient decay

For smooth functions, the coefficients $c_k$ decay rapidly to zero as $k$ increases.
The rate of decay depends on the regularity of $f$: analytic functions enjoy
exponential decay, while functions with $p$ continuous derivatives see algebraic
decay of order $O(k^{-p})$.

Because the truncated series omits all terms from $c_n$ onward, the magnitude of the
**last included coefficient** $|c_{n-1}|$ serves as a proxy for the truncation error.
If $|c_{n-1}|$ is already small, the omitted tail is even smaller.

!!! warning "Not a formal bound"

    Strictly speaking, small trailing coefficients do not *guarantee* that the
    interpolant has converged.  In practice, however, once the last few
    coefficients fall below a small threshold (e.g. $10^{-4}$), the
    interpolation error is almost always at most that size.  The authors of
    Ruiz & Zeron (2021) note that they have never encountered an exception to
    this rule in a real-world problem (Section 3.4).

### Computing coefficients via DCT-II

PyChebyshev uses **Type I Chebyshev nodes** (also called Gauss--Chebyshev points):

$$x_i = \cos\!\left(\frac{(2i - 1)\,\pi}{2n}\right), \quad i = 1, \ldots, n$$

For values sampled at these nodes, the Chebyshev coefficients $c_k$ can be recovered
exactly using the **Discrete Cosine Transform (DCT-II)**, which runs in
$O(n \log n)$ time via FFT.

### Reference

This error estimation approach follows the *ex ante* method described in
Ruiz & Zeron (2021), Section 3.4.

## Multi-Dimensional Error

For a $D$-dimensional interpolant on a tensor grid with $n_d$ nodes in dimension
$d$, the error estimate generalizes as follows:

1. **Extract 1-D slices.** For each dimension $d$, fix all other indices and extract
   every 1-D slice of the tensor along dimension $d$.
2. **Compute per-slice error.** For each slice, compute the Chebyshev coefficients
   via DCT-II and take the magnitude of the last coefficient $|c_{n_d - 1}|$.
3. **Maximize over slices.** Take the maximum $|c_{n_d - 1}|$ across all slices for
   dimension $d$.
4. **Sum across dimensions.** The total error estimate is the sum of per-dimension
   maxima:

$$\hat{E} = \sum_{d=1}^{D} \max_{\text{slices along } d} |c_{n_d - 1}|$$

This summation reflects the fact that interpolation error can accumulate from each
dimension independently when contracting the tensor one axis at a time.

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

    The slider error estimate captures **per-slide interpolation error only**.
    Error from cross-group interactions -- inherent to the additive sliding
    decomposition -- is **not** included. For strongly coupled functions, the
    true error may be significantly larger than the reported estimate.

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

For $\sin(x)$, which is entire (analytic everywhere), the coefficients decay
exponentially, so you should see the estimate drop by several orders of magnitude as
$n$ grows.

## Tensor Train Error Estimation

`ChebyshevTT` also supports `error_estimate()`. The estimate is computed from the
Chebyshev coefficient cores: for each dimension, the last coefficient slice of the
core is extracted and its norm serves as the per-dimension error proxy. These are
summed across dimensions, following the same logic as the full tensor case.

```python
from pychebyshev import ChebyshevTT

tt = ChebyshevTT(my_func, 5, domain, [11]*5, max_rank=10)
tt.build()
print(f"TT error estimate: {tt.error_estimate():.2e}")
```

## API Reference

For full method signatures and parameter details, see:

- [`ChebyshevApproximation.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevApproximation.error_estimate) -- error estimate for standard tensor interpolation.
- [`ChebyshevSlider.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevSlider.error_estimate) -- error estimate for sliding approximation (sum of per-slide errors).
- [`ChebyshevTT.error_estimate()`](../api/reference.md#pychebyshev.ChebyshevTT.error_estimate) -- error estimate for Tensor Train approximation (from coefficient cores).
