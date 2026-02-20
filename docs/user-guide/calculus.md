# Chebyshev Calculus

## Motivation

Once you have a Chebyshev interpolant, you can compute integrals, find roots,
and optimize -- all without re-evaluating the original (expensive) function.
These operations work directly on the stored tensor values using spectral
methods:

- **Integration** via Fejér-1 quadrature weights (DCT-III)
- **Rootfinding** via companion matrix eigenvalues
- **Optimization** via derivative rootfinding + endpoint evaluation

The key insight is that the Chebyshev interpolant *is* a polynomial, and
polynomial integrals, roots, and extrema can be computed exactly from
the expansion coefficients -- no additional function evaluations needed.

In quantitative finance, this enables computing expected exposures
($\mathbb{E}[\max(V, 0)]$ via integration), finding break-even levels
(roots of $V - K$), and locating worst-case scenarios (optimization) --
all from a single pre-built proxy, without calling the pricing engine again.

!!! tip "Cross-reference"
    See [Chebyshev Algebra](algebra.md) for combining interpolants and
    [Extrusion & Slicing](extrude-slice.md) for dimension manipulation.

## Supported Classes

| Class | `integrate()` | `roots()` | `minimize()` / `maximize()` |
|-------|:---:|:---:|:---:|
| `ChebyshevApproximation` | Yes | Yes | Yes |
| `ChebyshevSpline` | Yes | Yes | Yes |
| `ChebyshevSlider` | No | No | No |
| `ChebyshevTT` | No | No | No |

## Integration

### Theory

Given a Chebyshev interpolant $p_n(x) = \sum_{k=0}^{n-1} c_k T_k(x)$ on
$[-1, 1]$, the definite integral is:

$$
\int_{-1}^{1} p_n(x)\, dx = 2\,c_0 + \sum_{\substack{k=2 \\ k \text{ even}}}^{n-1} \frac{2\,c_k}{1 - k^2}
$$

since $\int_{-1}^{1} T_k(x)\,dx = 2/(1-k^2)$ for even $k$ and $0$ for odd $k$.

This is exact for the polynomial interpolant -- the only error is from the
original Chebyshev approximation, not from the quadrature.

In practice, we compute **Fejér-1 quadrature weights** $w_j$ such that:

$$
\int_{-1}^{1} p(x)\, dx = \sum_{j=0}^{n-1} w_j \, p(x_j)
$$

where $x_j$ are the Chebyshev Type I nodes.  The weights are computed in
$O(n \log n)$ via the DCT-III algorithm of Waldvogel (2006):

$$
w_j = \frac{1}{n}\left[I_0 + 2\sum_{k=1}^{n-1} I_k \cos\!\left(\frac{\pi\, k\,(2j+1)}{2n}\right)\right], \quad I_k = \begin{cases} \frac{2}{1-k^2} & k \text{ even} \\ 0 & k \text{ odd} \end{cases}
$$

The vector $[I_0, I_1, \ldots, I_{n-1}]$ is the input to a DCT-III, giving
all $n$ weights in a single $O(n \log n)$ transform.

**Domain scaling.** For a general interval $[a, b]$:

$$
\int_a^b f(x)\, dx \approx \frac{b - a}{2} \sum_{j=0}^{n-1} w_j \, v_j
$$

where $v_j$ are the stored tensor values at the Chebyshev nodes.

**Multi-dimensional integration.** For a $d$-dimensional interpolant,
integration along each dimension is a tensor contraction via
`np.tensordot` (routed through BLAS GEMV):

$$
\int_{\Omega} p(\mathbf{x})\, d\mathbf{x} = \sum_{i_1,\ldots,i_d} v_{i_1,\ldots,i_d} \prod_{k=1}^{d} w_{i_k}^{(k)} \cdot s_k
$$

where $s_k = (b_k - a_k)/2$ is the half-width of dimension $k$.

### Usage

```python
import math
from pychebyshev import ChebyshevApproximation

def f(x, _):
    return math.sin(x[0]) * math.cos(x[1])

cheb = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [11, 11])
cheb.build()

# Full integral over all dimensions -> float
total = cheb.integrate()

# Partial integral -- integrate out dim 1, get 1D interpolant
cheb_1d = cheb.integrate(dims=[1])
value = cheb_1d.vectorized_eval([0.5], [0])

# Integrate multiple dimensions at once
cheb_3d = ChebyshevApproximation(h, 3, domains, n_nodes)
cheb_3d.build()
cheb_1d = cheb_3d.integrate(dims=[0, 2])
```

### Sub-Interval Integration

By default, `integrate()` integrates over the full domain of each
dimension.  The `bounds` parameter restricts integration to a
sub-interval $[a', b'] \subseteq [a, b]$.

**Domain mapping.**  Physical bounds $[a', b']$ on domain $[a, b]$ are
mapped to reference coordinates via the standard affine transformation:

$$
t_a = \frac{2(a' - a)}{b - a} - 1, \qquad t_b = \frac{2(b' - a)}{b - a} - 1
$$

so that $[a', b'] \mapsto [t_a, t_b] \subseteq [-1, 1]$.

**Theory.**  Fejér-1 weights use moments $I_k = \int_{-1}^{1} T_k(t)\,dt$.
For a sub-interval $[t_a, t_b] \subseteq [-1, 1]$, replace with
sub-interval moments:

$$
I_k = \int_{t_a}^{t_b} T_k(t)\,dt
$$

These are computed in closed form via the Chebyshev polynomial
antiderivative (Trefethen 2013, Ch. 19):

$$
\int T_k(x)\,dx = \frac{T_{k+1}(x)}{2(k+1)} - \frac{T_{k-1}(x)}{2(k-1)} + C \quad (k \geq 2)
$$

with special cases $\int T_0\,dx = x$ and $\int T_1\,dx = x^2/2$.

??? note "Proof sketch"
    Since $\frac{d}{dx}T_n(x) = n\,U_{n-1}(x)$ and the Chebyshev
    identity $U_k(x) - U_{k-2}(x) = 2\,T_k(x)$ holds, differentiating
    the right-hand side gives:

    $$
    \frac{d}{dx}\!\left[\frac{T_{k+1}}{2(k+1)} - \frac{T_{k-1}}{2(k-1)}\right]
    = \frac{(k+1)\,U_k}{2(k+1)} - \frac{(k-1)\,U_{k-2}}{2(k-1)}
    = \frac{U_k - U_{k-2}}{2} = T_k \quad\checkmark
    $$

The sub-interval moments are then:

- $k = 0$: $I_0 = t_b - t_a$
- $k = 1$: $I_1 = (t_b^2 - t_a^2) / 2$
- $k \geq 2$: $I_k = \frac{1}{2}\!\left[\frac{T_{k+1}(t_b) - T_{k+1}(t_a)}{k+1} - \frac{T_{k-1}(t_b) - T_{k-1}(t_a)}{k-1}\right]$

where $T_j$ values are evaluated via the three-term recurrence
$T_{k+1}(x) = 2x\,T_k(x) - T_{k-1}(x)$, which is stable for
$|x| \leq 1$.

The moments vector is then processed through the same DCT-III →
weights pipeline as full-domain Fejér-1 (Waldvogel 2006).  When
$[t_a, t_b] = [-1, 1]$, the moments reduce to the standard Fejér-1
moments exactly (since $I_k = 2/(1-k^2)$ for even $k$, $0$ for odd $k$).

**Additivity.**  Sub-interval integration satisfies the interval
additivity property: for any $a \leq c \leq b$,

$$
\int_a^c f + \int_c^b f = \int_a^b f
$$

This follows from the linearity of the DCT-III pipeline and the
additivity of the Chebyshev antiderivative over adjacent sub-intervals.

**Splines.** For splines, each piece's sub-domain is clipped against
the requested bounds.  Pieces with no overlap are skipped; pieces with
partial overlap receive sub-interval bounds; pieces fully contained use
standard Fejér-1 weights.

```python
import math
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(lambda x, _: math.sin(x[0]), 1, [[0, 2 * math.pi]], [25])
cheb.build()

# Sub-interval: first half-period
half = cheb.integrate(bounds=(0.0, math.pi))   # ≈ 2.0

# Per-dimension bounds for multi-D
cheb_2d = ChebyshevApproximation(
    lambda x, _: x[0] ** 2 + math.cos(x[1]),
    2, [[-1, 1], [-1, 1]], [11, 11],
)
cheb_2d.build()
result = cheb_2d.integrate(dims=[0, 1], bounds=[(0.0, 1.0), None])
```

### `integrate()` API

| Parameter | Type | Description |
|-----------|------|-------------|
| `dims` | `int`, `list[int]`, or `None` | Dimensions to integrate out. `None` = all. |
| `bounds` | `tuple`, `list[tuple/None]`, or `None` | Sub-interval bounds. `None` = full domain. Single `(lo, hi)` for one dim, or list with positional correspondence to `dims`. |

**Returns**: `float` if all dimensions are integrated; otherwise a
lower-dimensional interpolant of the same type (`ChebyshevApproximation` or
`ChebyshevSpline`).

**Errors**:

- `RuntimeError` if `build()` has not been called
- `ValueError` if any dimension index is out of range, duplicated, or
  bounds are outside the domain

## Rootfinding

### Theory

Roots of a Chebyshev interpolant are found via the **colleague matrix**
(Good 1961), a Chebyshev analogue of the companion matrix.  Given the
expansion $p_n(x) = \sum_{k=0}^{n-1} c_k T_k(x)$, the roots of $p_n$
are the eigenvalues of an $(n-1) \times (n-1)$ tridiagonal-plus-rank-1
matrix.

PyChebyshev delegates to `numpy.polynomial.chebyshev.chebroots()`, which
constructs this colleague matrix and computes eigenvalues via LAPACK QR.
Complex and out-of-domain roots are filtered out, and the remaining real
roots are mapped from $[-1, 1]$ to the physical domain $[a, b]$.

For multi-dimensional interpolants, the interpolant is first sliced to 1-D
using the existing `slice()` infrastructure (see
[Extrusion & Slicing](extrude-slice.md)), then rootfinding proceeds on the
resulting 1-D polynomial.

### Usage

```python
import math
from pychebyshev import ChebyshevApproximation

# 1D roots
cheb = ChebyshevApproximation(lambda x, _: math.sin(x[0]), 1, [[-4, 4]], [25])
cheb.build()
roots = cheb.roots()  # array([-pi, 0, pi])

# Multi-D: fix other dims, find roots along dim 0
cheb_2d = ChebyshevApproximation(
    lambda x, _: math.sin(x[0]) * math.cos(x[1]),
    2, [[-4, 4], [-2, 2]], [25, 15],
)
cheb_2d.build()
roots = cheb_2d.roots(dim=0, fixed={1: 0.5})
```

### `roots()` API

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` or `None` | Dimension along which to find roots. Defaults to `0` for 1-D. |
| `fixed` | `dict` or `None` | For multi-D: `{dim_index: value}` for **all** dimensions except `dim`. |

**Returns**: Sorted `ndarray` of root locations in the physical domain.

**Errors**:

- `RuntimeError` if `build()` has not been called
- `ValueError` if `dim` is out of range, `fixed` is incomplete (missing
  dimensions), or a fixed value is outside its domain

## Optimization

### Theory

Minima and maxima of a polynomial on a closed interval occur either at
**critical points** (where the derivative is zero) or at the **endpoints**.
PyChebyshev:

1. Computes derivative values at the Chebyshev nodes via the spectral
   differentiation matrix $\mathcal{D}$ (Berrut & Trefethen 2004)
2. Finds all roots of the derivative via the colleague matrix (critical points)
3. Evaluates the original interpolant at all critical points and domain endpoints
4. Returns the global minimum or maximum

For multi-dimensional interpolants, the interpolant is first sliced to 1-D
(same as rootfinding), then optimization proceeds on the 1-D polynomial.

### Usage

```python
import math
from pychebyshev import ChebyshevApproximation

cheb = ChebyshevApproximation(lambda x, _: math.sin(x[0]), 1, [[-4, 4]], [25])
cheb.build()

val, loc = cheb.minimize()   # (-1.0, -pi/2)
val, loc = cheb.maximize()   # (1.0, pi/2)

# Multi-D: fix other dims
cheb_2d = ChebyshevApproximation(
    lambda x, _: x[0] ** 2 + x[1],
    2, [[-1, 1], [-1, 1]], [11, 11],
)
cheb_2d.build()
val, loc = cheb_2d.minimize(dim=0, fixed={1: 0.5})  # (0.5, 0.0)
```

### `minimize()` / `maximize()` API

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` or `None` | Dimension along which to optimize. Defaults to `0` for 1-D. |
| `fixed` | `dict` or `None` | For multi-D: `{dim_index: value}` for **all** dimensions except `dim`. |

**Returns**: `(value, location)` tuple of two `float`s -- the optimal value
and its coordinate in the target dimension.

**Errors**:

- `RuntimeError` if `build()` has not been called
- `ValueError` if `dim` is out of range, `fixed` is incomplete, or a fixed
  value is outside its domain

## Spline Support

All three calculus operations are supported on `ChebyshevSpline`:

- **Integration**: sums the integrals of each piece (pieces cover disjoint
  sub-domains). Partial integration sums piece contributions along the
  integrated dimension and returns a lower-dimensional spline.
- **Roots**: finds roots in each piece independently, then merges and
  deduplicates at knot boundaries.
- **Optimization**: computes per-piece extrema and returns the global
  minimum or maximum across all pieces.

```python
from pychebyshev import ChebyshevSpline

spline = ChebyshevSpline(
    lambda x, _: abs(x[0]) - 0.3,
    1, [[-1, 1]], [15], [[0.0]],
)
spline.build()

integral = spline.integrate()
roots = spline.roots()          # array([-0.3, 0.3])
val, loc = spline.minimize()    # (-0.3, 0.0)
```

The API signatures and return types are identical to `ChebyshevApproximation`.
Partial integration on a spline returns a lower-dimensional `ChebyshevSpline`.

## Derivatives, Error Estimation, and Serialization

**Partial integration** returns a fully functional interpolant (either
`ChebyshevApproximation` or `ChebyshevSpline`).  All existing features
work on the result:

- **Derivatives**: `vectorized_eval(point, [1])` computes derivatives in
  the surviving dimensions via the spectral differentiation matrices.
- **Error estimation**: `error_estimate()` recomputes from the DCT
  coefficients of the reduced tensor.
- **Serialization**: `save()` and `load()` work as usual.
- **Further calculus**: you can call `integrate()`, `roots()`,
  `minimize()`, or `maximize()` on the result.

```python
# Partial integrate, then take derivatives and estimate error
cheb_2d = ChebyshevApproximation(f, 2, [[-1, 1], [-1, 1]], [15, 15])
cheb_2d.build()

cheb_1d = cheb_2d.integrate(dims=[0])
deriv = cheb_1d.vectorized_eval([0.5], [1])
err = cheb_1d.error_estimate()
cheb_1d.save("partial_integral.pkl")
```

`roots()`, `minimize()`, and `maximize()` return scalar values (arrays or
tuples), so derivatives/serialization are not applicable to their outputs.

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Fejér-1 weights | $O(n \log n)$ | DCT-III (Waldvogel 2006), computed once per dimension |
| 1-D integration | $O(n)$ | Dot product of weights and values |
| Multi-D integration | $O(n^d)$ | Sequential `np.tensordot` contractions (BLAS GEMV) |
| 1-D rootfinding | $O(n^3)$ | Companion matrix eigenvalues (LAPACK QR) |
| 1-D optimization | $O(n^3)$ | Derivative roots + endpoint evaluation |

For typical node counts ($n \leq 50$), the $O(n^3)$ companion matrix
eigenvalue computation is negligible (sub-millisecond). All operations
reuse existing infrastructure: DCT coefficients, spectral differentiation
matrices, and barycentric evaluation.

## Limitations

- **Not yet supported on `ChebyshevSlider` or `ChebyshevTT`.**
  Calculus on Tensor Train interpolants would require TT-specific
  coefficient extraction; slider calculus would need per-slide integration
  with additive recombination.
- **Multi-D rootfinding** (2D Bezout resultants) is not implemented. Only
  1-D slices are supported via the `dim` + `fixed` interface.
- **Result has `function=None`** -- partial integration results cannot call
  `build()` again, since there is no underlying function reference.

## References

1. Waldvogel, J. (2006), "Fast Construction of the Fejér and
   Clenshaw--Curtis Quadrature Rules", *BIT Numerical Mathematics*
   46(2):195--202.

2. Trefethen, L.N. (2013), *Approximation Theory and Approximation
   Practice*, SIAM, Chapters 18--21.

3. Good, I.J. (1961), "The colleague matrix, a Chebyshev analogue of the
   companion matrix", *The Quarterly Journal of Mathematics* 12(1):61--68.

4. Berrut, J.-P. & Trefethen, L.N. (2004), "Barycentric Lagrange
   Interpolation", *SIAM Review* 46(3):501--517.
