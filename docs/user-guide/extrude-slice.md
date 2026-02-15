# Extrusion & Slicing

## Motivation -- Portfolio Combination

In practice, trades depend on different subsets of risk factors.  Trade A
might depend on (spot, rate) while Trade B depends on (spot, vol).  The
v0.7.0 algebra operators require operands on the **same grid**, so these
two proxies cannot be added directly.

**Extrusion** solves this by adding new dimensions where the function is
constant.  After extruding both trades to a common 3D grid (spot, rate, vol),
they can be combined with the standard `+`, `-`, `*`, `/` operators:

```python
portfolio = trade_a_3d + trade_b_3d
```

**Slicing** is the inverse: it fixes a dimension at a specific value, reducing
dimensionality via barycentric interpolation.  Together, extrusion and slicing
form the bridge between Chebyshev proxies on heterogeneous risk-factor sets.

!!! tip "When to use extrude/slice"
    Use extrusion when you need to combine Chebyshev interpolants that
    live on **different** sets of dimensions.  Use slicing to project a
    high-dimensional interpolant onto a lower-dimensional subspace (e.g.,
    fixing a parameter at its current market value).

## Mathematical Basis

### Partition of Unity

The barycentric basis functions satisfy a fundamental identity:

$$
\sum_{j=0}^{n} \ell_j(x) = 1
$$

for all $x$ in the domain.  This is because any polynomial interpolation scheme
reproduces constant functions exactly -- the constant $1$ is interpolated by
$\sum_j 1 \cdot \ell_j(x) = 1$.

!!! note "Reference"
    Berrut & Trefethen (2004), "Barycentric Lagrange Interpolation",
    SIAM Review 46(3):501--517, Section 2.

### Extrusion Proof

Given a $d$-dimensional CT with values $v_{i_1,\ldots,i_d}$, the extruded
$(d+1)$-dimensional CT inserts a new axis at position $k$ with $M$ Chebyshev
nodes, replicating values:

$$
\hat{v}_{i_1,\ldots,i_{k-1},\,j,\,i_k,\ldots,i_d} = v_{i_1,\ldots,i_d}
\quad \forall\; j = 0,\ldots,M-1
$$

Evaluating at any point $(x_1,\ldots,x_{k-1},x^*,x_k,\ldots,x_d)$:

$$
p(\mathbf{x}) = \sum_{\text{all indices}} v_{i_1,\ldots,i_d}
\cdot \ell_j^{(k)}(x^*) \cdot \prod_{m \neq k} \ell_{i_m}^{(m)}(x_m)
$$

Since the values don't depend on $j$, the $j$-sum factors out:

$$
= \underbrace{\left(\sum_{j=0}^{M-1} \ell_j^{(k)}(x^*)\right)}_{= 1
\;\text{(partition of unity)}} \cdot \sum_{i_1,\ldots,i_d} v_{i_1,\ldots,i_d}
\prod_{m \neq k} \ell_{i_m}^{(m)}(x_m) = p_{\text{orig}}(\mathbf{x}_{\text{orig}})
$$

**Result**: The extruded CT evaluates to the same value as the original,
regardless of the new coordinate.  Extrusion is exact.

### Slicing Proof

Given a $d$-dimensional CT, fixing dimension $k$ at value $x^*$:

$$
p(x_1,\ldots,x_{k-1},x^*,x_{k+1},\ldots,x_d) = \sum_{i_1,\ldots,i_d}
v_{i_1,\ldots,i_d} \prod_{m=1}^{d} \ell_{i_m}^{(m)}(x_m) \bigg|_{x_k = x^*}
$$

Factoring out the $k$-th dimension:

$$
= \sum_{i_1,\ldots,i_{k-1},i_{k+1},\ldots,i_d}
\underbrace{\left(\sum_{i_k} v_{i_1,\ldots,i_d}
\cdot \ell_{i_k}^{(k)}(x^*)\right)}_{\hat{v}_{i_1,\ldots,i_{k-1},i_{k+1},\ldots,i_d}}
\prod_{m \neq k} \ell_{i_m}^{(m)}(x_m)
$$

The contracted values $\hat{v}$ define a valid $(d-1)$-dimensional CT.

**Fast path**: When $x^*$ coincides with a Chebyshev node $x_m^{(k)}$
(within tolerance $10^{-14}$), the basis function simplifies to
$\ell_{i_k}^{(k)}(x_m) = \delta_{i_k,m}$, so
$\hat{v} = v_{\ldots,m,\ldots}$ -- a simple `np.take` (no arithmetic
needed).

### Extrude-then-Slice = Identity

If we extrude along dimension $k$ and then slice at any value $x^*$
along $k$:

$$
\text{slice}(\text{extrude}(T, k), k, x^*) = T
$$

*Proof.*  Extrusion replicates values along $k$, then slicing contracts via
$\sum_j v \cdot \ell_j(x^*) = v \cdot 1 = v$.

### Error Bounds

- **Extrusion**: No approximation error introduced (exact operation).
- **Slicing**: The sliced CT evaluates the polynomial interpolant at $x_k = x^*$.
  No additional error beyond the original approximation error:
  if $\|f - p\|_\infty \leq \epsilon$, then
  $\|f(\cdot, x^*) - p(\cdot, x^*)\|_\infty \leq \epsilon$.

!!! note "Book reference"
    Extrusion and slicing of Chebyshev Tensors is described in Section 24.2.1,
    Listing 24.15 (slice) and Listing 24.16 (extrude) of Ruiz & Zeron (2021),
    *Machine Learning for Risk Calculations*, Wiley Finance.

## Quick Start

```python
import math
from pychebyshev import ChebyshevApproximation

def f(x, _): return math.sin(x[0]) + x[1]
def g(x, _): return math.cos(x[0]) * x[1]

# Trade A depends on (spot, rate)
trade_a = ChebyshevApproximation(f, 2, [[80, 120], [0.01, 0.08]], [11, 11])
trade_a.build()

# Trade B depends on (spot, vol)
trade_b = ChebyshevApproximation(g, 2, [[80, 120], [0.15, 0.35]], [11, 11])
trade_b.build()

# Extrude both to 3D: (spot, rate, vol)
trade_a_3d = trade_a.extrude((2, (0.15, 0.35), 11))  # add vol dim
trade_b_3d = trade_b.extrude((1, (0.01, 0.08), 11))  # add rate dim

# Combine into portfolio
portfolio = trade_a_3d + trade_b_3d
price = portfolio.vectorized_eval([100.0, 0.05, 0.25], [0, 0, 0])
```

## Supported Operations

| Class | `extrude()` | `slice()` |
|-------|-------------|-----------|
| `ChebyshevApproximation` | Yes | Yes |
| `ChebyshevSpline` | Yes | Yes |
| `ChebyshevSlider` | Yes | Yes |
| `ChebyshevTT` | No | No |

## Extrude API

```python
result = cheb.extrude(params)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | tuple or list of tuples | `(dim_index, (lo, hi), n_nodes)` |

- `dim_index` -- position in the **output** space (0 = prepend, `d` = append)
- `(lo, hi)` -- domain bounds for the new dimension
- `n_nodes` -- number of Chebyshev nodes (must match other CTs for later algebra)

**Returns**: A new interpolant of the same type, already built, with
`function=None`.

**Errors**:

- `RuntimeError` if the interpolant has not been built
- `ValueError` if `dim_index` is out of range, duplicated, `lo >= hi`, or `n_nodes < 2`

!!! example "Multi-extrude: 1D to 3D"
    ```python
    cheb_1d = ChebyshevApproximation(f, 1, [[-1, 1]], [15])
    cheb_1d.build()

    # Add two dimensions at once
    cheb_3d = cheb_1d.extrude([
        (1, (0.0, 5.0), 11),
        (2, (-2.0, 2.0), 9),
    ])
    ```

## Slice API

```python
result = cheb.slice(params)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | tuple or list of tuples | `(dim_index, value)` |

- `dim_index` -- dimension to fix (0-indexed in the current object)
- `value` -- point at which to fix (must be within the domain)

**Returns**: A new interpolant of the same type, already built, with
`function=None`.

**Errors**:

- `RuntimeError` if the interpolant has not been built
- `ValueError` if `value` is outside the domain, `dim_index` is out of range,
  duplicated, or if slicing all dimensions

!!! example "Slice 3D to 1D"
    ```python
    cheb_3d = ChebyshevApproximation(f, 3, [[-1, 1]] * 3, [11, 11, 11])
    cheb_3d.build()

    # Fix dim 1 and dim 2
    cheb_1d = cheb_3d.slice([(1, 0.5), (2, -0.3)])
    ```

!!! tip "Fast path at exact nodes"
    When the slice value coincides with a Chebyshev node (within $10^{-14}$),
    the contraction reduces to `np.take` -- a simple array index with no
    floating-point arithmetic.

## Compatibility with Algebra

Extrusion is the key enabler for the v0.7.0 algebra operators.  After
extruding two CTs to a common grid, all standard operators work:

```python
# Different risk factors
ct_a = ChebyshevApproximation(f, 2, [[80, 120], [0.01, 0.08]], [11, 11])
ct_b = ChebyshevApproximation(g, 2, [[80, 120], [0.15, 0.35]], [11, 11])
ct_a.build(); ct_b.build()

# Extrude to common 3D: (spot, rate, vol)
ct_a_3d = ct_a.extrude((2, (0.15, 0.35), 11))
ct_b_3d = ct_b.extrude((1, (0.01, 0.08), 11))

# Now all algebra operators work
portfolio = 0.6 * ct_a_3d + 0.4 * ct_b_3d
hedged    = ct_a_3d - ct_b_3d
scaled    = 2.0 * ct_a_3d
```

The compatibility requirements from [Chebyshev Algebra](algebra.md) apply
to the extruded results: same domain, node counts, derivative order, and
number of dimensions.

## Derivatives

**Extrusion**: Derivatives in the original dimensions are preserved.
Derivatives with respect to the new dimension are zero (the function is
constant along the new axis).  This follows from
$\mathcal{D}_k \cdot [c, c, \ldots, c]^T = \mathbf{0}$.

**Slicing**: Derivatives in the remaining dimensions are preserved.  The
sliced CT has valid spectral differentiation matrices for all surviving
dimensions.

```python
# Extrude: derivative w.r.t. new dim is zero
ct_3d = ct_2d.extrude((2, (0.0, 1.0), 11))
assert abs(ct_3d.vectorized_eval([0.5, 0.3, 0.7], [0, 0, 1])) < 1e-12

# Slice: derivative in remaining dim preserved
ct_1d = ct_2d.slice((1, 0.5))
d_dx = ct_1d.vectorized_eval([0.3], [1])  # first derivative w.r.t. dim 0
```

## Serialization

Extruded and sliced interpolants support `save()` and `load()` just like
any other built interpolant:

```python
ct_3d = ct_2d.extrude((2, (0.0, 1.0), 11))
ct_3d.save("extruded.pkl")

loaded = ChebyshevApproximation.load("extruded.pkl")
loaded.vectorized_eval([0.5, 0.3, 0.7], [0, 0, 0])  # works identically
```

## Class-Specific Notes

### ChebyshevSpline

When extruding a spline, each piece is extruded independently.  The new
dimension gets `knots=[]` (no interior knots) and a single interval.

When slicing a spline, only the pieces whose interval along the sliced
dimension contains the slice value survive.  Each surviving piece is then
sliced via its underlying `ChebyshevApproximation.slice()`.

### ChebyshevSlider

When extruding a slider, the new dimension becomes its own single-dim
slide group with `tensor_values = np.full(n, pivot_value)`, so the slide
contributes zero: $s_{\text{new}}(x) - pv = 0$ for all $x$.  The
partition indices for existing dimensions are remapped accordingly.

When slicing a slider, two cases arise:

- **Multi-dim group**: The slide's `ChebyshevApproximation` is sliced at
  the local dimension index within the group.
- **Single-dim group**: The slide is evaluated at the slice value, giving
  a constant $s_{g^*}(v)$.  The shift $\delta = s_{g^*}(v) - pv$ is
  absorbed into `pivot_value` and each remaining slide's tensor values.

## Limitations

- **No `ChebyshevTT` support** -- extrusion and slicing for Tensor Train
  interpolants are not currently implemented.
- **Operand must be built** -- `build()` must have been called before
  calling `extrude()` or `slice()`.
- **No cross-type operations** -- you cannot extrude a `ChebyshevSpline`
  and then add it to a `ChebyshevApproximation`.
- **Result has `function=None`** -- the extruded/sliced interpolant cannot
  call `build()` again, since it has no underlying function reference.
