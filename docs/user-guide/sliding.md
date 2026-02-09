# Sliding Technique

The **Sliding Technique** enables Chebyshev approximation of high-dimensional functions by decomposing them into a sum of low-dimensional interpolants. This sidesteps the curse of dimensionality at the cost of losing cross-group interactions.

## Motivation

A full tensor Chebyshev interpolant on \(n\) dimensions with \(m\) nodes per dimension requires \(m^n\) function evaluations. For \(n = 10\) and \(m = 11\), that is over 25 billion evaluations — clearly infeasible.

The sliding technique partitions the dimensions into small groups and builds a separate Chebyshev interpolant (a **slide**) for each group, with all other dimensions fixed at a **pivot point**. The total cost becomes the *sum* of the group grid sizes rather than their *product*.

## Algorithm

Given \(f: \mathbb{R}^n \to \mathbb{R}\), a pivot point \(\mathbf{z} = (z_1, \ldots, z_n)\), and a partition of dimensions into \(k\) groups:

1. Evaluate the **pivot value** \(v = f(\mathbf{z})\).
2. For each group \(i\), build a **slide** \(s_i\) — a Chebyshev interpolant on the group's dimensions, with all other dimensions fixed at their pivot values.
3. Evaluate using the additive formula:

\[
f(\mathbf{x}) \approx v + \sum_{i=1}^{k} \bigl[ s_i(\mathbf{x}_{G_i}) - v \bigr]
\]

where \(\mathbf{x}_{G_i}\) denotes the components of \(\mathbf{x}\) belonging to group \(i\).

## When to Use Sliding

Sliding works well when:

- The function is **additively separable** or nearly so (e.g., \(\sin(x_1) + \sin(x_2) + \sin(x_3)\)).
- Cross-group interactions are **weak** relative to within-group effects.
- The number of dimensions is **too large** for full tensor interpolation (say, \(n > 6\)).

Sliding does **not** work well when:

- Variables in different groups are **strongly coupled** (e.g., Black-Scholes where \(S\), \(T\), and \(\sigma\) interact multiplicatively).
- High accuracy is required far from the pivot point.

!!! tip "Choosing the partition"
    Group variables that have strong non-linear interactions together. For example, if \(f = x_1^3 x_2^2 + x_3\), group \((x_1, x_2)\) in one slide and \(x_3\) in another.

## Usage

```python
import math
from pychebyshev import ChebyshevSlider

# Additively separable function
def f(x, _):
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])

slider = ChebyshevSlider(
    function=f,
    num_dimensions=3,
    domain=[[-1, 1], [-1, 1], [-1, 1]],
    n_nodes=[11, 11, 11],
    partition=[[0], [1], [2]],       # each dim is its own slide
    pivot_point=[0.0, 0.0, 0.0],
)
slider.build()

# Evaluate function value
val = slider.eval([0.5, 0.3, -0.2], [0, 0, 0])

# Evaluate derivative w.r.t. x0
dfdx0 = slider.eval([0.5, 0.3, -0.2], [1, 0, 0])
```

### Multi-dimensional slides

For functions with within-group coupling, use larger groups:

```python
def g(x, _):
    return x[0]**3 * x[1]**2 + math.sin(x[2]) + math.sin(x[3])

slider = ChebyshevSlider(
    function=g,
    num_dimensions=4,
    domain=[[-2, 2], [-2, 2], [-1, 1], [-1, 1]],
    n_nodes=[12, 12, 8, 8],
    partition=[[0, 1], [2], [3]],    # 2D + 1D + 1D
    pivot_point=[0.0, 0.0, 0.0, 0.0],
)
slider.build()
```

### Build cost comparison

```python
# Full tensor: 12 * 12 * 8 * 8 = 9,216 evaluations
# Sliding:     12*12 + 8 + 8   = 160 evaluations  (57x fewer)
print(f"Slider build evaluations: {slider.total_build_evals}")
```

## Derivatives

The slider supports analytical derivatives through its slides. Only the slide containing the differentiated dimension contributes:

\[
\frac{\partial}{\partial x_j} f(\mathbf{x}) \approx \frac{\partial}{\partial x_j} s_i(\mathbf{x}_{G_i})
\]

where \(j \in G_i\). The pivot value \(v\) is constant and drops out.

```python
# Multiple derivatives at once
results = slider.eval_multi(
    [0.5, 0.3, -0.2],
    [
        [0, 0, 0],  # function value
        [1, 0, 0],  # d/dx0
        [0, 1, 0],  # d/dx1
        [0, 0, 1],  # d/dx2
    ],
)
```

## Limitations

### Cross-group derivatives are zero

Because slides are independent functions of disjoint variable groups, **mixed partial derivatives across groups are exactly zero**. For example, with partition `[[0, 1], [2]]`:

- \(\frac{\partial^2 f}{\partial x_0 \partial x_1}\) — computed within the `[0, 1]` slide (correct)
- \(\frac{\partial^2 f}{\partial x_0 \partial x_2}\) — returns 0 (x₀ and x₂ are in different slides)

This is mathematically correct for the sliding approximation, but may differ from the true function's cross-derivatives. If cross-group sensitivities matter, group those variables together or use full tensor interpolation.

### Accuracy degrades far from pivot

The sliding approximation is most accurate near the pivot point. As the evaluation point moves away from the pivot in multiple dimensions simultaneously, cross-coupling errors accumulate. For strongly coupled functions like Black-Scholes, this can produce 20-50% errors at domain boundaries.

## API Reference

::: pychebyshev.slider.ChebyshevSlider
    options:
      show_source: false
      docstring_style: numpy
      members_order: source
