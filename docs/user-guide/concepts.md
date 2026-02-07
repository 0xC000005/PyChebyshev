# Mathematical Concepts

## Why Chebyshev Interpolation?

Polynomial interpolation with equally-spaced points suffers from **Runge's phenomenon** — wild oscillations near interval endpoints that worsen as polynomial degree increases. Chebyshev nodes solve this by clustering near boundaries:

$$x_i = \cos\left(\frac{(2i-1)\pi}{2n}\right), \quad i = 1, \ldots, n$$

The Lebesgue constant for Chebyshev nodes grows only logarithmically: $\Lambda_n \leq \frac{2}{\pi}\log(n+1) + 1$, versus exponential growth for equidistant points.

## Spectral Convergence

For functions analytic in a **Bernstein ellipse** with parameter $\rho > 1$, the interpolation error decays exponentially:

$$|f(x) - p_N(x)| = O(\rho^{-N})$$

Each additional node multiplies accuracy by a constant factor $\rho$.

## Barycentric Interpolation Formula

The interpolating polynomial is expressed as:

$$p(x) = \frac{\sum_{i=0}^{n} \frac{w_i f_i}{x - x_i}}{\sum_{i=0}^{n} \frac{w_i}{x - x_i}}$$

where the barycentric weights $w_i = 1 / \prod_{j \neq i}(x_i - x_j)$ depend **only on node positions**, not on function values. This enables full pre-computation.

## Multi-Dimensional Extension

For a $d$-dimensional function, PyChebyshev uses **dimensional decomposition**:

1. Start with the full tensor of function values at all node combinations
2. Contract one dimension at a time using barycentric interpolation
3. Each contraction reduces dimensionality by 1 (5D → 4D → ... → scalar)

This avoids the curse of dimensionality in the evaluation step — query cost scales linearly with the number of dimensions.

## Analytical Derivatives

Derivatives are computed using **spectral differentiation matrices** $D^{(k)}$:

$$D^{(1)}_{ij} = \frac{w_j / w_i}{x_i - x_j} \quad (i \neq j), \qquad D^{(1)}_{ii} = -\sum_{k \neq i} D^{(1)}_{ik}$$

Given function values $\mathbf{f}$ at nodes, $D^{(1)} \mathbf{f}$ gives exact derivative values at those same nodes. These derivative values are then interpolated to the query point using the barycentric formula.
