# Mathematical Concepts

## Why Chebyshev Interpolation?

Polynomial interpolation with equally-spaced points suffers from **Runge's phenomenon** (Runge 1901) — wild oscillations near interval endpoints that worsen as polynomial degree increases. Chebyshev nodes solve this by clustering near boundaries:

$$x_i = \cos\left(\frac{(2i-1)\pi}{2n}\right), \quad i = 1, \ldots, n$$

The Lebesgue constant for Chebyshev nodes grows only logarithmically: $\Lambda_n \leq \frac{2}{\pi}\log(n+1) + 1$ (Trefethen 2013, Ch. 15), versus exponential growth for equidistant points.

## Spectral Convergence

For functions analytic in a **Bernstein ellipse** with parameter $\rho > 1$, the interpolation error decays exponentially:

$$|f(x) - p_N(x)| = O(\rho^{-N})$$

Each additional node multiplies accuracy by a constant factor $\rho$.

### Bernstein ellipse

A Bernstein ellipse is an ellipse in the complex plane with foci at $x = -1$ and
$x = +1$. The parameter $\rho$ equals the sum of the semi-major and semi-minor axis
lengths. Functions analytic inside a larger ellipse (larger $\rho$) converge faster.

**Practical implication:** The convergence rate depends on how far the function's
nearest singularity (pole, branch cut, discontinuity) is from the real interval
$[-1, 1]$ in the complex plane. For example:

- $f(x) = e^x$ is entire (no singularities) -- $\rho = \infty$, superexponential convergence.
- $f(x) = 1/(1 + 25x^2)$ has poles at $x = \pm i/5$ -- the Bernstein ellipse must
  avoid these poles, limiting $\rho$ and slowing convergence.
- Black-Scholes option prices are analytic in all parameters over typical domains,
  giving large $\rho$ and rapid convergence with 10--15 nodes per dimension.

For the full theory, see Trefethen (2013), *Approximation Theory and Approximation
Practice*, SIAM, Chapter 8.

## Barycentric Interpolation Formula

The interpolating polynomial is expressed in the **second-form barycentric formula**
(Berrut & Trefethen 2004):

$$p(x) = \frac{\sum_{i=0}^{n} \frac{w_i f_i}{x - x_i}}{\sum_{i=0}^{n} \frac{w_i}{x - x_i}}$$

where the barycentric weights $w_i = 1 / \prod_{j \neq i}(x_i - x_j)$ depend **only on node positions**, not on function values. This enables full pre-computation.

## Multi-Dimensional Extension

For a $d$-dimensional function, PyChebyshev uses **dimensional decomposition**:

1. Start with the full tensor of function values at all node combinations
2. Contract one dimension at a time using barycentric interpolation
3. Each contraction reduces dimensionality by 1 (5D → 4D → ... → scalar)

This avoids the curse of dimensionality in the evaluation step — query cost scales linearly with the number of dimensions.

## Analytical Derivatives

Derivatives are computed using **spectral differentiation matrices** $D^{(k)}$
(Berrut & Trefethen 2004, §9):

$$D^{(1)}_{ij} = \frac{w_j / w_i}{x_i - x_j} \quad (i \neq j), \qquad D^{(1)}_{ii} = -\sum_{k \neq i} D^{(1)}_{ik}$$

Given function values $\mathbf{f}$ at nodes, $D^{(1)} \mathbf{f}$ gives exact derivative values at those same nodes. These derivative values are then interpolated to the query point using the barycentric formula.

## References

- Berrut, J.-P. & Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation."
  *SIAM Review* 46(3):501--517.
- Runge, C. (1901). "Über empirische Funktionen und die Interpolation zwischen
  äquidistanten Ordinaten." *Zeitschrift für Mathematik und Physik* 46:224--243.
- Trefethen, L. N. (2013). *Approximation Theory and Approximation Practice.*
  SIAM.
