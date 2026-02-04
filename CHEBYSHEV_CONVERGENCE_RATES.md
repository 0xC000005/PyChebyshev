# Chebyshev Convergence Rates: Analytic vs Non-Analytic Functions

## Summary

Chebyshev interpolation convergence rates depend critically on function smoothness. While analytic functions achieve exponential convergence (spectral accuracy), non-analytic functions exhibit algebraic convergence rates proportional to their differentiability.

## Convergence Hierarchy

### 1. Analytic Functions: Exponential Convergence

**Rate**: O(ρ^(-N)) where ρ > 1 is the Bernstein ellipse parameter

If function f is analytic and bounded in a Bernstein ellipse, Chebyshev interpolants converge geometrically [1]:

$$|f(x) - p_N(x)| = O(\rho^{-N})$$

**Example**: Black-Scholes pricing achieves machine precision (~10^-16) with 10-15 nodes per dimension.

### 2. Smooth Non-Analytic Functions: Algebraic Convergence

**Rate**: O(N^(-k)) where k = number of continuous derivatives

For functions with k continuous derivatives (and k-th derivative of bounded variation), convergence follows algebraic rate [2, 3]:

$$|f(x) - p_N(x)| = O(N^{-k})$$

This result is attributed to Mastroianni & Szabados (1995) and established through integration by parts, showing Chebyshev coefficients decay at rate O(n^(-k-1)) [3].

**Examples** [4]:
- |x|^5 (C^5 continuous): convergence rate O(N^-5)
- |x|^π (C^π continuous): convergence rate O(N^-π)
- sin(|x|^5.5) (C^5.5 continuous): convergence rate O(N^-5.5)

**Practical implication**: Reducing error by 10× requires ~10× more nodes (algebraic) versus ~3 more nodes for analytic functions (exponential).

### 3. Piecewise Smooth Functions: Degraded Convergence

For functions continuous on [-1,1] with finite interior discontinuities in derivatives:
- Pointwise convergence at continuous points [5]
- Convergence severely slowed by discontinuities [4]
- Coefficients decay slowly; spectral convergence lost

**Solution**: Domain splitting at discontinuities, using separate Chebyshev interpolants on each smooth piece [6].

### 4. Discontinuous Functions: Gibbs Phenomenon

**Rate**: No uniform convergence

For functions with jump discontinuities (e.g., sign(x), step functions):
- No convergence in maximum norm [5]
- Persistent overshoot near discontinuities (~9% overshoot, converging to 1.2823 for unit jump) [4]
- Pointwise convergence to f(x) where continuous, and to ½(f(x⁻) + f(x⁺)) at jumps [5]

## Comparison Table

| Function Class | Example | Convergence Rate | Nodes for 0.01% Error |
|----------------|---------|------------------|----------------------|
| Analytic | e^x, Black-Scholes | O(ρ^-N) | ~10-15 |
| C^5 smooth | \|x\|^5 | O(N^-5) | ~100 |
| C^2 smooth | \|x\|^2 | O(N^-2) | ~1,000 |
| C^0 continuous | \|x\| | O(N^-1) | ~10,000 |
| Discontinuous | sign(x) | None (Gibbs) | ∞ |

## Implications for Option Pricing

**European options** (Black-Scholes): Analytic everywhere → spectral convergence achieved with modest node counts (11^5 = 161,051 nodes achieves 0.000% error in this work).

**American options**: Early exercise boundary creates non-smoothness → algebraic convergence at best. Consider:
- Domain splitting at exercise boundary
- Regularization techniques
- Alternative methods (FDM, adaptive mesh refinement)

**Barrier/Digital options**: Discontinuous payoffs → Gibbs phenomenon. Require special treatment (Gegenbauer reconstruction, domain splitting, or non-polynomial methods).

## Key Insight

Chebyshev interpolation is **ideally suited for analytic functions** like European option pricing, where it achieves near-machine precision with reasonable computational cost. For non-analytic functions, convergence degrades gracefully but predictably according to smoothness class.

## References

[1] Trefethen, L. N. (2019). *Approximation Theory and Approximation Practice, Extended Edition*. SIAM. Chapter 8: Convergence for Analytic Functions.

[2] Trefethen, L. N. (2019). *Approximation Theory and Approximation Practice, Extended Edition*. SIAM. Chapter 7: Convergence for Differentiable Functions.

[3] Trefethen, L. N. (2017). "Lecture 3: Chebyshev Series." Oxford University. Available: https://people.maths.ox.ac.uk/trefethen/outline3_2017.pdf

[4] "4. Chebfun and Approximation Theory." *Chebfun Guide*. Available: https://www.chebfun.org/docs/guide/guide04.html

[5] "Convergence Rates for Interpolating Functions." *Chebfun Examples*. Available: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/23972/versions/22/previews/chebfun/examples/approx/html/Convergence.html

[6] Driscoll, T. A. & Hale, N. (2014). "Optimal Domain Splitting for Interpolation by Chebyshev Polynomials." Available: https://tobydriscoll.net/_docs/driscoll-optimal-domain-splitting-2014.pdf

---

*Last updated: 2025-10-27*
