"""Quick start example: approximate a 2D function and compute derivatives."""

import math

from pychebyshev import ChebyshevApproximation


def f(x, _):
    """A smooth 2D function: sin(x) * exp(-y)."""
    return math.sin(x[0]) * math.exp(-x[1])


# Build interpolant
cheb = ChebyshevApproximation(
    f,
    num_dimensions=2,
    domain=[[-3, 3], [0, 2]],
    n_nodes=[15, 15],
)
cheb.build()

# Evaluate at a test point
point = [1.0, 0.5]
exact = f(point, None)
approx = cheb.vectorized_eval(point, [0, 0])

print(f"Exact:  {exact:.10f}")
print(f"Approx: {approx:.10f}")
print(f"Error:  {abs(approx - exact):.2e}")

# Derivative df/dx
dfdx_exact = math.cos(point[0]) * math.exp(-point[1])
dfdx_approx = cheb.vectorized_eval(point, [1, 0])
print(f"\ndf/dx exact:  {dfdx_exact:.10f}")
print(f"df/dx approx: {dfdx_approx:.10f}")
print(f"df/dx error:  {abs(dfdx_approx - dfdx_exact):.2e}")
