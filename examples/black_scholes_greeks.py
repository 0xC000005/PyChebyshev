"""5D Black-Scholes example: price + all Greeks via Chebyshev interpolation."""

import math

from scipy.stats import norm

from pychebyshev import ChebyshevApproximation


def black_scholes_call(x, _):
    """Analytical Black-Scholes call price."""
    S, K, T, sigma, r = x
    q = 0.02  # dividend yield
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# Build 5D interpolant
cheb = ChebyshevApproximation(
    black_scholes_call,
    num_dimensions=5,
    domain=[[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
    n_nodes=[11, 11, 11, 11, 11],
)
cheb.build()

# Query point: ATM option
point = [100, 100, 1.0, 0.25, 0.05]

# Compute price + all Greeks in one call
price, delta, gamma, vega, rho = cheb.vectorized_eval_multi(point, [
    [0, 0, 0, 0, 0],  # Price
    [1, 0, 0, 0, 0],  # Delta = dV/dS
    [2, 0, 0, 0, 0],  # Gamma = d²V/dS²
    [0, 0, 0, 1, 0],  # Vega  = dV/dσ
    [0, 0, 0, 0, 1],  # Rho   = dV/dr
])

print("5D Black-Scholes at ATM (S=K=100, T=1, σ=0.25, r=0.05)")
print(f"  Price: {price:.6f}")
print(f"  Delta: {delta:.6f}")
print(f"  Gamma: {gamma:.6f}")
print(f"  Vega:  {vega:.6f}")
print(f"  Rho:   {rho:.6f}")
