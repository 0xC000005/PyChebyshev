"""Shared test fixtures for PyChebyshev tests."""

import math

import pytest

from pychebyshev import ChebyshevApproximation


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def sin_sum_3d(x, _):
    """sin(x) + sin(y) + sin(z)"""
    return math.sin(x[0]) + math.sin(x[1]) + math.sin(x[2])


def _bs_call_price(S, K, T, r, sigma, q=0.0):
    """Analytical Black-Scholes call price (no external dependency)."""
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_call_delta(S, K, T, r, sigma, q=0.0):
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.cdf(d1)


def _bs_call_gamma(S, K, T, r, sigma, q=0.0):
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def _bs_call_vega(S, K, T, r, sigma, q=0.0):
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)


def _bs_call_rho(S, K, T, r, sigma, q=0.0):
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * T * math.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cheb_sin_3d():
    """Pre-built 3D sin(x)+sin(y)+sin(z) interpolant."""
    cheb = ChebyshevApproximation(
        sin_sum_3d, 3, [[-1, 1], [-1, 1], [1, 3]], [10, 8, 4]
    )
    cheb.build(verbose=False)
    return cheb


@pytest.fixture
def cheb_bs_3d():
    """Pre-built 3D Black-Scholes C(S, T, sigma) interpolant."""
    K, r, q = 100.0, 0.05, 0.02

    def bs(x, _):
        return _bs_call_price(S=x[0], K=K, T=x[1], r=r, sigma=x[2], q=q)

    cheb = ChebyshevApproximation(
        bs, 3, [[50, 150], [0.1, 2.0], [0.1, 0.5]], [15, 12, 10]
    )
    cheb.build(verbose=False)
    return cheb


@pytest.fixture
def cheb_bs_5d():
    """Pre-built 5D Black-Scholes V(S, K, T, sigma, r) interpolant."""
    q = 0.02

    def bs_5d(x, _):
        return _bs_call_price(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=q)

    cheb = ChebyshevApproximation(
        bs_5d, 5,
        [[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]],
        [11, 11, 11, 11, 11],
    )
    cheb.build(verbose=False)
    return cheb
