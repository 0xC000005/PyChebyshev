"""Shared test fixtures for PyChebyshev tests."""

import math

import pytest

from pychebyshev import ChebyshevApproximation, ChebyshevSpline, ChebyshevTT


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


# ---------------------------------------------------------------------------
# TT Fixtures
# ---------------------------------------------------------------------------

def _bs_5d_func(x, _):
    """5D BS call price: V(S, K, T, sigma, r), q=0.02."""
    return _bs_call_price(S=x[0], K=x[1], T=x[2], r=x[4], sigma=x[3], q=0.02)


_TT_5D_BS_DOMAIN = [[80, 120], [90, 110], [0.25, 1.0], [0.15, 0.35], [0.01, 0.08]]
_TT_5D_BS_NODES = [11, 11, 11, 11, 11]


@pytest.fixture(scope="module")
def tt_sin_3d():
    """Pre-built 3D sin ChebyshevTT (TT-Cross)."""
    tt = ChebyshevTT(
        sin_sum_3d, 3, [[-1, 1], [-1, 1], [-1, 1]], [11, 11, 11],
        max_rank=5,
    )
    tt.build(verbose=False, seed=42)
    return tt


@pytest.fixture(scope="module")
def tt_bs_5d():
    """Pre-built 5D Black-Scholes ChebyshevTT (TT-Cross)."""
    tt = ChebyshevTT(
        _bs_5d_func, 5, _TT_5D_BS_DOMAIN, _TT_5D_BS_NODES,
        max_rank=15, max_sweeps=5,
    )
    tt.build(verbose=False, seed=42)
    return tt


@pytest.fixture(scope="module")
def tt_sin_3d_svd():
    """Pre-built 3D sin ChebyshevTT (TT-SVD)."""
    tt = ChebyshevTT(
        sin_sum_3d, 3, [[-1, 1], [-1, 1], [-1, 1]], [11, 11, 11],
        max_rank=5,
    )
    tt.build(verbose=False, method="svd")
    return tt


# ---------------------------------------------------------------------------
# Spline Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spline_abs_1d():
    """Pre-built 1D |x| with knot at x=0."""
    def f(x, _):
        return abs(x[0])

    sp = ChebyshevSpline(f, 1, [[-1, 1]], [15], [[0.0]])
    sp.build(verbose=False)
    return sp


@pytest.fixture(scope="module")
def spline_bs_2d():
    """Pre-built 2D discounted call payoff max(S-K,0)*exp(-rT) with knot at K=100."""
    def f(x, _):
        return max(x[0] - 100.0, 0.0) * math.exp(-0.05 * x[1])

    sp = ChebyshevSpline(f, 2, [[80, 120], [0.25, 1.0]], [15, 15], [[100.0], []])
    sp.build(verbose=False)
    return sp
