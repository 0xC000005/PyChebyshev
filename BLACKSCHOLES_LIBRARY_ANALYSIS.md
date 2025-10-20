# BlackScholes Library Implementation Analysis

## Summary

The `blackscholes` Python library (version 0.2.0+) uses **analytical closed-form formulas**, NOT numerical PDE methods. It implements the classic Black-Scholes-Merton analytical solution for European options.

## Implementation Details

### Core Pricing Method

The library computes option prices using the **closed-form Black-Scholes-Merton formula**:

**Call Option Price** (`call.py:24-29`):
```python
def price(self) -> float:
    """Fair value of Black-Scholes call option."""
    return (
        self.S * exp(-self.q * self.T) * self._cdf(self._d1)
        - self._cdf(self._d2) * exp(-self.r * self.T) * self.K
    )
```

This is the standard analytical formula:
```
C = S·e^(-q·T)·N(d₁) - K·e^(-r·T)·N(d₂)
```

Where:
- S = spot price
- K = strike price
- T = time to maturity
- r = risk-free rate
- q = dividend yield
- N(·) = cumulative normal distribution function

### d₁ and d₂ Calculation

The probability parameters are computed analytically (`base.py:268-277`):

```python
@property
def _d1(self) -> float:
    """1st probability factor that acts as a multiplication factor for stock prices."""
    return (1.0 / (self.sigma * sqrt(self.T))) * (
        log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
    )

@property
def _d2(self) -> float:
    """2nd probability parameter that acts as a multiplication factor for discounting."""
    return self._d1 - self.sigma * sqrt(self.T)
```

Standard formulas:
```
d₁ = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

### Normal Distribution Functions

The library implements standard normal PDF and CDF (`base.py:12-19`):

```python
@staticmethod
def _pdf(x: float) -> float:
    """PDF of standard normal distribution."""
    return exp(-(x**2) / 2.0) / sqrt(2.0 * pi)

@staticmethod
def _cdf(x):
    """CDF of standard normal distribution."""
    return (1.0 + erf(x / sqrt(2.0))) / 2.0
```

Uses the **error function** (erf) from Python's math library for the CDF:
```
N(x) = [1 + erf(x/√2)] / 2
```

This is a fast, accurate approximation - no numerical integration required.

### Greeks Calculation

All Greeks are computed using **analytical derivatives** of the pricing formula:

**Example - Delta** (`call.py:31-37`):
```python
def delta(self) -> float:
    """Rate of change in option price
    with respect to the forward price (1st derivative).
    """
    return exp(-self.q * self.T) * self._cdf(self._d1)
```

**Example - Gamma** (`base.py:76-84`):
```python
def gamma(self) -> float:
    """Rate of change in delta with respect to the underlying asset price (2nd derivative)."""
    return (
        exp(-self.q * self.T)
        * self._pdf(self._d1)
        / (self.S * self.sigma * sqrt(self.T))
    )
```

**Example - Vega** (`base.py:96-100`):
```python
def vega(self) -> float:
    """Rate of change in option price with respect to the volatility of the asset."""
    return self.S * self._pdf(self._d1) * sqrt(self.T)
```

All Greeks up to **3rd order** are implemented using analytical formulas:
- 1st order: Delta, Vega, Theta, Rho, Epsilon
- 2nd order: Gamma, Vanna, Vomma, Charm
- 3rd order: Speed, Zomma, Color, Ultima

## What This Library Does NOT Do

1. ❌ **No PDE solving**: Does not discretize or solve the Black-Scholes PDE numerically
2. ❌ **No finite differences**: Does not use explicit/implicit schemes, Crank-Nicolson, etc.
3. ❌ **No iterative methods**: No time-stepping, no grid-based methods
4. ❌ **No Monte Carlo**: Pure analytical formulas only
5. ❌ **No numerical integration**: CDF uses error function, not quadrature
6. ❌ **No root finding**: Doesn't solve for implied volatility iteratively (though this could be added)

## Computational Characteristics

### Performance
- **Extremely fast**: Each pricing call is O(1) - constant time
- **No convergence issues**: Analytical formulas always return exact result
- **No stability concerns**: No numerical discretization errors
- **Machine precision**: Limited only by floating-point arithmetic (~10⁻¹⁵ relative error)

### Limitations
- **European options only**: Cannot price American options (no early exercise)
- **Single underlying**: No multi-asset basket options
- **Constant parameters**: Assumes constant volatility, interest rate
- **No path dependence**: Cannot handle barriers, Asian options, etc.
- **Model assumptions**: Requires Black-Scholes assumptions (log-normal prices, etc.)

## When to Use Numerical Methods Instead

You would need **numerical PDE methods** or other techniques for:

1. **American Options**
   - Early exercise requires backward induction
   - Methods: Finite differences, binomial trees, LSM Monte Carlo

2. **Path-Dependent Options**
   - Barriers, Asian options, lookbacks
   - Methods: Monte Carlo, PDE with additional state variables

3. **Multi-Asset Options**
   - Basket options, worst-of, spread options
   - Methods: Multi-dimensional PDE, Monte Carlo, Chebyshev tensors

4. **Non-Constant Parameters**
   - Time-dependent volatility or rates
   - Stochastic volatility (Heston, SABR)
   - Methods: Numerical PDE, characteristic functions, Monte Carlo

5. **Non-Log-Normal Models**
   - Jump-diffusion, Variance Gamma, rough volatility
   - Methods: Monte Carlo, Fourier methods, numerical PDE

6. **Exotic Payoffs**
   - Digitals (can be done analytically but sensitive)
   - Complex payoff structures
   - Methods depends on features

## Relationship to Chebyshev Acceleration Research

The `blackscholes` library provides **baseline analytical pricing** that could be:

1. **Accelerated with Chebyshev interpolation** for parametric pricing
   - Pre-compute at Chebyshev nodes in (S, K, T, σ) space
   - Interpolate for fast repeated evaluations
   - Useful when pricing thousands of options

2. **Used as reference for validation**
   - Numerical methods should converge to analytical solutions
   - Greeks can be compared for accuracy testing

3. **Extended to handle cases beyond analytical formulas**
   - American options: Dynamic Chebyshev method
   - Multi-dimensional: Chebyshev tensors
   - Complex models: Chebyshev + PDE methods

## Conclusion

The `blackscholes` library is a **pure analytical implementation** that:
- Evaluates closed-form formulas instantly
- Provides exact results (up to floating-point precision)
- Suitable for European vanilla options under Black-Scholes assumptions
- Does NOT solve PDEs numerically or use iterative methods

For problems requiring numerical methods (American options, path dependence, exotic models), you would need different libraries or implement custom solvers, potentially using the Chebyshev acceleration techniques documented in `CHEBYSHEV_ACCELERATION.md`.

## Library Information

- **Package**: blackscholes
- **PyPI**: https://pypi.org/project/blackscholes/
- **GitHub**: https://github.com/CarloLepelaars/blackscholes
- **Version in project**: 0.2.0+
- **License**: Open source
- **Author**: Carlo Lepelaars
- **Key Features**: Up to 3rd order Greeks, option strategies, Black-76 variant
