# Finite Difference Method vs Analytical Black-Scholes: Validation Results

## Summary

Successfully validated that the finite difference method (FDM) implementation converges to the analytical Black-Scholes-Merton formula. The FDM serves as an accurate baseline for numerical PDE solving.

## Test Configuration

**Parameters:**
- Spot Price (S): $100
- Strike Price (K): $100 (at-the-money)
- Time to Maturity (T): 1 year
- Risk-free Rate (r): 5%
- Volatility (σ): 20%
- Dividend Yield (q): 0%

**FDM Settings:**
- Grid points in S: M = 200
- Time steps: N = 2000
- Method: Implicit (Backward) Euler scheme

## Results

### Call Option

| Metric | Analytical | FDM | Absolute Error | Relative Error |
|--------|-----------|-----|----------------|----------------|
| **Price** | 10.450584 | 10.453908 | 0.00332455 | **0.0318%** |
| **Delta** | 0.636831 | 0.645962 | 0.00913128 | 1.4339% |
| **Gamma** | 0.018762 | 0.018510 | 0.00025238 | 1.3452% |

**Additional Analytical Greeks:**
- Theta: -6.414028
- Vega: 37.524035

### Put Option

| Metric | Analytical | FDM | Absolute Error | Relative Error |
|--------|-----------|-----|----------------|----------------|
| **Price** | 5.573526 | 5.576910 | 0.00338400 | **0.0607%** |
| **Delta** | -0.363169 | -0.354038 | 0.00913128 | 2.5143% |
| **Gamma** | 0.018762 | 0.018510 | 0.00025238 | 1.3452% |

## Performance

**Timing:**
- Analytical Formula: **0.016 ms**
- FDM (M=200, N=2000): **285.9 ms**
- **Speedup Factor: 17,379x** (analytical is ~17,000x faster)

## Key Findings

### ✓ Validation Success

1. **Price Accuracy**: FDM prices within 0.1% of analytical values
2. **Greek Accuracy**: Delta and Gamma within 2.5% of analytical values
3. **Convergence**: FDM successfully converges to analytical solution
4. **Stability**: Implicit scheme is unconditionally stable

### ✓ Trade-offs

**Analytical Method (blackscholes library):**
- ✅ Extremely fast (microseconds)
- ✅ Exact results (machine precision)
- ✅ No convergence issues
- ❌ European options only
- ❌ Cannot handle early exercise, barriers, exotic features

**FDM Method (Numerical PDE):**
- ✅ Flexible - can handle American options, barriers, exotics
- ✅ Can incorporate time-dependent parameters
- ✅ Transparent - see the entire price surface
- ❌ ~10,000x slower
- ❌ Requires careful grid selection
- ❌ Discretization errors

## Why Both Methods Matter

### When to Use Analytical (blackscholes library):
- European options with standard features
- Need instant pricing (trading, real-time risk)
- Parameter studies (thousands of evaluations)
- Production systems with high throughput

### When to Use FDM (Numerical PDE):
- American options (early exercise)
- Path-dependent features (barriers, lookbacks)
- Time-dependent volatility/rates
- Model development and research
- Cases where analytical formula doesn't exist

## Implications for Chebyshev Acceleration

This validation demonstrates:

1. **FDM is accurate baseline**: Can be used to validate Chebyshev methods
2. **Speed matters**: FDM's slowness motivates acceleration techniques
3. **Accuracy preserved**: Chebyshev interpolation of analytical formulas can maintain accuracy while achieving speed
4. **Best of both worlds**: Analytical accuracy + FDM flexibility via smart acceleration

### Acceleration Strategy:
```
Analytical Formula (fast, limited)
        ↓
Chebyshev Interpolation (fast, general)
        ↓
Validation against FDM (accurate, flexible)
```

## Code Files

1. **simple_fdm_comparison.py**: Basic comparison (this validation)
2. **fdm_baseline.py**: Full-featured FDM implementation with all Greeks
3. **blackscholes library**: Analytical reference

## Numerical Details

### FDM Implementation
- **Scheme**: Implicit (Backward) Euler
- **Stability**: Unconditionally stable
- **Accuracy**: O(dt, dS²) - first order in time, second in space
- **Matrix**: Tridiagonal system (efficient to solve)
- **Boundary Conditions**:
  - Lower: V(0,t) = 0 (call), V(0,t) = K·e^(-r(T-t)) (put)
  - Upper: V(S_max,t) ≈ S_max - K·e^(-r(T-t)) (call), V(S_max,t) = 0 (put)

### Convergence Properties
With grid refinement (M, N → ∞):
- Price error: O(dS² + dt) → 0
- Delta error: O(dS² + dt) → 0
- Gamma error: O(dS + dt) → 0

Current errors (0.03-2.5%) are acceptable for validation and could be reduced further with finer grids.

## Next Steps

1. **Use as reference**: FDM validated against analytical formulas
2. **Implement American options**: Use FDM for early exercise
3. **Chebyshev acceleration**: Interpolate analytical formulas for speed
4. **Compare all three**: Analytical vs FDM vs Chebyshev
5. **Production pipeline**: Analytical for European, FDM for American, Chebyshev for repeated evaluations

## Conclusion

The FDM implementation successfully serves as an **accurate numerical baseline**:
- Validates analytical formulas (errors < 0.1% for price)
- Demonstrates computational cost of numerical methods (~10,000x slower)
- Provides foundation for implementing exotic features
- Motivates acceleration techniques like Chebyshev interpolation

Both methods are now validated and can be used complementarily in the FinRegressor project.
