# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinRegressor is a research project focused on **numerical methods for option pricing**, specifically comparing and accelerating Black-Scholes PDE solutions. The project explores:

1. **Analytical solutions** (closed-form Black-Scholes formulas)
2. **Numerical PDE methods** (finite difference methods)
3. **Chebyshev acceleration techniques** (for high-dimensional parametric pricing)

## Development Environment

- **Python Version**: 3.13+
- **Package Manager**: uv (modern Python package manager)
- **Virtual Environment**: `.venv/` (managed by uv)

## Common Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Run the FDM convergence study (main demo)
uv run python fdm_baseline.py
```

### Key Example Scripts
```bash
# Finite difference convergence study (accuracy vs speed)
uv run python fdm_baseline.py

# Main entry point (minimal example)
uv run python main.py
```

### Dependency Management
```bash
# Add a new dependency
uv add <package-name>

# Update dependencies
uv lock --upgrade
```

## Project Structure & Architecture

### Core Implementation Files

**`fdm_baseline.py`** - Main implementation file
- `BlackScholesFDM` class: Finite difference solver for Black-Scholes PDE
  - Implements implicit Euler (unconditionally stable)
  - Implements Crank-Nicolson scheme (2nd order accurate)
  - Calculates Greeks via finite differences (Delta, Gamma, Theta, Vega, Rho)
- `compare_with_analytical()`: Validates FDM against analytical formulas
- `convergence_study()`: Demonstrates accuracy/speed trade-off with varying grid resolutions

**Key Methods**:
- `solve_implicit()`: Backward Euler scheme (1st order in time, 2nd in space)
- `solve_crank_nicolson()`: Crank-Nicolson scheme (2nd order in time and space)
- Grid-based Greek calculations using central differences

### Research Documentation

**`CHEBYSHEV_ACCELERATION.md`** - Comprehensive research on Chebyshev methods
- **Part 1**: Basic Chebyshev methods for option pricing
  - Parametric interpolation (offline/online phases)
  - Dynamic Chebyshev method for American options
  - Super-time-stepping for PDE acceleration
  - Spectral/pseudo-spectral methods
- **Part 2**: Advanced Chebyshev tensor methods
  - Tensor Train (TT) format for high-dimensional problems
  - TT-cross approximation algorithm
  - Tensor completion techniques
  - Applications: multi-asset options, XVA, initial margin (SIMM)
  - Performance benchmarks from academic literature
- **Part 3**: Detailed analysis of key arXiv papers
  - arXiv:1505.04648 - Parametric option pricing
  - arXiv:1805.00898 - Ultra-efficient risk calculations
  - arXiv:1902.04367 - Low-rank tensor approximations
  - arXiv:1808.08221 - Dynamic initial margin via tensors
  - arXiv:2103.01934 - High-dimensional Bermudan options

**`BLACKSCHOLES_LIBRARY_ANALYSIS.md`** - Analysis of the `blackscholes` library
- Confirms it uses analytical closed-form formulas (NOT numerical PDE)
- Documents all Greeks calculations (up to 3rd order)
- Performance: ~17,000x faster than FDM but limited to European options

**`FDM_COMPARISON_RESULTS.md`** - Validation results
- FDM vs analytical comparison: < 0.1% error on price
- Performance benchmarks
- Trade-offs between methods

**`verify_fdm_math.md`** - Mathematical verification
- Complete derivation of finite difference discretization
- Proof that implementation is mathematically correct
- Validation of boundary conditions, terminal conditions, Greeks formulas

## Key Dependencies

- **blackscholes** (0.2.0+): Analytical Black-Scholes formulas (ground truth reference)
- **numpy**: Array operations, grid setup
- **scipy**: Sparse matrix operations (`scipy.sparse.diags`, `scipy.sparse.linalg.spsolve`)
- **matplotlib**: Visualization (for future plotting)
- **pandas**: Data manipulation (for future analysis)

## Architecture & Design Principles

### Three-Tier Approach

1. **Analytical Layer** (`blackscholes` library)
   - Instant pricing (microseconds)
   - Exact results (machine precision)
   - Limited to European vanilla options

2. **Numerical PDE Layer** (`fdm_baseline.py`)
   - Flexible for exotic features
   - Accurate baseline (~0.03% error with fine grids)
   - Slow (~1000x slower than analytical)
   - Used for validation and options without analytical formulas

3. **Acceleration Layer** (documented in `CHEBYSHEV_ACCELERATION.md`)
   - Chebyshev interpolation for parametric pricing
   - Tensor methods for high-dimensional problems
   - Combines analytical speed with numerical flexibility

### Important Implementation Details

**Grid Configuration in FDM**:
- Space grid: `S_grid = np.linspace(0, S_max, M+1)` where `S_max = 3*K`
- Time grid: `t_grid = np.linspace(0, T, N+1)`
- Typical resolution: M=200-800, N=2000-8000 depending on accuracy needs

**PDE Discretization**:
- Uses implicit scheme (unconditionally stable)
- Coefficients: `α = 0.5·dt·(σ²i² - (r-q)i)`, `β = -1 - dt·(σ²i² + r)`, `γ = 0.5·dt·(σ²i² + (r-q)i)`
- Solves tridiagonal system: `A·V^j = b` using scipy's sparse solver

**Greeks Calculation**:
- Delta: Central difference `(V[i+1] - V[i-1])/(2·dS)`
- Gamma: Second-order central `(V[i+1] - 2·V[i] + V[i-1])/dS²`
- Theta: Forward difference in time
- Vega/Rho: Finite difference requiring PDE re-solve with perturbed parameters

### Convergence & Accuracy

From `convergence_study()` results:
- **Coarse grid** (M=50, N=500): ~0.6% price error, 0.09s runtime
- **Fine grid** (M=200, N=2000): ~0.04% price error, 0.5s runtime
- **Ultra fine** (M=800, N=8000): ~0.002% price error, 6s runtime
- Trade-off: 69× slower gives 273× better accuracy

**Error scaling**: O(dt, dS²) truncation error

## Validation Strategy

All numerical methods are validated against analytical Black-Scholes formulas:
1. Run FDM solver with fine grid
2. Compare price, Delta, Gamma to `blackscholes` library
3. Verify errors are within expected bounds (< 2.5% for Greeks, < 0.1% for price)
4. Check convergence: errors decrease as grid is refined

## When to Use Each Method

**Use Analytical** (`blackscholes` library):
- European vanilla options only
- Need instant results
- Parameter sweeps (thousands of evaluations)

**Use FDM** (`fdm_baseline.py`):
- American options (early exercise)
- Barriers, exotics
- Research/validation
- When analytical formula doesn't exist

**Use Chebyshev** (future implementation):
- Repeated evaluations with varying parameters
- High-dimensional problems (multi-asset, many parameters)
- Real-time risk systems
- XVA/regulatory capital calculations

## Research References

The project is based on extensive research documented in `CHEBYSHEV_ACCELERATION.md`:
- 7+ key arXiv papers analyzed
- Production methods from major financial institutions
- Speedups of 10,000× - 40,000× demonstrated in literature
- Applications: parametric pricing, XVA, SIMM, volatility calibration
