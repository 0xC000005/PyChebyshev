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

# Tensor interpolation demo (5D Chebyshev with CP decomposition)
uv run python chebyshev_tensor_demo.py

# MoCaX library test (requires setup - see MoCaX Setup section below)
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH" && uv run python mocax_test.py

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

### MoCaX Setup (Optional)

MoCaX is a proprietary library for multi-dimensional Chebyshev approximation.

**📖 See [`MOCAX_SETUP_GUIDE.md`](MOCAX_SETUP_GUIDE.md) for complete installation and usage documentation.**

Quick setup:

1. **Extract the Python package** (one-time setup):
```bash
cd MoCaXSuite-1.2.0/MoCaXSuite-1.2.0/MoCaX/Linux/gmake/64bit/Python/MoCaXLibrary
unzip -q mocaxpy-4.3.1.linux-x86_64.zip
mkdir -p mocax_lib
cp -r usr/lib/python2.7/site-packages/mocaxpy mocax_lib/
cp libmocaxc.so mocax_lib/
cp -r mocax_lib /path/to/FinRegressor/
```

2. **Run MoCaX tests**:
```bash
# Set LD_LIBRARY_PATH to find the shared library
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
uv run python mocax_test.py
```

The test script validates MoCaX installation with three comprehensive tests:

**Test 1: Simple 3D Function** (sin(x) + sin(y) + sin(z))
- Validates basic Chebyshev approximation
- 0.024% approximation error
- Analytical derivatives accurate to 1e-7

**Test 2: Black-Scholes 3D** (S, T, σ with fixed K, r)
- 3D parametric pricing
- < 0.01% price error vs analytical
- Delta: 0.001% error, Vega: 1.97% error
- ~6× slower than pure analytical

**Test 3: 5D Parametric Black-Scholes** (S, K, T, σ, r) - **Showcases True Capabilities**
- The challenging case where we used linear interpolation in tensor demo
- 161,051 function evaluations (11^5 nodes) in ~1.5s
- **Results across 14 test cases:**
  - Price accuracy: **0.000% max error** (spectral accuracy!)
  - Delta: 0.000%, Gamma: 0.000%, Rho: 0.000%
  - Vega: 1.98% (vs 7.60% with linear interpolation)
  - Strike sensitivity ∂V/∂K: 0.000% (analytical derivative!)
- **Comparison with our tensor approach:**
  - Linear interpolation: 3.22% mean Vega error, finite difference Greeks
  - MoCaX Chebyshev: 1.98% Vega error, analytical Greeks
  - Demonstrates true polynomial evaluation vs piecewise linear
- **See [`MOCAX_5D_RESULTS.md`](MOCAX_5D_RESULTS.md) for detailed results**

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

**`chebyshev_tensor_demo.py`** - Multi-dimensional tensor interpolation
- `TensorOptionPricer` class: 5D Chebyshev interpolation using CP decomposition
  - Builds full 5D tensor V(S, K, T, σ, r) at Chebyshev nodes
  - Compresses using CP (CANDECOMP/PARAFAC) decomposition
  - Achieves 287× storage reduction with <1% reconstruction error
  - Fast Greeks via finite difference on interpolated surface
- `comprehensive_tensor_comparison()`: Validates tensor interpolation across 19 configurations
- **Performance**: 1-8% error on Greeks, 855× faster than FDM for 1000 queries

**`mocax_test.py`** - MoCaX library integration test
- Tests MoCaX (Multi-dimensional Chebyshev Approximation) library
- Demonstrates Black-Scholes pricing with automatic differentiation for Greeks
- Requires external `mocax_lib/` directory (proprietary library)
- **Note**: MoCaX provides Chebyshev approximation with analytical derivatives (vs numerical for TensorOptionPricer)

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

**`TENSOR_COMPARISON_SUMMARY.md`** - Detailed comparison of interpolation methods
- **Step-by-step explanation** of 1D vs 5D Chebyshev interpolation
- Why 1D interpolation fails when parameters vary (40-108% errors)
- How 5D tensor methods solve the multi-parameter problem (1-8% errors)
- Complete mathematical intuition with concrete examples
- Computational trade-offs and break-even analysis
- **Key insight**: 1D interpolation only works when varying a single parameter at a fixed point; 5D tensor interpolation works across entire parameter space

## Key Dependencies

### Core Numerical Libraries
- **numpy** (2.3.4+): Array operations, grid setup, tensor manipulation
- **scipy** (1.16.2+): Sparse matrix operations (`scipy.sparse.diags`, `scipy.sparse.linalg.spsolve`), interpolation (`RegularGridInterpolator`)
- **pandas** (2.3.3+): Data manipulation and results analysis
- **matplotlib** (3.10.7+): Visualization and plotting

### Option Pricing
- **blackscholes** (0.2.0+): Analytical Black-Scholes formulas (ground truth reference for validation)

### Tensor & Interpolation Methods
- **tensorly** (0.9.0+): CP/Tucker decomposition for multi-dimensional tensor compression
- **teneva** (0.14.9+): Tensor Train methods (alternative to CP for higher dimensions)
- **chebpy** (0.2+): Chebyshev polynomial utilities
- **scikit-fdiff** (0.7.0+): Finite difference schemes for PDEs

### Performance
- **numba** (0.62.1+): JIT compilation for performance-critical loops
- **tqdm** (4.67.1+): Progress bars for long computations

### External (Optional)
- **MoCaX**: Multi-dimensional Chebyshev approximation (proprietary, requires manual installation in `mocax_lib/`)

## Architecture & Design Principles

### Four-Tier Approach

1. **Analytical Layer** (`blackscholes` library)
   - Instant pricing (~10μs)
   - Exact results (machine precision)
   - Limited to European vanilla options
   - **Use case**: Single evaluations, ground truth validation

2. **Tensor Interpolation Layer** (`chebyshev_tensor_demo.py`)
   - 5D Chebyshev interpolation with CP decomposition
   - **Offline phase**: Build compressed tensor in ~1.2s (161,051 evaluations)
   - **Online phase**: Query in ~1ms, Greeks in ~2ms
   - 1-8% error on Greeks across entire parameter space
   - 855× faster than FDM for repeated queries
   - **Use case**: Risk systems, parameter sweeps, portfolio Greeks

3. **Numerical PDE Layer** (`fdm_baseline.py`)
   - Flexible for exotic features (American, barriers, etc.)
   - Accurate baseline (~0.03% error with fine grids)
   - Slow (~500ms per solve)
   - Used for validation and options without analytical formulas
   - **Use case**: One-off exotic pricing, high-accuracy validation

4. **Advanced Acceleration** (documented in `CHEBYSHEV_ACCELERATION.md`)
   - Tensor Train (TT) methods for d>5 dimensions
   - TT-cross adaptive sampling
   - Dynamic tensors for American options
   - **Use case**: Multi-asset options, XVA, regulatory capital

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
- Vega/Rho (FDM): Finite difference requiring PDE re-solve with perturbed parameters (~1s each)
- Vega/Rho (Tensor): Finite difference on interpolated surface (~2ms each, 500× faster!)

**Tensor Interpolation Configuration**:
- Dimensions: 5D → V(S, K, T, σ, r)
- Nodes per dimension: 11 (Chebyshev nodes)
- Total grid: 11^5 = 161,051 points
- CP rank: 10 (controls accuracy vs compression trade-off)
- Compression: 287× (161,051 → 560 parameters)
- Reconstruction error: ~0.5%
- Interpolation method: Multi-linear on regular grid

### Convergence & Accuracy

From `convergence_study()` results:
- **Coarse grid** (M=50, N=500): ~0.6% price error, 0.09s runtime
- **Fine grid** (M=200, N=2000): ~0.04% price error, 0.5s runtime
- **Ultra fine** (M=800, N=8000): ~0.002% price error, 6s runtime
- Trade-off: 69× slower gives 273× better accuracy

**Error scaling**: O(dt, dS²) truncation error

From `chebyshev_tensor_demo.py` tensor interpolation:
- **Price error**: Mean 0.79%, Max 1.94%
- **Vega error**: Mean 3.22%, Max 7.60% (vs 40% for 1D interpolation!)
- **Rho error**: Mean 1.36%, Max 4.28% (vs 109% for 1D interpolation!)
- **Speedup**: 855× faster than FDM for 1000 queries
- **Compression**: 287× storage reduction with 0.56% reconstruction error

## Validation Strategy

All numerical methods are validated against analytical Black-Scholes formulas:
1. Run FDM solver with fine grid
2. Compare price, Delta, Gamma to `blackscholes` library
3. Verify errors are within expected bounds (< 2.5% for Greeks, < 0.1% for price)
4. Check convergence: errors decrease as grid is refined

## When to Use Each Method

**Use Analytical** (`blackscholes` library):
- ✅ European vanilla options only
- ✅ Need instant results (~10μs)
- ✅ Single evaluations or simple parameter sweeps
- ✅ Ground truth validation
- ❌ Don't use for American options or exotics

**Use Tensor Interpolation** (`chebyshev_tensor_demo.py`):
- ✅ **Multiple parameters vary simultaneously** (key advantage!)
- ✅ Need Greeks (Vega, Rho) across diverse scenarios
- ✅ Many queries (>10) to amortize 1.2s offline cost
- ✅ Can tolerate 1-8% error
- ✅ Parameters within interpolation range
- ❌ Don't use for single query (not worth offline cost)
- ❌ Don't use for out-of-range extrapolation

**Use FDM** (`fdm_baseline.py`):
- ✅ American options (early exercise)
- ✅ Barriers, path-dependent options, exotics
- ✅ Need highest accuracy (<0.1% error)
- ✅ Research/validation baseline
- ✅ One-off calculations
- ❌ Don't use for repeated similar queries (use tensor interpolation instead)

**Use Advanced Tensor Methods** (from `CHEBYSHEV_ACCELERATION.md`):
- ✅ Multi-asset options (d>5 dimensions)
- ✅ XVA calculations (CVA, DVA, FVA)
- ✅ Regulatory capital (SIMM)
- ✅ High-dimensional parameter spaces
- ✅ Production risk systems

## Research References

The project is based on extensive research documented in `CHEBYSHEV_ACCELERATION.md`:
- 7+ key arXiv papers analyzed (1505.04648, 1805.00898, 1902.04367, 1808.08221, 2103.01934)
- Production methods from major financial institutions
- Speedups of 10,000× - 40,000× demonstrated in literature
- Applications: parametric pricing, XVA, SIMM, volatility calibration

**Implemented methods**:
- ✅ 5D Chebyshev tensor interpolation with CP decomposition (`chebyshev_tensor_demo.py`)
- ✅ Finite difference methods for Black-Scholes PDE (`fdm_baseline.py`)
- 🔬 MoCaX integration (under testing, requires proprietary library)

**Future directions** (documented but not yet implemented):
- Tensor Train (TT) decomposition for d>5 dimensions
- TT-cross adaptive sampling
- Dynamic Chebyshev for American options
- Super-time-stepping for PDE acceleration

## Important Notes for Development

### Multi-Parameter Greeks: Why 5D Matters

**Critical insight from TENSOR_COMPARISON_SUMMARY.md**:
- 1D Chebyshev interpolation (varying only σ or r at a fixed point) gives 40-108% errors on Greeks when other parameters change
- **Why?** Because Vega(S, K, T, σ, r) depends on ALL parameters, not just σ!
- **Solution**: Build 5D tensor covering entire parameter space
- **Result**: Errors reduced to 1-8% uniformly across all parameter variations

**When implementing new Greeks calculations**:
1. Always consider whether parameters will vary in practice
2. If yes → use 5D tensor interpolation (not 1D!)
3. If no (truly fixed point) → 1D is 30× faster

### Tensor Rank Selection

The `rank` parameter in CP decomposition controls accuracy vs compression:
- **rank=5**: High compression (574×), moderate error (~10%)
- **rank=10**: Balanced (287×), good error (1-8%) ← **current default**
- **rank=20**: Lower compression (143×), excellent error (<1%)

Empirical rule: Start with rank=10, increase if errors exceed requirements.

### Performance Optimization

**Offline phase** (building tensor):
- Use analytical formulas when available (161,051 evals in 0.28s)
- If using FDM, this becomes the bottleneck (would take 22 hours!)
- Consider parallel evaluation for non-analytical cases

**Online phase** (queries):
- Tensor interpolation: ~1ms per price, ~2ms per Greek
- Break-even: >1-2 queries makes tensor worth it
- For 1000 queries: 855× faster than FDM
