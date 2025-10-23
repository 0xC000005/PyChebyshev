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

# Chebyshev baseline (uses NumPy's Chebyshev.interpolate with partial pre-computation)
uv run python chebyshev_baseline.py

# Chebyshev barycentric (true barycentric interpolation matching MoCaX algorithm)
uv run python chebyshev_barycentric.py

# MoCaX library test (requires setup - see MoCaX Setup section below)
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH" && uv run python mocax_test.py
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

**`chebyshev_baseline.py`** - NumPy Chebyshev with partial pre-computation
- `ChebyshevApproximation` class: Multi-dimensional interpolation using `Chebyshev.interpolate()`
  - Uses dimensional decomposition (3D→2D→1D) to collapse multi-dimensional tensors
  - Pre-computes polynomials for innermost dimension only (14,641 objects for 5D)
  - Re-interpolates outer dimensions on each query (values depend on query point)
  - **Performance**: Accurate but requires O(N log N) polynomial fitting per query
  - **Limitation**: Polynomial coefficients depend on both nodes AND values

**`chebyshev_barycentric.py`** - True barycentric interpolation (matches MoCaX algorithm)
- `ChebyshevApproximation` class: Barycentric interpolation with full pre-computation
  - Pre-computes barycentric weights for ALL dimensions (just 55 floats for 5D)
  - Weights depend ONLY on node positions, NOT function values
  - Uniform O(N) evaluation for all dimensions (no polynomial fitting during queries)
  - **Performance**: 0.000% price error, 1.980% max Greek error
  - **Key advantage**: Algorithmically equivalent to MoCaX, fair comparison

**`mocax_test.py`** - MoCaX library integration test
- Tests MoCaX (Multi-dimensional Chebyshev Approximation) proprietary library
- Three comprehensive tests: simple 3D, Black-Scholes 3D, and 5D parametric BS
- Demonstrates automatic differentiation for Greeks (analytical, not finite difference)
- Requires external `mocax_lib/` directory (see MOCAX_SETUP_GUIDE.md)
- **Results**: Spectral accuracy (0.000% price error), 1.98% Vega error on 5D test

### Research Documentation

**`MOCAX_ALGORITHM_ANALYSIS.md`** - Comprehensive analysis of MoCaX algorithm
- **Part 1**: Overview of MoCaX (Multi-dimensional Chebyshev Approximation)
  - Algorithmic Pricing Acceleration (APA) and Greeks Acceleration (AGA)
  - Standard MoCaX (full Chebyshev tensors) vs MoCaX Extend (Tensor Train format)
- **Part 2**: Core Algorithm - Barycentric Chebyshev Interpolation
  - Detailed explanation of barycentric formula: `p(x) = Σ[w_i * f_i / (x - x_i)] / Σ[w_i / (x - x_i)]`
  - Why barycentric is superior to Lagrange form (O(N) vs O(N²))
  - Source code evidence from `mocaxc_utils.private.h`
- **Part 3**: Dimensional Decomposition Strategy
  - 5D → 4D → 3D → 2D → 1D collapse process
  - How to evaluate V(S, K, T, σ, r) by repeated 1D interpolation
- **Part 4**: Key Optimization - Pre-computed Weights
  - Barycentric weights depend ONLY on node positions (not function values!)
  - Formula: `w_i = 1 / ∏(j≠i) (x_i - x_j)`
  - Why polynomial coefficients can't be pre-computed for all dimensions
  - Complexity: O(N^d) barycentric vs O(N^(d+1)) with refitting

**`MOCAX_SETUP_GUIDE.md`** - Installation and usage guide
- Complete setup instructions for MoCaX proprietary library
- How to extract and configure the Python bindings
- Test suite documentation and expected results

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

### Three-Tier Implementation Strategy

1. **Analytical Layer** (`blackscholes` library)
   - Instant pricing (~10μs)
   - Exact results (machine precision)
   - Limited to European vanilla options
   - **Use case**: Single evaluations, ground truth validation

2. **Chebyshev Approximation Layer** (two implementations for comparison)

   **A. Baseline (`chebyshev_baseline.py`)**
   - Uses `numpy.polynomial.chebyshev.Chebyshev.interpolate()`
   - Dimensional decomposition with partial pre-computation
   - Pre-computes polynomials for innermost dimension only (14,641 objects for 5D)
   - Re-interpolates outer dimensions on each query (O(N log N) per dimension)
   - **Limitation**: Polynomial coefficients depend on values, can't fully pre-compute

   **B. Barycentric (`chebyshev_barycentric.py`)**
   - Implements barycentric interpolation formula manually
   - Pre-computes weights for ALL dimensions (just 55 floats for 5D)
   - Uniform O(N) evaluation for all dimensions
   - **Performance**: 0.000% price error, 1.980% max Greek error
   - **Key insight**: Barycentric weights depend only on nodes, not values!
   - **Algorithmically equivalent to MoCaX** - fair comparison benchmark

3. **Numerical PDE Layer** (`fdm_baseline.py`)
   - Finite difference methods for Black-Scholes PDE
   - Flexible for exotic features (American, barriers, path-dependent)
   - Accurate baseline (~0.03% error with fine grids)
   - Slow (~500ms per solve)
   - **Use case**: Validation, exotic options without analytical formulas

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
- Delta: Central difference `(V[i+1] - V[i-1])/(2·dS)` (FDM grid-based)
- Gamma: Second-order central `(V[i+1] - 2·V[i] + V[i-1])/dS²` (FDM grid-based)
- Theta: Forward difference in time (FDM grid-based)
- Vega/Rho (FDM): Finite difference requiring PDE re-solve with perturbed parameters (~1s each)
- Vega/Rho (Chebyshev): Finite difference on interpolated surface (~2ms each, 500× faster!)
- Derivatives (Barycentric): 5-point stencil with adaptive epsilon for numerical stability

**Chebyshev Interpolation Configuration**:
- Dimensions: 5D → V(S, K, T, σ, r)
- Nodes per dimension: 11 (Chebyshev nodes from `chebpts1()`)
- Total grid: 11^5 = 161,051 points
- **Baseline**: 14,641 polynomial objects (innermost dimension only)
- **Barycentric**: 55 weights (ALL dimensions, just 440 bytes!)
- Evaluation: Dimensional decomposition (5D → 4D → 3D → 2D → 1D → scalar)

### Convergence & Accuracy

**From FDM** (`convergence_study()` results):
- **Coarse grid** (M=50, N=500): ~0.6% price error, 0.09s runtime
- **Fine grid** (M=200, N=2000): ~0.04% price error, 0.5s runtime
- **Ultra fine** (M=800, N=8000): ~0.002% price error, 6s runtime
- Trade-off: 69× slower gives 273× better accuracy
- **Error scaling**: O(dt, dS²) truncation error

**From Chebyshev Barycentric** (`chebyshev_barycentric.py` 5D test):
- **Price error**: 0.000% max (spectral accuracy!)
- **Delta**: 0.000%, **Gamma**: 1.590%, **Vega**: 1.980%, **Rho**: 0.000%
- Build time: ~0.35s (161,051 evaluations using analytical formulas)
- Pre-computed weights: 55 floats (440 bytes) for all dimensions
- Uniform O(N) evaluation approach

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

**Use Chebyshev Baseline** (`chebyshev_baseline.py`):
- ✅ Learning dimensional decomposition concept
- ✅ Using NumPy's built-in Chebyshev functions
- ✅ Partial pre-computation acceptable
- ❌ Not optimal for production (re-interpolates outer dimensions)
- ❌ Use barycentric instead for fair MoCaX comparison

**Use Chebyshev Barycentric** (`chebyshev_barycentric.py`):
- ✅ **Fair comparison with MoCaX algorithm** (algorithmically equivalent!)
- ✅ **Multiple parameters vary simultaneously** (key advantage!)
- ✅ Need Greeks (Vega, Rho) across diverse scenarios
- ✅ Many queries (>10) to amortize ~0.35s offline cost
- ✅ Can tolerate ~2% error on Greeks
- ✅ Parameters within interpolation range
- ✅ Minimal memory footprint (55 weights for 5D)
- ❌ Don't use for single query (not worth offline cost)
- ❌ Don't use for out-of-range extrapolation

**Use FDM** (`fdm_baseline.py`):
- ✅ American options (early exercise)
- ✅ Barriers, path-dependent options, exotics
- ✅ Need highest accuracy (<0.1% error)
- ✅ Research/validation baseline
- ✅ One-off calculations
- ❌ Don't use for repeated similar queries (use Chebyshev instead)

**Use MoCaX** (proprietary library, see `MOCAX_SETUP_GUIDE.md`):
- ✅ Production risk systems requiring extreme performance
- ✅ Automatic differentiation for Greeks (analytical, not finite difference)
- ✅ Multi-asset options (d>5 dimensions with MoCaX Extend)
- ✅ XVA calculations (CVA, DVA, FVA)
- ✅ Regulatory capital (SIMM)

## Research References

The project focuses on comparing Chebyshev approximation methods:

**Core Documentation**:
- `MOCAX_ALGORITHM_ANALYSIS.md`: Detailed analysis of MoCaX's barycentric algorithm
- `TENSOR_COMPARISON_SUMMARY.md`: Why 5D matters (1D gives 40-108% errors when parameters vary)
- `MOCAX_SETUP_GUIDE.md`: Complete MoCaX installation and usage guide

**Implemented Methods**:
- ✅ Finite difference methods for Black-Scholes PDE (`fdm_baseline.py`)
- ✅ NumPy Chebyshev baseline with partial pre-computation (`chebyshev_baseline.py`)
- ✅ Barycentric interpolation matching MoCaX algorithm (`chebyshev_barycentric.py`)
- ✅ MoCaX integration tests (`mocax_test.py`, requires proprietary library)

**Key Findings**:
- Barycentric weights can be pre-computed for ALL dimensions (depend only on nodes)
- Polynomial coefficients can't be fully pre-computed (depend on both nodes and values)
- 5D Chebyshev barycentric: 0.000% price error, 1.98% max Greek error
- MoCaX 5D test: 0.000% price error, 1.98% Vega error (spectral accuracy)

## Important Notes for Development

### Multi-Parameter Greeks: Why 5D Matters

**Critical insight from TENSOR_COMPARISON_SUMMARY.md**:
- 1D Chebyshev interpolation (varying only σ or r at a fixed point) gives 40-108% errors on Greeks when other parameters change
- **Why?** Because Vega(S, K, T, σ, r) depends on ALL parameters, not just σ!
- **Solution**: Build 5D tensor covering entire parameter space
- **Result**: Errors reduced to 1-8% uniformly across all parameter variations

**When implementing new Greeks calculations**:
1. Always consider whether parameters will vary in practice
2. If yes → use 5D Chebyshev interpolation (not 1D!)
3. If no (truly fixed point) → 1D is faster but limited

### Barycentric vs Polynomial Coefficients: Key Insight

**Why barycentric weights can be pre-computed for ALL dimensions**:
- Barycentric weight: `w_i = 1 / ∏(j≠i) (x_i - x_j)` → depends ONLY on node positions
- Polynomial coefficients: depend on BOTH node positions AND function values
- During evaluation, outer dimension "values" are intermediate results (depend on query point)
- Therefore: Coefficients must be recomputed, but weights don't!

**Complexity comparison** (5D with 11 nodes per dimension):
- **Baseline**: 14,641 polynomial objects (innermost only) + refitting for outer dimensions
- **Barycentric**: 55 weights (ALL dimensions) + no refitting needed
- **Result**: Uniform O(N) evaluation vs mixed O(N log N) + O(N)

### Performance Optimization

**Offline phase** (building approximation):
- Evaluate function at Chebyshev nodes: 11^5 = 161,051 points
- Use analytical formulas when available (~0.28s for Black-Scholes)
- If using FDM, this becomes the bottleneck (would take ~22 hours!)
- Pre-compute barycentric weights (trivial, ~1ms)

**Online phase** (queries):
- Barycentric interpolation: ~1ms per price, ~2ms per Greek
- Break-even: >1-2 queries makes pre-computation worth it
- Evaluation: Dimensional decomposition (5D → 4D → 3D → 2D → 1D → scalar)
