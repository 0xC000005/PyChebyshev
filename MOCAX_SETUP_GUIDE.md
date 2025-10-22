# MoCaX Suite Setup Guide

Complete guide to installing and using the MoCaX Intelligence library for multi-dimensional Chebyshev approximation in option pricing.

**Version**: MoCaX Suite 1.2.0 (Library version 4.3.1)
**Platform**: Linux x86_64
**Python**: 3.13+ (compatible with Python 2.7+)

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Understanding the Library](#understanding-the-library)
8. [Performance Tuning](#performance-tuning)
9. [References](#references)

---

## Overview

**MoCaX (Multi-dimensional Chebyshev Approximation eXtended)** is a proprietary library for building high-accuracy function approximations using Chebyshev tensors. It's particularly powerful for:

- **Parametric option pricing** (accelerating expensive PDE solvers)
- **Risk calculations** (XVA, CVA, initial margin)
- **Volatility surface interpolation**
- **Any expensive function** that needs to be evaluated many times with different parameters

**Key Features**:
- True Chebyshev polynomial evaluation (spectral accuracy)
- Analytical derivatives up to 2nd order
- Automatic differentiation for Greeks
- Serialization/deserialization for production deployment
- C++ core with Python/Java bindings

---

## Prerequisites

### System Requirements

- **OS**: Linux (64-bit)
- **Architecture**: x86_64
- **RAM**: 4GB+ (more for high-dimensional problems)
- **Disk**: ~200MB for MoCaX Suite

### Software Requirements

```bash
# Check your system
uname -m        # Should output: x86_64
python3 --version  # Python 3.x (3.13+ recommended)

# Required Python packages (install via uv or pip)
numpy >= 1.20
scipy >= 1.6
```

### Optional Dependencies

For the Black-Scholes test suite:
```bash
uv add blackscholes  # Analytical formulas for validation
uv add pandas        # Data analysis
uv add matplotlib    # Plotting
```

---

## Installation Steps

### Step 1: Download and Extract MoCaX Suite

You should have the `MoCaXSuite-1.2.0.zip` file (obtained from mocaxintelligence.org or the textbook).

```bash
# Navigate to your project directory
cd /path/to/FinRegressor

# Extract the suite (if not already done)
unzip MoCaXSuite-1.2.0.zip
```

**Directory structure**:
```
MoCaXSuite-1.2.0/
├── MoCaX/
│   ├── Linux/
│   │   └── gmake/64bit/
│   │       ├── Python/
│   │       │   ├── MoCaXLibrary/
│   │       │   │   ├── libmocaxc.so           # Shared library
│   │       │   │   └── mocaxpy-4.3.1.linux-x86_64.zip  # Python bindings
│   │       │   └── MoCaXExamples/             # Example scripts
│   │       ├── C++/                           # C++ interface
│   │       └── Java/                          # Java interface
│   ├── Windows/                               # Windows binaries
│   └── MoCaXDocumentationPDF/                 # User manuals
├── MoCaXExtend/                               # Tensor Train extensions
└── README.txt
```

### Step 2: Extract Python Bindings

Navigate to the Python library directory and extract the bindings:

```bash
cd MoCaXSuite-1.2.0/MoCaXSuite-1.2.0/MoCaX/Linux/gmake/64bit/Python/MoCaXLibrary

# Extract the Python package
unzip -q mocaxpy-4.3.1.linux-x86_64.zip

# Verify extraction
ls usr/lib/python2.7/site-packages/mocaxpy/
# Should show: __init__.py, Mocax.py, MocaxCInterface.py, MocaxDomain.py, etc.
```

### Step 3: Create mocax_lib Directory

Copy the library files to a location your Python scripts can access:

```bash
# Create mocax_lib in MoCaXLibrary directory
mkdir -p mocax_lib

# Copy Python module
cp -r usr/lib/python2.7/site-packages/mocaxpy mocax_lib/

# Copy shared library
cp libmocaxc.so mocax_lib/

# Verify structure
ls -la mocax_lib/
# Should show: mocaxpy/ and libmocaxc.so
```

### Step 4: Copy to Project Root

Move the `mocax_lib` directory to your project root for easy access:

```bash
# Return to project root
cd /path/to/FinRegressor

# Copy mocax_lib
cp -r MoCaXSuite-1.2.0/MoCaXSuite-1.2.0/MoCaX/Linux/gmake/64bit/Python/MoCaXLibrary/mocax_lib .

# Verify it's in the right place
ls -la mocax_lib/
```

**Final project structure**:
```
FinRegressor/
├── mocax_lib/                  # ← MoCaX library
│   ├── libmocaxc.so
│   └── mocaxpy/
│       ├── __init__.py
│       ├── Mocax.py
│       ├── MocaxCInterface.py
│       └── ...
├── mocax_test.py               # Test suite
├── run_mocax_test.sh           # Helper script
└── ...
```

### Step 5: Set Up Environment

The shared library (`libmocaxc.so`) must be in the library search path.

**Option A: Using the helper script** (recommended):

```bash
# The script is already set up
cat run_mocax_test.sh
# #!/bin/bash
# export LD_LIBRARY_PATH="$(pwd)/mocax_lib:$LD_LIBRARY_PATH"
# uv run python mocax_test.py

chmod +x run_mocax_test.sh
```

**Option B: Manual export**:

```bash
# Add to your shell session
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"

# Or add to ~/.bashrc for persistence
echo 'export LD_LIBRARY_PATH="/path/to/FinRegressor/mocax_lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

**Option C: System-wide installation** (requires sudo):

```bash
# Copy to system library path
sudo cp mocax_lib/libmocaxc.so /usr/local/lib/

# Update library cache
sudo ldconfig

# Copy Python module to site-packages (optional)
python3 -m site  # Find your site-packages directory
cp -r mocax_lib/mocaxpy /path/to/site-packages/
```

---

## Verification

### Quick Test: Import MoCaX

```bash
# Test 1: Check library loads
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
python3 -c "import sys; sys.path.insert(0, 'mocax_lib'); import mocaxpy; print(f'✓ MoCaX {mocaxpy.get_version_id()}')"

# Expected output:
# [INFO] Loading MOCAX library with version 4.3.1
# ✓ MoCaX 4.3.1
```

### Full Test Suite

Run the comprehensive test suite (3 tests, ~1-2 minutes):

```bash
./run_mocax_test.sh
```

**Expected results**:

```
======================================================================
MoCaX Installation and Integration Test Suite
======================================================================

======================================================================
TEST 1: Simple 3D Function (sin(x) + sin(y) + sin(z))
======================================================================
✓ Test PASSED: MoCaX approximation is accurate

======================================================================
TEST 2: Black-Scholes Call Option with MoCaX
======================================================================
✓ Test PASSED: Maximum error 0.002% is acceptable

======================================================================
TEST 3: 5D Parametric Black-Scholes (S, K, T, σ, r)
======================================================================
✓ Test PASSED: Price error 0.000%, Greek error 1.980%

======================================================================
TEST SUMMARY
======================================================================
  Simple 3D Function             ✓ PASSED
  Black-Scholes Call (3D)        ✓ PASSED
  5D Parametric Black-Scholes    ✓ PASSED
======================================================================
✓ All tests PASSED - MoCaX is ready to use!
```

### Understanding Test Results

**Test 1: Simple 3D Function**
- Validates basic Chebyshev approximation
- Tests analytical derivatives
- Build time: ~1ms
- Approximation error: <0.1%

**Test 2: Black-Scholes 3D (S, T, σ)**
- 3D parametric option pricing
- 1,800 function evaluations (15×12×10 nodes)
- Build time: ~10ms
- Price error: <0.01%, Vega error: ~2%

**Test 3: 5D Parametric Black-Scholes (S, K, T, σ, r)** ⭐
- Full parametric pricing across 5 dimensions
- 161,051 function evaluations (11^5 nodes)
- Build time: ~1.5s
- Price error: 0.000% (spectral accuracy!)
- Greeks: Delta/Gamma/Rho <0.001%, Vega ~2%
- **This is the showcase test** demonstrating MoCaX's capabilities

---

## Usage Examples

### Example 1: Simple Function Approximation

```python
import sys
sys.path.insert(0, 'mocax_lib')
import mocaxpy
import math

# Define your expensive function
def my_function(x, additional_data):
    """
    Example: f(x, y, z) = sin(x) + cos(y) + x*y*z
    x is a list/array: x[0]=x, x[1]=y, x[2]=z
    """
    return math.sin(x[0]) + math.cos(x[1]) + x[0]*x[1]*x[2]

# Setup
num_dimensions = 3
domain_values = [
    [-1.0, 1.0],   # x range
    [-1.0, 1.0],   # y range
    [0.0, 2.0]     # z range
]
domain = mocaxpy.MocaxDomain(domain_values)

# Accuracy control
n_values = [10, 10, 8]  # Chebyshev nodes per dimension
ns = mocaxpy.MocaxNs(n_values)
max_derivative_order = 2  # Enable 2nd derivatives

# Build approximation
mocax_obj = mocaxpy.Mocax(
    my_function,
    num_dimensions,
    domain,
    None,  # error_threshold (None = use ns directly)
    ns,
    max_derivative_order=max_derivative_order
)

# Evaluate at a point
test_point = [0.5, 0.3, 1.2]

# Function value
deriv_id = mocax_obj.get_derivative_id([0, 0, 0])
value = mocax_obj.eval(test_point, deriv_id)
print(f"f({test_point}) = {value}")

# First derivative ∂f/∂x
deriv_id = mocax_obj.get_derivative_id([1, 0, 0])
df_dx = mocax_obj.eval(test_point, deriv_id)
print(f"∂f/∂x = {df_dx}")

# Second derivative ∂²f/∂x²
deriv_id = mocax_obj.get_derivative_id([2, 0, 0])
d2f_dx2 = mocax_obj.eval(test_point, deriv_id)
print(f"∂²f/∂x² = {d2f_dx2}")

# Mixed derivative ∂²f/∂x∂y
deriv_id = mocax_obj.get_derivative_id([1, 1, 0])
d2f_dxdy = mocax_obj.eval(test_point, deriv_id)
print(f"∂²f/∂x∂y = {d2f_dxdy}")

# Clean up
del mocax_obj
```

### Example 2: Black-Scholes with Greeks

```python
import sys
sys.path.insert(0, 'mocax_lib')
import mocaxpy
from blackscholes import BlackScholesCall

# Black-Scholes wrapper
def bs_pricer(x, additional_data):
    """x = [S, K, T, sigma, r]"""
    S, K, T, sigma, r = x[0], x[1], x[2], x[3], x[4]
    option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=0.02)
    return option.price()

# Define 5D domain
domain_values = [
    [80.0, 120.0],   # S
    [90.0, 110.0],   # K
    [0.25, 1.0],     # T
    [0.15, 0.35],    # sigma
    [0.01, 0.08]     # r
]
domain = mocaxpy.MocaxDomain(domain_values)

# Build approximation (takes ~1-2 seconds)
n_values = [11, 11, 11, 11, 11]  # 161,051 evaluations
ns = mocaxpy.MocaxNs(n_values)

mocax_bs = mocaxpy.Mocax(
    bs_pricer,
    5,  # dimensions
    domain,
    None,
    ns,
    max_derivative_order=2
)

# Query at a point (instant!)
point = [100.0, 100.0, 1.0, 0.25, 0.05]

# Price
deriv_id = mocax_bs.get_derivative_id([0, 0, 0, 0, 0])
price = mocax_bs.eval(point, deriv_id)
print(f"Price: {price:.6f}")

# Delta: ∂V/∂S
deriv_id = mocax_bs.get_derivative_id([1, 0, 0, 0, 0])
delta = mocax_bs.eval(point, deriv_id)
print(f"Delta: {delta:.6f}")

# Gamma: ∂²V/∂S²
deriv_id = mocax_bs.get_derivative_id([2, 0, 0, 0, 0])
gamma = mocax_bs.eval(point, deriv_id)
print(f"Gamma: {gamma:.6f}")

# Vega: ∂V/∂σ
deriv_id = mocax_bs.get_derivative_id([0, 0, 0, 1, 0])
vega = mocax_bs.eval(point, deriv_id)
print(f"Vega: {vega:.6f}")

# Rho: ∂V/∂r
deriv_id = mocax_bs.get_derivative_id([0, 0, 0, 0, 1])
rho = mocax_bs.eval(point, deriv_id)
print(f"Rho: {rho:.6f}")

del mocax_bs
```

### Example 3: Serialization for Production

```python
# Build once, save to disk
mocax_obj = mocaxpy.Mocax(expensive_function, dims, domain, None, ns, max_derivative_order=2)
mocax_obj.serialize("pricing_model.mcx")
del mocax_obj

# Later: load from disk (instant!)
pricing_model = mocaxpy.Mocax.deserialize("pricing_model.mcx")

# Use in production
for scenario in scenarios:
    price = pricing_model.eval(scenario, deriv_id)
    process(price)

del pricing_model
```

---

## Troubleshooting

### Issue 1: "cannot open shared object file: No such file or directory"

**Error**:
```
OSError: libmocaxc.so: cannot open shared object file
```

**Solution**:
```bash
# Make sure LD_LIBRARY_PATH is set
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"

# Or use the helper script
./run_mocax_test.sh
```

### Issue 2: "ModuleNotFoundError: No module named 'mocaxpy'"

**Error**:
```
ModuleNotFoundError: No module named 'mocaxpy'
```

**Solution**:
```python
# Add mocax_lib to Python path before importing
import sys
sys.path.insert(0, 'mocax_lib')
import mocaxpy
```

Or in your script:
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mocax_lib'))
```

### Issue 3: "'module' object is not callable" with blackscholes

**Error**:
```
TypeError: 'module' object is not callable
```

**Cause**: Old import style `import blackscholes as bs; bs.call(...)`

**Solution**: Use class-based API
```python
from blackscholes import BlackScholesCall
option = BlackScholesCall(S=100, K=100, T=1, r=0.05, sigma=0.25, q=0.02)
price = option.price()
```

### Issue 4: Slow Build Times

**Symptom**: 5D test takes >5 seconds to build

**Causes & Solutions**:

1. **Too many Chebyshev nodes**:
   - `n_values = [15, 15, 15, 15, 15]` = 759,375 evaluations
   - Reduce to `[11, 11, 11, 11, 11]` = 161,051 evaluations
   - Or use adaptive: `[11, 11, 9, 9, 7]` = 68,607 evaluations

2. **Slow original function**:
   - If using FDM solver, each evaluation is expensive
   - Use analytical formulas in the build phase if available
   - Consider parallelization for offline build

3. **Python overhead**:
   - Python callbacks have overhead
   - For production, consider C++ implementation

### Issue 5: Low Accuracy Results

**Symptom**: Errors > 5% on function values or Greeks

**Diagnostics**:

1. **Check function smoothness**:
   - Chebyshev approximation works best for smooth functions
   - Discontinuities or kinks reduce accuracy
   - Use more nodes near problematic regions

2. **Increase Chebyshev nodes**:
   ```python
   # Increase from [9, 9, 7] to [11, 11, 9]
   n_values = [11, 11, 9]
   ```

3. **Check domain boundaries**:
   - Extrapolation outside domain is unreliable
   - Ensure test points are within domain_values ranges

4. **Use error threshold**:
   ```python
   # Let MoCaX adaptively choose nodes
   mocax_obj = mocaxpy.Mocax(
       function,
       dims,
       domain,
       error_threshold=1e-4,  # Target error
       ns,  # Starting point
       max_derivative_order=2
   )
   ```

---

## Understanding the Library

### Chebyshev Nodes

MoCaX samples your function at **Chebyshev nodes** (also called Gauss-Lobatto points):

```
xᵢ = cos(iπ/n)  for i = 0, 1, ..., n

Mapped to [a, b]:
xᵢ = (b - a)/2 · cos(iπ/n) + (a + b)/2
```

**Why Chebyshev nodes?**
- Optimal for polynomial interpolation
- Minimize interpolation error (Runge phenomenon)
- Clustered near boundaries (where functions often have more variation)

### Chebyshev Polynomials

The approximation is a tensor product of Chebyshev polynomials:

```
V(x₁, x₂, ..., xₙ) ≈ Σᵢ₁...ᵢₙ cᵢ₁...ᵢₙ · Tᵢ₁(x₁) · Tᵢ₂(x₂) · ... · Tᵢₙ(xₙ)

where Tₙ(x) = cos(n · arccos(x))  (Chebyshev polynomial of 1st kind)
```

**Properties**:
- Orthogonal on [-1, 1]
- Bounded: |Tₙ(x)| ≤ 1
- Recurrence: Tₙ₊₁(x) = 2x·Tₙ(x) - Tₙ₋₁(x)
- Derivative: T'ₙ(x) = n·Uₙ₋₁(x) (Chebyshev polynomial of 2nd kind)

### Analytical Derivatives

MoCaX computes derivatives analytically using Chebyshev differentiation:

```
∂V/∂xᵢ = Σⱼ cⱼ · ∂Tⱼ(xᵢ)/∂xᵢ

Since T'ₙ(x) is known analytically, derivatives are exact (up to coefficient accuracy)
```

**Advantages over finite differences**:
- No truncation error from h → 0
- No need to re-evaluate function at x ± h
- Higher-order derivatives equally fast
- Mixed derivatives: ∂²V/∂xᵢ∂xⱼ directly available

### Derivative IDs

```python
# Derivative order vector [n₁, n₂, ..., nₐ]
# nᵢ = order of derivative w.r.t. dimension i

get_derivative_id([0, 0, 0])    # Function value
get_derivative_id([1, 0, 0])    # ∂f/∂x₁
get_derivative_id([0, 1, 0])    # ∂f/∂x₂
get_derivative_id([2, 0, 0])    # ∂²f/∂x₁²
get_derivative_id([1, 1, 0])    # ∂²f/∂x₁∂x₂
get_derivative_id([1, 0, 1])    # ∂²f/∂x₁∂x₃
```

Maximum order: sum(nᵢ) ≤ max_derivative_order

### Storage and Compression

**Coefficients vs Grid Values**:

| Method | Storage | Access Pattern |
|--------|---------|----------------|
| Grid values | N₁ × N₂ × ... × Nₐ values | Direct lookup |
| Chebyshev coefficients | N₁ × N₂ × ... × Nₐ coefficients | Polynomial evaluation |

**Coefficient decay**: For smooth functions, high-order coefficients → 0 exponentially
- Allows compression (drop small coefficients)
- Controlled approximation error
- Typical: 10-100× compression with <1% error

**MoCaX internal storage**:
- Stores coefficients (not raw values)
- Compact binary format
- Serializes to .mcx files for deployment

---

## Performance Tuning

### Choosing Number of Nodes

**Rule of thumb**:
```
Error ≈ C · ρ⁻ⁿ  (exponential convergence for smooth functions)

where ρ > 1 depends on function analyticity
```

**Guidelines**:

| Target Error | Nodes per Dim | Notes |
|--------------|---------------|-------|
| 10% | 5-7 | Quick approximation |
| 1% | 8-10 | Good accuracy |
| 0.1% | 11-13 | High accuracy |
| 0.01% | 15-20 | Very high accuracy |

**Diminishing returns**: Going from n=11 to n=15 gives 2-3× accuracy improvement but 2.8× more evaluations

### Dimension-Dependent Nodes

Use fewer nodes in less sensitive dimensions:

```python
# Adaptive approach
n_values = [
    15,  # Dimension 1: most sensitive (e.g., spot price S)
    12,  # Dimension 2: moderately sensitive (e.g., volatility σ)
    10,  # Dimension 3: less sensitive (e.g., time T)
    8,   # Dimension 4: least sensitive (e.g., dividend q)
]
# Total: 15 × 12 × 10 × 8 = 14,400 evaluations (vs 15⁴ = 50,625 uniform)
```

**How to determine sensitivity**:
1. Run with uniform nodes first
2. Analyze coefficient decay in each dimension
3. Reduce nodes in dimensions with faster decay

### Build vs Query Trade-off

**Break-even analysis**:
```
Break-even queries = Build_time / (Analytical_time - MoCaX_time)

Example (5D Black-Scholes):
- Build: 1.5s = 1,500,000 μs
- Analytical: 1 μs per query
- MoCaX: 220 μs per query
- Break-even: 1,500,000 / (1 - 220) = NEGATIVE → Never breaks even!

But if no analytical formula exists (FDM solver @ 500ms):
- Build: 1.5s
- FDM: 500,000 μs per query
- MoCaX: 220 μs per query
- Break-even: 1,500,000 / (500,000 - 220) ≈ 3 queries → Always use MoCaX!
```

**When MoCaX wins**:
- No analytical formula (exotics, early exercise, multi-asset)
- Expensive PDE solvers (FDM, FEM: milliseconds to seconds per solve)
- Monte Carlo (thousands of paths)
- Need many queries (>10-100 for typical setup)

**When analytical wins**:
- Fast closed-form formula exists (Black-Scholes: microseconds)
- Single or few queries
- Simple scenarios

### Parallelization

**Build phase** (if original function is very expensive):

```python
# Deferred construction pattern
mocax_obj = mocaxpy.Mocax(
    None,  # No function
    num_dimensions,
    domain,
    None,
    ns,
    max_derivative_order=2
)

# Get evaluation points
points = mocax_obj.get_evaluation_points()

# Evaluate in parallel (using multiprocessing, etc.)
values = parallel_map(expensive_function, points)

# Complete construction
mocax_obj.set_original_function_values(values)
```

**Query phase**:
- Each `eval()` call is independent
- Parallelize over scenarios/portfolio positions
- Thread-safe (check documentation for your version)

### Memory Management

```python
# Always delete when done
del mocax_obj

# For long-running processes
import gc
gc.collect()  # Force garbage collection
```

**Memory footprint**:
- Coefficients: O(N₁ × N₂ × ... × Nₐ × 8 bytes)
- 5D with [11, 11, 11, 11, 11]: 161,051 coefficients × 8 bytes ≈ 1.2 MB
- 10D with [9]¹⁰: 3.5 billion coefficients × 8 bytes ≈ 28 GB (use Tensor Train instead!)

---

## Advanced Topics

### Tensor Train (MoCaX Extend)

For high-dimensional problems (d > 5-6), use Tensor Train format:

```
Located in: MoCaXExtend/
Purpose: Rank-adaptive low-rank tensor approximations
Max dimensions: 20+ dimensions
Compression: 1000-10,000× compared to full tensor
```

See: `MoCaXExtend/` documentation for TT-cross algorithms

### Adaptive Refinement

Use error threshold instead of fixed nodes:

```python
mocax_obj = mocaxpy.Mocax(
    function,
    dims,
    domain,
    error_threshold=1e-4,  # Target L∞ error
    initial_ns,            # Starting guess
    max_derivative_order=2
)
# MoCaX will adaptively add nodes until error < 1e-4
```

### Domain Decomposition

For functions with different behaviors in different regions:

1. Split domain into subdomains
2. Build separate MoCaX objects for each
3. Query appropriate object based on point location

```python
# Example: Split at x = 0
mocax_left = mocaxpy.Mocax(func, dims, domain_left, None, ns, 2)
mocax_right = mocaxpy.Mocax(func, dims, domain_right, None, ns, 2)

def evaluate(point):
    if point[0] < 0:
        return mocax_left.eval(point, deriv_id)
    else:
        return mocax_right.eval(point, deriv_id)
```

---

## References

### Documentation

- **User Manual**: `MoCaXSuite-1.2.0/MoCaX/MoCaXDocumentationPDF/`
- **Project Website**: https://mocaxintelligence.org
- **YouTube Channel**: youtube.com/mocax
- **Textbook**: "Machine Learning for Risk Calculations" by I. Ruiz & M. Zeron (Wiley)

### Academic Papers

See `CHEBYSHEV_ACCELERATION.md` for 190+ pages of research including:
- arXiv:1505.04648 - Parametric option pricing
- arXiv:1805.00898 - Ultra-efficient risk calculations (40,000× speedup)
- arXiv:1808.08221 - SIMM initial margin via tensors
- arXiv:1902.04367 - Low-rank tensor approximations

### Example Code

- **This project**: `mocax_test.py` (comprehensive test suite)
- **MoCaX Suite**: `MoCaXExamples/` (simple-example-py, sliding-example-py)
- **Research**: `chebyshev_tensor_demo.py` (our tensor approach for comparison)

### Related Documents

- `MOCAX_5D_RESULTS.md` - Detailed results from 5D Black-Scholes test
- `TENSOR_COMPARISON_SUMMARY.md` - Comparison of Chebyshev methods
- `CHEBYSHEV_ACCELERATION.md` - Comprehensive research survey
- `CLAUDE.md` - Project documentation and architecture

---

## Quick Reference Card

### Installation TL;DR

```bash
cd MoCaXSuite-1.2.0/.../Python/MoCaXLibrary
unzip -q mocaxpy-4.3.1.linux-x86_64.zip
mkdir -p mocax_lib
cp -r usr/lib/python2.7/site-packages/mocaxpy mocax_lib/
cp libmocaxc.so mocax_lib/
cp -r mocax_lib /path/to/your/project/
cd /path/to/your/project
export LD_LIBRARY_PATH="$PWD/mocax_lib:$LD_LIBRARY_PATH"
./run_mocax_test.sh  # Verify installation
```

### Python Template

```python
import sys
sys.path.insert(0, 'mocax_lib')
import mocaxpy

def my_func(x, additional_data):
    return your_expensive_function(x[0], x[1], ...)

domain = mocaxpy.MocaxDomain([[a1, b1], [a2, b2], ...])
ns = mocaxpy.MocaxNs([n1, n2, ...])
mocax = mocaxpy.Mocax(my_func, dims, domain, None, ns, max_derivative_order=2)

deriv_id = mocax.get_derivative_id([0, 0, ...])  # Function value
value = mocax.eval(point, deriv_id)

deriv_id = mocax.get_derivative_id([1, 0, ...])  # First derivative
derivative = mocax.eval(point, deriv_id)

del mocax
```

### Common Pitfalls

❌ Forgetting `LD_LIBRARY_PATH` → "cannot open shared object"
❌ Not adding `mocax_lib` to `sys.path` → "ModuleNotFoundError"
❌ Using `additional_data` without ctypes → Segmentation fault
❌ Extrapolating outside domain → Unreliable results
❌ Too few nodes → High error (use n ≥ 11 per dimension)
❌ Too many nodes → Curse of dimensionality (use Tensor Train for d > 6)

---

## Support and Contributing

### Getting Help

1. **Documentation**: Check the PDF manual first
2. **Examples**: Study the example scripts
3. **Test suite**: Run `mocax_test.py` to see working code
4. **Website**: mocaxintelligence.org for updates

### Reporting Issues

When reporting problems, include:
- MoCaX version (`mocaxpy.get_version_id()`)
- Python version (`python3 --version`)
- OS and architecture (`uname -a`)
- Minimal reproducible example
- Error messages and stack traces

### Citation

If using MoCaX in academic work:

```bibtex
@book{ruiz2020machine,
  title={Machine Learning for Risk Calculations: A Practitioner's View},
  author={Ruiz, Ignacio and Zeron, Mariano},
  year={2020},
  publisher={Wiley}
}
```

---

**Last Updated**: 2025-10-22
**Guide Version**: 1.0
**MoCaX Version**: 4.3.1

For questions or suggestions, refer to the MoCaX Intelligence website or the textbook.
